"""Distill the ParT r2 teacher into the FPGA-compatible EdgeConv student.

Lives in teacher_deepmet to reuse the shard pipeline, r2 loss and training recipe;
the student architecture itself (StudentGraphMETNetwork) is loaded from
../student_deepmet/model/graph_met_network.py unchanged, so the deployed network
stays exactly the FlowGNN-compatible EdgeConv design.

Distillation run:
    python trainL1_student_distill.py --ckpts ../student_ckpts_distill_ParT_r1

From-scratch baseline (same student, same loss, no teacher):
    python trainL1_student_distill.py --ckpts ../student_ckpts_scratch_r1 --teacher_ckpt ''

Loss: loss_fn_huber_response (identical to the teacher r2 run)
      + beta * masked MSE(student w, teacher s*w).

Student head change vs the legacy pipeline: w = relu(puppiWeight + out) with the
final linear zero-initialized (same trick as both teachers) — on FPGA this is one
extra scalar add before the existing ReLU. Step 0 == PUPPI baseline exactly.
The graph is the same dR<0.4 radius graph but built with wrapped delta-phi, so
the +-pi seam is connected (the legacy radius_graph on raw phi was not).
"""
import argparse
import importlib.util
import json
import math
import os
import os.path as osp
import time
from time import strftime, gmtime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import model.net as net
from model.part_met_network import PartMETNetwork
from model.shard_loader import ShardLoader
from trainL1_ParT import evaluate_teacher, puppi_forward

_spec = importlib.util.spec_from_file_location(
    'student_graph_met_network',
    osp.join(osp.dirname(osp.abspath(__file__)),
             '..', 'student_deepmet', 'model', 'graph_met_network.py'))
_student_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_student_mod)
StudentGraphMETNetwork = _student_mod.StudentGraphMETNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../data/data4L1/data_ttbar/shards_v2')
parser.add_argument('--ckpts', default='../student_ckpts_distill_ParT_r1')
parser.add_argument('--teacher_ckpt', default='../teacher_ckpts_ParT_r2/best.pth.tar',
                    help="ParT teacher checkpoint; pass '' for the no-distillation baseline")
parser.add_argument('--beta', type=float, default=20.0,
                    help='weight of the distillation MSE term')
parser.add_argument('--restore_file', default=None)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--mem_fraction', type=float, default=0.18)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--min_lr', type=float, default=3e-5)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--lam', type=float, default=15.0)
parser.add_argument('--sigma0', type=float, default=30.0)
parser.add_argument('--ema', type=float, default=0.999)
parser.add_argument('--clip', type=float, default=1.0)

SCALE_MOMENTUM = 128.


class StudentNet(nn.Module):
    """Padded-batch adapter around the unchanged EdgeConv student.

    forward(x_pad, mask) -> (w (B,N), s (B,)) — same contract as the teachers,
    so the step-0 gate and evaluate_teacher() work unchanged. s is fixed to 1:
    the deployed student has no scale head; the teacher's s is folded into the
    distillation target via effective_weights().
    """

    def __init__(self, device, radius=0.4):
        super().__init__()
        self.radius = radius
        norm = torch.tensor([1. / SCALE_MOMENTUM, 1. / SCALE_MOMENTUM,
                             1. / SCALE_MOMENTUM, 1., 1., 1.], device=device)
        self.graphnet = StudentGraphMETNetwork(6, 2, norm, output_dim=1,
                                               hidden_dim=32, conv_depth=2)
        # zero-init the final linear: step 0 -> w = relu(puppi + 0) = puppi
        nn.init.zeros_(self.graphnet.output[-1].weight)
        nn.init.zeros_(self.graphnet.output[-1].bias)

    def build_graph(self, x_pad, mask):
        B, N, _ = x_pad.shape
        eta = x_pad[..., 3]
        phi = x_pad[..., 4]
        deta = eta.unsqueeze(2) - eta.unsqueeze(1)
        dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
        dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi
        adj = (deta ** 2 + dphi ** 2 < self.radius ** 2)
        adj &= mask.unsqueeze(2) & mask.unsqueeze(1)
        adj &= ~torch.eye(N, dtype=torch.bool, device=adj.device).unsqueeze(0)
        idx = torch.full((B, N), -1, dtype=torch.long, device=x_pad.device)
        idx[mask] = torch.arange(int(mask.sum()), device=x_pad.device)
        b, i, j = adj.nonzero(as_tuple=True)
        edge_index = torch.stack([idx[b, j], idx[b, i]])  # symmetric adj -> undirected
        batch_vec = torch.arange(B, device=x_pad.device).unsqueeze(1).expand(B, N)[mask]
        return edge_index, batch_vec

    def forward(self, x_pad, mask):
        B, N, _ = x_pad.shape
        edge_index, batch_vec = self.build_graph(x_pad, mask)
        flat = x_pad[mask]                       # (M, 8), fresh tensor
        x_cont = flat[:, :6].clone()             # graphnet multiplies in place
        x_cat = flat[:, 6:8].long()
        out = self.graphnet(x_cont, x_cat, edge_index, batch_vec)
        w_flat = F.relu(flat[:, 5] + out)
        w = x_pad.new_zeros(B, N)
        w[mask] = w_flat
        s = torch.ones(B, device=x_pad.device)
        return w, s


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    os.makedirs(args.ckpts, exist_ok=True)

    train_dl = ShardLoader(args.data, 'train', args.batch, shuffle=True, device=device)
    val_dl = ShardLoader(args.data, 'val', args.batch, shuffle=False, device=device)
    print('train events: %d (%d batches), val events: %d' %
          (train_dl.n_events, len(train_dl), val_dl.n_events))

    teacher = None
    if args.teacher_ckpt:
        teacher = PartMETNetwork().to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt['state_dict'])  # EMA weights
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print('teacher loaded from', args.teacher_ckpt)
    else:
        print('NO teacher: from-scratch baseline run')

    model = StudentNet(device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('student params=%d (teacher: %s)' %
          (n_params, sum(p.numel() for p in teacher.parameters()) if teacher else 'n/a'))

    # step-0 sanity gate: zero-init head must reproduce the PUPPI baseline
    model.eval()
    x_pad, mask, y = next(iter(val_dl))
    with torch.no_grad():
        w, s = model(x_pad, mask)
    wp, sp = puppi_forward(x_pad, mask)
    assert torch.allclose(w, wp, atol=1e-5), 'step-0 weights != puppiWeight — wiring bug'
    assert torch.allclose(s, sp, atol=1e-6), 'step-0 scale != 1 — wiring bug'
    print('step-0 sanity check passed: untrained student == PUPPI baseline')
    val_dl.epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch
    floor = args.min_lr / args.lr

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema_model = StudentNet(device).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    first_epoch = 0
    best_val = float('inf')
    if args.restore_file is not None:
        ckpt = torch.load(osp.join(args.ckpts, args.restore_file + '.pth.tar'),
                          map_location=device)
        model.load_state_dict(ckpt['raw_state_dict'])
        ema_model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optim_dict'])
        scheduler.load_state_dict(ckpt['sched_dict'])
        first_epoch = ckpt['epoch']
        train_dl.epoch = first_epoch
        with open(osp.join(args.ckpts, 'metrics_val_best.json')) as f:
            best_val = json.load(f)['loss']
        print('restored from epoch', first_epoch)

    log_path = osp.join(args.ckpts, 'loss.log')
    loss_log = open(log_path, 'a' if first_epoch > 0 else 'w')
    if first_epoch == 0:
        loss_log.write('# student %s started %s, params=%d, beta=%g lam=%g\n'
                       % ('distill' if teacher else 'scratch',
                          strftime('%Y-%m-%d %H:%M:%S', gmtime()), n_params, args.beta,
                          args.lam))
        loss_log.write('epoch,train_loss,train_task,train_distill,val_loss,val_res,val_resp\n')
        loss_log.flush()

    for epoch in range(first_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss, tr_task, tr_dist = [], [], []
        for x_pad, mask, y in train_dl:
            optimizer.zero_grad()
            w, s = model(x_pad, mask)
            task, res_t, resp_t = net.loss_fn_huber_response(w, s, x_pad, mask, y,
                                                 sigma0=args.sigma0, lam=args.lam)
            if teacher is not None:
                with torch.no_grad():
                    w_t = teacher.effective_weights(x_pad, mask)
                distill = ((w - w_t)[mask] ** 2).mean()
            else:
                distill = torch.zeros((), device=device)
            loss = task + args.beta * distill
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for pe, p in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(args.ema).add_(p, alpha=1 - args.ema)
                for be, b in zip(ema_model.buffers(), model.buffers()):
                    be.copy_(b)
            tr_loss.append(loss.item())
            tr_task.append(task.item())
            tr_dist.append(distill.item())
        mem = (' gpu %.1f/%.1fGB' % (torch.cuda.max_memory_allocated() / 2 ** 30,
                                     torch.cuda.max_memory_reserved() / 2 ** 30)
               if device.type == 'cuda' else '')
        print('epoch %02d train loss %.4f (task %.4f distill %.5f) lr %.2e [%.0fs]%s' %
              (epoch, np.mean(tr_loss), np.mean(tr_task), np.mean(tr_dist),
               optimizer.param_groups[0]['lr'], time.time() - t0, mem), flush=True)

        ema_model.eval()
        val_metrics, resolutions = evaluate_teacher(ema_model, val_dl, device,
                                               args.lam, args.sigma0)
        loss_log.write('%d,%.4f,%.4f,%.5f,%.4f,%.4f,%.4f\n' %
                       (epoch, np.mean(tr_loss), np.mean(tr_task), np.mean(tr_dist),
                        val_metrics['loss'], val_metrics['res_term'],
                        val_metrics['resp_term']))
        loss_log.flush()

        state = {'epoch': epoch,
                 'state_dict': ema_model.state_dict(),
                 'raw_state_dict': model.state_dict(),
                 'optim_dict': optimizer.state_dict(),
                 'sched_dict': scheduler.state_dict(),
                 'arch': 'student_edgeconv'}
        utils.save_checkpoint(state, is_best=False, checkpoint=args.ckpts)
        utils.save_dict_to_json(val_metrics, osp.join(args.ckpts, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(args.ckpts, 'last.resolutions'))
        if val_metrics['loss'] <= best_val:
            print('new best val loss')
            best_val = val_metrics['loss']
            utils.save_checkpoint(state, is_best=True, checkpoint=args.ckpts)
            utils.save_dict_to_json(val_metrics, osp.join(args.ckpts, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(args.ckpts, 'best.resolutions'))

    loss_log.close()
    print('training done')


if __name__ == '__main__':
    main()
