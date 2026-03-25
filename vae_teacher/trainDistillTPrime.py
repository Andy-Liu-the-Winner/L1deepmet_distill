"""
Phase 4 — Distil T' → Student S.
==================================
T' (TeacherPrime) is fully frozen.  The Student S is trained with:

  L = α * task_loss(S_out, data.x, data.y, batch)
    + (1-α) * MSE(S_out, T'_out)

where T'_out is the per-particle weight prediction from the VAE-bridged teacher,
computed on the same S-format input that S sees.  This removes the input-dimension
mismatch that prevented using Teacher T directly as the distillation supervisor.

Checkpoints saved to: ../student_ckpts_tprime/

Usage
-----
    cd vae_teacher/
    python trainDistillTPrime.py \
        --data          ../data/data4L1/data_ttbar \
        --teacher_ckpt  ../teacher_ckpts_L1_withPV/best.pth.tar \
        --vae_ckpt      ../teacher_prime_ckpts/best_vae.pth.tar \
        --ckpts         ../student_ckpts_tprime
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # vae_teacher/ (local model/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../teacher_deepmet'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../student_deepmet'))

import json
import argparse
import numpy as np
import warnings
warnings.simplefilter('ignore')
from time import strftime, gmtime

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph
from tqdm import tqdm

import utils
import model.data_loader as data_loader
from model.net import StudentNet, loss_fn_response_tune, metrics  # student_deepmet
from evaluate import evaluate

import teacher_model.net as _teacher_net          # teacher_model/ copied from teacher_deepmet
TeacherNet = _teacher_net.Net                      # cat_dim=2, no PV

from model.vae_encoder import VAEEncoder
from model.teacher_prime import TeacherPrime

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data',         default='../data/data4L1/data_ttbar')
parser.add_argument('--teacher_ckpt', default='../teacher_ckpts_L1_withPV/best.pth.tar',
                    help='Phase-1 Teacher T checkpoint (with PV)')
parser.add_argument('--vae_ckpt',     default='../teacher_prime_ckpts/best_vae.pth.tar',
                    help='Phase-2 VAEEncoder checkpoint')
parser.add_argument('--ckpts',        default='../student_ckpts_tprime')
parser.add_argument('--restore_file', default=None)
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight on task loss (1-alpha goes to distillation MSE)')

scale_momentum = 128.
deltaR         = 0.4
deltaR_dz      = 0.3
max_epochs     = 5
batch_size     = 64


# ---------------------------------------------------------------------------
def extract_features_student(data):
    """S-format: x_cat = [pdgid, charge] — no PV."""
    x_cont = data.x[:, :6]
    pdgid  = data.x[:, 6:7].long()
    charge = data.x[:, 7:8].long()
    x_cat  = torch.cat([pdgid, charge], dim=1)
    return x_cont, x_cat


def build_edge_index(data, deltaR):
    phi      = torch.atan2(data.x[:, 2], data.x[:, 1])
    etaphi   = torch.cat([data.x[:, 3:4], phi[:, None]], dim=1)
    edge_idx = radius_graph(etaphi, r=deltaR, batch=data.batch,
                            loop=False, max_num_neighbors=255)
    return to_undirected(edge_idx)


def _model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


# ---------------------------------------------------------------------------
def train(student, teacher_prime, device, optimizer, scheduler,
          loss_fn, alpha, dataloader, epoch):
    student.train()
    teacher_prime.eval()

    loss_avg_arr = []
    loss_avg     = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            x_cont, x_cat = extract_features_student(data)
            edge_index     = build_edge_index(data, deltaR)

            # T' forward (fully frozen — no grad needed)
            with torch.no_grad():
                tp_out, _, _ = teacher_prime(x_cont, x_cat, edge_index, data.batch)

            # Student forward
            s_out = student(x_cont, x_cat, edge_index, data.batch)

            # Combined loss
            l_task   = loss_fn(s_out, data.x, data.y, data.batch)
            l_distil = F.mse_loss(s_out, tp_out)
            loss     = alpha * l_task + (1.0 - alpha) * l_distil

            loss.backward()
            optimizer.step()

            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    scheduler.step(np.mean(loss_avg_arr))
    mean_loss = np.mean(loss_avg_arr)
    print('Training epoch: {:02d}, loss: {:.4f}'.format(epoch, mean_loss))
    return mean_loss


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.ckpts, exist_ok=True)

    dataloaders = data_loader.fetch_dataloader(
        data_dir=args.data, batch_size=batch_size, validation_split=0.25
    )
    train_dl = dataloaders['train']
    test_dl  = dataloaders['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    norm = torch.tensor(
        [1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 1., 1., 1.]
    ).to(device)
    student_norm = norm.clone()

    # ----- Build and freeze T' -----
    teacher_net = TeacherNet(6, 2, norm).to(device)  # cat_dim=2, no PV
    ckpt_t = torch.load(args.teacher_ckpt, map_location=device)
    teacher_net.load_state_dict(ckpt_t['state_dict'])
    teacher_net.eval()

    vae_enc = VAEEncoder(continuous_dim=6, norm=norm, hidden_dim=32).to(device)
    ckpt_v  = torch.load(args.vae_ckpt, map_location=device)
    vae_enc.load_state_dict(ckpt_v['vae_state_dict'])

    teacher_prime = TeacherPrime(vae_enc, teacher_net.graphnet).to(device)
    teacher_prime.eval()
    for p in teacher_prime.parameters():
        p.requires_grad = False

    print('T\' loaded.  T\' total params:',
          sum(p.numel() for p in teacher_prime.parameters()))

    # ----- Build Student S -----
    student = StudentNet(6, 2, student_norm).to(device)

    t_params = sum(p.numel() for p in teacher_prime.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print('Student params: {:,d}  ({:.2%} of T\')'.format(s_params, s_params / t_params))
    print('Student size:   {:.3f} MB'.format(_model_size_mb(student)))

    with open(os.path.join(args.ckpts, 'model_size.txt'), 'w') as f:
        f.write('Teacher-prime params: {}\n'.format(t_params))
        f.write('Student params:       {}\n'.format(s_params))
        f.write('Relative size:        {:.4f}\n'.format(s_params / t_params))
        f.write('Student size MB:      {:.3f}\n'.format(_model_size_mb(student)))

    # ----- Optimiser -----
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False
    )

    first_epoch          = 0
    best_validation_loss = 1e7
    loss_fn              = loss_fn_response_tune

    if args.restore_file is not None:
        restore_ckpt = os.path.join(args.ckpts, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, student, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Resuming from epoch', first_epoch)
        with open(os.path.join(args.ckpts, 'metrics_val_best.json')) as f:
            best_validation_loss = json.load(f)['loss']

    if first_epoch == 0:
        loss_log = open(os.path.join(args.ckpts, 'loss.log'), 'w')
        loss_log.write('# T\'-distilled student  α={}  '.format(args.alpha) +
                       strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        loss_log.write('epoch,train_loss,val_loss\n')
    else:
        loss_log = open(os.path.join(args.ckpts, 'loss.log'), 'a')

    for epoch in range(first_epoch + 1, max_epochs + 1):
        train_loss = train(
            student, teacher_prime, device, optimizer, scheduler,
            loss_fn, args.alpha, train_dl, epoch
        )

        utils.save_checkpoint(
            {'epoch': epoch,
             'state_dict': student.state_dict(),
             'optim_dict': optimizer.state_dict(),
             'sched_dict': scheduler.state_dict()},
            is_best=False,
            checkpoint=args.ckpts,
        )

        test_metrics, resolutions = evaluate(
            student, device, loss_fn, test_dl, metrics, deltaR, deltaR_dz, args.ckpts
        )

        val_loss = test_metrics['loss']
        loss_log.write('{:d},{:.4f},{:.4f}\n'.format(epoch, train_loss, val_loss))
        loss_log.flush()

        is_best = val_loss <= best_validation_loss
        if is_best:
            print('  ✓ New best val loss: {:.4f}'.format(val_loss))
            best_validation_loss = val_loss
            utils.save_checkpoint(
                {'epoch': epoch,
                 'state_dict': student.state_dict(),
                 'optim_dict': optimizer.state_dict(),
                 'sched_dict': scheduler.state_dict()},
                is_best=True,
                checkpoint=args.ckpts,
            )
            utils.save_dict_to_json(
                test_metrics, os.path.join(args.ckpts, 'metrics_val_best.json')
            )
            utils.save(resolutions, os.path.join(args.ckpts, 'best.resolutions'))

        utils.save_dict_to_json(
            test_metrics, os.path.join(args.ckpts, 'metrics_val_last.json')
        )
        utils.save(resolutions, os.path.join(args.ckpts, 'last.resolutions'))

    loss_log.close()
    print('Done. Best student checkpoint:', os.path.join(args.ckpts, 'best.pth.tar'))
