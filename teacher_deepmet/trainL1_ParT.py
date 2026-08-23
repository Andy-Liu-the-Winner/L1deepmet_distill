"""L1-ParT teacher training (see teacher_ParT_DESIGN.md).

python trainL1_ParT.py --data ../data/data4L1/data_ttbar/shards_v2 \
                     --ckpts ../teacher_ckpts_ParT --arch part

Prerequisite: run repack_L1_data.py once (sbatch repack_L1_job.slurm).
"""
import argparse
import json
import math
import os
import os.path as osp
import time
from time import strftime, gmtime

import numpy as np
import torch

import utils
import model.net as net
from model.part_met_network import PartMETNetwork
from model.graphnet_met_network import GraphMETTeacher
from model.shard_loader import ShardLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../data/data4L1/data_ttbar/shards_v2')
parser.add_argument('--ckpts', default='../teacher_ckpts_ParT')
parser.add_argument('--arch', default='part', choices=['part', 'graph'])
parser.add_argument('--restore_file', default=None)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--mem_fraction', type=float, default=0.18,
                    help="hard cap on GPU memory as fraction of device total; keeps the "
                         "allocator recycling below the MPS fair-share limit")
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--min_lr', type=float, default=3e-5)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--lam', type=float, default=5.0)
parser.add_argument('--sigma0', type=float, default=30.0)
parser.add_argument('--ema', type=float, default=0.999)
parser.add_argument('--clip', type=float, default=1.0)


def build_model(arch):
    if arch == 'part':
        return PartMETNetwork()
    return GraphMETTeacher()


def puppi_forward(x_pad, mask, y=None):
    """The zero-knowledge baseline: w = puppiWeight, s = 1."""
    w = x_pad[..., 5] * mask
    s = torch.ones(x_pad.shape[0], device=x_pad.device)
    return w, s


def evaluate_teacher(forward_fn, loader, device, lam, sigma0):
    """Loss components + resolution histograms, binned exactly as v1 evaluate.py so the
    saved .resolutions files work with the existing plotting machinery."""
    loss_arr, res_arr, resp_arr = [], [], []
    u_perp_all, u_par_all, R_all, qT_all = [], [], [], []
    with torch.no_grad():
        for x_pad, mask, y in loader:
            w, s = forward_fn(x_pad, mask)
            loss, res_t, resp_t = net.loss_fn_huber_response(w, s, x_pad, mask, y,
                                                 sigma0=sigma0, lam=lam)
            loss_arr.append(loss.item())
            res_arr.append(res_t.item())
            resp_arr.append(resp_t.item())

            # same math as v1 net.resolution(): vector = -MET, v_qT = truth
            METx = s * (w * x_pad[..., 1]).sum(1)
            METy = s * (w * x_pad[..., 2]).sum(1)
            vx, vy = -METx, -METy
            qx, qy = y[:, 0], y[:, 1]
            qt2 = (qx ** 2 + qy ** 2).clamp(min=1e-12)
            response = (vx * qx + vy * qy) / qt2
            v_par_x, v_par_y = response * qx, response * qy
            u_par = torch.sqrt(v_par_x ** 2 + v_par_y ** 2) - torch.sqrt(qt2)
            u_perp = torch.sqrt((vx - v_par_x) ** 2 + (vy - v_par_y) ** 2)
            u_perp_all.append(u_perp.cpu().numpy())
            u_par_all.append(u_par.cpu().numpy())
            R_all.append(response.cpu().numpy())
            qT_all.append(torch.sqrt(qt2).cpu().numpy())

    u_perp_arr = np.concatenate(u_perp_all)
    u_par_arr = np.concatenate(u_par_all)
    R_arr = np.concatenate(R_all)
    qT_arr = np.concatenate(qT_all)

    # binning copied verbatim from v1 evaluate.py
    max_x = 400
    x_n = 40
    bin_edges = np.arange(0, max_x, 10)
    inds = np.digitize(qT_arr, bin_edges)
    qT_hist = []
    for i in range(1, len(bin_edges)):
        qT_hist.append((bin_edges[i] + bin_edges[i - 1]) / 2.)

    u_perp_hist, u_perp_scaled_hist = [], []
    u_par_hist, u_par_scaled_hist, R_hist = [], [], []
    for i in range(1, len(bin_edges)):
        R_i = R_arr[np.where(inds == i)[0]]
        R_hist.append(np.mean(R_i) if len(R_i) > 0 else 0)
        u_perp_i = u_perp_arr[np.where(inds == i)[0]]
        u_perp_scaled_i = u_perp_i / np.mean(R_i) if len(R_i) > 0 else u_perp_i
        u_perp_hist.append((np.quantile(u_perp_i, 0.84) - np.quantile(u_perp_i, 0.16)) / 2.
                           if len(u_perp_i) > 0 else 0)
        u_perp_scaled_hist.append((np.quantile(u_perp_scaled_i, 0.84)
                                   - np.quantile(u_perp_scaled_i, 0.16)) / 2.
                                  if len(u_perp_scaled_i) > 0 else 0)
        u_par_i = u_par_arr[np.where(inds == i)[0]]
        u_par_scaled_i = u_par_i / np.mean(R_i) if len(R_i) > 0 else u_par_i
        u_par_hist.append((np.quantile(u_par_i, 0.84) - np.quantile(u_par_i, 0.16)) / 2.
                          if len(u_par_i) > 0 else 0)
        u_par_scaled_hist.append((np.quantile(u_par_scaled_i, 0.84)
                                  - np.quantile(u_par_scaled_i, 0.16)) / 2.
                                 if len(u_par_scaled_i) > 0 else 0)

    resolution_hists = {'MET': {
        'u_perp_resolution': np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=u_perp_hist),
        'u_perp_scaled_resolution': np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=u_perp_scaled_hist),
        'u_par_resolution': np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=u_par_hist),
        'u_par_scaled_resolution': np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=u_par_scaled_hist),
        'R': np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=R_hist),
    }}
    metrics_mean = {
        'loss': float(np.mean(loss_arr)),
        'res_term': float(np.mean(res_arr)),
        'resp_term': float(np.mean(resp_arr)),
        'response_plateau_50_300': float(np.mean([r for c, r in zip(qT_hist, R_hist)
                                                  if 50 <= c <= 300])),
    }
    print('- Eval: ' + ' ; '.join('%s: %.4f' % (k, v) for k, v in metrics_mean.items()),
          flush=True)
    return metrics_mean, resolution_hists


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

    model = build_model(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('arch=%s params=%d' % (args.arch, n_params))

    # ---- step-0 sanity gate: zero-init heads must reproduce the PUPPI baseline ----
    model.eval()
    x_pad, mask, y = next(iter(val_dl))
    with torch.no_grad():
        w, s = model(x_pad, mask)
    wp, sp = puppi_forward(x_pad, mask)
    assert torch.allclose(w, wp, atol=1e-5), 'step-0 weights != puppiWeight — wiring bug'
    assert torch.allclose(s, sp, atol=1e-6), 'step-0 scale != 1 — wiring bug'
    print('step-0 sanity check passed: untrained model == PUPPI baseline')
    val_dl.epoch = 0  # reset (no-op for unshuffled, but keep deterministic)

    # ---- PUPPI baseline resolutions, once, for the comparison plots ----
    baseline_file = osp.join(args.ckpts, 'puppi_baseline.resolutions')
    if not osp.exists(baseline_file):
        print('computing PUPPI baseline on val...')
        base_metrics, base_res = evaluate_teacher(puppi_forward, val_dl, device,
                                             args.lam, args.sigma0)
        utils.save(base_res, baseline_file)
        utils.save_dict_to_json(base_metrics, osp.join(args.ckpts, 'metrics_puppi_baseline.json'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch  # 1 epoch
    floor = args.min_lr / args.lr

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema_model = build_model(args.arch).to(device)
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
        loss_log.write('# %s teacher training started %s, params=%d, lam=%g sigma0=%g\n'
                       % (args.arch, strftime('%Y-%m-%d %H:%M:%S', gmtime()), n_params,
                          args.lam, args.sigma0))
        loss_log.write('epoch,train_loss,train_res,train_resp,val_loss,val_res,val_resp\n')
        loss_log.flush()

    for epoch in range(first_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss, tr_res, tr_resp = [], [], []
        for x_pad, mask, y in train_dl:
            optimizer.zero_grad()
            w, s = model(x_pad, mask)
            loss, res_t, resp_t = net.loss_fn_huber_response(w, s, x_pad, mask, y,
                                                 sigma0=args.sigma0, lam=args.lam)
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
            tr_res.append(res_t.item())
            tr_resp.append(resp_t.item())
        mem = (' gpu %.1f/%.1fGB' % (torch.cuda.max_memory_allocated() / 2 ** 30,
                                     torch.cuda.max_memory_reserved() / 2 ** 30)
               if device.type == 'cuda' else '')
        print('epoch %02d train loss %.4f (res %.4f resp %.4f) lr %.2e [%.0fs]%s' %
              (epoch, np.mean(tr_loss), np.mean(tr_res), np.mean(tr_resp),
               optimizer.param_groups[0]['lr'], time.time() - t0, mem), flush=True)

        ema_model.eval()
        val_metrics, resolutions = evaluate_teacher(ema_model, val_dl, device,
                                               args.lam, args.sigma0)
        loss_log.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' %
                       (epoch, np.mean(tr_loss), np.mean(tr_res), np.mean(tr_resp),
                        val_metrics['loss'], val_metrics['res_term'],
                        val_metrics['resp_term']))
        loss_log.flush()

        state = {'epoch': epoch,
                 'state_dict': ema_model.state_dict(),  # EMA = the eval/deploy copy
                 'raw_state_dict': model.state_dict(),
                 'optim_dict': optimizer.state_dict(),
                 'sched_dict': scheduler.state_dict(),
                 'arch': args.arch}
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
