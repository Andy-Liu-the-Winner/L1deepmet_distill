"""
Compare Teacher T vs T' (VAE-bridged) on L1 test data.
Both take identical input (6 cont, 2 cat — no PV).
Measures how well T' approximates T's output distribution.

Run from vae_teacher/:
    python compare_teachers.py --data ../data/data4L1/data_ttbar
"""
import argparse, os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph

# ── paths ────────────────────────────────────────────────────────────────────
VAEROOT = osp.dirname(__file__)
sys.path.insert(0, osp.join(VAEROOT, '../student_deepmet'))

import model.net as student_net
import model.data_loader as data_loader
from evaluate import evaluate

from model.teacher_prime import TeacherPrime
import teacher_model.net as _tn

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',         default='../data/data4L1/data_ttbar')
parser.add_argument('--teacher_ckpt', default='../teacher_ckpts_L1_fixed/best.pth.tar')
parser.add_argument('--vae_ckpt',     default='../teacher_prime_ckpts/best_vae.pth.tar')
parser.add_argument('--out_dir',      default='../teacher_comparison_results')
parser.add_argument('--batch_size',   type=int, default=40)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scale_momentum = 128
norm = torch.tensor([1./scale_momentum]*3 + [1., 1., 1.]).to(device)

deltaR, deltaR_dz = 0.4, 0.3
loss_fn = student_net.loss_fn_response_tune
metrics = student_net.metrics

# ── thin wrapper: T' returns (weights, mu, logvar) — strip to single tensor ─
class TeacherPrimeEval(nn.Module):
    def __init__(self, tp): super().__init__(); self.tp = tp
    def forward(self, x_cont, x_cat, edge_index, batch):
        out, _, _ = self.tp(x_cont, x_cat, edge_index, batch)
        return out

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
dataloaders = data_loader.fetch_dataloader(data_dir=args.data,
                                           batch_size=args.batch_size,
                                           validation_split=0.2)
test_dl = dataloaders['test']

# ── load T ────────────────────────────────────────────────────────────────────
print("Loading Teacher T...")
teacher = _tn.Net(6, 2, norm).to(device)
ckpt_t  = torch.load(args.teacher_ckpt, map_location=device)
teacher.load_state_dict(ckpt_t['state_dict'])
teacher.eval()

# ── load T' ───────────────────────────────────────────────────────────────────
print("Loading Teacher T'...")
tp_raw = TeacherPrime.from_checkpoints(args.teacher_ckpt, args.vae_ckpt, device, norm=norm)
tp_raw.eval()
tprime = TeacherPrimeEval(tp_raw).to(device)

# ── evaluate ──────────────────────────────────────────────────────────────────
print("Evaluating T...")
_, res_T = evaluate(teacher, device, loss_fn, test_dl, metrics,
                    deltaR, deltaR_dz, args.out_dir)

print("Evaluating T'...")
_, res_Tp = evaluate(tprime, device, loss_fn, test_dl, metrics,
                     deltaR, deltaR_dz, args.out_dir)

# ── output agreement metric ───────────────────────────────────────────────────
print("\nComputing output agreement (T vs T' per-particle weights)...")
mse_list, mae_list = [], []
teacher.eval(); tprime.eval()
with torch.no_grad():
    for data in test_dl:
        data = data.to(device)
        x_cont = data.x[:, :6]
        x_cat  = torch.cat([data.x[:, 6:7].long(), data.x[:, 7:8].long()], dim=1)
        phi     = torch.atan2(data.x[:, 2], data.x[:, 1])
        etaphi  = torch.cat([data.x[:, 3:4], phi[:, None]], dim=1)
        ei      = to_undirected(radius_graph(etaphi, r=deltaR, batch=data.batch,
                                             loop=False, max_num_neighbors=255))
        t_out   = teacher(x_cont, x_cat, ei, data.batch)
        tp_out  = tprime(x_cont, x_cat, ei, data.batch)
        mse_list.append(torch.mean((t_out - tp_out)**2).item())
        mae_list.append(torch.mean(torch.abs(t_out - tp_out)).item())

print(f"  MSE(T, T') on weights : {np.mean(mse_list):.6f}")
print(f"  MAE(T, T') on weights : {np.mean(mae_list):.6f}")

# ── plots ─────────────────────────────────────────────────────────────────────
def get_bin_centers(hist):
    e = hist[1]; return (e[:-1] + e[1:]) / 2

metrics_to_plot = {
    'u_perp_resolution':        r'$\sigma(u_\perp)$ [GeV]',
    'u_par_resolution':         r'$\sigma(u_\parallel)$ [GeV]',
    'u_perp_scaled_resolution': r'$\sigma(u_\perp)/R$',
    'u_par_scaled_resolution':  r'$\sigma(u_\parallel)/R$',
    'R':                        r'Response $R$',
}

for key, ylabel in metrics_to_plot.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, res, color in [
        ('Teacher T',   res_T,  'steelblue'),
        ("Teacher T'",  res_Tp, 'tomato'),
    ]:
        h  = res['MET'][key]
        xc = get_bin_centers(h)
        ax.plot(xc, h[0], label=label, color=color, linewidth=2)
    ax.set_xlabel(r'$q_T$ [GeV]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 400)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"T vs T' Comparison — {key}")
    fname = osp.join(args.out_dir, f'compare_{key}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")

print(f"\nAll plots saved to {args.out_dir}/")
