"""
Compare original L1 student vs T'-distilled student on test data.
Run from vae_teacher/:
    python compare_students.py --data ../data/data4L1/data_ttbar
"""
import argparse, os, sys
import os.path as osp
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph

sys.path.insert(0, osp.join(osp.dirname(__file__), '../student_deepmet'))
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data',          default='../data/data4L1/data_ttbar')
parser.add_argument('--ckpt_orig',     default='../student_ckpts_L1_fixed/best.pth.tar',
                    help='Original L1 student checkpoint')
parser.add_argument('--ckpt_tprime',   default='../student_ckpts_tprime/best.pth.tar',
                    help='T\'-distilled student checkpoint')
parser.add_argument('--out_dir',       default='../student_comparison_results')
parser.add_argument('--batch_size',    type=int, default=40)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scale_momentum = 128

norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum,
                     1., 1., 1., 1., 1.]).to(device)
student_norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum,
                              1., 1., 1.]).to(device)

deltaR, deltaR_dz = 0.4, 0.3
loss_fn = net.loss_fn_response_tune
metrics = net.metrics

# ── data ─────────────────────────────────────────────────────────────────────
print("Loading data...")
dataloaders = data_loader.fetch_dataloader(data_dir=args.data,
                                           batch_size=args.batch_size,
                                           validation_split=0.2)
test_dl = dataloaders['test']

# ── helper to load StudentNet ────────────────────────────────────────────────
def load_student(ckpt_path):
    model = net.StudentNet(6, 2, student_norm).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model

# ── evaluate both ────────────────────────────────────────────────────────────
print("Evaluating original L1 student...")
model_orig   = load_student(args.ckpt_orig)
_, res_orig  = evaluate(model_orig, device, loss_fn, test_dl, metrics,
                        deltaR, deltaR_dz, args.out_dir)

print("Evaluating T'-distilled student...")
model_tprime  = load_student(args.ckpt_tprime)
_, res_tprime = evaluate(model_tprime, device, loss_fn, test_dl, metrics,
                         deltaR, deltaR_dz, args.out_dir)

# ── plot ─────────────────────────────────────────────────────────────────────
def get_bin_centers(hist):
    edges = hist[1]
    return (edges[:-1] + edges[1:]) / 2

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
        ('Original L1 student',    res_orig,   'steelblue'),
        ("T'-distilled student",   res_tprime, 'tomato'),
    ]:
        h = res['MET'][key]
        xc = get_bin_centers(h)
        ax.plot(xc, h[0], label=label, color=color, linewidth=2)
    ax.set_xlabel(r'$q_T$ [GeV]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 400)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'L1 Student Comparison — {key}')
    fname = osp.join(args.out_dir, f'compare_{key}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")

print(f"\nAll plots saved to {args.out_dir}/")
