"""Compare teacher v1 / PUPPI baseline / ParT / GraphNet teachers on the shared validation split.

python compare_all_teachers.py \
    --v1_ckpts ../teacher_ckpts_L1_fixed \
    --part_ckpts ../teacher_ckpts_ParT \
    --output ../teacher_comparison_new

Reads the .resolutions files (v1 format) and overlays response, sigma(u_perp),
sigma(u_par). Add --graphnet_ckpts once the graph variant is trained.
"""
import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load

parser = argparse.ArgumentParser()
parser.add_argument('--v1_ckpts', default='../teacher_ckpts_L1_fixed')
parser.add_argument('--part_ckpts', default='../teacher_ckpts_ParT')
parser.add_argument('--graphnet_ckpts', default=None)
parser.add_argument('--output', default='../teacher_comparison_new')


def curves():
    args = parser.parse_args()
    out = []
    v1 = osp.join(args.v1_ckpts, 'best.resolutions')
    if osp.exists(v1):
        out.append(('Teacher v1 (EdgeConv)', 'tab:blue', '-', load(v1)))
    base = osp.join(args.part_ckpts, 'puppi_baseline.resolutions')
    if osp.exists(base):
        out.append(('PUPPI baseline', 'tab:gray', '--', load(base)))
    part = osp.join(args.part_ckpts, 'best.resolutions')
    if osp.exists(part):
        out.append(('Teacher ParT', 'tab:red', '-', load(part)))
    if args.graphnet_ckpts:
        graphnet = osp.join(args.graphnet_ckpts, 'best.resolutions')
        if osp.exists(graphnet):
            out.append(('Teacher GraphNet', 'tab:green', '-', load(graphnet)))
    return args, out


def plot(args, entries, key, ylabel, title, fname, ylim, hline=None):
    plt.figure(figsize=(10, 6))
    for label, color, ls, res in entries:
        yy, xx = res['MET'][key][0][0:40], res['MET'][key][1][0:40]
        plt.plot(xx, yy, color=color, linestyle=ls, label=label, linewidth=2)
    if hline is not None:
        plt.axhline(y=hline, color='black', linestyle='-.', linewidth=1, label='Ideal')
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.axis([0, 400] + ylim)
    plt.grid(True, alpha=0.3)
    path = osp.join(args.output, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('saved', path)


def main():
    args, entries = curves()
    assert entries, 'no .resolutions files found'
    os.makedirs(args.output, exist_ok=True)
    plot(args, entries, 'R', 'Response', 'MET Response — teacher comparison',
         'compare_response.png', [0, 1.3], hline=1.0)
    plot(args, entries, 'u_perp_resolution', r'$\sigma(u_{\perp})$ [GeV]',
         'Perpendicular resolution — teacher comparison',
         'compare_u_perp.png', [0, 40])
    plot(args, entries, 'u_par_resolution', r'$\sigma(u_{\parallel})$ [GeV]',
         'Parallel resolution — teacher comparison',
         'compare_u_par.png', [0, 60])


if __name__ == '__main__':
    main()
