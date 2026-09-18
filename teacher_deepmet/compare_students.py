"""Compare the distilled / from-scratch EdgeConv students against PUPPI, v1 and
the ParT teacher (the distillation ideal) on the shared validation split.

python compare_students.py --output ../student_comparison_results

Plots the SCALED resolutions (resolution / response) — the calibration-fair
metric — plus the response curve.
"""
import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='../student_comparison_results')

ENTRIES = [
    ('PUPPI baseline',            '../teacher_ckpts_ParT_r2/puppi_baseline.resolutions', 'tab:gray',   '--', 1.5),
    ('Teacher v1 (6.4k)',         '../teacher_ckpts_L1_fixed/best.resolutions',          'tab:blue',   '-',  1.5),
    ('Student scratch (6.4k)',    '../student_ckpts_scratch_r1/best.resolutions',        'tab:orange', '--', 2.0),
    ('Student distilled (6.4k)',  '../student_ckpts_distill_ParT_r1/best.resolutions',   'tab:purple', '-',  2.5),
    ('Teacher ParT (138k, ideal)','../teacher_ckpts_ParT_r2/best.resolutions',           'tab:red',    '-',  2.0),
]


def plot(args, entries, key, ylabel, title, fname, ylim, hline=None):
    plt.figure(figsize=(10, 6))
    for label, color, ls, lw, res in entries:
        yy, xx = res['MET'][key][0][0:40], res['MET'][key][1][0:40]
        plt.plot(xx, yy, color=color, linestyle=ls, label=label, linewidth=lw)
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
    args = parser.parse_args()
    entries = [(lbl, c, ls, lw, load(p)) for lbl, p, c, ls, lw in ENTRIES if osp.exists(p)]
    assert entries, 'no .resolutions files found'
    os.makedirs(args.output, exist_ok=True)
    plot(args, entries, 'R', 'Response', 'MET response — student vs teachers',
         'student_response.png', [0, 1.3], hline=1.0)
    plot(args, entries, 'u_perp_scaled_resolution',
         r'scaled $\sigma(u_{\perp})$ [GeV]',
         'Scaled perpendicular resolution — student vs teachers',
         'student_u_perp_scaled.png', [0, 45])
    plot(args, entries, 'u_par_scaled_resolution',
         r'scaled $\sigma(u_{\parallel})$ [GeV]',
         'Scaled parallel resolution — student vs teachers',
         'student_u_par_scaled.png', [0, 90])


if __name__ == '__main__':
    main()
