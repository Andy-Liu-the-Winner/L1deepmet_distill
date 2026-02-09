import sys
sys.path.append('/home/export/xuantinl/L1deepmet_distill/teacher_deepmet')
from utils import load
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_ckpts', default='../teacher_ckpts_L1_20260128',
                    help="Teacher checkpoints folder")
parser.add_argument('--student_ckpts', default='../student_ckpts_L1_20260128',
                    help="Student checkpoints folder")
parser.add_argument('--output', default='.', help="Output directory for plots")

args = parser.parse_args()

# Plot training loss curves
def plot_loss(ckpts_dir, label, color, output_dir):
    loss_file = os.path.join(ckpts_dir, 'loss.log')
    if not os.path.exists(loss_file):
        print(f"Loss file not found: {loss_file}")
        return None, None

    epochs, train_loss, val_loss = [], [], []
    with open(loss_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('epoch'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 3:
                epochs.append(int(parts[0]))
                train_loss.append(float(parts[1]))
                val_loss.append(float(parts[2]))
    return epochs, train_loss, val_loss

# Plot loss curves
plt.figure(figsize=(10, 6))

teacher_data = plot_loss(args.teacher_ckpts, 'Teacher', 'blue', args.output)
student_data = plot_loss(args.student_ckpts, 'Student', 'red', args.output)

if teacher_data[0]:
    plt.plot(teacher_data[0], teacher_data[1], 'b-', label='Teacher Train', linewidth=2)
    plt.plot(teacher_data[0], teacher_data[2], 'b--', label='Teacher Val', linewidth=2)

if student_data[0]:
    plt.plot(student_data[0], student_data[1], 'r-', label='Student Train', linewidth=2)
    plt.plot(student_data[0], student_data[2], 'r--', label='Student Val', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves - L1 DeepMET')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(args.output, 'loss_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {args.output}/loss_curves.png")

# Plot resolution comparisons if available
def plot_resolutions(ckpts_dir, restore_file, label, color, linestyle='-'):
    res_file = os.path.join(ckpts_dir, f'{restore_file}.resolutions')
    if not os.path.exists(res_file):
        print(f"Resolution file not found: {res_file}")
        return None
    return load(res_file)

# Try to load resolution data
teacher_res = plot_resolutions(args.teacher_ckpts, 'best', 'Teacher', 'blue')
student_res = plot_resolutions(args.student_ckpts, 'best', 'Student', 'red')

if teacher_res or student_res:
    # Resolution perpendicular
    plt.figure(figsize=(10, 6))
    if teacher_res and 'MET' in teacher_res:
        xx = teacher_res['MET']['u_perp_resolution'][1][0:40]
        yy = teacher_res['MET']['u_perp_resolution'][0][0:40]
        plt.plot(xx, yy, 'b-', label='Teacher', linewidth=2)
    if student_res and 'MET' in student_res:
        xx = student_res['MET']['u_perp_resolution'][1][0:40]
        yy = student_res['MET']['u_perp_resolution'][0][0:40]
        plt.plot(xx, yy, 'r-', label='Student', linewidth=2)
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\perp})$ [GeV]')
    plt.title('Perpendicular Resolution - L1 DeepMET')
    plt.legend()
    plt.axis([0, 400, 0, 35])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output, 'resolution_perp.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}/resolution_perp.png")

    # Resolution parallel
    plt.figure(figsize=(10, 6))
    if teacher_res and 'MET' in teacher_res:
        xx = teacher_res['MET']['u_par_resolution'][1][0:40]
        yy = teacher_res['MET']['u_par_resolution'][0][0:40]
        plt.plot(xx, yy, 'b-', label='Teacher', linewidth=2)
    if student_res and 'MET' in student_res:
        xx = student_res['MET']['u_par_resolution'][1][0:40]
        yy = student_res['MET']['u_par_resolution'][0][0:40]
        plt.plot(xx, yy, 'r-', label='Student', linewidth=2)
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\parallel})$ [GeV]')
    plt.title('Parallel Resolution - L1 DeepMET')
    plt.legend()
    plt.axis([0, 400, 0, 60])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output, 'resolution_parallel.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}/resolution_parallel.png")

    # Response
    plt.figure(figsize=(10, 6))
    if teacher_res and 'MET' in teacher_res:
        xx = teacher_res['MET']['R'][1][0:40]
        yy = teacher_res['MET']['R'][0][0:40]
        plt.plot(xx, yy, 'b-', label='Teacher', linewidth=2)
    if student_res and 'MET' in student_res:
        xx = student_res['MET']['R'][1][0:40]
        yy = student_res['MET']['R'][0][0:40]
        plt.plot(xx, yy, 'r-', label='Student', linewidth=2)
    plt.axhline(y=1.0, color='black', linestyle='-.', label='Ideal')
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Response')
    plt.title('Response - L1 DeepMET')
    plt.legend()
    plt.axis([0, 400, 0, 1.2])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output, 'response.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}/response.png")

print("\nPlotting complete!")
