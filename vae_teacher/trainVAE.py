"""
Phase 2 — Train VAEEncoder so that T'(S_input) ≈ T(T_input).
=============================================================
The VAEEncoder learns to produce a latent z (from S-format input) that, when
passed through Teacher T's frozen conv/output layers, yields the same per-particle
weights as Teacher T receiving its own (full-PV) input.

Loss
----
  L_pred  = MSE( T'_output,  T_output )        end-to-end output matching
  L_kl    = β * KL( q(z|S) || N(0,I) )         latent regularisation
  L_task  = γ * loss_fn_response_tune(T'_out)  optional physics task loss

  L_total = L_pred + L_kl + L_task

Typical β = 0.001, γ = 0.1  (keep L_pred dominant).

Checkpoints
-----------
Saves {'vae_state_dict': ..., 'optim_dict': ..., 'epoch': ...}
to  ../teacher_prime_ckpts/best_vae.pth.tar

Usage
-----
    cd vae_teacher/
    python trainVAE.py \
        --data        ../data/data4L1/data_ttbar \
        --teacher_ckpt ../teacher_ckpts_L1_withPV/best.pth.tar \
        --ckpts        ../teacher_prime_ckpts
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # vae_teacher/ (local model/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../teacher_deepmet'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../student_deepmet'))

import argparse
import numpy as np
import warnings
warnings.simplefilter('ignore')
from time import strftime, gmtime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph
from tqdm import tqdm

import utils
import model.data_loader as data_loader

import teacher_model.net as _teacher_net          # teacher_model/ copied from teacher_deepmet
TeacherNet = _teacher_net.Net                      # cat_dim=2, no PV
loss_fn_response_tune = _teacher_net.loss_fn_response_tune

from model.vae_encoder import VAEEncoder, kl_loss
from model.teacher_prime import TeacherPrime

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data',          default='../data/data4L1/data_ttbar')
parser.add_argument('--teacher_ckpt',  default='../teacher_ckpts_L1_withPV/best.pth.tar')
parser.add_argument('--ckpts',         default='../teacher_prime_ckpts')
parser.add_argument('--restore_file',  default=None)
parser.add_argument('--beta',  type=float, default=0.001,
                    help='KL divergence weight')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='Task loss (response-tune) weight')

scale_momentum = 128.
deltaR         = 0.4
deltaR_dz      = 0.3
max_epochs     = 10
batch_size     = 64


# ---------------------------------------------------------------------------
def extract_features_teacher(data, device):
    """Teacher T features: x_cat has 2 columns (pdgid, charge) — matches existing checkpoint."""
    x_cont = data.x[:, :6]
    pdgid  = data.x[:, 6:7].long()
    charge = data.x[:, 7:8].long()
    x_cat  = torch.cat([pdgid, charge], dim=1)
    return x_cont, x_cat


def extract_features_student(data):
    """Student S features: x_cat has 2 columns (pdgid, charge), no PV."""
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


# ---------------------------------------------------------------------------
def train_epoch(teacher_prime, teacher, device, optimizer, scheduler,
                beta, gamma, loss_fn_task, dataloader, epoch):
    teacher_prime.train()
    teacher.eval()

    loss_avg_arr  = []
    loss_avg      = utils.RunningAverage()
    pred_avg      = utils.RunningAverage()
    kl_avg        = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            x_cont_T, x_cat_T = extract_features_teacher(data, device)
            x_cont_S, x_cat_S = extract_features_student(data)
            edge_index         = build_edge_index(data, deltaR)

            # Teacher T forward (frozen, no grad)
            with torch.no_grad():
                t_out = teacher(x_cont_T, x_cat_T, edge_index, data.batch)

            # T' forward (VAEEncoder is trainable; T's layers are frozen)
            tp_out, mu, logvar = teacher_prime(
                x_cont_S, x_cat_S, edge_index, data.batch
            )

            # Losses
            l_pred = F.mse_loss(tp_out, t_out)
            l_kl   = kl_loss(mu, logvar)
            l_task = loss_fn_task(tp_out, data.x, data.y, data.batch)

            loss = l_pred + beta * l_kl + gamma * l_task

            loss.backward()
            optimizer.step()

            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            pred_avg.update(l_pred.item())
            kl_avg.update(l_kl.item())

            t.set_postfix(
                total='{:.3f}'.format(loss_avg()),
                pred='{:.3f}'.format(pred_avg()),
                kl='{:.4f}'.format(kl_avg()),
            )
            t.update()

    scheduler.step(np.mean(loss_avg_arr))
    mean_loss = np.mean(loss_avg_arr)
    print('Epoch {:02d}: total={:.4f}  pred={:.4f}  kl={:.4f}'.format(
        epoch, mean_loss, pred_avg(), kl_avg()
    ))
    return mean_loss


# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(teacher_prime, teacher, device, beta, gamma, loss_fn_task, dataloader):
    teacher_prime.eval()
    teacher.eval()

    losses = []
    for data in dataloader:
        data = data.to(device)

        x_cont_T, x_cat_T = extract_features_teacher(data, device)
        x_cont_S, x_cat_S = extract_features_student(data)
        edge_index         = build_edge_index(data, deltaR)

        t_out               = teacher(x_cont_T, x_cat_T, edge_index, data.batch)
        tp_out, mu, logvar  = teacher_prime(x_cont_S, x_cat_S, edge_index, data.batch)

        l_pred = F.mse_loss(tp_out, t_out)
        l_kl   = kl_loss(mu, logvar)
        l_task = loss_fn_task(tp_out, data.x, data.y, data.batch)

        losses.append((l_pred + beta * l_kl + gamma * l_task).item())

    return float(np.mean(losses))


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

    # ----- Load Teacher T (frozen) -----
    # Existing checkpoints were trained with cat_dim=2 (no PV) — use teacher_deepmet Net
    teacher = TeacherNet(6, 2, norm).to(device)
    ckpt_t  = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt_t['state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print('Teacher T loaded from', args.teacher_ckpt)
    print('Teacher params:', sum(p.numel() for p in teacher.parameters()))

    # ----- Build VAEEncoder + TeacherPrime -----
    vae_enc      = VAEEncoder(continuous_dim=6, norm=norm, hidden_dim=32).to(device)
    teacher_prime = TeacherPrime(vae_enc, teacher.graphnet).to(device)

    vae_params = sum(p.numel() for p in vae_enc.parameters())
    print('VAEEncoder params:', vae_params)

    # Only the VAEEncoder is trainable
    optimizer = torch.optim.AdamW(vae_enc.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False
    )

    first_epoch          = 0
    best_validation_loss = 1e7

    if args.restore_file is not None:
        restore_path = os.path.join(args.ckpts, args.restore_file + '_vae.pth.tar')
        ckpt_r = torch.load(restore_path, map_location=device)
        vae_enc.load_state_dict(ckpt_r['vae_state_dict'])
        optimizer.load_state_dict(ckpt_r['optim_dict'])
        first_epoch = ckpt_r['epoch']
        if 'best_val_loss' in ckpt_r:
            best_validation_loss = ckpt_r['best_val_loss']
        print('Resumed from epoch', first_epoch)

    if first_epoch == 0:
        loss_log = open(os.path.join(args.ckpts, 'vae_loss.log'), 'w')
        loss_log.write('# VAE training  β={} γ={}  '.format(args.beta, args.gamma) +
                       strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        loss_log.write('epoch,train_loss,val_loss\n')
    else:
        loss_log = open(os.path.join(args.ckpts, 'vae_loss.log'), 'a')

    loss_fn_task = loss_fn_response_tune

    for epoch in range(first_epoch + 1, max_epochs + 1):
        train_loss = train_epoch(
            teacher_prime, teacher, device, optimizer, scheduler,
            args.beta, args.gamma, loss_fn_task, train_dl, epoch
        )
        val_loss = validate(
            teacher_prime, teacher, device,
            args.beta, args.gamma, loss_fn_task, test_dl
        )

        print('  Val loss: {:.4f}'.format(val_loss))

        # Save latest checkpoint
        ckpt_payload = {
            'epoch':          epoch,
            'vae_state_dict': vae_enc.state_dict(),
            'optim_dict':     optimizer.state_dict(),
            'best_val_loss':  best_validation_loss,
        }
        torch.save(ckpt_payload,
                   os.path.join(args.ckpts, 'last_vae.pth.tar'))

        loss_log.write('{:d},{:.4f},{:.4f}\n'.format(epoch, train_loss, val_loss))
        loss_log.flush()

        is_best = val_loss <= best_validation_loss
        if is_best:
            print('  ✓ New best VAE val loss: {:.4f}'.format(val_loss))
            best_validation_loss = val_loss
            ckpt_payload['best_val_loss'] = best_validation_loss
            torch.save(ckpt_payload,
                       os.path.join(args.ckpts, 'best_vae.pth.tar'))

    loss_log.close()
    print('Done. Best VAE checkpoint:', os.path.join(args.ckpts, 'best_vae.pth.tar'))
    print()
    print('Next step — run trainDistillTPrime.py to distill T\' → S.')
