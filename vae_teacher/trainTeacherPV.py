"""
Phase 1 — Train full Teacher T with PV association enabled.
============================================================
The full Teacher T is GraphMETNetwork (defined in student_deepmet/model/) with
embed_pv active, taking cat_dim=3 (pdgid, charge, PV association).

Current L1 data has 8 columns: [pt, px, py, eta, d0, dz, pdgid, charge].
There is no PV column.  We append a synthetic pv_col = zeros (PV quality 0)
so the architecture can be validated and trained.  Replace with real offline
data containing PV association when available.

Usage
-----
    cd vae_teacher/
    python trainTeacherPV.py \
        --data  ../data/data4L1/data_ttbar \
        --ckpts ../teacher_ckpts_L1_withPV

Checkpoint saved to:  ../teacher_ckpts_L1_withPV/best.pth.tar
"""

import sys, os
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

import utils                          # teacher_deepmet/utils.py
import model.data_loader as data_loader   # teacher_deepmet/model/data_loader.py
from evaluate import evaluate

# Full teacher: GraphMETNetwork with embed_pv active (student_deepmet definition)
from model.net import Net as TeacherNet   # student_deepmet/model/net.py -> Net(6,3,norm)
from model.net import loss_fn_response_tune, metrics

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data',         default='../data/data4L1/data_ttbar')
parser.add_argument('--ckpts',        default='../teacher_ckpts_L1_withPV')
parser.add_argument('--restore_file', default=None,
                    help="'best' or 'last' to resume training")

scale_momentum = 128.
deltaR         = 0.4
deltaR_dz      = 0.3
max_epochs     = 10
batch_size     = 64


# ---------------------------------------------------------------------------
def extract_features(data, device):
    """
    Extract features from a data batch.
    x_cont : [N, 6]   pt, px, py, eta, d0, dz
    x_cat  : [N, 3]   pdgid, charge, PV (synthetic zeros if not in data)
    """
    x_cont = data.x[:, :6]                          # [N, 6]

    pdgid  = data.x[:, 6:7].long()                  # [N, 1]
    charge = data.x[:, 7:8].long()                  # [N, 1]

    if data.x.shape[1] >= 9:
        # Real PV column present
        pv_col = data.x[:, 8:9].long()              # [N, 1]
    else:
        # Synthetic PV = 0 (all particles treated as "fromPV quality 0")
        pv_col = torch.zeros(data.x.shape[0], 1, dtype=torch.long, device=device)

    x_cat = torch.cat([pdgid, charge, pv_col], dim=1)   # [N, 3]
    return x_cont, x_cat


def build_edge_index(data, deltaR):
    phi      = torch.atan2(data.x[:, 2], data.x[:, 1])
    etaphi   = torch.cat([data.x[:, 3:4], phi[:, None]], dim=1)
    edge_idx = radius_graph(etaphi, r=deltaR, batch=data.batch,
                            loop=False, max_num_neighbors=255)
    return to_undirected(edge_idx)


# ---------------------------------------------------------------------------
def train(model, device, optimizer, scheduler, loss_fn, dataloader, epoch):
    model.train()
    loss_avg_arr = []
    loss_avg     = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            x_cont, x_cat = extract_features(data, device)
            edge_index     = build_edge_index(data, deltaR)

            result = model(x_cont, x_cat, edge_index, data.batch)
            loss   = loss_fn(result, data.x, data.y, data.batch)

            loss.backward()
            optimizer.step()

            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    scheduler.step(np.mean(loss_avg_arr))
    mean_loss = np.mean(loss_avg_arr)
    print('Training epoch: {:02d}, MSE: {:.4f}'.format(epoch, mean_loss))
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

    # Normalisation — teacher does NOT normalise inside the network
    # (normalisation is applied by StudentNet but not by Net/TeacherNet)
    norm = torch.tensor(
        [1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 1., 1., 1.]
    ).to(device)

    # Full teacher: 6 continuous, 3 categorical (pdgid, charge, PV)
    model = TeacherNet(6, 3, norm).to(device)
    print('Teacher T (with PV) params:',
          sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False
    )

    first_epoch           = 0
    best_validation_loss  = 1e7
    loss_fn               = loss_fn_response_tune

    if args.restore_file is not None:
        restore_ckpt = os.path.join(args.ckpts, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Resuming from epoch', first_epoch)
        with open(os.path.join(args.ckpts, 'metrics_val_best.json')) as f:
            best_validation_loss = json.load(f)['loss']

    if first_epoch == 0:
        loss_log = open(os.path.join(args.ckpts, 'loss.log'), 'w')
        loss_log.write('# Teacher-T-with-PV training  ' +
                       strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        loss_log.write('epoch,train_loss,val_loss\n')
    else:
        loss_log = open(os.path.join(args.ckpts, 'loss.log'), 'a')

    for epoch in range(first_epoch + 1, max_epochs + 1):
        train_loss = train(model, device, optimizer, scheduler,
                           loss_fn, train_dl, epoch)

        utils.save_checkpoint(
            {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optim_dict': optimizer.state_dict(),
             'sched_dict': scheduler.state_dict()},
            is_best=False,
            checkpoint=args.ckpts,
        )

        test_metrics, resolutions = evaluate(
            model, device, loss_fn, test_dl, metrics, deltaR, deltaR_dz, args.ckpts
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
                 'state_dict': model.state_dict(),
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
    print('Done. Best checkpoint:', os.path.join(args.ckpts, 'best.pth.tar'))
