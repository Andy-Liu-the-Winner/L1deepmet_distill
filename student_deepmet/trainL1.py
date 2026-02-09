import json
import os.path as osp
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm
import argparse
import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate

# Import teacher's model definition (different architecture - no embed_pv)
import teacher_model.net as teacher_net
import warnings
warnings.simplefilter('ignore')
from time import strftime, gmtime

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--data', default='../data/data4L1/data_ttbar',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='../student_ckpts_L1',
                    help="Name of the ckpts folder")
parser.add_argument('--teacher_ckpt', default='../teacher_ckpts_L1_20260128/best.pth.tar',
                    help="Path to teacher checkpoint")

scale_momentum = 128

def _model_size_mb(m):
    """Compute the size of a model in MB."""
    ps = sum(p.nelement() * p.element_size() for p in m.parameters())
    bs = sum(b.nelement() * b.element_size() for b in m.buffers())
    return (ps + bs) / 1024**2


def train(model, teacher, device, optimizer, scheduler, loss_fn, dataloader, epoch, deltaR):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            sample_weight = None

            # L1 data: 6 continuous features (pt, px, py, eta, d0, dz), 2 categorical (pdgid, charge)
            x_cont = data.x[:,:6]
            x_cat = torch.cat([
                data.x[:,6:7].long(),  # pdgid
                data.x[:,7:8].long(),  # charge
            ], dim=1)

            # Student uses same features as teacher for L1 data
            student_x_cont = x_cont
            student_x_cat = x_cat

            # L1 data format: [pt, px, py, eta, d0, dz, pdgid, charge]
            phi = torch.atan2(data.x[:,2], data.x[:,1])
            etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1)
            edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=False, max_num_neighbors=255)
            edge_index = to_undirected(edge_index)

            with torch.no_grad():
                teacher_out = teacher(x_cont, x_cat, edge_index, data.batch)

            # student output
            student_out = model(student_x_cont, student_x_cat, edge_index, data.batch)

            # student's original loss
            loss_s = loss_fn(student_out, data.x, data.y, data.batch)
            # distillation loss (MSE)
            loss_d = F.mse_loss(student_out, teacher_out)
            a = 0.5
            loss = a * loss_s + (1 - a) * loss_d

            loss.backward()
            optimizer.step()
            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    scheduler.step(np.mean(loss_avg_arr))
    print('Training epoch: {:02d}, MSE: {:.4f}'.format(epoch, np.mean(loss_avg_arr)))
    return np.mean(loss_avg_arr)

if __name__ == '__main__':
    args = parser.parse_args()

    dataloaders = data_loader.fetch_dataloader(data_dir=args.data,
                                               batch_size=64,
                                               validation_split=.25)

    print(dataloaders.__len__())

    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    print(train_dl.dataset.__len__())
    print(test_dl.dataset.__len__())
    print(type(train_dl))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # L1 data: 6 continuous features
    norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 1., 1., 1.]).to(device)
    student_norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 1., 1., 1.]).to(device)

    # 1) load teacher (freeze it) - L1 teacher has 6 continuous, 2 categorical
    # Use teacher's model definition (different architecture from student's)
    teacher = teacher_net.Net(6, 2, norm).to(device)
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(teacher_ckpt['state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2) instantiate student
    student_n_features_cont = 6
    student_n_features_cat = 2

    model = net.StudentNet(student_n_features_cont, student_n_features_cat, student_norm).to(device)

    # Parameter count comparison
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher params: {teacher_params:,d}")
    print(f"Student params: {student_params:,d}")
    print(f"Student is {student_params/teacher_params:.2%} of teacher size")

    # Model storage size comparison
    teacher_size = _model_size_mb(teacher)
    student_size = _model_size_mb(model)
    print(f"Teacher size: {teacher_size:.3f} MB")
    print(f"Student size: {student_size:.3f} MB")
    print(f"Size reduction: {100 * (1 - student_size/teacher_size):.2f}%")

    # Save model sizes
    os.makedirs(args.ckpts, exist_ok=True)
    size_file = osp.join(args.ckpts, 'model_size.txt')
    with open(size_file, 'w') as f:
        f.write(f"Teacher params: {teacher_params}\n")
        f.write(f"Student params: {student_params}\n")
        f.write(f"Relative size: {student_params/teacher_params:.4f}\n")
        f.write(f"Teacher size: {teacher_size:.3f} MB\n")
        f.write(f"Student size: {student_size:.3f} MB\n")
        f.write(f"Size reduction: {100 * (1 - student_size/teacher_size):.2f}%\n")

    print('model initialized')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False)
    first_epoch = 0
    max_epochs = 5
    best_validation_loss = 10e7
    deltaR = 0.4
    deltaR_dz = 0.3

    loss_fn = net.loss_fn_response_tune
    metrics = net.metrics

    model_dir = args.ckpts

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch', first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']

    if first_epoch == 0:
        loss_log = open(model_dir+'/loss.log', 'w')
        loss_log.write('# loss log for training starting in '+strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        loss_log.write('epoch, loss, val_loss\n')
        loss_log.flush()
    else:
        loss_log = open(model_dir+'/loss.log', 'a')

    for epoch in range(first_epoch+1, max_epochs+1):
        print('Epoch:', epoch)
        print('Current best loss:', best_validation_loss)

        train_loss = train(model, teacher, device, optimizer, scheduler, loss_fn, train_dl, epoch, deltaR)

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'sched_dict': scheduler.state_dict()},
                              is_best=False,
                              checkpoint=model_dir)

        # Evaluate for one epoch on validation set
        test_metrics, resolutions = evaluate(model, device, loss_fn, test_dl, metrics, deltaR, deltaR_dz, model_dir)

        validation_loss = test_metrics['loss']
        loss_log.write('%d,%.2f,%.2f\n'%(epoch, train_loss, validation_loss))
        loss_log.flush()
        is_best = (validation_loss <= best_validation_loss)

        if is_best:
            print('Found new best loss!')
            best_validation_loss = validation_loss

            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'sched_dict': scheduler.state_dict()},
                                  is_best=True,
                                  checkpoint=model_dir)

            utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

        utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(model_dir, 'last.resolutions'))

    loss_log.close()
