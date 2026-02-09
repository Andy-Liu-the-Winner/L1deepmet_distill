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
import warnings
warnings.simplefilter('ignore')
from time import strftime, gmtime

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--data', default='/hildafs/projects/phy230010p/share/NanoAOD/Znunu/',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='/hildafs/projects/phy230010p/andy_liu/fep/ckpts_znunu',
                    help="Name of the ckpts folder")

scale_momentum = 128

def _model_size_mb(m): # idea from https://discuss.pytorch.org/t/finding-model-size/130275/2 by ptrblck
    """Compute the size of a model in MB."""
    ps = sum(p.nelement() * p.element_size() for p in m.parameters())
    bs = sum(b.nelement() * b.element_size() for b in m.buffers())
    return (ps + bs) / 1024**2



def train(model, device, optimizer, scheduler, loss_fn, dataloader, epoch):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            # print("data:",data)
            optimizer.zero_grad()
            data = data.to(device)
            # sample_weight = torch.full((data.y.shape[0],), 1.0, dtype=torch.float32, device=device)
            sample_weight = None
            x_cont = data.x[:,:8] #include puppi
            #x_cont = data.x[:,:7] #remove puppi
            x_cat = data.x[:,8:].long()
            # def student x_cont and x_cat
            student_x_cont = data.x[:,:6]
            student_x_cat = data.x[:,6:].long()

            phi = torch.atan2(data.x[:,1], data.x[:,0])
            etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1)        
            # NB: there is a problem right now for comparing hits at the +/- pi boundary
            edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=False, max_num_neighbors=255)
            edge_index = to_undirected(edge_index)  # Make the edge index undirected
            # result = model(x_cont, x_cat, edge_index, data.batch)
            # loss = loss_fn(result, data.x, data.y, data.batch)

            with torch.no_grad():
                teacher_out = teacher(x_cont, x_cat, edge_index, data.batch)

            # student output
            student_out = model(student_x_cont, student_x_cat, edge_index, data.batch)

            # student’s original loss
            loss_s = loss_fn(student_out, data.x, data.y, data.batch)
            # distillation loss (MSE)
            loss_d = F.mse_loss(student_out, teacher_out)
            a = 0.5
            loss = a * loss_s + (1 - a) * loss_d

            loss.backward()
            optimizer.step()
            # update the average loss
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

    norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 1., 1., 1., 1., 1.]).to(device)   # pt, px, py: scale by 128      
    student_norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 1., 1., 1.]).to(device)   
    # 1) load teacher (freeze it)
    teacher = net.Net(8, 3, norm).to(device) # initiate a fresh net architecture
    teacher_ckpt = torch.load(os.path.join(args.ckpts, 'best_teacher.pth.tar')) 
    teacher.load_state_dict(teacher_ckpt['state_dict']) # overwrite the random weights
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2) instantiate student
    student_n_features_cont = 6
    student_n_features_cat = 2

    model = net.StudentNet(student_n_features_cont, student_n_features_cat, student_norm).to(device)
    

    #  Parameter‐count comparison
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher params: {teacher_params:,d}")
    print(f"Student params: {student_params:,d}")
    print(f"Student is {student_params/teacher_params:.2%} of teacher size")
    # Save parameter counts to text file
    size_file = osp.join(args.ckpts, 'model_size.txt')
    with open(size_file, 'w') as f:
        print("Writing to:", size_file)
        f.write(f"Teacher params: {teacher_params}\n")
        f.write(f"Student params: {student_params}\n")
        f.write(f"Relative size: {student_params/teacher_params:.4f}\n")

    # model storage size comparison
    # 1) compute sizes in MB
    teacher_size = _model_size_mb(teacher)
    student_size = _model_size_mb(model)

    # 2) print them and the relative reduction
    print(f"Teacher size: {teacher_size:.3f} MB")
    print(f"Student size: {student_size:.3f} MB")
    print(f"Size reduction: {100 * (1 - student_size/teacher_size):.2f}%")

    # Save model sizes to text file
    size_file = osp.join(args.ckpts, 'model_size.txt')
    with open(size_file, 'a') as f:
        print("Writing to:", size_file)
        f.write(f"Teacher size: {teacher_size:.3f} MB\n")
        f.write(f"Student size: {student_size:.3f} MB\n")
        f.write(f"Size reduction: {100 * (1 - student_size/teacher_size):.2f}%\n")


    print('model initialized')
    #model = net.Net(7, 3).to(device) #remove puppi
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False)
    first_epoch = 0
    max_epochs = 50
    best_validation_loss = 10e7
    deltaR = 0.4
    deltaR_dz = 0.3

    # loss_fn = net.loss_fn_response_binned
    loss_fn = net.loss_fn_response_tune
    # loss_fn = net.loss_fn
    metrics = net.metrics

    model_dir = args.ckpts
    # loss_log = open(model_dir+'/loss.log', 'w')
    # loss_log.write('# loss log for training starting in '+strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
    # loss_log.write('epoch, loss, val_loss\n')
    # loss_log.flush()

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch',first_epoch)
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
        print('Current time:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Number of parameters:', sum(p.numel() for p in model.parameters()))
        print('Model size:', _model_size_mb(model), 'MB')
        print('Number of batches in one epoch:', len(train_dl))
        print('Number of batches in validation set:', len(test_dl))
        print('Number of training samples:', len(train_dl.dataset))
        print('Number of validation samples:', len(test_dl.dataset))

        print('Current best loss:', best_validation_loss)
        if '_last_lr' in scheduler.state_dict():
            print('Learning rate:', scheduler.state_dict()['_last_lr'][0])

        # compute number of batches in one epoch (one full pass over the training set)
        train_loss = train(model, device, optimizer, scheduler, loss_fn, train_dl, epoch)

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'sched_dict': scheduler.state_dict()},
                              is_best=False,
                              checkpoint=model_dir)

        # Evaluate for one epoch on validation set
        test_metrics, resolutions = evaluate(model, device, loss_fn, test_dl, metrics, deltaR,deltaR_dz, model_dir)

        validation_loss = test_metrics['loss']
        loss_log.write('%d,%.2f,%.2f\n'%(epoch,train_loss, validation_loss))
        loss_log.flush()
        is_best = (validation_loss<=best_validation_loss)

        # If best_eval, best_save_path
        if is_best: 
            print('Found new best loss!') 
            best_validation_loss=validation_loss

            # Save weights
            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'sched_dict': scheduler.state_dict()},
                                  is_best=True,
                                  checkpoint=model_dir)
            
            # Save best val metrics in a json file in the model directory
            utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

        utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(model_dir, 'last.resolutions'))

    loss_log.close()

