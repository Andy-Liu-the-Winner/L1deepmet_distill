"""Repack the ~1M single-event .pt files into event shards for the redesigned teachers (ParT/GraphNet).

Replicates the v1 train/val split exactly (torch.manual_seed(42) + randperm over the
sorted-glob file list, 25% val), then writes train/val events into separate shard
files so training can shuffle at shard + intra-shard level with sequential I/O.

Each shard is a dict:
    x       : (sum_n, 8) float32, all events concatenated
    lengths : (n_events,) int32
    y       : (n_events, 2) float32  (genMETx, genMETy)

Run on a worker node (I/O heavy):  sbatch repack_L1_job.slurm
"""
import argparse
import glob
import json
import os
import os.path as osp
import time

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../data/data4L1/data_ttbar',
                    help="data folder containing processed/ single-event .pt files")
parser.add_argument('--out', default=None,
                    help="output folder (default: <data>/shards_v2)")
parser.add_argument('--shard_size', type=int, default=2048)
parser.add_argument('--val_split', type=float, default=0.25)
parser.add_argument('--seed', type=int, default=42)


def main():
    args = parser.parse_args()
    out = args.out or osp.join(args.data, 'shards_v2')
    os.makedirs(out, exist_ok=True)

    # identical glob + sort as model/data_loader.py METDataset.existing_pt_names
    processed_dir = osp.join(args.data, 'processed')
    files = sorted(glob.glob(processed_dir + '/*_file*_slice_*_nevent_1000_*.pt'))
    N = len(files)
    assert N > 0, 'no event files found in ' + processed_dir

    # identical split as model/data_loader.py fetch_dataloader (random_split under seed 42)
    split = int(np.floor(args.val_split * N))
    torch.manual_seed(args.seed)
    perm = torch.randperm(N).tolist()
    train_idx, val_idx = perm[:N - split], perm[N - split:]

    def write_split(name, idxs):
        n_shards = (len(idxs) + args.shard_size - 1) // args.shard_size
        t0 = time.time()
        for shard_no in range(n_shards):
            path = osp.join(out, '%s_shard_%05d.pt' % (name, shard_no))
            if osp.exists(path):
                continue  # idempotent: safe to resume a killed job
            chunk = idxs[shard_no * args.shard_size:(shard_no + 1) * args.shard_size]
            xs, ys, lens = [], [], []
            for i in chunk:
                d = torch.load(files[i])
                x = d.x.float()
                xs.append(x)
                lens.append(x.shape[0])
                ys.append(d.y.float().view(-1)[:2])
            torch.save({'x': torch.cat(xs, dim=0),
                        'lengths': torch.tensor(lens, dtype=torch.int32),
                        'y': torch.stack(ys, dim=0)},
                       path + '.tmp')
            os.rename(path + '.tmp', path)
            if shard_no % 20 == 0:
                rate = (shard_no + 1) * args.shard_size / max(time.time() - t0, 1e-9)
                print('%s shard %d/%d (%.0f ev/s)' % (name, shard_no, n_shards, rate),
                      flush=True)
        return n_shards

    n_train_shards = write_split('train', train_idx)
    n_val_shards = write_split('val', val_idx)

    meta = {
        'n_events': N,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'seed': args.seed,
        'val_split': args.val_split,
        'shard_size': args.shard_size,
        'n_train_shards': n_train_shards,
        'n_val_shards': n_val_shards,
        'first_train_idx': train_idx[:5],
        'first_val_idx': val_idx[:5],
        'columns': ['pt', 'px', 'py', 'eta', 'phi', 'puppiWeight', 'pdgid', 'charge'],
    }
    with open(osp.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('done:', json.dumps(meta))


if __name__ == '__main__':
    main()
