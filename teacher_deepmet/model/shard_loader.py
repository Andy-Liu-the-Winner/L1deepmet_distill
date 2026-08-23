"""Loader for the repacked event shards (see repack_L1_data.py).

Yields padded batches (x_pad (B,N,8), mask (B,N) bool, y (B,2)) with two-level
shuffling for training: shard order + event order within each shard. Sequential
shard I/O keeps an epoch compute-bound instead of I/O-bound.
"""
import glob
import json
import os.path as osp

import torch


class ShardLoader:
    def __init__(self, shard_dir, split, batch_size, shuffle, seed=0, device=None):
        self.files = sorted(glob.glob(osp.join(shard_dir, '%s_shard_*.pt' % split)))
        assert self.files, 'no %s shards in %s' % (split, shard_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.epoch = 0
        self.seed = seed
        with open(osp.join(shard_dir, 'meta.json')) as f:
            meta = json.load(f)
        self.n_events = meta['n_train'] if split == 'train' else meta['n_val']

    def __len__(self):
        return (self.n_events + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = (torch.randperm(len(self.files), generator=g).tolist()
                       if self.shuffle else range(len(self.files)))
        for si in shard_order:
            shard = torch.load(self.files[si])
            lengths = shard['lengths'].long()
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
            n_ev = len(lengths)
            ev_order = (torch.randperm(n_ev, generator=g).tolist()
                        if self.shuffle else range(n_ev))
            for b0 in range(0, n_ev, self.batch_size):
                idxs = list(ev_order)[b0:b0 + self.batch_size]
                if not idxs:
                    continue
                lens = [int(lengths[i]) for i in idxs]
                # bucket padded size to multiples of 32: few distinct tensor shapes ->
                # the CUDA caching allocator reuses blocks instead of accumulating
                # stale ones per unique N (which blew past the MPS memory share)
                nmax = max(64, ((max(lens) + 31) // 32) * 32)
                B = len(idxs)
                x_pad = torch.zeros(B, nmax, 8)
                mask = torch.zeros(B, nmax, dtype=torch.bool)
                for j, i in enumerate(idxs):
                    x_pad[j, :lens[j]] = shard['x'][offsets[i]:offsets[i] + lens[j]]
                    mask[j, :lens[j]] = True
                y = shard['y'][idxs]
                if self.device is not None:
                    x_pad = x_pad.to(self.device, non_blocking=True)
                    mask = mask.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                yield x_pad, mask, y
