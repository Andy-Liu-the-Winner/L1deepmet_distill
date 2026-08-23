"""Unit tests for the L1-ParT teacher (CPU, a few real events — safe on the head node).

python test_ParT.py [--data ../data/data4L1/data_ttbar]

Tests (from teacher_ParT_DESIGN.md §5):
  1. step-0: untrained model reproduces the PUPPI baseline exactly (zero-init heads)
  2. rotation invariance: global phi rotation leaves per-particle weights unchanged
     and rotates the predicted MET vector by the same angle
  3. padding invariance: an event's weights don't depend on batch padding
  4. loss_fn_huber_response is finite and backprops
"""
import argparse
import glob
import math

import torch

import model.net as net
from model.part_met_network import PartMETNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../data/data4L1/data_ttbar')


def load_events(data_dir, n=8):
    files = sorted(glob.glob(data_dir + '/processed/*_file*_slice_*_nevent_1000_*.pt'))[:n]
    assert files, 'no event files found'
    return [torch.load(f) for f in files]


def pad_events(events, extra_pad=0):
    lens = [d.x.shape[0] for d in events]
    nmax = max(lens) + extra_pad
    x_pad = torch.zeros(len(events), nmax, 8)
    mask = torch.zeros(len(events), nmax, dtype=torch.bool)
    for j, d in enumerate(events):
        x_pad[j, :lens[j]] = d.x.float()
        mask[j, :lens[j]] = True
    y = torch.stack([d.y.float().view(-1)[:2] for d in events])
    return x_pad, mask, y


def rotate(x_pad, alpha):
    out = x_pad.clone()
    pt, phi = x_pad[..., 0], x_pad[..., 4]
    phi2 = torch.remainder(phi + alpha + math.pi, 2 * math.pi) - math.pi
    out[..., 1] = pt * torch.cos(phi2)
    out[..., 2] = pt * torch.sin(phi2)
    out[..., 4] = phi2
    return out


def met(w, s, x_pad):
    return torch.stack([s * (w * x_pad[..., 1]).sum(1),
                        s * (w * x_pad[..., 2]).sum(1)], dim=1)


def main():
    args = parser.parse_args()
    torch.manual_seed(0)
    events = load_events(args.data)
    x_pad, mask, y = pad_events(events)
    model = PartMETNetwork()
    model.eval()

    # 1. step-0 == PUPPI baseline
    with torch.no_grad():
        w, s = model(x_pad, mask)
    assert torch.allclose(w, x_pad[..., 5] * mask, atol=1e-6), 'step-0 w != puppi'
    assert torch.allclose(s, torch.ones_like(s)), 'step-0 s != 1'
    print('PASS step-0 == PUPPI baseline')

    # perturb heads so the remaining tests exercise a non-trivial network
    with torch.no_grad():
        for p in model.delta_head[-1].parameters():
            p.add_(torch.randn_like(p) * 0.1)
        for p in model.scale_head.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # 2. rotation invariance / equivariance
    alpha = 1.234
    xr = rotate(x_pad, alpha)
    with torch.no_grad():
        w0, s0 = model(x_pad, mask)
        w1, s1 = model(xr, mask)
    dw = (w0 - w1).abs().max().item()
    assert dw < 1e-4, 'weights not rotation-invariant: max dev %g' % dw
    assert torch.allclose(s0, s1, atol=1e-5), 'scale not rotation-invariant'
    m0, m1 = met(w0, s0, x_pad), met(w1, s1, xr)
    rot = torch.tensor([[math.cos(alpha), -math.sin(alpha)],
                        [math.sin(alpha), math.cos(alpha)]])
    dm = (m1 - m0 @ rot.T).abs().max().item()
    assert dm < 1e-2, 'MET not rotation-equivariant: max dev %g' % dm
    print('PASS rotation invariance (dw=%.2g) / MET equivariance (dm=%.2g)' % (dw, dm))

    # 3. padding invariance
    xp, mp, _ = pad_events(events, extra_pad=17)
    with torch.no_grad():
        w2, s2 = model(xp, mp)
    dp = (w0 - w2[:, :w0.shape[1]]).abs().max().item()
    assert dp < 1e-4, 'weights depend on padding: max dev %g' % dp
    assert w2[:, w0.shape[1]:].abs().max().item() == 0, 'nonzero weight on padding'
    print('PASS padding invariance (dp=%.2g)' % dp)

    # 4. loss finite + gradients flow
    model.train()
    w, s = model(x_pad, mask)
    loss, res_t, resp_t = net.loss_fn_huber_response(w, s, x_pad, mask, y)
    assert torch.isfinite(loss), 'loss not finite'
    loss.backward()
    grads = [p.grad.abs().max().item() for p in model.parameters() if p.grad is not None]
    assert all(math.isfinite(g) for g in grads), 'non-finite gradients'
    assert any(g > 0 for g in grads), 'no gradient signal'
    print('PASS loss finite (%.4f = res %.4f + 5*resp %.4f), gradients flow'
          % (loss.item(), res_t.item(), resp_t.item()))

    n_params = sum(p.numel() for p in model.parameters())
    print('ALL TESTS PASSED (params: %d)' % n_params)


if __name__ == '__main__':
    main()
