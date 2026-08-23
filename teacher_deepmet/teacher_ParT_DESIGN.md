# Teacher Redesign — "L1-ParT": Compact Particle Transformer for L1 MET

Spec for a redesigned teacher to replace `GraphMETNetwork` (hidden_dim=32, 2×EdgeConv, ~6.4k params).
Written 2026-08-02 after auditing the data and the baseline's performance.

## 0. Facts this design is built on (all verified against the data)

1. **The data format documented in the README is wrong.** Source is `L1PuppiCands`
   (`generate_npz.py`), true columns are:

   | col | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
   |-----|---|---|---|---|---|---|---|---|
   | actual | pt | px | py | eta | **phi** | **puppiWeight** | pdgid | charge |

   Col 4 is phi (px = pt·cos(col4) to machine precision), col 5 is puppiWeight
   (=1.0 for all charged particles; continuous in (0,1] for photons / K_L).
   The loss-function column indices (px=col1, py=col2) are correct and unaffected.

2. **The current teacher barely beats the trivial baseline.** On 2000 events, the plain
   PUPPI-weighted sum gives response (qT>50) = 0.775; the trained teacher's response
   plateau is ~0.78 (see `teacher_comparison_results/compare_R.png`). σ(u⊥) ≈ 20 GeV.
   All the GNN capacity is currently buying ~nothing on response.

3. **Events are small**: 16–110 particles, median 42, ≤ ~128 always. Full O(N²)
   self-attention over one event costs ≈ 110² = 12k pairs — negligible.

4. **Training-loop defects** in `trainL1.py` / `data_loader.py`:
   - `DataLoader(..., shuffle=False)` for the **train** split — identical batch order every epoch.
   - 998,545 single-event `.pt` files → one `torch.load` per event per epoch (I/O-bound;
     this is why only 5–10 epochs are run).
   - `radius_graph` on (eta, phi) has no φ-wraparound: particles across ±π are never connected
     (acknowledged in a code comment).
   - Loss balance: resolution term is a *mean* in GeV² (~2000 scale) while the response term is
     c=4000 × a *sum* over events in the batch — the response term dominates the total loss (~70k)
     by >30×, yet response still sits at 0.78, i.e. the optimization is fighting itself.

## 1. Model architecture

**Type:** small Transformer encoder over the particle set with pairwise geometric attention
bias (Particle-Transformer / ParT style), replacing the radius-graph + EdgeConv stack.
No graph construction at all.

### Inputs (rotation-invariant by construction)

Per-particle continuous features — **do not feed px, py, or raw phi**:
- `log(pt)`  (pt spans 1.25–250 GeV, log-normalizes the heavy tail)
- `eta`
- `puppiWeight`  ← currently fed but mislabeled as "dz"; it is the single strongest prior

Categorical embeddings (as today):
- `|pdgid|` remapped over vocabulary {11, 13, 22, 130, 211} → Embedding(5, 8)
- `charge + 1` → Embedding(3, 8)

Node encoder: `Linear(3+8+8 → d) → GELU → LayerNorm`, d = 64.

Phi enters **only** through pairwise quantities, so the predicted per-particle weights are
invariant under a global phi rotation — exactly the symmetry of the task
(MET = Σ wᵢ·p⃗ᵢ is then automatically rotation-equivariant). The current model must waste
capacity learning this invariance from data and can never learn it exactly.

### Pairwise attention bias (the ParT trick)

For each pair (i,j) compute:
- `Δη = ηᵢ − ηⱼ`
- `Δφ = wrap(φᵢ − φⱼ)` into (−π, π]  ← fixes the ±π boundary bug for free
- `ΔR = √(Δη² + Δφ²)`, use `log(ΔR + ε)`
- `log(min(ptᵢ, ptⱼ))` and `log(ptᵢ · ptⱼ)`

MLP: `Linear(5 → 32) → GELU → Linear(32 → n_heads)` → added to the pre-softmax attention
logits of every layer (shared bias MLP across layers is fine and cheaper).
This replaces the hard ΔR=0.4 cut with a *learned, soft* neighborhood, and gives every
layer a global receptive field — MET is a global quantity; 2 EdgeConv hops at ΔR=0.4
cannot see across the event.

Implementation note: pad events to max N in batch (≤128), build the (B, N, N) bias tensor,
use a padding mask in `torch.nn.functional.scaled_dot_product_attention` (additive bias =
pair-bias + −inf on padding). No torch_cluster / torch_scatter needed in the model.

### Trunk

- 4 pre-LN Transformer blocks: MHA (4 heads, d=64) + FFN (64→128→64), GELU, residual.
- ~200k params total. VRAM at batch 256, N=128, d=64 is < 1 GB — far under the 16 GB cap.

### Outputs (physics-anchored)

Two heads:

1. **Per-particle weight, residual on PUPPI:**
   `wᵢ = relu(puppiWeightᵢ + δᵢ)`, δᵢ from `Linear(d → d/2) → GELU → Linear(d/2 → 1)`
   with the final layer **zero-initialized** → the model starts *exactly* at the PUPPI-MET
   baseline (response 0.775) instead of at noise. Keeps ReLU semantics the student
   distillation already assumes (weights ≥ 0, unbounded above).

2. **Per-event response-calibration scalar:**
   attention-pool the particle embeddings (learned query), then
   `s = 1 + 0.5·tanh(Linear(d → 1))`, zero-initialized so s starts at 1.
   Final prediction used in the loss: `MET = s · Σ wᵢ p⃗ᵢ`.
   This is the cheapest possible mechanism to close the 0.78 → 1.0 response gap: a global
   multiplicative miscalibration cannot be fixed efficiently by per-particle weights alone
   (they'd all have to inflate coherently), but one scalar head fixes it directly.
   For distillation later, the student can either learn `s·wᵢ`-equivalent weights from the
   teacher's *effective* per-particle weights `s·wᵢ` (export those as the distillation target
   so the student interface is unchanged).

## 2. Loss (rebalanced)

```
res_term  = mean( huber( (METx + genMETx)/σ0 ) + huber( (METy + genMETy)/σ0 ) ),  σ0 = 30 GeV
resp_term = mean( |response − 1|  over events with qT > 50 GeV )      # MEAN, not sum
loss      = res_term + λ · resp_term,  λ = 5 (sweep 1–20 once)
```

- Same sign/response conventions as `loss_fn_response_tune` (response = −MET·q⃗T/|qT|²).
- Huber (delta=1 on the normalized residual) instead of pure MSE: the qT spectrum is steep
  and TTbar has a long tail; MSE lets rare 300+ GeV events dominate gradients.
- Both terms are per-event means → batch-size-independent balance, single interpretable λ.
- Optional (second iteration): per-qT-bin weights as in `loss_fn_weighted` to flatten the
  effective spectrum; only add if high-qT response still sags.

## 3. Training recipe

- **Repack the dataset first**: concatenate the 998,545 single-event `.pt` files into ~1000
  sharded files of ~1000 events each (store padded tensors + n_particles, or a list of Data
  objects saved together). One-time ~30 min job (run it on a worker node via
  `srun --mem=4G`, not on Falcon). This turns an I/O-bound epoch into a compute-bound one
  and is what makes 30+ epochs feasible at all.
- `shuffle=True` on the train loader (fixing the existing bug).
- batch 256, AdamW lr 3e-4, weight_decay 0.01, cosine decay to 3e-5 with 1-epoch warmup,
  30 epochs, gradient clip 1.0.
- EMA of model weights (decay 0.999) for the eval/checkpoint copy — cheap, reliably better.
- Keep the existing SLURM flow (`train_L1_job.slurm` pattern); GPU footprint stays ≪ 16 GB.
- Keep the same train/val split seed (42, 25%) so v1 vs ParT comparisons are on identical events.

## 4. Evaluation protocol (must-pass gates vs teacher v1)

Use the existing `resolution()` metric and `plots_L1_20260128/plot_L1.py` machinery, same
validation split:

| Metric | Teacher v1 | Trivial PUPPI | ParT target |
|--------|-----------|---------------|-----------|
| response plateau (qT 50–300) | ~0.78 | 0.775 | **0.95–1.0** |
| σ(u⊥), qT < 150 | ~20 GeV | — | **≤ 17 GeV** |
| σ(u∥), qT < 150 | (see plots) | — | beat v1 |

Also report the two loss components separately per epoch (res_term, resp_term) — the v1 log
only stores the blended number, which hid the imbalance.

Sanity check at step 0: with zero-init heads the untrained model must reproduce the PUPPI
baseline numbers exactly. If it doesn't, the wiring is wrong.

## 5. Implementation order (each step independently testable)

1. Dataset repacking script + new loader (padded batches, no torch_cluster). Verify event
   count and that per-event tensors match the old loader on 100 random events.
2. Model file `model/part_met_network.py` (node encoder, pair-bias MLP, 4 blocks, 2 heads).
   Unit test: rotation invariance — rotate all φ by a random angle, per-particle weights
   must be bit-identical (within fp tolerance) and MET must rotate by the same angle.
3. New loss in `model/net.py` (`loss_fn_huber_response`), keep old ones untouched.
4. `trainL1_ParT.py` (clone of `trainL1.py` with the recipe above; drop radius_graph entirely).
5. Train, then run the comparison plots against `teacher_ckpts_L1_fixed`.

### Fallback (if you want a minimal-diff variant instead)

Keep `GraphMETNetwork` but: fix inputs (drop px/py/phi, keep log pt/eta/puppi), fix φ-wrap
in graph building (build radius graph on (η, sinφ, cosφ) with chord radius), hidden_dim 64
/ depth 3 with residuals, new loss, shuffle=True, repacked data, 30 epochs. Expect most of
the response gain (loss + calibration head matter most) but less resolution gain (still a
local receptive field).

## Why this will exceed teacher v1 — summary of the tactics

1. **Found the real data semantics** (phi + puppiWeight, not d0/dz) → puppiWeight becomes an
   explicit physics prior and the model warm-starts at the PUPPI baseline instead of noise.
2. **Symmetry built in, not learned**: rotation-invariant inputs; phi only enters as Δφ.
3. **Global receptive field with learned neighborhoods** (attention + pairwise ΔR/Δφ bias)
   instead of a hard ΔR=0.4 graph with a broken ±π seam; MET is a global observable.
4. **A dedicated 1-parameter-per-event calibration head** attacks the dominant failure mode
   (response stuck at 0.78) directly, instead of asking 50 per-particle weights to conspire.
5. **Loss actually balanced** (per-event means, Huber, normalized scales) so the response
   term can act without drowning the resolution term.
6. **Trained properly**: shuffling fixed, I/O fixed → 30 epochs instead of 5, cosine + EMA.
7. Capacity 6.4k → ~200k params — still tiny, but the teacher's job in distillation is to be
   the quality ceiling; the L1-hardware constraints live in the *student*, not here.
