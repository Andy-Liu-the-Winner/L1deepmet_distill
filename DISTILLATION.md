# L1 DeepMET Distillation — Handoff

**Audience:** whoever picks up the L1 trigger MET distillation work next.
**Status as of 2026-09-18:** teacher side is done and reproducible; the distillation
experiment is complete and its headline result is a **null result** (see §7). The
recommended deployment artifact is the *from-scratch* student, not the distilled one.

Everything below is reproducible from the checkpoints and shards on this cluster. All
numbers in §6 and §7 were recomputed from the committed `.resolutions` files on
2026-09-18, not copied from older notes.

---

## 1. What the project is

L1 trigger MET regression for CMS Phase-2. A network assigns a **per-particle weight**
`w_i` to each L1 PUPPI candidate and MET is the weighted vector sum:

```
METx = s * Σ_i w_i · px_i
METy = s * Σ_i w_i · py_i
```

`s` is a per-event calibration scalar (teachers only — see §4.2).

The deployed network must stay **EdgeConv / FlowGNN-FPGA-compatible**, which is the
entire reason for the teacher/student split: the teacher is free to be any
architecture (it only has to define the quality ceiling), while the student carries
the hardware constraints.

The real goal metric is the **MET trigger turn-on curve at fixed rate**, which is
driven by scaled resolution (resolution / response) in the 50–150 GeV `qT` band.
That is why §6 reports *scaled* resolutions, not raw ones.

---

## 2. Repo map

```
teacher_deepmet/            # everything for the redesigned teachers AND the distillation run
  model/
    part_met_network.py       ← NEW: ParT transformer teacher (§4.2)
    graphnet_met_network.py   ← NEW: GraphNet teacher, architecture control (§4.3)
    shard_loader.py           ← NEW: padded-batch loader over event shards
    graph_met_network.py      # v1 EdgeConv (GraphMETNetwork / StudentGraphMETNetwork)
    net.py                    # losses; loss_fn_huber_response is the current one (§5)
  repack_L1_data.py         ← NEW: 1M single-event .pt files -> shards (§3.2)
  trainL1_ParT.py           ← NEW: teacher training (--arch part|graph)
  trainL1_student_distill.py← NEW: THE DISTILLATION SCRIPT (§7)
  test_ParT.py              ← NEW: unit tests (rotation invariance, padding, step-0 gate)
  compare_all_teachers.py   # 4-way teacher plots
  compare_students.py       # student-vs-teacher plots
  teacher_ParT_DESIGN.md    ← design spec written before implementation; read this
  teacher_GraphNet_DESIGN.md
  trainL1.py, evaluate.py   # v1 pipeline (superseded, kept for reference)

student_deepmet/            # v1 student pipeline (superseded) + the deployed architecture
  model/graph_met_network.py  # StudentGraphMETNetwork — imported unchanged by the distill script

vae_teacher/                # DEAD END, kept for the record (§9)

teacher_ckpts_ParT_r2/          # ← the distillation teacher (best.pth.tar holds EMA weights)
teacher_ckpts_GraphNet_r1/
teacher_ckpts_L1_fixed/         # v1 teacher
student_ckpts_scratch_r1/       # ← SHIP THIS ONE
student_ckpts_distill_ParT_r1/
teacher_comparison_new/         # 4-way teacher plots
student_comparison_results/     # student plots + pre-registration + measured results
```

Checkpoint directories are gitignored (`*_ckpts*/`) — they live on the cluster only.

---

## 3. Data

### 3.1 True column format

The original README documented the wrong format. The verified format, from
`L1PuppiCands` in `generate_npz.py`, is:

| col | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----|---|---|---|---|---|---|---|---|
| | pt | px | py | eta | **phi** | **puppiWeight** | pdgid | charge |

Columns 4 and 5 are **phi and puppiWeight**, not d0/dz. This was verified numerically
(`px = pt·cos(col4)` to machine precision; col5 is exactly 1.0 for every charged
particle and continuous in (0,1] for photons / K_L).

This matters twice over:

1. All pre-2026-02 loss functions and phi calculations used wrong column indices.
   Fixed in `CHANGES_L1_FIX.md`.
2. `puppiWeight` is the single strongest physics prior in the input, and knowing it is
   what makes the residual-on-PUPPI head (§4.2) possible.

Truth labels are `(genMETx, genMETy)`. Sign convention: `MET = Σ w·p ≈ −genMET`.

### 3.2 Shards

The original dataset is 998,545 single-event `.pt` files — one `torch.load` per event
per epoch, which made an epoch I/O-bound and capped the v1 runs at 5–10 epochs.

`repack_L1_data.py` concatenates them into 2048-event shards:

```bash
cd teacher_deepmet && sbatch repack_L1_job.slurm     # one-time, ~30 min
```

Output `data/data4L1/data_ttbar/shards_v2/`: 365 train + 122 val shards + `meta.json`.
995,000 events, 746,250 train / 248,750 val, **seed 42, 25 % val — the identical split
the v1 runs used**, so v1 and the new models are compared on exactly the same events.

`ShardLoader` yields `(x_pad (B,N,8), mask (B,N) bool, y (B,2))` with two-level shuffling
(shard order + intra-shard order) and sequential shard I/O. Padded `N` is bucketed to
multiples of 32 so the CUDA caching allocator reuses blocks instead of accumulating a
stale block per distinct event size.

Events hold 16–110 particles (median 42), so full O(N²) attention costs ≤ ~12k pairs per
event — negligible, which is what makes the transformer affordable here.

---

## 4. Architectures

### 4.1 v1 EdgeConv baseline — `GraphMETNetwork` / `StudentGraphMETNetwork`

`hidden_dim=32`, `conv_depth=2`, embeddings for pdgid/charge (and PV, unused for L1),
2 × `EdgeConv` over a `radius_graph` with ΔR < 0.4, output head → `ReLU` weights.

**6,417 trainable parameters.** Note: for L1 data the v1 *teacher* and the *student* are
the same size — the PV-association embedding that would have distinguished them does not
exist in L1 data. (Earlier plot legends labelled the v1 teacher "173k"; that label was
wrong and has been corrected.)

Three defects of the v1 setup, all fixed in the new pipeline:

- `radius_graph` on raw (η, φ) never connects particles across the ±π seam.
- Train loader ran with `shuffle=False` — identical batch order every epoch.
- Loss imbalance: the resolution term was a mean in GeV² (~2000) while the response term
  was `c=4000 ×` a *sum* over the batch (~70k), so the response term dominated by >30×
  and the two terms fought each other. Response still sat at 0.78.

### 4.2 ParT teacher (NEW) — `model/part_met_network.py`

**This is the transformer that did not previously exist in this repo.** It replaces the
radius-graph + EdgeConv stack with a compact Particle-Transformer encoder. 137,926
parameters, `d=64`, 4 heads, 4 pre-LN blocks, FFN 64→128→64.

Design spec: `teacher_deepmet/teacher_ParT_DESIGN.md` (written before implementation,
with the reasoning and the pass/fail gates). The four ideas that make it work:

**(a) Rotation invariance by construction.** The per-particle inputs are
`log(pt)`, `eta`, `puppiWeight` plus `Embedding(5,8)` on `|pdgid|` over vocabulary
{11,13,22,130,211} and `Embedding(3,8)` on `charge+1`. Node encoder is
`Linear(3+16 → 64) → GELU → LayerNorm`.

`px`, `py` and absolute `phi` **never enter the network.** Phi appears only through
pairwise differences, so the predicted weights are exactly invariant under a global phi
rotation and MET is exactly equivariant. v1 had to learn that symmetry from data and
could never learn it exactly. `test_ParT.py` asserts it numerically.

**(b) Pairwise geometric attention bias — the ParT trick.** For every pair (i,j):

```
Δη,  Δφ = wrap(φ_i − φ_j) ∈ (−π, π],  log(ΔR + ε),  log min(pt_i,pt_j),  log(pt_i·pt_j)
```

fed through `Linear(5→32) → GELU → Linear(32→n_heads)` and **added to the pre-softmax
attention logits of every block** (one bias MLP shared across blocks). This replaces the
hard ΔR = 0.4 cut with a learned, soft neighbourhood, fixes the ±π seam for free via the
wrap, and gives every layer a global receptive field — which matters because MET is a
global observable that 2 EdgeConv hops at ΔR = 0.4 cannot see.

Attention is written out manually (`AttnBlock`) rather than using
`scaled_dot_product_attention`, because the cluster env is torch 1.12. Padded **keys**
get `−1e9`; padded queries keep finite rows and are discarded at the output.

**(c) Two physics-anchored, zero-initialized heads.**

```
w_i = relu(puppiWeight_i + δ_i)          δ from Linear(64→32) → GELU → Linear(32→1)
s   = 1 + 0.5·tanh(scale_head(pooled))   pooled = attention pool with a learned query
```

Both final linears are **zero-initialized**, so an *untrained* model reproduces the PUPPI
baseline exactly (`w = puppiWeight`, `s = 1`). Training starts from response 0.816
instead of from noise, and `relu` keeps the weight semantics the student assumes
(`w ≥ 0`, unbounded above). Both training scripts assert this at step 0 and abort on
mismatch — it is a wiring check that has caught real bugs.

The `s` head exists because response was stuck at ~0.78: a *global multiplicative*
miscalibration cannot be fixed efficiently by per-particle weights (they would all have
to inflate coherently), but one scalar per event fixes it directly.

**(d) `effective_weights(x_pad, mask) → s·w`** is the distillation export: it folds the
calibration scalar into the per-particle weights so the student — which has no `s` head —
sees a single per-particle target. See §7.

### 4.3 GraphNet teacher (NEW) — `model/graphnet_met_network.py`

128,642 parameters. Same inputs, same heads, same loss, same training recipe as ParT;
**only the trunk differs** — 3 message-passing blocks with a global exchange node `g`
(updated from mean/max pools each block and broadcast back to every node) over a dense
on-the-fly ΔR < 0.4 adjacency with wrapped Δφ.

Its purpose is a controlled GNN-vs-Transformer comparison with everything else held
fixed, and it is what tells us *where* the ParT advantage comes from (§6). Deviation from
its design doc: adjacency is computed on the fly rather than precomputed into the shards
— mathematically equivalent, avoids a second repack, costs ~2× ParT's epoch time.

### 4.4 The student — the deployment target

`StudentGraphMETNetwork`, imported **unchanged** from
`student_deepmet/model/graph_met_network.py` by the distillation script (via
`importlib`, deliberately, so the deployed network cannot silently drift from the trained
one). `hidden_dim=32`, `conv_depth=2`, 6,417 parameters — **4.7 % of the ParT teacher**.

The distillation script wraps it in `StudentNet`, a padded-batch adapter that:

- builds the ΔR < 0.4 radius graph with **wrapped Δφ** (so the ±π seam is connected,
  unlike v1) and flattens padded batches to the `(M, ...)` layout EdgeConv wants;
- applies the same residual-on-PUPPI head, `w = relu(puppiWeight + out)` with the final
  linear zero-initialized. On FPGA this is one extra scalar add before the existing ReLU;
- returns `s ≡ 1`, since the deployed student has no scale head. The teacher's `s` is
  folded into the distillation target instead.

`StudentNet.forward` returns `(w, s)` — the same contract as the teachers — so the step-0
gate and `evaluate_teacher()` work on it unchanged.

---

## 5. Loss — `net.loss_fn_huber_response`

```python
res_term  = mean( huber((METx + genMETx)/σ0) + huber((METy + genMETy)/σ0) )   # σ0 = 30 GeV
resp_term = mean over qT bins of  mean(|response − 1|)                        # bins: 50-100,
                                                                              # 100-200, 200-300, 300+
loss      = res_term + λ · resp_term                                          # λ = 15
```

Every choice here is load-bearing:

- **Huber, not MSE:** the qT spectrum is steep and ttbar has a long tail; MSE lets rare
  300+ GeV events dominate the gradient.
- **Both terms are per-event means,** so the balance is batch-size independent and λ is a
  single interpretable knob.
- **The response term is averaged over qT bins with equal weight, not over events.** This
  is the r1→r2 fix: with an event-level mean, the steeply falling qT spectrum meant the
  low-qT events set the calibration and high-qT response sagged (r1: R = 0.99 at 50 GeV,
  0.75 at 200 GeV). Binning fixed it — r2 holds R ≥ 0.83 across 50–300 GeV.
- `res_term` and `resp_term` are returned separately and logged per epoch. The v1 log
  stored only the blended number, which is exactly what hid the v1 loss imbalance.

Response below the 50 GeV threshold is unconstrained and overshoots (1.0–1.3). Harmless
for trigger use, but do not quote it as a result.

---

## 6. Teacher results (validation split, 248,750 events)

Scaled resolutions σ/R in GeV for `qT` ∈ 50–150 / 150–300 GeV, and the response plateau
averaged over 50–300 GeV. Lower resolution is better; response should be 1.

| model | params | scaled σ(u⊥) | scaled σ(u∥) | plateau R |
|---|---|---|---|---|
| PUPPI baseline | — | 29.7 / 29.2 | 49.9 / 68.3 | 0.816 |
| v1 teacher | 6.4k | 25.5 / 27.5 | 43.5 / 62.5 | 0.782 |
| GraphNet teacher | 129k | 24.3 / 24.5 | 39.5 / 54.6 | 0.866 |
| **ParT teacher (r2)** | **138k** | **24.9 / 24.2** | **31.8 / 53.8** | **0.874** |

Reproduce with:

```bash
cd teacher_deepmet
python compare_all_teachers.py --output ../teacher_comparison_new
```

**How to read this:** GraphNet ties ParT on u⊥ and response but loses decisively on
u∥ at 50–150 GeV (39.5 vs 31.8) — the turn-on-critical bin. So the shared
loss / heads / training recipe accounts for most of the gain over v1, and attention's
global soft neighbourhoods add the remaining u∥ edge on top.

The design doc's 0.95 response gate was not met. Response is calibratable downstream and
scaled resolution is the calibration-fair metric, which r2 wins on every entry; if pure
response ever matters, λ ≈ 30 or a longer schedule is the obvious lever (the loss was
still falling at epoch 30; val best was epoch 22).

`teacher_ckpts_ParT_r2/best.pth.tar` holds **EMA** weights in `state_dict` and the raw
weights in `raw_state_dict`. Always load `state_dict`.

---

## 7. The distillation experiment

### 7.1 Implementation — `teacher_deepmet/trainL1_student_distill.py`

Plain **output-level (response-based) distillation** on per-particle weights. No
temperature, no logit matching (the outputs are regression weights, not a softmax), no
intermediate-feature matching.

```python
task, res_t, resp_t = net.loss_fn_huber_response(w, s, x_pad, mask, y, sigma0=30, lam=15)

with torch.no_grad():
    w_t = teacher.effective_weights(x_pad, mask)     # = s_teacher * w_teacher
distill = ((w - w_t)[mask] ** 2).mean()              # masked: padding excluded

loss = task + beta * distill                         # beta = 20
```

Five details that matter if you modify this:

1. **`effective_weights`, not `w`.** The teacher's calibration scalar `s` is folded into
   the target, because the student has no `s` head and would otherwise be asked to match
   an uncalibrated pattern.
2. **The MSE is masked** (`[mask]`) — averaging over padded slots would scale the loss
   with the padding bucket size and make it batch-composition dependent.
3. **The teacher is fully frozen** (`eval()` + `requires_grad_(False)`) and loaded from
   the EMA `state_dict`.
4. **β = 20 is a strong pull, not a nudge.** At epoch 30 the training loss is
   `task 5.85 + 20 × 0.2287 = 10.42`, so the distillation term is **44 % of the total
   loss** — the student really was pushed hard toward the teacher, which is why the null
   result in §7.2 is informative rather than a sign of an under-weighted term.
   `student_ckpts_distill_ParT_r1/loss.log` logs `train_task` and `train_distill`
   separately for exactly this check.
5. **The scratch baseline is the same script** with `--teacher_ckpt ''` — identical
   architecture, loss, schedule, seed and data order. This is what makes the comparison
   in §7.2 a clean A/B rather than a comparison against a differently-trained model.

Recipe for both runs, identical to the ParT r2 teacher run: AdamW, lr 3e-4 with 1-epoch
linear warmup then cosine decay to 3e-5, weight decay 0.01, batch 128, 30 epochs, grad
clip 1.0, EMA 0.999 on the eval/checkpoint copy, GPU capped at 18 % of the device.

```bash
cd teacher_deepmet
sbatch train_student_distill_job.slurm --ckpts ../student_ckpts_distill_ParT_r1
sbatch train_student_distill_job.slurm --ckpts ../student_ckpts_scratch_r1 --teacher_ckpt ''
python compare_students.py --output ../student_comparison_results
```

### 7.2 Result: distillation gave zero net gain

| model | params | scaled σ(u⊥) | scaled σ(u∥) | plateau R |
|---|---|---|---|---|
| PUPPI baseline | — | 29.7 / 29.2 | 49.9 / 68.3 | 0.816 |
| v1 teacher | 6.4k | 25.5 / 27.5 | 43.5 / 62.5 | 0.782 |
| **student, scratch** | **6.4k** | **25.2 / 25.6** | **43.1 / 57.8** | **0.855** |
| student, distilled | 6.4k | 25.1 / 25.6 | 42.8 / 57.9 | 0.846 |
| ParT teacher (ideal) | 138k | 24.9 / 24.2 | 31.8 / 53.8 | 0.874 |

Every scratch-vs-distilled delta is ≤ 0.3 GeV (≤ 0.6 %), i.e. run-to-run noise, and the
distilled student is slightly **worse** on plateau response (0.846 vs 0.855).

This was **pre-registered** before the runs finished
(`student_comparison_results/prediction_20260804.md`): the predicted bands for all five
metrics were correct, but the explicit bet that "distillation gives 2–3 GeV on scaled
u∥" was wrong — the actual effect is ~0.2 GeV.

### 7.3 Why — the diagnostic

Mimicry worked; transfer did not. Per-particle weight MSE against the teacher target
`s·w`, and mean predicted weight, over 3,072 validation events:

| | weight MSE vs teacher | mean weight |
|---|---|---|
| PUPPI | 0.470 | 0.916 |
| student, scratch | 0.350 | 1.027 |
| **student, distilled** | **0.230** | **0.908** |
| teacher (target) | — | 0.906 |

The distilled student matches the teacher's weights 34 % better than the scratch student
and reproduces the teacher's mean weight to 0.2 % — **and not a single physics metric
moved.**

The conclusion is specific and worth carrying forward: **the teacher's per-particle
weight pattern is not the carrier of its advantage.** The residual ParT edge (≈11 GeV on
u∥ at 50–150 GeV) lives in global-event-context structure that two local EdgeConv hops
cannot represent *at any weight target*. Pushing harder on weight matching cannot reach
it.

There is also a good reason distillation had nothing left to add here: the task loss is
already a direct, low-noise supervised signal over 746k events. Distillation pays for
itself in weak-signal or low-data regimes. Here it only adds a constraint the student
cannot satisfy, and the pull toward the globally-calibrated `s·w` target slightly
degrades response.

Reproduce the diagnostic with `/tmp`-style throwaway code against
`PartMETNetwork.effective_weights` and `StudentNet` — it runs on CPU in a couple of
minutes over ~24 val batches.

---

## 8. What to do next

In priority order.

1. **Ship the scratch-trained student** (`student_ckpts_scratch_r1/best.pth.tar`). Same
   metrics as the distilled one, and no teacher dependency anywhere in the pipeline. It
   beats the v1 teacher on every metric at the same parameter count — that gain came from
   the loss, the residual-on-PUPPI head, the seam fix and the training recipe, **not**
   from distillation.
2. **Give the student a cheap global feature.** This is the highest-value open move.
   The scratch student already closes 94 %/72 % of the PUPPI → ParT gap on u⊥, but only
   **38 %** on u∥ at 50–150 GeV. GraphNet's global exchange node recovered 57 % of that
   same u∥ gap — so the mechanism is confirmed, not speculative. An event-level scalar context (sum-pT,
   multiplicity) broadcast to every node, or an s-head equivalent, is the cheapest FPGA-
   compatible version. Check with the FPGA/FlowGNN side what a broadcast scalar costs
   before designing around it.
3. **Then, if you want, retry distillation** — but on *intermediate features* against the
   GraphNet teacher's global node, not on output weights. §7.3 says output-weight matching
   is saturated; the untested hypothesis is that the *global context vector* is the
   transferable object. GraphNet was kept in the repo for exactly this.
4. **Evaluate on the real goal metric.** Everything so far is resolution/response. Nobody
   has yet produced MET trigger turn-on curves at fixed rate, which is the number the
   collaboration actually cares about.
5. Optional: a λ ≈ 30 or longer-schedule ParT run if pure response (not scaled
   resolution) ever becomes the target.

---

## 9. Dead ends and gotchas — read before you re-derive them

- **`vae_teacher/` is a dead end**, kept for the record. It bridged a full-PV teacher `T`
  to a no-PV student via a VAE encoder producing a latent that feeds `T`'s frozen
  conv/output layers. It was built for a world where the teacher sees PV association and
  the student does not — which is not the L1 world, where nobody has PV. Superseded by
  the ParT teacher. Do not spend time on it without a specific reason.
- **The data format in the pre-2026-08 README was wrong** (d0/dz vs phi/puppiWeight). If
  you find any script indexing cols 4/5 as d0/dz, it is stale. See §3.1.
- **Always load `state_dict` (EMA), not `raw_state_dict`,** from the new checkpoints.
- **Torch is 1.12 on this cluster** (`/home/export/xuantinl/envs/env`) — no
  `scaled_dot_product_attention`, hence the manual attention in `AttnBlock`.
- **GPU budget is capped at ~20 % of the shared A100** (`--mem_fraction 0.18` plus
  `--gres=mps:20` in the SLURM scripts). GraphNet peaks at 12.6/14.3 GB, which is close to
  the cap; ParT and the student are far below it. Do not raise this without asking.
- **Keep the seed-42 / 25 % split.** Every number in this document depends on all models
  being evaluated on the identical validation events.
- **Run `python test_ParT.py`** after touching the teacher. It checks the step-0 PUPPI
  gate, rotation invariance, MET equivariance, padding invariance, and gradient flow.

---

## 10. Commit map

The history is ordered so each step is independently reviewable.

| commit | what |
|---|---|
| `7ef372a` | `loss_fn_huber_response` — Huber + qT-binned response loss (§5) |
| `db8185d` | shard repacking pipeline + `ShardLoader` (§3.2) |
| `79f77af` | **ParT transformer teacher** + design doc + unit tests (§4.2) |
| `9a3a81e` | GraphNet teacher with global exchange node (§4.3) |
| `ee64d76` | 4-way teacher comparison (§6) |
| `c2f9163` | **ParT → EdgeConv student distillation pipeline** (§7.1) |
| `93622e2` | student comparison + pre-registered prediction |
| `fd00f65` | VAE-bridged teacher `T'` — superseded (§9) |
| `4036f33` | L1 column-index fix + original student pipeline (§3.1) |

Pre-2026-08 commits operate on the v1 pipeline and the wrong column assumption; read
`CHANGES_L1_FIX.md` before trusting anything older than `4036f33`.
