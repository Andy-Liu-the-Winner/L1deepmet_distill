# Teacher Redesign — "L1-GraphNet": Graph-based counterpart to L1-ParT

Companion spec to `teacher_ParT_DESIGN.md`. Same data facts, same loss, same training recipe,
same output heads — **only the trunk differs**: message passing on an explicit graph instead
of full attention. Purpose: a controlled architecture comparison (GNN vs Transformer) where
everything else is held fixed.

Read `teacher_ParT_DESIGN.md` §0 first (true data format `[pt, px, py, eta, phi, puppiWeight,
pdgid, charge]`, baseline numbers, training-loop bugs). Everything there applies here.

## 1. Model architecture ("L1-GraphNet")

### Inputs — identical to ParT

- Continuous: `log(pt)`, `eta`, `puppiWeight` (no px/py/raw phi → rotation-invariant)
- `|pdgid|` vocab {11,13,22,130,211} → Embedding(5, 8); `charge+1` → Embedding(3, 8)
- Node encoder: `Linear(3+8+8 → d) → GELU → LayerNorm`, d = 64

### Graph construction — fixed and PRECOMPUTED

The v1 radius graph is built on raw (η, φ) → particles across the ±π seam are never
connected. Fix: build the radius graph on the 3D embedding **(η, cosφ, sinφ)** with the
same r = 0.4. Chord distance 2·sin(Δφ/2) ≈ Δφ for Δφ ≤ 0.4 (error < 0.3%), so this is the
same ΔR graph but with correct wraparound.

Key tactic: **the graph depends only on (η, φ), which never change → compute edge_index and
edge_attr ONCE in the repacking job and store them in the shards.** No torch_cluster call at
train time at all (v1 rebuilds the graph on every batch, every epoch).

Edge attributes (3 dims, stored): `Δη`, `wrap(Δφ) ∈ (−π,π]`, `ΔR`.
`to_undirected` as in v1; `max_num_neighbors=255`.

### Trunk — 3 message-passing blocks with a global exchange node

Per block (custom `MessagePassing`, aggr='mean'; write once, reuse 3×, unshared weights):

```
message  m_ij = MLP_e([x_i, x_j, e_ij])            # Linear(2d+3→d) → GELU → Linear(d→d)
aggregate a_i = mean_j m_ij
update    x_i ← LayerNorm( x_i + MLP_n([x_i, a_i, g]) )   # Linear(3d→d) → GELU → Linear(d→d)
global    g   ← g + MLP_g([mean_i x_i, max_i x_i])        # Linear(2d→d) → GELU → Linear(d→d)
```

- `g` is a per-event context vector (init = zeros, updated after each block, broadcast to all
  nodes of that event via `data.batch` indexing; use `scatter_mean` / `scatter_max` for pools).
  This is the graph-world substitute for attention's global receptive field: MET is a global
  quantity, and v1's two local hops at ΔR=0.4 cannot see across the event. Without `g` the
  comparison against ParT would be unfair by construction.
- Residual + LayerNorm on nodes (v1 has BatchNorm and a *single Linear* as the whole EdgeConv
  message — no nonlinearity in the message at all).
- d = 64, 3 blocks → ~160k params, same budget class as ParT (~200k). Keep them within
  ~±30% of each other; don't "win" the comparison with parameter count.

### Outputs — identical to ParT

1. `w_i = relu(puppiWeight_i + δ_i)`, δ head `Linear(d→d/2) → GELU → Linear(d/2→1)`,
   final layer zero-init → step 0 reproduces the PUPPI baseline exactly.
2. Per-event scale from the global node: `s = 1 + 0.5·tanh(Linear(d→1)(g))`, zero-init.
   Loss uses `MET = s · Σ w_i p_i`. Export `s·w_i` as the distillation target.

## 2. Loss, training, evaluation — identical to ParT

Use the same `loss_fn_huber_response` (Huber/σ0=30 + λ·mean|response−1| over qT>50, λ=5), the same
repacked shards, `shuffle=True`, batch 256, AdamW 3e-4 → cosine to 3e-5, 1-epoch warmup,
30 epochs, grad clip 1.0, EMA 0.999, split seed 42. Same must-pass gates:
response plateau ≥ 0.95, σ(u⊥) ≤ 17 GeV, and the step-0 sanity check (untrained model ==
PUPPI baseline numbers, since both heads are zero-init).

Unit tests (same as ParT plus one graph-specific):
- Rotation invariance: rotate all φ by a random angle → rebuild graph → per-particle weights
  identical within fp tolerance (edge_attr Δφ is rotation-invariant, so this must hold).
- Seam test: take an event, rotate φ so a dense cluster straddles ±π → edge count must be
  unchanged (v1 fails this).

## 3. Files and comparison protocol

- `model/graphnet_met_network.py` — the model above.
- `trainL1_ParT.py` should take `--arch {part, graph}` so both trainings share the loader,
  loss, recipe, and logging code path; only the model class differs.
- Checkpoints: `../teacher_ckpts_ParT/`, `../teacher_ckpts_GraphNet/`.
- Final comparison (same val split, existing `resolution()` + plot machinery), 4 curves per
  plot: **v1 teacher / PUPPI baseline / GraphNet / ParT** for response vs qT, σ(u⊥),
  σ(u∥), plus a table with params, epoch time, and the two loss components.

Expected reading of the outcome:
- Both new teachers ≫ v1 on response → confirms the gains are mostly loss/heads/training (shared parts).
- ParT ≳ GraphNet on resolution → global soft neighborhoods beat a fixed ΔR graph.
- GraphNet ≈ ParT everywhere → the graph inductive bias is sufficient at N~50; pick the
  graph version if its epoch time is better (precomputed edges make it competitive).
- Distillation angle: GraphNet is architecturally homologous to the EdgeConv student
  (message passing → message passing), which may make intermediate-feature distillation
  easier later; ParT is the stronger-ceiling bet. Whichever wins on the gates becomes the
  distillation teacher.
