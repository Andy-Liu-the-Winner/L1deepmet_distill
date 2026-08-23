# Pre-registered prediction: distilled EdgeConv student (2026-08-04)

Written BEFORE seeing any student results (jobs 30335-30338 running).
Student: 6,417 params (4.7% of ParT teacher), 2×EdgeConv hidden-32, local dR<0.4
graph, no global node, no s-head. Trained with r2 loss (lam=15 binned response)
+ beta=20 distillation vs ParT r2 effective_weights.

## Reference numbers (val, scaled res 50-150/150-300 GeV, plateau R 50-300)

| model | s.u_perp | s.u_par | plateau R |
|---|---|---|---|
| PUPPI | 29.7/29.2 | 49.9/68.3 | 0.816 |
| v1 teacher (173k) | 25.6/27.5 | 43.5/62.5 | 0.782 |
| GraphNet teacher (129k) | 24.3/24.5 | 39.6/54.6 | 0.866 |
| ParT teacher (138k, ideal) | 24.9/24.2 | 31.8/53.8 | 0.874 |

## Predictions

Physics reasoning: u_perp is mostly LOCAL pileup suppression -> small student can
do it. u_par needs event-wide coordination (2 local hops can't see across the
event; GraphNet ablation showed the global node matters) -> student will lose
most of the teacher's u_par gain. Response: the loss response term + distill
targets bake the average calibration into per-particle weights, but without an
s-head the student cannot do event-by-event calibration -> plateau close to but
below the teachers.

| quantity | prediction (distilled) | prediction (scratch) |
|---|---|---|
| plateau R 50-300 | 0.84 +- 0.02 | 0.83 +- 0.02 |
| scaled u_perp 50-150 | 26.5 +- 1.5 | 27.5 +- 1.5 |
| scaled u_perp 150-300 | 26.5 +- 1.5 | 27.5 +- 1.5 |
| scaled u_par 50-150 | 44 +- 3 | 47 +- 3 |
| scaled u_par 150-300 | 59 +- 3 | 62 +- 3 |

## Headline bets

1. Distilled student ~= v1 teacher overall despite 27x fewer params (beats it on
   response and high-qT u_perp; ties within ~2 GeV on u_par).
2. Distillation net gain (scratch - distilled) is real but modest: ~2-3 GeV on
   scaled u_par, ~1 GeV on u_perp, ~0.01 on plateau R.
3. Gap to ideal (ParT) stays large on u_par 50-150 (student ~44 vs teacher 31.8):
   capacity/receptive-field limited, not training limited. If this bet is wrong
   and the student lands near GraphNet (~40), the global-node capacity argument
   weakens and an s-head-like cheap global feature for the student becomes the
   obvious next move.
