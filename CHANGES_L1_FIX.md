# L1 DeepMET Bug Fixes

## Column Index Mismatch

The L1 data format is `[pt, px, py, eta, d0, dz, pdgid, charge]`, but the code treated column 0 as `px` and column 1 as `py`. In reality, column 0 is `pt`, column 1 is `px`, and column 2 is `py`.

This broke all MET calculations (summing `weights * pt` instead of `weights * px`) and phi computations (`atan2(px, pt)` instead of `atan2(py, px)`).

### Files changed

**teacher_deepmet/model/net.py**
- All loss functions (`loss_fn_weighted`, `loss_fn_response_tune`, `loss_fn_response_binned`, `loss_fn`) and metric functions (`u_perp_par_loss`, `resolution`): changed `px=prediction[:,0]` / `py=prediction[:,1]` to `px=prediction[:,1]` / `py=prediction[:,2]`

**teacher_deepmet/trainL1.py, evaluate.py, train_teacher_optimized.py**
- `phi = torch.atan2(data.x[:,2], data.x[:,1])` (was `atan2(data.x[:,1], data.x[:,0])`)

**student_deepmet/model/net.py**
- Same column index fixes as teacher

**student_deepmet/trainL1.py, evaluate.py**
- Same phi fix as teacher

## Student Activation

`StudentNet.forward()` used `torch.sigmoid(weights)` which caps output to [0,1]. The teacher uses `nn.ReLU()` which is unbounded. The distillation loss (MSE between student and teacher outputs) forced the student to collapse since it could never match teacher weights above 1.

Changed to `F.relu(weights)` in `student_deepmet/model/net.py`.

## Debug Print Removal

Removed all commented-out debug print statements from the core training and evaluation files:

- `teacher_deepmet/model/net.py` — shape/value prints in all loss functions and `resolution()`
- `teacher_deepmet/trainL1.py` — per-batch data print, learning rate print
- `teacher_deepmet/evaluate.py` — result shape prints, per-bin type/value prints that caused >30 min eval time
- `student_deepmet/model/net.py` — shape/value prints in all loss functions and `resolution()`
- `student_deepmet/evaluate.py` — result shape prints, per-bin type/value prints
