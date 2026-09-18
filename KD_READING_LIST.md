# Knowledge distillation — annotated reading list for this project

Companion to [`DISTILLATION.md`](DISTILLATION.md). Every entry says **why it matters for
*this* project**, not just what it is. Links verified 2026-09-18.

Our situation, so the relevance notes make sense:

- **Regression, not classification.** We match per-particle weights with MSE. There is no
  softmax, so Hinton's temperature has no meaning here — most of the KD literature does
  not transfer unchanged.
- **Teacher:** ParT transformer, 138k params, with rotation invariance built in and a
  global receptive field. **Student:** EdgeConv GNN, 6.4k params, two local ΔR<0.4 hops,
  no global context. 21× capacity gap *and* an architecture-family gap.
- **Result so far:** output-level distillation was a null result. The student matched the
  teacher's weights 34 % better than a scratch student and no physics metric moved
  (`DISTILLATION.md` §7).

---

## Read these three first (≈1 hour)

1. **Stanton et al., "Does Knowledge Distillation Really Work?"** (NeurIPS 2021) —
   [arXiv:2106.05945](https://arxiv.org/abs/2106.05945)
   The paper that names our exact problem: **student fidelity and student generalization
   are different things.** They find students that generalize better without matching the
   teacher; we found the mirror image — a student that matched the teacher and gained
   nothing. Either way the lesson is the same, and it is the single most important
   citation for defending our null result to a referee or a group meeting.

2. **Romero et al., "FitNets: Hints for Thin Deep Nets"** (ICLR 2015) —
   [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
   The origin of middle-layer distillation: pick a teacher "hint" layer and a student
   "guided" layer, add a learned regressor to reconcile the widths, pretrain the student
   up to that layer, then do normal KD. Our next experiment is a FitNets-shaped one —
   teacher `d=64` vs student `hidden_dim=32` means the regressor is mandatory.

3. **Bal et al., "Distilling particle knowledge for fast reconstruction at high-energy
   physics experiments"** (Mach. Learn.: Sci. Technol. 5 025033, 2024) —
   [arXiv:2311.12551](https://arxiv.org/abs/2311.12551)
   The closest published prior art to what we are doing: an **event-level GNN teacher
   distilled into a small particle-level student ("DistillNet")** for deciding whether a
   particle comes from the primary vertex, explicitly aimed at CPU/FPGA deployment at
   HL-LHC. Same problem shape, same deployment constraint, adjacent physics. Read it
   before designing the next run.

---

## A. Foundations — where our loss actually comes from

Our objective is `MSE(w_student, s·w_teacher)`, which descends from the *regression*
branch of KD, not the softmax branch. Worth knowing which ancestor you are citing.

- **Buciluă, Caruana & Niculescu-Mizil, "Model Compression"** (KDD 2006) —
  [ACM](https://dl.acm.org/doi/10.1145/1150402.1150464)
  The actual original, nine years before Hinton. An ensemble labels a synthetic transfer
  set and a small net regresses onto those labels. This is structurally what we do.
- **Ba & Caruana, "Do Deep Nets Really Need to be Deep?"** (NeurIPS 2014) —
  [arXiv:1312.6184](https://arxiv.org/abs/1312.6184)
  L2 regression on teacher *logits* rather than softened probabilities. Our masked MSE on
  weights is the direct analogue; cite this, not Hinton, for the form of our loss.
- **Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network"** (2015) —
  [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
  The canonical reference. Read for the "dark knowledge" framing and because everyone
  will expect you to have read it — but note that temperature-softened softmax targets do
  not exist in a regression task, so §2 of it does not apply to us.

## B. Why our run came out null — the diagnostic literature

- **Cho & Hariharan, "On the Efficacy of Knowledge Distillation"** (ICCV 2019) —
  [arXiv:1910.01348](https://arxiv.org/abs/1910.01348)
  Bigger teachers do not monotonically make better students; past a point the **capacity
  gap** makes distillation *hurt*. We are at a 21× gap (138k → 6.4k), well into the
  regime they describe. Their mitigation is early-stopping the teacher.
- **Mirzadeh et al., "Improved Knowledge Distillation via Teacher Assistant"** (AAAI 2020)
  — [arXiv:1902.03393](https://arxiv.org/abs/1902.03393)
  The standard capacity-gap fix: insert an intermediate-size teacher. We already have a
  natural candidate — **GraphNet (129k) is a message-passing network, the same family as
  the student**, so it is a teacher assistant in *architecture* space even though it is
  not smaller. Distilling ParT → GraphNet → student is a cheap, well-motivated experiment.
- **Beyer et al., "Knowledge distillation: A good teacher is patient and consistent"**
  (CVPR 2022) — [arXiv:2106.05237](https://arxiv.org/abs/2106.05237)
  Treats KD as **function matching** and shows the gains often need *very* long schedules —
  far longer than a normal training run — plus consistent teacher/student input views.
  **This is a real caveat on our null result:** our runs were 30 epochs. Before concluding
  distillation cannot help, someone should try one long-schedule run. Cheap to do.

## C. Middle-layer / feature distillation — this is the next step

Our finding was that the teacher's *output weights* are not the carrier of its advantage.
That is precisely the argument for moving to intermediate representations. Ordered by how
well they fit our case.

- **Tung & Mori, "Similarity-Preserving Knowledge Distillation"** (ICCV 2019) —
  [CVF PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf)
  **The best structural fit in this whole list.** Instead of matching activations, match
  the *pairwise similarity matrix* of activations. Two consequences for us: (a) it
  sidesteps the 64-vs-32 width mismatch entirely — similarity matrices are the same shape
  regardless of embedding width; (b) an N×N particle-similarity matrix within an event is
  exactly the kind of object the ParT pairwise attention bias computes, i.e. plausibly
  *the* carrier of the global context the student is missing. Start here.
- **Park et al., "Relational Knowledge Distillation"** (CVPR 2019) —
  [arXiv:1904.05068](https://arxiv.org/abs/1904.05068)
  Same instinct, different penalties: distance-wise and angle-wise relations among
  examples. Useful as the second variant to try against SPKD; the angle-wise term has an
  obvious geometric reading for particles in η–φ.
- **Zagoruyko & Komodakis, "Paying More Attention to Attention"** (ICLR 2017) —
  [arXiv:1612.03928](https://arxiv.org/abs/1612.03928)
  Transfer *attention maps* from teacher to student. Directly available to us: our teacher
  is attention-based, so the maps are literally sitting there. The student has no
  attention, but its EdgeConv edge set is a natural target for an attention-shaped
  penalty — the teacher's soft neighbourhood supervising the student's hard ΔR<0.4 one.
- **Heo et al., "A Comprehensive Overhaul of Feature Distillation"** (ICCV 2019) —
  [arXiv:1904.01866](https://arxiv.org/abs/1904.01866)
  The practical engineering paper: *where* to tap features, what transform to apply, and
  a partial-L2 distance that only penalizes in the directions that matter. Read it when
  you are actually wiring the hooks, to avoid re-deriving known mistakes.
- **Tian, Krishnan & Isola, "Contrastive Representation Distillation"** (ICLR 2020) —
  [arXiv:1910.10699](https://arxiv.org/abs/1910.10699)
  Strong method, but the more useful part for us is the paper's **benchmark table**: it
  reimplements a dozen feature-distillation methods under one protocol, which is the
  fastest way to decide what is worth our GPU time.
- **Kim et al., "Factor Transfer"** (NeurIPS 2018) —
  [arXiv:1802.04977](https://arxiv.org/abs/1802.04977)
  Compress teacher features through a learned autoencoder ("paraphraser") before matching.
  Relevant because it is the idea `vae_teacher/` was groping toward — read it before
  anyone proposes reviving that directory.

## D. Regression-specific KD — because A and C are all classification

- **Saputra et al., "Distilling Knowledge From a Deep Pose Regressor Network"** (ICCV 2019)
  — [arXiv:1908.00858](https://arxiv.org/abs/1908.00858)
  The key idea we are currently **missing**: in regression there are no soft labels and
  the teacher is not always right, so weight the imitation loss by teacher confidence
  (*Attentive Imitation Loss*) and treat teacher error as an **upper bound** — stop
  penalizing the student once it beats the teacher. Our loss follows the teacher
  unconditionally, including on events where the teacher is wrong, which is a plausible
  contributor to the distilled student's slightly *worse* response (0.846 vs 0.855). Also
  contains *Attentive Hint Training*, the regression version of FitNets.
- **Chen et al., "Learning Efficient Object Detection Models with Knowledge Distillation"**
  (NeurIPS 2017) —
  [NeurIPS PDF](https://papers.nips.cc/paper/2017/hash/e1e32e235eee1f970470a3a6658dfdd5-Abstract.html)
  Source of the **teacher-bounded regression loss** for bounding-box regression — the same
  "only imitate when the teacher is better than ground truth" trick, in the detection
  setting. Short, and the loss drops straight into our training step.

## E. GNN-specific KD — the student is a GNN

- **Yang et al., "Distilling Knowledge from Graph Convolutional Networks"** (CVPR 2020) —
  [arXiv:2003.10477](https://arxiv.org/abs/2003.10477)
  Introduces **Local Structure Preserving (LSP)**: match the *distribution of similarities
  over each node's neighbourhood* rather than node embeddings. The first KD method written
  for GNNs, and the natural baseline for any structural distillation we try.
- **Joshi et al., "On Representation Knowledge Distillation for Graph Neural Networks"**
  (IEEE TNNLS 2022) — [arXiv:2111.04964](https://arxiv.org/abs/2111.04964)
  Argues LSP's *local* structure is the wrong target and proposes **G-CRD**, which
  preserves *global* topology contrastively — and beats LSP across 4 datasets and 14 GNN
  architectures. Given that our whole diagnosis is "the missing knowledge is global, not
  local", this is the most directly on-point method paper in section E.
- **Tian et al., "Knowledge Distillation on Graphs: A Survey"** (2023) —
  [arXiv:2302.00219](https://arxiv.org/abs/2302.00219)
  Use as a map, not a read. Good for checking whether an idea already exists before
  spending a week on it.

## F. Cross-architecture — transformer teacher, non-transformer student

- **Liu et al., "Cross-Architecture Knowledge Distillation"** (ACCV 2022) —
  [arXiv:2207.05273](https://arxiv.org/abs/2207.05273)
  Exactly our mismatch (transformer → convolutional student). Rather than matching
  features directly, they project both into shared spaces with a *partially cross
  attention projector* and a group-wise linear projector. The projector design is the
  transferable part; our student is a GNN rather than a CNN but the geometry of the
  problem is the same.
- **Touvron et al., "Training data-efficient image transformers & distillation through
  attention" (DeiT)** — [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
  Mostly of interest for the *reverse* direction (CNN teacher → transformer student) and
  the distillation-token mechanism. Skim; useful background on why architecture-family
  mismatch is its own problem rather than a detail.

## G. HEP precedents — cite these in any writeup

- **Bal et al., "Distilling particle knowledge for fast reconstruction at HEP
  experiments"** (MLST 2024) — [arXiv:2311.12551](https://arxiv.org/abs/2311.12551)
  Listed again from the top three because it is the paper to benchmark ourselves against:
  event-level GNN teacher → small particle-level student, PV-origin task, CPU/FPGA target.
- **Liu et al., "Efficient and Robust Jet Tagging at the LHC with Knowledge Distillation"**
  (ML4PS @ NeurIPS 2023) — [arXiv:2311.14160](https://arxiv.org/abs/2311.14160)
  Distillation transfers a teacher's **symmetry inductive bias** (Lorentz equivariance) to
  a student that lacks it, improving robustness. Sharply relevant: our ParT teacher has
  exact *rotation* invariance built in and our EdgeConv student does not. If any structural
  knowledge should be transferable here, that symmetry is the prime candidate — and it is
  a hypothesis we have not tested.

---

## If you only run one more experiment

Ranked by expected information per GPU-hour:

1. **Give the student a global feature first** (`DISTILLATION.md` §8, item 2). It is not
   distillation at all, and the evidence that it works is already in hand — GraphNet's
   global node recovered 57 % of the u∥ gap. Do this before anything in this list.
2. **Then SPKD or G-CRD on particle-similarity matrices**, with a FitNets-style regressor
   for the 64→32 width gap — testing the specific hypothesis that the missing knowledge is
   relational/global, which is what §7.3 of the handoff doc concluded.
3. **Add the teacher-bounded regression loss** from Saputra / Chen to whatever you run.
   It is a few lines, it is theoretically the right thing for a regression task, and it
   targets the one metric distillation actively hurt.
4. **One long-schedule rerun of the existing setup** before anyone writes "distillation
   does not work here" in a note — Beyer et al. is a real caveat on a 30-epoch conclusion.
