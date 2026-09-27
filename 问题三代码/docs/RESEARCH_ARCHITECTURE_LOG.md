# Architecture research log

This log records external ideas consulted during the architecture iterations.
The project adapts concepts to its own tensor contract and does not copy an
external model wholesale.

## 2026-09-25: shared/private, deep supervision, and weak-modality robustness

- MISA, ACM Multimedia 2020: <https://github.com/declare-lab/MISA>
  - MIT licensed.
  - Consulted ideas: modality projection into a common dimension,
    shared/private decomposition, reconstruction regularization, and fusion of
    the decomposed components.
- Self-MM, AAAI 2021: <https://github.com/thuiar/Self-MM>
  - MIT licensed.
  - Consulted ideas: separate unimodal prediction paths, fusion prediction,
    and deep supervision that prevents a weak branch from being starved.
- TFR-Net, ACM Multimedia 2022: <https://github.com/thuiar/TFR-Net>
  - MIT licensed.
  - Consulted ideas: aligned representation learning and robustness to missing
    or unreliable modalities through reconstruction-style auxiliary learning.
- MAG-BERT, ACL 2020:
  <https://github.com/WasifurRahman/BERT_multimodal_transformer>
  - Consulted idea: inject a norm-bounded, gated non-verbal shift into a
    pretrained language representation instead of treating all modalities as
    equally reliable. The repository does not expose a root license file, so
    no source code was copied; the project implementation is original and
    uses the paper-level architectural idea only.

### Dataset-specific observation

The label table is zero-inflated: every Neutral example has regression target
exactly 0, every Negative target is below 0, and every Positive target is above
0. The direct three-class head currently has its largest error on Neutral.

### Proposed architecture innovation

Add a zero-inflated ordinal hurdle head on top of the fused representation:

1. predict Neutral versus Polar;
2. conditionally predict Negative versus Positive;
3. estimate signed non-zero magnitude for regression;
4. adaptively mix the structured posterior with the original evidence head;
5. distribute the structured residual through learned modality gates so the
   additive explanation interface remains valid.

This combines the task factorization suggested by the exact label semantics
with the existing shared/private and deeply supervised HAFusion backbone. It is
an architectural hypothesis, not a claim of improvement until real validation
training confirms it.

### Pretrained text-anchor branch

The frozen 768-dimensional text features limit adaptation to this dataset. A
new branch fine-tunes DeBERTa-v3-small end to end, pools both the first token
and masked token mean, and supports a MAG-style bounded visual shift followed
by a MISA-style shared/private component mixer. Experiments first isolate the
text tower, then enable visual fusion, and finally add the hurdle expert only
if validation evidence supports it.

## 2026-09-25: tri-modal prototype routing

- Supervised Contrastive Learning, NeurIPS 2020:
  <https://github.com/HobbitLong/SupContrast>
  - BSD-2-Clause licensed official implementation.
  - Consulted the positive-pair mask, temperature-scaled similarity, and
    singleton-anchor handling in `losses.py`. The implementation in this
    project is an original single-view adaptation rather than copied source.
- MISA, ACM Multimedia 2020: <https://github.com/declare-lab/MISA>
  - Revisited the shared/private decomposition for all three modalities. The
    pretrained branch previously omitted audio and only decomposed text plus
    vision.

### EXP018 architectural hypothesis

1. encode audio and vision with independent temporal encoders;
2. estimate sample-wise modality reliability before applying norm-bounded
   MAG shifts to the DeBERTa text anchor;
3. mix text/audio/vision plus shared and private components as seven tokens;
4. let three learned label queries cross-attend to all token-level evidence;
5. compare each class-conditioned representation with its own prototype and
   blend that posterior with the direct head through a learnable residual mix;
6. use a small supervised-contrastive auxiliary objective to compact the
   Neutral representation without making the prototype path the sole head.

This is intended to address EXP017's 0.3859 Neutral recall while preserving
the strong polar-class decisions. It remains a hypothesis until the recorded
validation run confirms or rejects it.

## 2026-09-25: causal utterance-context adaptation

- DialogueRNN, AAAI 2019:
  <https://github.com/declare-lab/conv-emotion/tree/master/DialogueRNN>
  - MIT licensed official implementation.
  - Consulted its separation of the current utterance, recurrent global
    context, and attention over prior utterances. No repository code was
    copied into this project.

### EXP019 architectural hypothesis

Sample identifiers encode a video and clip order. Within each already-isolated
split, 55.0% of training samples and 67.2% of validation samples have at least
one previous utterance from the same video. EXP019 constructs a label-free,
causal context from at most two earlier clips, jointly encodes it with the
current utterance, pools the current tokens separately, and injects the prior
context through its own norm-bounded reliability gate. Samples without prior
clips receive a zero context gate. Split boundaries and labels are never used
to construct context, so this does not introduce train/validation/test leakage.

## 2026-09-25: parameter-efficient and adaptive hyper-modality branches

- LoRA, ICLR 2022: <https://github.com/microsoft/LoRA>
  - MIT licensed official implementation; the project uses the maintained
    `peft` implementation to insert low-rank query/value adapters while
    keeping the pretrained backbone frozen.
- ALMT, EMNLP 2023: <https://github.com/Haoyu-ha/ALMT>
  - MIT licensed official implementation.
  - Consulted its language-guided adaptive hyper-modality learning: language
    tokens query acoustic and visual tokens layer by layer, then the learned
    hyper-modality representation is fused back with language.
- DORN, CVPR 2018: <https://github.com/hufu6371/DORN>
  - Consulted the paper/repository-level ordinal decomposition only. The root
    repository declares no license, so no source was copied.

### Implemented hypotheses

- EXP023 applies LoRA adapters to DeBERTa query/value projections and reduces
  trainable parameters from about 146M to about 5.1M.
- EXP024 adds a rank-consistent ordered interval expert with learned Negative /
  Neutral and Neutral / Positive boundaries.
- EXP025 replaces pooled non-verbal injection with four ALMT-inspired adaptive
  hyper-modality tokens built from aligned text/audio/vision sequences, then
  injects the hyper representation into the raw-text DeBERTa anchor through a
  reliability gate. The implementation is original and mask-aware.

## 2026-09-25: capacity-controlled large language anchor

- DeBERTa, ICLR 2021, and DeBERTaV3, ICLR 2023:
  <https://github.com/microsoft/DeBERTa>
  - MIT licensed official implementation.
  - Consulted the disentangled attention and replaced-token-detection backbone
    scaling results. The project uses the maintained Hugging Face checkpoint;
    no model source was copied.
- LoRA, ICLR 2022: <https://github.com/microsoft/LoRA>
  - Reused through PEFT to control trainable capacity on an 8 GB GPU.

### EXP026 architectural hypothesis

EXP025 reached 0.9932 train accuracy but only 0.6387 validation accuracy, so
adding a deeper randomly initialized cross-modal branch worsened the actual
bottleneck. EXP026 instead scales the pretrained language anchor from
DeBERTa-v3-base to DeBERTa-v3-large while training only rank-16 query/value
adapters. The existing bounded audio/vision residual and class-prototype router
remain small task-specific heads. This separates pretrained representation
capacity from dataset-sized parameter growth and tests a qualitatively
different architecture route rather than another scalar hyperparameter sweep.

## 2026-09-25: confidence-aware dynamic residual routing

- MMTM, CVPR 2020: <https://github.com/haamoon/mmtm>
  - MIT licensed official implementation.
  - Consulted its sample-dependent cross-modal squeeze-and-excitation idea.
- MAG-BERT, ACL 2020:
  <https://github.com/WasifurRahman/BERT_multimodal_transformer>
  - Revisited its bounded residual injection principle. No source was copied.

### EXP027 architectural hypothesis

Across EXP017--EXP020 the learned global multimodal residual scale remains near
0.119 for every sample. Error inspection shows why a single scalar is
insufficient: some validation utterances have misleading literal polarity and
need acoustic/visual correction, while ordinary text-dominant samples should
not receive the same shift. EXP027 adds a zero-initialized, sample-wise router
conditioned on the text and multimodal candidates, their absolute/product
interaction, both modality reliabilities, and text uncertainty. It starts
exactly at the EXP020 scalar route and learns only per-sample deviations, so it
can be warm-started without discarding the validated backbone.

### EXP037 architectural hypothesis: context-aware dynamic routing

EXP019 has the strongest Neutral recall (0.625) but loses polar-class accuracy;
EXP027/EXP029 improve the overall trade-off by making the multimodal residual
sample dependent. EXP037 combines those complementary structures: the current
utterance receives only causal, label-free earlier clips from the same video,
while a confidence-aware router controls the acoustic/visual residual after
context adaptation. Because EXP019 already reaches 0.964 train accuracy, its
language encoder stays frozen and only the context/fusion heads and new router
are adapted. The hypothesis is that context supplies Neutral evidence while
the router prevents it from uniformly suppressing strong polar cues.

## 2026-09-25: direction-aware dialogue evidence with an explicit reject route

- DAG-ERC, ACL-IJCNLP 2021: <https://github.com/shenwzh3/DAG-ERC>
  - Official PyTorch repository, Apache-2.0, inspected at commit
    `ba7639dd12cdfa77cbde762f22f94266b30073d2`.
  - Consulted its directed utterance graph, relation-aware attention, and
    recurrent aggregation of selected earlier utterances. No source was copied;
    this project adapts the paper-level idea to split-local text views and a
    shared pretrained encoder.
- DialogueRNN, AAAI 2019: <https://github.com/declare-lab/conv-emotion/tree/master/DialogueRNN>
  - MIT licensed official implementation, repository HEAD inspected at
    `6128ca20e9c736605cce7e99d5d95db0356c35f5`.
  - Revisited its separation of the current utterance from conversational state.

### EXP080 architectural hypothesis

EXP077 improved its standalone macro-F1 over the causal EXP019 model, but its
single merged context gate reached mean reliability 0.8931 on validation and the
model still confused 132 of 184 Neutral samples with polar classes. Merging both
directions into one text pair gives the gate no way to identify which neighbour
is helpful or to reject both.

EXP080 therefore encodes the current utterance, previous context, and following
context as three distinct views through the same pinned DeBERTa backbone. Two
direction-specific bounded residuals compete with an explicit reject route.
Unavailable directions are hard-masked, and direction-tagged tokens are also
exposed to the class-prototype evidence queries. This is a structural context
selection change rather than a threshold or ensemble-weight search.

## 2026-09-25: signed transition and trust-region Neutral correction

- DAG-ERC, ACL-IJCNLP 2021: <https://github.com/shenwzh3/DAG-ERC>
  - Apache-2.0 licensed official code at commit
    `ba7639dd12cdfa77cbde762f22f94266b30073d2`.
  - Reused the paper-level idea that directed relation messages should represent
    a transition between utterances rather than inject an absolute neighbour.
- DialogueRNN, AAAI 2019:
  <https://github.com/declare-lab/conv-emotion/tree/master/DialogueRNN>
  - MIT licensed official code at commit
    `6128ca20e9c736605cce7e99d5d95db0356c35f5`.
  - Reused the separation between the current utterance state and conversation
    state. The implementation here is original.

EXP082/083 encoded signed current-minus-neighbour changes, magnitudes,
interactions, and sentiment-posterior changes, but their unrestricted residual
overfit. EXP084 converted the branch into a trust-region Neutral-vs-Polar
correction: it preserves polar ordering, is hard-bounded by parent uncertainty,
and becomes exactly zero on reject. Even with those invariants it did not beat
EXP018, so the direction is closed rather than tuned further.

## 2026-09-25: retrieval-memory direction under investigation

- kNN-LM, ICLR 2020: <https://github.com/urvashik/knnlm>
  - MIT licensed official code, HEAD inspected at
    `8afab92bfcc8be28eccdf41fb82582a80977346e`.
  - Consulted its datastore-query pattern, distance-derived neighbour posterior,
    and probability-space interpolation with a parametric model.
- Tip-Adapter, ECCV 2022: <https://github.com/gaopengcuhk/Tip-Adapter>
  - Official repository HEAD inspected at
    `d0e2d6f8c5feb8b6ce937b757810761f7155d4d5`.
  - Consulted only the paper/repository-level cache-key and one-hot cache-value
    design because the repository exposes no root license file; no source code
    was copied.

A validation-only feasibility diagnostic using the existing frozen multimodal
features and a class-balanced cache raised EXP064 from 0.7019 / 0.6837 to
0.7102 / 0.6916. This is not yet a selected result because the diagnostic
searched cache settings on validation. It does establish that label memory is
complementary to the seven-model ensemble and motivates a clean train-only
selection protocol for the next architecture loop.

## 2026-09-25: grouped-OOF expert routing and deployment consistency

- MMoE, KDD 2018: <https://github.com/drawbridge/keras-mmoe>
  - MIT licensed official implementation inspected at commit
    `2718f56e4313716fd3a86e9510bc2df0cc366238`.
  - Consulted its task-specific softmax gates over a shared expert bank.
- MMTM, CVPR 2020: <https://github.com/haamoon/mmtm>
  - MIT licensed official implementation inspected at commit
    `1c81cfefad5532cfb39193b8af3840ac3346e897`.
  - Reused the paper-level idea that cross-stream reliability should be
    sample-dependent. No external source was copied.

The first strict grouped-OOF parent combines four task-trained architectures and
reaches 0.6819 accuracy / 0.6557 Macro-F1 in OOF and 0.6882 / 0.6655 on locked
validation. Free multinomial stacking raised OOF accuracy but reduced Macro-F1
and failed on validation, so EXP101 replaces unrestricted logits with a
29-parameter router. Each sentiment class owns a softmax convex gate over the
experts, and the routed posterior is mixed with the uniform parent through a
sample-wise gate capped at 0.45. Uniform initialization is exactly the parent;
the router cannot emit a free residual. EXP101 improves grouped OOF to 0.6919 /
0.6572 and locked validation to 0.6951 / 0.6710, but remains below the success
gate. Exact Neutral preservation (EXP102) and fold-local classwise sparse top-3
routing (EXP103) did not improve the primary OOF ranking and were closed before
another validation read.

The HAFusion OOF audit also exposed a reproducibility failure: EXP098 inherited
the YAML batch size 500 and ran only five batches per epoch. It is invalid, not
an architecture result. EXP099 reran the intended batch size 16 and produced a
valid 0.6510 / 0.6176 OOF artifact; adding it to the four-expert parent reduced
OOF Macro-F1, so the branch was rejected.

EXP104 was intentionally stopped before a fold completed when its from-scratch
training was found not to match EXP027's warm-start deployment recipe. EXP105
implements the corrected nested parent-to-child procedure entirely inside each
outer training fold: three fixed EXP020 parent epochs, compatible state transfer,
one fixed EXP027 router epoch, then held-out video prediction. This preserves the
architectural training path without using held-out fold metrics for checkpoint
selection.

The completed EXP105 OOF result (0.6707 / 0.6451) showed that deployment-faithful
dynamic routing is not complementary enough to the core expert bank. In contrast,
the independently cross-fitted EXP063 NLI Neutral-hurdle expert was weak alone
(EXP108: 0.6480 / 0.6265) yet raised the uniform five-expert system to 0.6884
accuracy / 0.6614 Macro-F1 OOF and 0.6909 / 0.6687 on locked validation
(EXP109/110). This same-direction transfer makes EXP109 the new level-one parent.

Ordinary task-conditional routing on these five experts (EXP111) traded Neutral
F1 for accuracy, while classwise top-4 sparsity (EXP112) reduced both primary
metrics. The next hypothesis therefore changes the training objective rather
than the expert weights or scalar optimizer settings: retain the convex expert
selector and bounded parent fallback, but add a differentiable per-class Dice
surrogate for Macro-F1, with a modest extra Neutral weight. The branch remains
fold-local and OOF-only until it exceeds EXP109 in both primary metrics. This is
an objective-level architectural change designed to align gradient learning
with the non-decomposable selection metric and protect the minority Neutral
class; it is not accepted until hard-label OOF metrics confirm it.

EXP113 rejected this hypothesis: grouped OOF reached only 0.6866 accuracy /
0.6540 Macro-F1, and Neutral F1 dropped from the EXP109 parent's 0.4810 to
0.4605. The soft objective drove mean fallback trust to 0.4452 against a 0.45
cap, so the failure is not lack of route capacity; it is unstable transfer of
fold-local routing decisions. No validation predictions were loaded. The
level-two convex-router family is closed, and the next experiment constructs
the missing EXP029 member with an exact fold-local EXP020 -> EXP027 -> frozen
EXP029 training path.

## 2026-09-25: strict-seven parent and architecture stress tests

EXP114 reproduced the complete EXP020 -> EXP027 -> frozen EXP029 path inside
each held-out video fold. Its 0.6586 / 0.6300 result established that the
single-split EXP029 checkpoint was not a standalone generalizable model, but
its error diversity still justified one fixed ensemble check. Adding it to the
existing bank produced EXP116/117, the fully cross-fitted strict-seven parent:
0.6913 / 0.6621 OOF and 0.7019 / 0.6837 on locked validation. This is the
current best integrity-preserving system; test remains unread.

EXP118--126 stress-tested qualitatively different structural assumptions over
that parent: two-query hierarchical pooling, uncertainty-bounded hurdle
residuals, frozen semantic prototypes, a heteroscedastic ordinal distribution,
bidirectional conversational context, low-rank tensor fusion, a repaired gated
HAFusionV2, and a semantic selective MMoE. Only the heteroscedastic ordinal
residual passed the OOF joint gate (EXP121), and it transferred negatively to
locked validation. EXP123 is particularly important: its full-OOF posterior
selection appeared positive, while true outer-fold selection fell to 0.6887 /
0.6587. This confirms that level-two configuration selection must itself be
cross-fitted.

EXP127 imposed a stronger semantic invariant: the current utterance alone
fixed Positive-versus-Negative odds, while context, audio, and vision could
only alter Neutral mass. The factorization was interpretable but unstable
across folds. EXP130 instead added a bounded low-rank tri-modal multiplicative
interaction to a fold-local DeBERTa-base parent. Its learned mix stayed small,
yet its errors remained correlated with the parent. Fixed ensemble checks
EXP128 and EXP131 reduced both OOF metrics, so neither branch was promoted.

EXP129 then quantified discrete ensemble-selection overfit. A seven-member
subset chosen against all OOF labels reached 0.6999 / 0.6708, whereas honest
outer-fold selection reached only 0.6895 / 0.6581. EXP132 found the same pattern
for class-balanced retrieval memory. A separately frozen, vision-only k=4
group-exclusive memory was therefore replicated as EXP133 and improved OOF to
0.6931 / 0.6634, but locked valid remained 0.7019 / 0.6832. The memory signal
is real but too small and domain-specific to replace EXP117.

## 2026-09-25: selective semantic confirmation

- SelectiveNet, ICML 2019: <https://github.com/geifmany/selectivenet>
  - Official repository HEAD inspected at
    `a6d0a8fd33dae61da910b61a2aae93102d2d4869`.
  - Consulted the selective-prediction principle. The repository exposes no
    root license file, so no source code was copied.
- Deep Gamblers, NeurIPS 2019:
  <https://github.com/Z-T-WANG/NIPS2019DeepGamblers>
  - MIT licensed official implementation inspected at commit
    `0d1b595611a8bc653fddfdf1419bd8dbde153532`.
  - Consulted the explicit reject-option principle; no source was copied.
- GoEmotions, ACL 2020:
  <https://github.com/google-research/google-research/tree/master/goemotions>
  - Apache-2.0 licensed reference implementation inspected through repository
    commit `d36068b845da4c2b24927fee2cea1e6ef98dadda`.

EXP134 permitted a polar-to-Neutral correction only when the locked parent was
near its Neutral boundary, the cross-fitted GoEmotions calibrator was strongly
Neutral, the raw ontology Neutral probability agreed, and every polar-emotion
probability stayed low. The correction was the smallest possible probability
projection that crossed the Neutral decision face and exactly preserved
Positive-versus-Negative odds. Rule selection was repeated inside each outer
video fold, with an explicit no-op reject route. The resulting honest OOF score
was 0.6892 / 0.6609 versus 0.6913 / 0.6621 for strict7. Three folds selected a
nonzero rule on their fit partition, but each rule regressed on its held-out
partition. Official validation was not loaded. This closes thresholded
GoEmotions confirmation and motivates a new independently pretrained base
representation rather than another level-two calibration rule.

## 2026-09-25: EmoBERTa emotion-space transfer

- EmoBERTa, ACL Findings 2021: <https://github.com/tae898/erc>
  - MIT licensed official repository inspected at commit
    `faf25370d8805a1975bda768cb94288d5d72c490`.
  - The `tae898/emoberta-large` and `tae898/emoberta-base` Hugging Face weights
    were pinned to revisions `8934b68e8b0d9fc3cd961cc7e7605533c7081e59`
    and `64377bdd2a1d7bc5ecdac9a4fbd219002663df1e` respectively.
  - Reused the paper-level idea of transferring a seven-emotion utterance
    representation trained on MELD/IEMOCAP. No external implementation code was
    copied.

EXP135 froze EmoBERTa-large, concatenated its seven emotion logits with its CLS
embedding, and fitted fold-local PCA-128 plus shrinkage LDA. Adding that expert
to strict7 reached 0.6916 accuracy / 0.6598 Macro-F1 OOF: the accuracy increase
did not compensate for the Macro-F1 loss, so validation stayed locked.

EXP136 instead made EmoBERTa-base trainable inside the existing reliability-
gated audio/vision architecture. Its native seven-way head was grouped into
Negative = anger/sadness/disgust/fear, Neutral = neutral/surprise, and Positive
= joy, then probability-mixed with the learned multimodal head. Fixed-three-
epoch five-fold grouped OOF reached 0.6713 / 0.6454, with Neutral F1 0.4616.
Although weak alone, its different pretraining improved the fixed strict8 OOF
ensemble very slightly to 0.6916 / 0.6623 (EXP137). The deployment-faithful
full-train replica (EXP138) was trained for the same fixed three epochs before
validation was materialized once; it scored 0.6566 / 0.6392 alone. EXP139 then
combined it with strict7 and fell to 0.6896 / 0.6702 on locked validation. The
OOF increment therefore did not transfer, EXP117 remains the best validated
system, and test remains sealed.

The next hypothesis uses the same pinned EmoBERTa-base but changes the model
graph rather than ensemble weights: current, previous, and following utterances
are encoded separately through the shared backbone. A learned three-way router
can reject context or choose one direction; unavailable directions are hard
masked. The current utterance retains the native seven-emotion probability
anchor, while routed context enters only through bounded representation shifts
and route-weighted prototype evidence. This directly tests EmoBERTa's reported
past-and-future context advantage under group-separated OOF evaluation.

EXP140 implemented that hypothesis as one end-to-end model. Relative to the
current-only EXP136 expert, grouped OOF improved from 0.6713 / 0.6454 to 0.6751
/ 0.6501, Neutral F1 improved from 0.4616 to 0.4747, and Pearson remained
0.7584. The router was active rather than collapsed: fold-zero mean reject,
previous, and following weights were 0.4491, 0.2580, and 0.2929. Adding EXP140
to strict7 under a predeclared uniform rule produced 0.6916 / 0.6625 OOF
(EXP141), a small gain in both primary metrics with no searched ensemble
weights. The full-train fixed-three-epoch replica reached 0.6621 / 0.6392 on
locked validation (EXP142), and the deployment ensemble reached only 0.6951 /
0.6748 (EXP143). Thus the architecture improved train-domain OOF but its
context decisions did not transfer to the validation domain; EXP117 remains
best and test remains sealed.

The follow-up architecture is a unified contextual valence-boundary model, not
a post-hoc rule. The current utterance alone produces Positive-versus-Negative
odds. Previous/following context, audio, and vision are allowed to change only
the Neutral mass through a learned interval expert; an explicit algebraic
invariant prevents them from reversing polarity. A bounded rank-four
tri-modal multiplicative interaction supplies joint non-verbal evidence to
that Neutral boundary. This combines direction-aware reject routing, emotion-
space transfer, structured valence factorization, and low-rank multimodal
interaction in a single differentiable graph. Its CUDA test verifies both
gradient flow and the polar-odds invariant before any OOF run.

EXP144 tested the hard invariant. Its first three held-out folds pooled to
0.6658 / 0.6403 versus 0.6717 / 0.6433 for the same EXP140 folds, so it was
stopped under an explicit structural-futility rule before spending two more
folds. The partial artifact is retained but is not a complete OOF result and
cannot enter any ensemble.

EXP145 replaced the hard constraint with a bounded uncertainty-aware polar
residual. This is a model-graph change: a learned trust route can alter current-
utterance polarity by at most 0.75 logits, while the Neutral interval remains
separate. It reached 0.6766 / 0.6492 alone. The fixed strict8 ensemble improved
all three class F1 values and reached 0.6937 / 0.6644 OOF (EXP146), but the
trust route averaged above 0.92 in inspected folds and 0.974 in the full-train
replica. That replica scored only 0.6374 / 0.6074 validation (EXP147), and the
deployment ensemble reached 0.6951 / 0.6735 (EXP148). The freely learned trust
route therefore transfers poorly despite honest OOF complementarity and is
closed.

The next architecture removes that unstable degree of freedom. It converts the
pinned EmoBERTa seven-emotion head into grouped Negative/Neutral/Positive
probabilities, takes the native log Positive/Negative odds as the primary
polarity prior, and permits only a bounded current-utterance adapter around the
prior. Previous/following context and non-verbal modalities remain confined to
the Neutral interval. This makes the transferred emotion classifier an actual
structural anchor instead of merely one probability-mixture member.

EXP149 completed five-fold grouped OOF for that native-prior graph at 0.6754
accuracy / 0.6467 Macro-F1, with Neutral F1 0.4644. The fixed strict7-plus-one
combination reached 0.6931 / 0.6636 (EXP150), exceeding EXP117 on both OOF
metrics without any weight or subset search. Although this is below EXP146's
OOF result, its polarity correction is structurally bounded instead of governed
by the fold-unstable free trust gate, so it remains eligible for one fixed-epoch
deployment transfer check.

Two larger context architectures then tested whether capacity or explicit
dialogue topology was the missing ingredient. EXP151 froze EmoBERTa-large node
representations and learned an independent previous/following directed dialogue
graph classifier. It reached only 0.5753 / 0.5432 OOF. EXP152 instead initialized
a graph residual at the strict7 parent and constrained it to alter only the
Neutral boundary; it still fell to 0.6878 / 0.6507. The dataset has about 2.2
utterances per dialogue group on average, so graph message passing has too
little repeated topology and too much fold-specific freedom. The dialogue-graph
route is closed.

EXP153 tested backbone scale rather than dialogue topology: EmoBERTa-large with
rank-eight LoRA, a learned four-layer scalar mix, prototype fusion, and a
bounded low-rank audio/vision interaction. The first two folds pooled to 0.6657
/ 0.6324, below EXP140's 0.6694 / 0.6424 on the same folds. It was stopped by
the declared structural-futility rule and is not a complete OOF result. This
rules out simple backbone scaling as the explanation for the remaining Neutral
error.

EXP154 finally applied a fixed, non-searched structural-consensus check to the
three independently OOF-qualified predictions: EXP117, the EXP121
heteroscedastic ordinal distribution residual, and the EXP133 group-exclusive
vision memory. Equal averaging produced 0.6913 accuracy / 0.6617 Macro-F1 OOF,
tying EXP117 accuracy while lowering Macro-F1. It failed the joint gate, so
locked validation was not loaded. The disagreement and memory corrections do
not compose additively; future work must change the representation and decision
graph rather than stack more post-hoc corrections.

The eligible deployment check for the native-prior model was then completed.
The fixed-three-epoch full-train replica scored 0.6442 accuracy / 0.6071
Macro-F1 on locked validation (EXP155), with Neutral F1 only 0.3881. Its fixed
strict8 deployment ensemble retained 0.7005 accuracy but reached only 0.6792
Macro-F1 (EXP156), below EXP117 by 0.45 percentage points. A bounded native
emotion prior therefore does not solve the domain-shifted Neutral boundary;
this EmoBERTa-polarity route is closed.

## 2026-09-25: agreement-aware dual-pretraining architecture

EXP157 changed the representation graph rather than adding another calibrated
posterior. A trainable DeBERTa-v3-base tower retained task-domain semantics,
while a separately tokenized and frozen EmoBERTa-base tower supplied a stable
seven-emotion coordinate system. A zero-initialized agreement module projected
the frozen CLS/mean representation into the task space, exposed it as a
class-prototype evidence token, and learned two independently bounded decision
axes: Neutral-vs-Polar (maximum 0.75 logit) and Positive-vs-Negative (maximum
0.25 logit). Both axes had sample-wise rejection gates; the external tower
could not overwrite the parent at initialization or during unstable training.
The CUDA test verified normalized probabilities, gradient flow into both new
residual paths, and zero gradients in the frozen emotion tower.

The five grouped folds used the deployment-faithful phased recipe: train the
DeBERTa dynamic multimodal parent for three fixed epochs, transfer compatible
weights, freeze the primary encoder, and train the dual-evidence child for one
fixed epoch. OOF reached 0.6748 accuracy / 0.6340 Macro-F1 (EXP157). The
rejection mechanism behaved as designed--mean emotion decision reliability was
only about 0.008--0.035 in inspected folds--but the remaining learned changes
favoured polar accuracy and reduced Neutral F1 to 0.4135. Its fixed addition to
strict7 reached only 0.6892 / 0.6581 (EXP158), below EXP117 on both metrics, so
locked validation was not loaded. Independent tokenization and safe evidence
rejection are implemented and reusable, but the EmoBERTa emotion space is not
the missing Neutral representation for this domain.

## 2026-09-25: sparse semantics, uncertainty hurdles, and latent subcenters

EXP159 introduced a deliberately non-neural error basis: fold-local word
unigrams/bigrams plus character 3--5-grams, a balanced logistic classifier, and
a sparse ridge intensity regressor. It reached only 0.5794 / 0.5275 OOF. Adding
it uniformly to strict7 raised accuracy slightly but lowered Macro-F1 to 0.6611
(EXP160), so the full sparse classifier was not transferred to validation.

EXP161 reused the sparse representation more selectively. A binary Neutral
hurdle combined sparse text with 78 numeric features derived from the seven
deep experts: posterior/log-posterior vectors, entropy, margin, Neutral logit,
regression predictions, ensemble means, and disagreement. The outer-fold model
was trained only on the other video groups. Its correction had an exact
Negative/Positive odds invariant and a bounded 0.75-logit Neutral residual. It
raised Neutral recall from 0.4960 to 0.6029 but reduced accuracy to 0.6728; a
fixed parent consensus also failed (EXP162).

A read-only trust-region diagnostic found a narrow 0.125 maximum shift that
tied parent accuracy and improved Macro-F1 on the discovery folds. Because that
choice was posterior, EXP163 prospectively froze it and rebuilt the meta folds
with independent seed 3407. The Macro-F1 gain reproduced at 0.6636, but accuracy
was 0.6901--four correct decisions below EXP117--so validation remained locked.
Error analysis localized the loss to Positive-to-Neutral changes. EXP165 then
froze a Positive-protection invariant and used a third independent meta-fold
seed (42); it reached only 0.6904 / 0.6614. Thus the sparse hurdle contains real
Neutral signal, but no replicated variant meets the joint accuracy constraint.

EXP164 tested whether Neutral heterogeneity was fundamentally a single-prototype
geometry problem. Each class received three normalized angular subcenters,
log-sum-exp class evidence, a class-internal diversity penalty, and a maximum
0.35 posterior mix. The complete dynamic DeBERTa parent was frozen and only the
subcenter head trained, preventing parent drift. After three folds, pooled
performance was 0.6565 / 0.6312 versus 0.6663 / 0.6392 for the same parent
folds; Neutral F1 improved by only 0.0011. The remaining two folds were stopped
under structural futility. Multi-center geometry is implemented and tested but
does not explain the remaining error.

## 2026-09-25: explicit conflict evidence and aligned temporal incongruity

EXP166 introduced a complete subjective-logic decision branch over the phased
DeBERTa dynamic-fusion parent. Text, audio, and vision each emitted a binary
polarity opinion and a non-negative evidence strength. Their reliable evidence
masses defined explicit conflict and ignorance coordinates. A zero-initialized,
uncertainty-gated head could move only the Neutral-vs-Polar logit, so the
parent's Negative/Positive odds were preserved exactly. CUDA tests verified the
identity initialization, probability normalization, invariant, and gradient
flow. Five-fold grouped OOF reached 0.6757 accuracy / 0.6404 Macro-F1. The head
learned non-zero shifts, but Neutral F1 remained 0.4391. Adding the model as a
fixed eighth strict7 member produced only 0.6892 / 0.6574 (EXP167), fixing 29
old errors while breaking 36 correct decisions. Locked validation was not read.

EXP168 moved cross-modal interaction before utterance pooling. It projected the
word-aligned 768-D text, 74-D audio, and 35-D vision sequences into a compact
space, encoded all three signed multiplicative agreements and absolute pairwise
disagreements, sparsely pooled valid frames, and exposed bounded Neutral and
polarity residual axes. The parent remained frozen and the new branch again
started as an exact identity. The first three held-out folds pooled to 0.6663 /
0.6339 versus approximately 0.6663 / 0.6387 for the same-fold dynamic parent.
The remaining folds were stopped under the declared structural-futility rule;
EXP168 is incomplete and cannot enter an ensemble. This closes two distinct
forms of explicit cross-modal incongruity for the current feature set.

A fixed robust-aggregation audit then compared label-free pooling rules over
the already cross-fitted strict7 expert bank. Median aggregation improved OOF
but regressed on locked validation. Equal consensus between the majority-vote
distribution and the probability-average distribution passed both gates at
0.6916 / 0.6635 OOF without learned weights, thresholds, or subset selection.
Its one permitted locked-valid transfer (EXP169) reached 0.7047 accuracy and
0.6852 Macro-F1, improving both EXP117 primary metrics. Neutral F1 was 0.5297;
the remaining failure is still Neutral separation rather than polar accuracy.
Test remains sealed because valid Macro-F1 is below 0.70.

## 2026-09-25: reliability-weighted feature decomposition and counterfactual rejection

EXP170 implemented the complete ConFEDE/HyCon-inspired route rather than a
scalar tuning variant. A weight-shared projector extracted common coordinates
from text, audio, and vision; three modality-private projectors were
orthogonalized sample-wise against those shared vectors. Three unimodal
classifiers, a shared/private consensus classifier, reliability-weighted
parent distillation, and cross-modal supervised contrastive learning supplied
deep supervision. The final decision remained an exact parent identity at
initialization through a zero-initialized 0.50-bounded residual. CUDA tests
verified identity, normalization, orthogonality, finite embeddings, end-to-end
loss wiring, and non-zero gradients through the residual, shared/private
projectors, modality classifiers, and consensus classifier. The branch adds
698,086 trainable parameters to the frozen phased parent.

Five grouped phased folds (parent 3 epochs, dynamic-router intermediate 1,
decomposition child 2) reached 0.6757 Accuracy / 0.6423 Macro-F1. The first
three folds narrowly beat the same-fold parent on both pooled metrics, so the
declared completion rule required all five folds. The full result was not
competitive. Fixed uniform and probability/vote strict8 combinations reached
only 0.6872/0.6559 (EXP171) and 0.6872/0.6572 (EXP172). This closes the current
shared/private decomposition implementation without pretending its auxiliary
loss decrease was a task-metric gain.

EXP173/174 tested one fixed consensus between EXP169 and the independently
replicated sparse Neutral hurdle. Both produced 0.6913/0.6633 and missed the
current OOF gate. EXP175 then changed the architecture from generic Neutral
detection to counterfactual rejection: its sparse lexical/deep-disagreement
head was fitted only where EXP169 predicted a polar class and could only move
such decisions toward Neutral. Error attribution showed the remaining losses
were concentrated in Positive-to-Neutral transitions. EXP176 therefore froze
two decision invariants: existing Neutral and Positive predictions could not
change, leaving only Negative-to-Neutral rejection. It tied EXP169 OOF
Accuracy at 0.6916 and improved Macro-F1 from 0.66350 to 0.66379, which passed
the non-degradation gate. The single locked-valid transfer reached only
0.7019/0.6822 and was rejected.

EXP179 added frozen Sentence-BERT embeddings to the sparse lexical and
eight-expert disagreement evidence, then used a fresh grouped meta-fold seed.
Despite the richer semantic representation, the 0.125 trust region produced
the same aggregate four boundary changes and the same 0.6916/0.6638 OOF result
as EXP176. Representation enrichment did not alter the actionable boundary.
Finally, two explicitly labelled validation-fitted deployment calibrations
were audited. The legacy composite objective reached 0.7102/0.6856 (EXP177),
and a target-aligned max-min objective reached 0.7060/0.6881 (EXP178). Neither
met the relaxed >0.69 Macro-F1 gate, so no test predictions were materialized.
EXP169 remains the honest frozen best at 0.7047/0.6852 valid.

## 2026-09-25: conditional innovation reconstruction and final Neutral arbitration

EXP180 implemented a TFR-Net-inspired conditional-innovation branch. Text
predicted the expected audio and visual fusion states; the residual innovations,
their interaction, and their disagreement drove an auxiliary classifier plus
bounded Neutral and polarity corrections. Reliability-weighted SmoothL1
reconstruction and zero-initialized decision axes made the branch a strict
parent identity at initialization. CUDA tests verified finite reconstruction,
probability normalization, identity initialization, and gradient flow through
both modality predictors, innovation projections, and decision axes. The first
three grouped folds pooled to 0.6717 Accuracy / 0.6352 Macro-F1, compared with
0.6663 / 0.6392 for the same-fold phased parent. Because only Accuracy improved,
the remaining folds were stopped under the predeclared structural-futility rule.

EXP181 then changed the final decision graph rather than adding another large
representation tower. A Neutral-authenticity arbitrator consumes 55 coordinates
from the strict seven-member bank: member log-posteriors, mean/dispersion/range,
votes, dual-channel consensus, member entropies, and regression disagreement.
Its logistic head is trained in a second grouped cross-fitting layer. The head
is asymmetric by construction: non-Neutral parent decisions cannot change, and
rejecting a Neutral decision removes only Neutral mass, exactly preserving the
parent Negative/Positive odds. OOF changed one decision and improved both parent
metrics from 0.691605/0.663501 to 0.691900/0.663814. The purely OOF-selected
threshold transferred as no validation changes, so EXP181 itself remained at
0.704670/0.685239 valid.

EXP183 retained the OOF-trained arbitrator weights and used validation only for
the final authenticity threshold, a standard single-parameter deployment
calibration. The frozen threshold 0.286166 rejected ten validation Neutral
decisions and reached 0.711538 Accuracy / 0.690806 Macro-F1. A pre-test lock
manifest was written before any new test posterior was generated. The frozen
seven checkpoints were then evaluated and combined once. Final held-out test
performance is 0.733150 Accuracy / 0.690347 Macro-F1, meeting the requested
strict >0.69 target on both metrics. Although the internal parent comparator was
slightly stronger on test, it was not selected post hoc; EXP183 remains the
leakage-honest delivered system.

## 2026-09-26: intrinsically explainable additive residuals and transfer audit

EXP184 introduced a parent-anchored Hierarchical Explainable Neural Additive
Model rather than another unconstrained stacker. Sixty-two named scalar
concepts are derived from seven frozen experts: member Neutral and polarity
log-odds, entropy, confidence, margin and predicted intensity, plus consensus
means, dispersion, ranges, votes, parent probabilities and intensity
disagreement. Every concept owns an independent one-dimensional neural shape
function on two axes: Neutral-vs-Polar and Positive-vs-Negative. The output
layers are zero-initialized, so the untrained model is exactly the EXP169
parent. Each contribution is bounded and baseline-centered as `f(x)-f(0)`.
Setting a standardized concept to zero therefore removes exactly its own logit
contribution. Five-fold deletion audits measured maximum absolute errors below
5e-7 and correlations effectively equal to 1.0.

The unrestricted EXP184 residual changed 352 OOF decisions. It corrected 130
parent errors, corrupted 167 correct decisions and moved 55 errors to another
wrong class, ending at 0.6807 Accuracy / 0.6460 Macro-F1. Locked validation was
not opened. EXP185 retained the same trained additive teachers but placed both
axes inside fixed OOF-selected trust regions of 0.35 and 0.40. Effective
contributions are scaled by the same factors, so the exact additive and
counterfactual identities remain valid. This reached 0.695140/0.667079 OOF and
legally unlocked validation, but validation regressed to 0.690934/0.672260.
The result shows that posterior-only nonlinear corrections can improve grouped
OOF while failing under the full-train deployment distribution. EXP185 is an
explainability contribution and diagnostic model, not the new champion.

The frozen EXP178 bias calibration, which had been selected before any test
access, was also reconstructed and evaluated historically after EXP183 had
already disclosed the test labels. Its +0.06 Neutral logit bias reached test
0.735901 Accuracy / 0.699935 Macro-F1. This is approximately 0.000065 below the
strict 0.70 target and is not rounded up. Because the same test labels are now
known, the number is explicitly descriptive rather than an independent
holdout claim.

EXP186 tested whether the full-train validation distribution could directly
train a deployment Neutral head. An L1 logistic model used the same 62 concepts
and retained 45 non-zero coefficients; the final class decomposition preserved
the parent Negative/Positive odds exactly. Its validation calibration-training
fit crossed both thresholds at 0.717033/0.701875, but a reverse-domain OOF audit
was only 0.685125/0.619107. The model and its -0.72 Neutral intercept correction
were hashed and locked before a separate test command. Descriptive test
performance was 0.719395/0.683995, confirming severe calibration-domain
overfit. EXP186 is rejected and its valid number must never be reported as
independent evaluation.

EXP187 reduced capacity to one transparent consistency rule. Starting from
EXP178's frozen +0.06 bias, a polar prediction could become Neutral only when
at least three of seven experts voted Neutral, absolute ensemble intensity was
at most 0.15 and the Neutral logit deficit was at most 0.30. It rescued one net
validation error, reaching 0.707418/0.690122, but failed the original joint OOF
gate at 0.688954/0.662187. A pre-evaluation lock was still written to audit
transfer. No test sample satisfied all conditions, so descriptive test stayed
at 0.735901/0.699935. The remaining gap cannot honestly be closed by selecting
another threshold on this already disclosed test set. Future work is therefore
restricted to new base evidence learned without test feedback; posterior bias,
generic meta-learning and handcrafted rescue routes are closed.

## 2026-09-26: native tri-modal Neutral energy and prior-free evidence

EXP188 moved the decision source back before the posterior bank. Text, audio,
and vision fusion states were each decomposed into named Neutral and Polar
subspaces. Per-modality bounded Neutral log-odds residuals were aggregated by a
reliability-weighted product of experts; disagreement only attenuated evidence,
and the final map changed Neutral mass while preserving the parent's
Negative/Positive odds exactly. The residual output layers were zero initialized,
so insertion was a strict parent identity. Exposed diagnostics include every
modality's energy, reliability, PoE weight and additive logit contribution.

The first mixed-update attempt was intentionally aborted after fold 0 because
the new shift remained nearly zero while inherited heads were also trainable.
The isolated retry trained only the energy module. Its complete three-fold
structural check reached 0.6613/0.6310, but fold-average shifts changed from
strongly positive to strongly negative. This identified fold-specific constant
drift rather than insufficient capacity.

EXP189 replaced the free residual MLP with bias-free signed Neutral-minus-Polar
energy axes. Projection LayerNorm affine parameters and projection biases were
also removed, preventing a hidden constant correction. A coherence gate and
the parent's Neutral Bernoulli boundary limited changes. The result was stable
but almost entirely negative, reaching only 0.6604/0.6266 in the three-fold
check. The model had learned the imbalanced training posterior rather than a
class-conditional evidence ratio.

EXP190 therefore computed the Polar/Neutral prior ratio separately inside each
outer training fold and used it in both modality evidence classification and
energy separation. No held-out labels entered this statistic. The resulting
shifts were genuinely bidirectional with fold means near zero, and Macro-F1
recovered to 0.6348, but Accuracy remained 0.6589. Against EXP169 it corrected
181 errors and corrupted 292 correct decisions, including 208 newly introduced
Polar-to-Neutral errors. Fixed inclusion as an eighth vote/probability member
produced only 0.6828/0.6546 (EXP191), so locked validation remained unopened.

EXP192 added a parameter-free sign unanimity veto: every available modality
must support the same direction, otherwise the correction and every modality
contribution are exactly zero. CUDA counterfactual tests verified that one
dissenting modality forces abstention. Only about one third of samples passed
the gate; Neutral precision rose slightly while recall fell, ending at
0.6586/0.6341. This closes further scalar, threshold, and consensus-mode search
for the current energy-PoE branch. EXP193 must change the raw evidence source or
introduce a genuinely video-group-invariant learning objective.

## 2026-09-26: video-group robustness and leave-one-out background evidence

EXP193 implemented real-video Group-DRO rather than treating utterances as
independent domains. Every video's total classification mass was equal and its
adversarial weight was updated from train-only epoch mean risk. The unbiased
minibatch estimator and diagnostics worked as designed, but 691 of 1,528 videos
are singletons. Three folds ended at 0.6480/0.6242 with strongly varying Neutral
recall. Per-video adversarial groups are therefore too granular for this corpus.

EXP194 introduced a deployment-consistent leave-one-out video background:
audio and vision references use every other utterance from the same video but
never the current utterance or a label. Singleton samples exactly abstain. A
zero-initialized residual compared current and background-centered modality
states before fusion. The branch learned real signal, but gates approached one
and correction norms reached 3--5, ending at 0.6766/0.6317. EXP195 multiplied
the learned gate by text-anchor uncertainty, reducing mean gates to roughly
0.4 and improving to 0.6786/0.6346. EXP196 then imposed a dimension-independent
0.35 L2 trust radius. It raised Accuracy to 0.6839 but reduced Neutral recall to
0.3179 and Macro-F1 to 0.6265, showing that representation-level background
correction is still a Polar confirmer even when geometrically bounded.

EXP197 moved the same video-relative evidence to a dedicated Neutral boundary.
Audio and vision deviations were projected into a shared space; availability,
text uncertainty and cross-modal agreement formed an interpretable trust gate.
Only Neutral mass changed, while Negative/Positive conditional odds were exact
invariants. The result was the best of this family at 0.6804/0.6353, but shifts
were almost entirely positive. The model learned global Neutral expansion, not
a trustworthy signed relative direction. Direct Group-DRO, free representation
deconfounding, trust-radius search and single-sample video-relative correction
are closed. The next candidate must learn signed within-video change from
outer-fold training pairs and may not use held-out labels for pair construction.

## 2026-09-26: signed within-video relation learning

EXP198 converted every informative train-only same-video Neutral/Polar relation
into a weighted ordered-pair objective. Shared current/reference modality maps
make each audio and vision difference antisymmetric; non-negative routing cannot
invent a sign. Inference uses only leave-one-out unlabeled peers, singleton
videos abstain, and the parent Negative/Positive odds remain exact. Shifts became
genuinely bidirectional, but raw pair AUC was only 0.5702 and OOF ended at
0.6828/0.6217.

EXP199 added a leave-one-out 768-dimensional semantic reference. The semantic
route dominated appropriately and pair AUC rose to 0.6912, establishing that
relative semantic affect transfers across held-out videos. A simultaneous
absolute Neutral anchor was not trustworthy: direct anchor supervision produced
negative drift and 0.6813/0.6303. EXP200 instead supervised the corrected parent
Neutral logit; drift flipped strongly positive and Accuracy fell, ending at
0.6766/0.6309. This is a structural two-sided failure, not a request for another
loss coefficient.

EXP201 removed the absolute anchor. Its shift stayed centered (mean 0.0044),
bidirectional and interpretable; pair AUC reached 0.6932. It was the most stable
variant at 0.6851/0.6278, but same-video ranking cannot place an entire video on
the global Neutral boundary and offers no evidence for 691 singleton videos.
Within-video pair ranking, semantic routing, absolute-anchor variants and their
scalar settings are closed. EXP202 must change the evidence population, with
cross-video class-balanced relation learning or a new pretrained multimodal
source as the leading options; test remains sealed.
