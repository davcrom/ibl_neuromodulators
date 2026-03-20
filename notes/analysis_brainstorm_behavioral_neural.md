# Disentangling NM-specific encoding from behavioral variation

**Date:** 2025-03-19

**Goal:** Find analyses that elegantly combine two considerations: (1) are
there behavioral differences across target_NM groups? and (2) do NM systems
respond differently to task events? Rather than treating these as separate
steps (first show behavior is matched, then show neural differences), integrate
behavior and neural responses into a unified analytical framework.

---

## Approach 1: Psychometric parameters as LMM regressors

### Data

`get_response_magnitudes()` produces a trial-level DataFrame: one row per (recording x
event x trial), with columns `response_early`, `contrast`, `side`,
`feedbackType`, `subject`, `target_NM`. Session-level psychometric fits
(`basic_performance()` -> `psych_50_threshold`, `psych_50_bias`,
`psych_50_lapse_left`, `psych_50_lapse_right`) get merged onto every trial row
by `eid`.

### Within-between decomposition

Threshold varies both across subjects (stable individual differences) and
across sessions within a subject (day-to-day fluctuation). These carry
different information. Decompose each psychometric parameter into:

- `threshold_between` = subject mean across sessions
- `threshold_within` = session value - subject mean

### Model specification

The current LMM is fit per (target_NM, event). Extend to:

```
response_early ~ log_contrast * C(side) * C(reward) * threshold_within
              + log_contrast * C(side) * C(reward) * threshold_between
              + (1 | subject)
```

This is a large model. In practice, focus on interpretable interactions. A
more realistic specification:

```
response_early ~ log_contrast * C(reward) + C(side)
              + log_contrast:threshold_within + C(reward):threshold_within
              + log_contrast:threshold_between + C(reward):threshold_between
              + (1 | subject)
```

### Interpretation of key terms

| Coefficient | Meaning |
|---|---|
| `log_contrast:threshold_within` | On days when this animal is less sensitive (higher threshold), does its neural contrast slope change? A positive interaction for DA would mean DA contrast encoding tracks daily behavioral fluctuations. |
| `C(reward):threshold_within` | On days when this animal performs worse, does the reward/omission signal change? |
| `log_contrast:threshold_between` | Do animals with chronically higher thresholds show different neural contrast encoding? This is the individual-differences version. |
| `C(reward):threshold_between` | Same, for reward encoding. |

### Comparing across target_NMs

Collect the coefficients for the interaction terms from each target_NM's fit.
The pattern of which interactions are significant, and their signs,
characterizes how each NM system's encoding depends on behavioral state. For
instance: if `log_contrast:threshold_within` is significant for VTA-DA but not
LC-NE, DA's contrast encoding fluctuates with daily performance while NE's
encoding is invariant to it.

### Practical concerns

- Threshold is bounded and skewed. Log-transform it before entering as a
  regressor.
- Check that within-subject variance in threshold is substantial enough to
  estimate within-subject slopes. If most variance is between-subject, the
  within terms will be unstable.
- Bias and lapse rates could be added analogously, but start with threshold
  alone because it has a clear, univariate interpretation (sensory
  sensitivity). If bias is used, it makes more sense as a moderator of the
  `C(side)` term than the `log_contrast` term.

---

## Approach 2: CCA of response vectors and psychometric parameters

### Data

`get_response_features()` produces a matrix X of shape (n_recordings, K)
where K ~ 40 features (event x contrast x side x feedback condition labels).
Each session also has psychometric parameters; build Y of shape
(n_recordings, P) where P includes `psych_50_threshold`, `psych_50_bias`,
`psych_50_lapse_left`, `psych_50_lapse_right`, and for biased sessions
`bias_shift`, `psych_20_threshold`, `psych_80_threshold`. Sessions with
multiple recordings (bilateral fibers) contribute multiple rows with different
X but identical Y. This shared Y introduces non-independence; handle either by
averaging neural features across fibers or by permutation testing that
preserves session structure.

### Method

With K ~ 40 features and likely 80-150 recordings, standard CCA will overfit.
Use sparse CCA (Witten, Tibshirani & Hastie 2009), which simultaneously
selects which neural features and which behavioral parameters contribute to
each canonical variate. Set regularization parameters by permutation: permute
the row pairings between X and Y, fit sparse CCA, and compare the permuted
canonical correlations to the real ones.

### Steps

1. Standardize X and Y (column-wise z-score).
2. Fit sparse CCA; extract canonical variate pairs (U1 = Xa1, V1 = Yb1),
   (U2, V2), ...
3. Test each canonical correlation against a permutation null (permute session
   labels, 1000 iterations).
4. For significant canonical variates: scatter-plot Uk vs Vk, color by
   target_NM.
5. Inspect the weight vectors ak and bk.

### Interpretation

The first canonical variate pair captures the dominant mode of covariation
between neural response profiles and behavioral parameters.

- If U1 separates target_NMs and V1 loads heavily on threshold: the NM systems
  differ in how their response profile covaries with behavioral sensitivity.
- If U1 does NOT separate target_NMs: neural-behavioral coupling has a shared
  structure across NM systems, and the NM-specific differences are orthogonal
  to behavioral variation. This is actually the more interesting outcome. It
  would mean that NM systems differ in their responses (known from decoding)
  but the axis of NM differentiation is orthogonal to the axis of behavioral
  variation -- a dissociation stronger than simply saying "behavior doesn't
  differ."

### Extension

After fitting CCA on pooled data, compute the canonical correlation separately
within each target_NM subgroup (using the global weight vectors). This tells
you how tightly each NM system's neural-behavioral coupling aligns with the
group-level mode.

---

## Approach 3: Variance partitioning via nested model comparison

### Core question

What fraction of each NM system's response variance is task-driven, what
fraction is state-dependent, and what fraction reflects state-modulated
encoding?

### Data

Same trial-level DataFrame as approach 1, with within-subject centered
psychometric parameters (`threshold_w`, `bias_w`) merged on.

### Model hierarchy

Fit a nested sequence of LMMs per (target_NM, event):

| Model | Formula | Captures |
|---|---|---|
| M0 | `response ~ 1 + (1\|subject)` | Baseline + individual differences |
| M1 | `response ~ log_contrast * C(side) * C(reward) + (1\|subject)` | + task encoding |
| M2 | `response ~ log_contrast * C(side) * C(reward) + threshold_w + bias_w + (1\|subject)` | + additive behavioral state |
| M3 | `response ~ log_contrast * C(reward) * threshold_w + C(side) * bias_w + (1\|subject)` | + state x task interactions |

where `threshold_w` and `bias_w` are session - subject mean to isolate
session-to-session fluctuations. Include `threshold_between` and
`bias_between` too if desired, but the within-subject terms are the
interesting ones: they ask whether fluctuations in behavioral state within an
animal predict changes in neural encoding.

### Variance increments

Compute marginal R-squared for each model and take the increments:

- dR2(M1 - M0) = task-driven variance
- dR2(M2 - M1) = additive state-dependent variance
- dR2(M3 - M2) = state-modulated encoding variance

### Interpretation

The critical quantity is dR2(M3 - M2): the state-modulated encoding fraction.
A target_NM with large dR2(M3 - M2) has encoding that shifts with behavioral
state (the way it represents contrast or reward changes on good vs bad days).
A target_NM with near-zero dR2(M3 - M2) has stable encoding regardless of
performance. This naturally integrates behavior and neural responses into one
comparison, and does not require showing that behavior is "the same" -- it
directly asks whether the behavior-neural coupling is different.

### Visualization

A stacked bar chart: x-axis = target_NM, y-axis = R-squared, with three
colored segments (task, state, state x task). This is a single panel that
tells the entire story. Supplement with a table of the specific interaction
coefficients from M3 to show the direction of effects.

---

## Approach 4: RSA with psychometric model RDMs

### Core question

Does each NM system's representational geometry across trial conditions match
the geometry predicted by the animal's own behavioral discriminability?

### Data

For each recording, compute the mean response (early window,
baseline-subtracted) for each of ~10 conditions: 5 contrast levels x 2
outcomes. This gives a 10-element vector per recording. Compute the pairwise
Euclidean distance between all condition means -> a 10x10 representational
dissimilarity matrix (RDM), or equivalently, its 45-element upper triangle.

### Model RDMs

Define candidate models for what the neural distances "should" look like:

1. **Contrast model**: distance(i, j) = |log(ci + 0.01) - log(cj + 0.01)|.
   Predicts neural geometry organized by stimulus strength.
2. **Outcome model**: distance = 0 within same outcome, 1 across. Predicts
   neural geometry organized by reward/omission.
3. **Psychometric model**: distance(i, j) = |P(correct | ci) - P(correct |
   cj)|, where P is from that session's own psychometric fit. This is the key
   model -- it asks whether neural distances between conditions reflect the
   animal's own behavioral discriminability.
4. **Interaction model**: contrast x outcome.

### Analysis

For each recording, compute the partial Spearman correlation between its
neural RDM and each model RDM (partialing out the other models). This gives,
for each recording, a set of coefficients indicating the unique contribution
of each model to explaining the neural geometry. Group by target_NM and test
whether the coefficients differ.

### Interpretation

- If VTA-DA's neural RDM is best explained by the outcome model, DA organizes
  its representation around reward vs omission.
- If LC-NE's neural RDM is best explained by the psychometric model, NE
  organizes its representation around the animal's behavioral certainty --
  conditions that the animal finds hard to distinguish are also close in NE
  neural space.
- If DR-5HT's neural RDM is best explained by the contrast model, 5HT's
  representation tracks physical stimulus properties regardless of outcome or
  behavioral state.

The psychometric model RDM is the bridge between behavior and neural activity:
it uses the animal's own behavior to define expected neural distances, then
asks which NM systems conform to this expectation. No need to separately
demonstrate that behavior is matched, because the behavioral fit is embedded
in the model.

### Practical notes

10 conditions with 5 contrasts x 2 outcomes gives 45 pairwise distances.
Reliable condition means require >= 10-15 trials per condition. Zero-contrast
correct and incorrect trials may have very different counts -- consider
collapsing zero contrast across outcomes, reducing to 9 conditions (4 nonzero
contrasts x 2 outcomes + 1 zero contrast).

---

## Summary

| Approach | Core question | Behavior-neural integration |
|---|---|---|
| 1. Psychometric regressors in LMM | Which specific interaction terms (contrast, reward) depend on behavioral state, per NM? | Direct: behavioral params moderate neural encoding |
| 2. CCA | What is the dominant mode of covariation between neural profiles and behavioral profiles? | Symmetric: finds shared axes |
| 3. Variance partitioning | What fraction of neural variance is state-modulated encoding, per NM? | Nested decomposition: single summary metric |
| 4. RSA with psychometric model RDM | Does each NM system's representational geometry align with the animal's own behavioral geometry? | Geometric: behavioral discriminability as reference frame |

Approaches 1 and 3 are closely related (3 is a structured summary of 1). Pair
**3** (for the overview figure) with **4** (for the geometric
interpretation), and use **2** for an exploratory, hypothesis-free view of the
joint space.
