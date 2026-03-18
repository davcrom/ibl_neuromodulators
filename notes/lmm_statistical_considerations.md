# Statistical Considerations: Linear Mixed-Effects Models for Events Analysis

## The Model

For each (target x event) group, we fit:

```
y_ij = beta_0 + beta_1 * log(c) + beta_2 * side + beta_3 * reward
       + beta_4 * log(c) * side + beta_5 * log(c) * reward + beta_6 * side * reward
       + beta_7 * log(c) * side * reward
       + u_0j + u_1j * log(c) + epsilon_ij
```

where `i` indexes trials, `j` indexes subjects, `c = contrast + 0.01`, and:

- **beta_0 -- beta_7** are **fixed effects**: population-level parameters shared
  across all subjects.
- **u_0j, u_1j** are **random effects**: subject-specific deviations from the
  population intercept and contrast slope, assumed drawn from a multivariate
  normal with mean zero and unstructured covariance.
- **epsilon_ij ~ N(0, sigma^2)** is residual noise.

The fixed effects capture population-average effects of contrast, laterality,
and reward. The random effects capture the fact that subjects differ in
baseline response (u_0j) and contrast sensitivity (u_1j).


## Why log(contrast)?

Weber-Fechner law: perceived intensity scales logarithmically with stimulus
magnitude. The `+ 0.01` offset handles zero-contrast trials (0.01 is below
the smallest non-zero contrast of 6.25%). This makes the contrast-response
relationship approximately linear in the predictor, which is what the linear
model assumes.


## Estimation: REML

Parameters are estimated via Restricted Maximum Likelihood (REML; Patterson &
Thompson, 1971), which proceeds in two stages:

1. **Variance components** (sigma^2_0, sigma_01, sigma^2_1, sigma^2): REML
   maximizes a modified likelihood that integrates out the fixed effects,
   removing the downward bias that ML has for variance components in small
   samples. Specifically, REML maximizes:

   ```
   L_REML(theta) = -1/2 [log|V(theta)| + log|X'V(theta)^{-1}X| + (y - X*beta_hat)'V(theta)^{-1}(y - X*beta_hat)]
   ```

   where **V(theta) = Z D Z' + sigma^2 I** is the marginal covariance of y,
   **D** is the random effects covariance, **Z** is the random effects design
   matrix, and **X** is the fixed effects design matrix. Optimization is over
   theta = (sigma^2_0, sigma_01, sigma^2_1, sigma^2).

2. **Fixed effects** (beta): Given the estimated variance components theta_hat,
   the fixed effects are estimated by generalized least squares:

   ```
   beta_hat = (X' V_hat^{-1} X)^{-1} X' V_hat^{-1} y
   ```

   This is the best linear unbiased estimator (BLUE) given V_hat.

The random effects are estimated as best linear unbiased predictors (BLUPs):

```
u_hat = D_hat Z' V_hat^{-1} (y - X * beta_hat)
```

These "shrink" subject-level estimates toward the population mean: subjects
with fewer trials or noisier data get pulled more toward the group average.


## Inference on Fixed Effects

The covariance of beta_hat is:

```
Cov(beta_hat) = (X' V_hat^{-1} X)^{-1}
```

Standard errors are the square roots of the diagonal. Wald z-statistics and
p-values follow:

```
z_k = beta_hat_k / SE(beta_hat_k)
p_k = 2 * Phi(-|z_k|)
```

This uses a normal (not t) reference distribution, which is the default in
statsmodels and standard for LMMs with reasonably large total N.

**Caveat for small samples:** Wald z-tests are somewhat anti-conservative with
few grouping levels (subjects). Luke (2017) showed via simulation that the
Kenward-Roger and Satterthwaite approximations for denominator degrees of
freedom produce Type I error rates closer to the nominal level for small
samples. These are available in R's `lmerTest` but not in statsmodels. With
typical photometry datasets (3-10 subjects), p-values near the significance
boundary should be interpreted cautiously.


## Confidence Intervals on Predictions

For a new design point x* (a specific contrast x side x reward combination),
the predicted population-mean response is:

```
y_hat* = x*' beta_hat
```

This is a **marginal prediction**: it averages over the random effects
distribution, giving the expected response for a "typical" subject.

The variance of this prediction comes from propagating uncertainty in beta_hat:

```
Var(y_hat*) = x*' Cov(beta_hat) x*
```

The 95% CI is then:

```
y_hat* +/- 1.96 * sqrt(x*' Cov(beta_hat) x*)
```

This CI reflects **uncertainty in the fixed effects only**. It tells you how
precisely we have estimated the population-average curve. It does **not**
include:

- Random effects variance (between-subject variability)
- Residual variance (trial-to-trial noise)

If you wanted a **prediction interval** for a new trial from a new subject,
you would add sigma^2_0 + sigma^2 to the variance. The CI we plot is the
right choice for "where is the population mean?", which is what the
fixed-effects model line represents.


## Variance Explained: Nakagawa R^2

Following Nakagawa & Schielzeth (2013):

```
R^2_marginal    = Var(X * beta_hat) / Var(y)
R^2_conditional = [Var(X * beta_hat) + Var(Z * u_hat)] / Var(y)
```

- **Marginal R^2**: proportion of total variance explained by the fixed
  effects (contrast, side, reward, interactions). Answers: "how much do the
  experimental manipulations explain?"
- **Conditional R^2**: proportion explained by fixed + random effects
  combined. Answers: "how much is explained when we also account for which
  subject it is?"

The gap (conditional - marginal) quantifies how much individual differences
in baseline and contrast sensitivity contribute beyond the task variables.


## Convergence and Fallback Strategy

The implementation tries a random intercept + random slope model first. With
few subjects (common in photometry, typically 3-10), the slope variance and
intercept-slope covariance can be poorly identified, causing the optimizer to
hit the parameter space boundary or fail to converge. When this happens, we
fall back to random intercept only (u_1j = 0), which is more stable but
assumes all subjects share the same contrast sensitivity.

- A **"boundary" warning** (variance component estimated at zero) is benign.
  It means the data do not support subject-level variation in that parameter,
  and the model effectively reduces to the simpler form.
- A **"failed to converge" warning** means the optimizer did not find a stable
  solution. Results should not be trusted; the group is skipped.

The choice to include a random slope for contrast (when it converges) follows
the "keep it maximal" principle (Barr et al., 2013): the random effects
structure should reflect the experimental design. Subjects genuinely differ in
contrast sensitivity, and failing to model this can inflate Type I error rates
for the fixed effect of contrast.


## Interpreting the Confidence Bands

The shaded CI around each model line (e.g., correct trials on the contra
side) reflects uncertainty in **all eight fixed-effect coefficients jointly**,
not just the intercept or just the slope. At a given point on the x-axis,
the prediction is a weighted sum of all beta terms:

```
y_hat = beta_0 + beta_1 * log(c) + beta_2 * side + beta_3 * reward + ...
```

The CI at that point propagates uncertainty from every coefficient in that
sum. At extreme values of log(contrast) — far from the data mean — the
slope uncertainty dominates and the band fans out.

### Why overlapping CIs do not contradict significant effects

Each band answers: "where could the true population mean for **this
condition** be?" It does **not** answer: "is there a difference between
conditions?" These are different questions. The correct-vs-incorrect
comparison depends on the uncertainty of beta_3 (the reward coefficient)
alone, which is much smaller than the total uncertainty in each individual
band. Each band carries uncertainty from all 8 coefficients; the difference
between the bands cancels out most of that shared uncertainty.

Analogy: measuring two people's heights with a wobbly ruler gives large
individual uncertainty, but the **difference** is precise because the wobble
affects both measurements equally.

### How to extract the CI in code

The CI is computed from the fixed-effects covariance matrix:

```python
# X_grid: design matrix for the prediction points (n_points x n_coefficients)
# fe_cov: fixed-effects covariance matrix (n_coefficients x n_coefficients)
# fe_params: fixed-effect coefficient vector (n_coefficients,)

predicted = X_grid @ fe_params                            # point estimate
se = np.sqrt(np.diag(X_grid @ fe_cov @ X_grid.T))        # standard error
ci_lower = predicted - 1.96 * se
ci_upper = predicted + 1.96 * se
```

`X_grid` is built from the Wilkinson formula's design info, which handles
the dummy coding of categorical variables and the interaction terms. The
covariance matrix `fe_cov` is the (X'V^{-1}X)^{-1} submatrix for the fixed
effects, obtained from `result.cov_params()`.

### Alternative: visualizing contrasts directly

To directly answer "is correct different from incorrect?", one can compute the
**difference** in predicted values between conditions and plot that with its
own CI. For example, the difference at each contrast level between reward=1
and reward=0 (holding side constant) depends only on beta_3 + beta_5*log(c) +
beta_6*side + beta_7*log(c)*side — a subset of the coefficients. The CI on
this difference is narrower because the shared uncertainty cancels.

This is the approach taken by estimated marginal means (see below).


## Comparison with emmeans and Estimated Marginal Means

### What emmeans does

The R package `emmeans` (Lenth, 2024) computes **estimated marginal means**
(EMMs): model predictions averaged over the levels of specified factors,
evaluated at a reference grid. For example, the EMM for "correct trials"
averages the model prediction over all contrast levels and both sides, yielding
a single number for the population-mean response on correct trials, marginalized
over the other predictors.

Key capabilities:
- **Marginal averaging**: predictions are averaged over factors not of
  interest, using equal weights (by default) regardless of how balanced the
  data actually are.
- **Pairwise contrasts**: direct comparisons between conditions (e.g.,
  correct vs. incorrect) with CIs that reflect only the uncertainty in the
  difference, not in each individual mean. This is why emmeans contrasts often
  show significant differences even when individual EMM CIs overlap.
- **Multiplicity correction**: Tukey, Bonferroni, or other adjustments for
  multiple comparisons are built in.
- **Kenward-Roger / Satterthwaite df**: emmeans can use these
  approximations for more accurate small-sample inference (see the caveat
  about Wald z-tests above).

### What our implementation does

We compute **conditional predictions**: the model's estimated response at
each observed contrast level, for each specific side x reward combination.
No averaging over other factors. This is closer to `emmeans::emmip()` (the
interaction plot) than to `emmeans::emmeans()` (the marginal means).

Our CIs are based on the Wald approximation with the full fixed-effects
covariance matrix, equivalent to what emmeans calls `summary(, infer=TRUE)`
with `lmerTest::lmer()` df = Inf (i.e., z-based rather than t-based).

### Key differences

| Aspect | Our approach | emmeans |
|--------|-------------|---------|
| What is plotted | Prediction at each contrast x side x reward | Marginal means averaged over specified factors |
| CI reflects | Uncertainty in all fixed effects jointly | Uncertainty in the (possibly marginal) mean |
| Contrasts | Not directly shown; bands may overlap despite significance | Explicit pairwise differences with own CIs |
| df approximation | Wald z (normal) | Kenward-Roger or Satterthwaite (t) |
| Multiplicity | Not applicable | Built-in corrections |

### Practical implications

Our plots are best for **visualizing the shape of the response function** —
how does the population-mean response vary with contrast, and does that shape
differ by side and feedback? The emmeans approach is better for **testing
specific hypotheses** — is there a main effect of reward, averaged over
contrast and side?

Both approaches extract information from the same model. They are
complementary, not competing.

### Python alternatives

- The `marginaleffects` package (Arel-Bundock, Greifer & Heiss, 2024) provides
  EMM-like functionality in Python, supporting statsmodels and other model
  classes. It can compute predictions, comparisons, and marginal means with a
  consistent API.
- EMMs can also be computed manually from the design matrix and coefficient
  covariance, as we do for conditional predictions — the only difference is
  constructing the reference grid and averaging over factor levels.


## Implementation Notes

- `statsmodels.regression.mixed_linear_model.MixedLM` with REML estimation.
- Categorical predictors (`side`, `reward`) are coded via `C()` in the
  Wilkinson formula, which defaults to treatment (dummy) coding.
- The `reward` predictor is derived from `feedbackType`: 1 (correct) -> 1,
  -1 (incorrect) -> 0.
- Data is filtered to unbiased blocks (probabilityLeft == 0.5), excluding
  no-go trials (choice != 0) and fast reactions (reaction_time > 0.05s).


## References

- Arel-Bundock, V., Greifer, N. & Heiss, A. (2024). How to interpret
  statistical models using marginaleffects for R and Python. *Journal of
  Statistical Software*, 111(9), 1-32.
  https://doi.org/10.18637/jss.v111.i09

- Barr, D.J., Levy, R., Scheepers, C. & Tily, H.J. (2013). Random effects
  structure for confirmatory hypothesis testing: Keep it maximal. *Journal of
  Memory and Language*, 68(3), 255-278.
  https://doi.org/10.1016/j.jml.2012.11.001

- Bates, D., Machler, M., Bolker, B. & Walker, S. (2015). Fitting linear
  mixed-effects models using lme4. *Journal of Statistical Software*, 67(1),
  1-48. https://doi.org/10.18637/jss.v067.i01

- Laird, N.M. & Ware, J.H. (1982). Random-effects models for longitudinal
  data. *Biometrics*, 38(4), 963-974. https://doi.org/10.2307/2529876

- Lenth, R.V. (2024). emmeans: Estimated marginal means, aka least-squares
  means. R package. https://CRAN.R-project.org/package=emmeans

- Luke, S.G. (2017). Evaluating significance in linear mixed-effects models
  in R. *Behavior Research Methods*, 49(4), 1494-1502.
  https://doi.org/10.3758/s13428-016-0809-y

- Nakagawa, S. & Schielzeth, H. (2013). A general and simple method for
  obtaining R2 from generalized linear mixed-effects models. *Methods in
  Ecology and Evolution*, 4(2), 133-142.
  https://doi.org/10.1111/j.2041-210x.2012.00261.x

- Patterson, H.D. & Thompson, R. (1971). Recovery of inter-block information
  when block sizes are unequal. *Biometrika*, 58(3), 545-554.
  https://doi.org/10.1093/biomet/58.3.545
