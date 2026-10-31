# Feature selection & regularization — reference

**Purpose.** Everything needed for **Part 5 of the assignment** (feature selection with Ridge/Lasso),
condensed from the lecture slides. Read this instead of re-reading the slide decks.

**Sources.** `docs/theory_lessons/T5_RidgeLasso.pdf` (47 slides, "Linear Model Selection &
Regularization", Elsa Ferreira Gomes, 2026/2027) and `Trabalho_APRAU_2026_27_step1.pdf` §5.
Slides cite ISLP chapter 6 as the book reference.

---

## 1. What the assignment asks (§5, verbatim meaning)

> Evaluate the impact of feature selection on **classification** performance using regularization
> methods (**Ridge and Lasso**). Compare the results obtained using **all available features** with
> those obtained using a **reduced set**, and critically discuss the differences.

Three deliverables: (a) all features vs reduced set, (b) using Ridge and Lasso, (c) a critical
discussion of the difference.

### Resolved: "Ridge and Lasso applied to classification"

This was an open question earlier in the project. **The slides answer it directly** (T5 slides 40–46,
section "Regularization for Classification"): regularization carries over to **Logistic Regression**
unchanged — only the base loss changes.

| Model | Base loss | Penalty added |
|---|---|---|
| Linear Regression | RSS = Σ(yᵢ − ŷᵢ)² | λΣβⱼ² (Ridge) · λΣ\|βⱼ\| (Lasso) |
| Logistic Regression | − log-likelihood | same three penalties |

So Part 5 = **L2 and L1 penalties on `LogisticRegression`**. No need to regress numeric class codes.
Slide 44 even shows the worked example on the Hitters dataset turned into a classification task.

---

## 2. Core theory

### Bias–variance trade-off (slide 2)
Generalization error = **bias** (wrong assumptions → underfit) + **variance** (sensitivity to the
training sample → overfit) + **irreducible error** (noise; only fixable by cleaning data).
↑ complexity → ↑ variance, ↓ bias. Regularization deliberately **adds bias to buy a larger reduction
in variance**.

### Why not plain least squares (slide 5)
- **Prediction accuracy**: when p > n there is no unique OLS estimate (variance infinite); even when n
  is not much bigger than p, variance is high.
- **Interpretability**: removing irrelevant features gives a model that is easier to explain.

### Three classes of methods (slide 6)
1. **Subset selection** — pick a subset of the p predictors, fit OLS on it.
2. **Regularization / shrinkage** — fit with all p predictors but shrink coefficients towards zero.
3. **Dimension reduction** — project onto an M-dimensional subspace (M < p), then fit.

The lecture (and the assignment) focus on **regularization**.

### Ridge — L2 (slides 8–12)
Minimizes **RSS + λΣβⱼ²**.
- Designed for **multicollinearity**: when predictors are highly correlated OLS becomes unstable.
- λ = 0 → exactly OLS. λ → ∞ → all coefficients shrink towards zero. 0 < λ < ∞ balances the two.
- **Never sets a coefficient exactly to zero** — all p predictors stay in the model.
- λ chosen by cross-validation.

### Lasso — L1 (slides 23–29)
Minimizes **RSS + λΣ|βⱼ|**.
- The only change is the **shape of the penalty**; that is what lets coefficients hit **exactly zero**.
- Therefore Lasso **performs variable selection** and yields **sparse** models.
- This is the method that actually does "feature selection" in the assignment's sense.

### Elastic Net (slides 34–38)
Minimizes **RSS + λ[(1−α)½Σβⱼ² + αΣ|βⱼ|]** — a linear blend of both penalties.
- `l1_ratio` (α): 0 = pure Ridge, 1 = pure Lasso, 0.5 = 50/50.
- **Two separate knobs**: `l1_ratio` controls *what kind* of penalty, λ/alpha controls *how much*.
- Slide 36 warning: l1_ratio alone does not determine sparsity — α matters just as much. In the
  Hitters case an l1_ratio of 0.99 (almost pure Lasso) zeroed only 1 variable because its α was small.

### Summary table (slide 38)

| Method | Penalty added to the loss | Sparse? | Tuning parameters |
|---|---|---|---|
| OLS | — | No | — |
| Ridge | λΣβⱼ² | No | λ |
| Lasso | λΣ\|βⱼ\| | **Yes** | λ |
| Elastic Net | λ[(1−α)½Σβⱼ² + αΣ\|βⱼ\|] | **Yes** | λ, α (l1_ratio) |

### Which to use (slide 37)

| Situation | Method |
|---|---|
| Many predictors genuinely matter, somewhat correlated | **Ridge** |
| Expect many coefficients to be truly zero; want sparsity/interpretability | **Lasso** |
| Predictors correlated **in groups** (Lasso picks one arbitrarily) | **Elastic Net** |
| Unsure, or p huge relative to n | **Elastic Net** |

---

## 3. Practical rules (non-negotiable)

1. **Standardize the predictors first.** Slide 16: *"Standardizing (scaling) predictors first is
   essential for Ridge/Lasso — the penalty term is sensitive to the scale of each variable."*
   Fit the scaler on the **training folds only**.
2. **Choose λ by cross-validation**, never by guessing (slides 22, 28).
3. **Watch the sklearn naming trap** (slide 43):

| Class | Parameter | Direction |
|---|---|---|
| `Ridge()` / `Lasso()` / `ElasticNet()` | `alpha` = λ | larger alpha → **more** shrinkage |
| `LogisticRegression()` | `C` = 1/λ | smaller C → **more** shrinkage (direction flips) |

4. Both extremes are bad: no shrinkage and too much shrinkage both lose to a well-chosen middle value
   (the U-shaped test-error curve, slide 21).

---

## 4. scikit-learn API used in the slides

```python
# Regression
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet
Ridge(alpha=4)                                   # fixed lambda
RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')
LassoCV(cv=10, max_iter=1000)
ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)

# Classification (Part 5 of the assignment)
from sklearn.linear_model import LogisticRegression
LogisticRegression(solver='saga', max_iter=10000, l1_ratio=0.0, C=1.0)   # Ridge  (L2)
LogisticRegression(solver='saga', max_iter=10000, l1_ratio=1.0, C=1.0)   # Lasso  (L1)
LogisticRegression(solver='saga', max_iter=10000, l1_ratio=0.5, C=1.0)   # Elastic Net

# Counting the surviving features
nz = np.sum(np.abs(clf.coef_[0]) > 1e-4)

# VIF (slide 31)
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

`solver='saga'` is required for L1 and elastic-net penalties in `LogisticRegression`.
⚠️ With 3 classes, `coef_` has shape (3, p) — count non-zeros per class or across all rows.

---

## 5. The Hitters case study — reference for what a good answer looks like

The slides work through 263 players, 19 predictors, target `Salary`, deliberately chosen because the
career-total variables are heavily collinear (CAtBat/CHits/CRuns correlations ≥ 0.98; VIF up to 503).

| Model | Test MSE | Non-zero coefficients |
|---|---|---|
| OLS (λ=0) | 133,595.0 | 19 / 19 |
| Ridge (λ=4) | 126,497.6 | 19 / 19 |
| RidgeCV (λ=2.32) | 126,176.9 | 19 / 19 |
| LassoCV | **124,887.9** | **13 / 19** |

As classification (`HighSalary`, balanced 134/129):

| Penalty | Accuracy | Non-zero |
|---|---|---|
| Ridge | 0.9057 | 19 / 19 |
| Lasso | 0.8868 | 10 / 19 |
| Elastic Net | 0.9057 | 16 / 19 |

**The argument structure to imitate:** the slides do not stop at the metric. They explain *which*
variables were zeroed and *why* — CAtBat and CHits are redundant with CRuns (corr ≥ 0.98), so once
CRuns is in the model the other two add nothing. Slide 45 then notes that the exact sparsity pattern
**depends on the task**, not only on the correlations: CRBI and CHmRun survive in regression but are
zeroed in classification.

That is the level of discussion Part 5 asks for: not "Lasso got X%", but *which features it dropped,
why that is consistent with the EDA evidence, and what it costs in performance.*

---

## 6. How this applies to our dataset

See `docs/eda_findings.md` for the numbers. The short version, and it is unusual:

- **Our VIF values are all below 1.05** (max 1.0441) once the four columns the EDA rejected are removed
  (`calcium_baseline`, `systolic_index`, `age_normalized`, `ldl_alternative`); the EDA computes this in
  §3.2. The Hitters example has VIF up to 503. The Hitters
  example has VIF up to 503. **Ridge's main justification does not apply to our data**, so it is
  predictable that Ridge will bring little or no gain. Say so, and back it with the VIF evidence.
- **Lasso is the method that should matter here**, not because of collinearity but because the EDA
  shows most features carry no signal at all — only 6 of 18 numeric features separate the classes.
- A defensible prediction to state *before* running it: Lasso should keep roughly `HbA1c_level`,
  `triglycerides`, and possibly `bmi`/`age`, and zero most of the rest. If it does, the EDA and the
  model agree — that agreement is itself a result worth reporting.

---

## 7. Checklist for Part 5

- [ ] Standardize **inside each CV fold** with an explicit loop. **No `Pipeline`, no `GridSearchCV`** —
      the user has not been taught them and they must not appear in the notebooks.
- [ ] Stratified splits (severe class imbalance — see `docs/eda_findings.md`).
- [ ] Tune `C` by cross-validation for both L1 and L2.
- [ ] Report **all features vs reduced set**, for both penalties.
- [ ] Metrics: macro-F1, balanced accuracy, per-class recall, confusion matrix (not accuracy alone).
- [ ] Report how many coefficients survived, and **name them**.
- [ ] Cross-check the surviving set against the EDA ranking and comment on agreement/disagreement.
- [ ] Explain the Ridge result using the VIF evidence.
- [ ] Credit the slides and ISLP ch. 6 in the references.

---

## 8. Results obtained (verified against `notebooks/05_feature_selection.ipynb`, 34 cells, 13 code)

**Seed: `RANDOM_STATE = 22`** everywhere (changed from 42 on 2026-10-31). The EDA notebooks were changed
too, but **no EDA number moved**: `RANDOM_STATE` is declared there and never used — the EDA has no
`random_state`, `np.random`, `.sample(` or `shuffle` in any cell, so it is fully deterministic. Every
number below is from the seed-22 run.

Setup: 4 EDA columns dropped → **29 variables (17 num + 12 cat) → 39 encoded columns**; stratified
75/25 holdout (1500/500); `saga`, `l1_ratio` ∈ {0, 1, 0.5}, `max_iter=10000`; manual stratified 5-fold
CV, scaler refitted inside every fold.

- **Two counts, never conflate them** — named with the total built in: `colunas mantidas (de 39)` =
  non-zero coefficient columns (a column counts if non-zero for at least one of the 3 classes) and
  `variaveis mantidas (de 29)` = original variables still in the model (counts if **any** of its dummy
  columns survives). They differ because each k-level categorical occupies k−1 dummies. The variable
  count is **meaningless for Ridge**: at C=0.001 Ridge "keeps" 29 variables yet scores 0.3155 (trivial).
- **Path, all three penalties, scored by CV** (`C` 1e-3..1e2, 12 points):
  variables kept — Ridge **flat at 29 everywhere** · Lasso 0/0/2/4/**11**/**15**/23/27/28/28/29/29 ·
  Elastic Net 0/1/2/10/16/22/26/28/28/29/29/29.
  CV F1 peaks at **0.6303** (Ridge at C=1.5199, EN at 4.3288); Lasso 0.6293 at 4.3288. sd ≈ 0.032.
  Under heavy penalty **Ridge wins** (0.5786 vs 0.5242 at C=0.0231). From C≈35 all three flatten at
  0.6244 = unregularized, so regularizing is worth **+0.0059** macro F1.
- **Section 5 does not recompute**: it takes the argmax of each curve from the section-4 path.
- **Refinement** (9 points between the two neighbours): Ridge stays at 1.5199 (gain 0), Lasso moves to
  5.6234 (+0.0010), EN to 3.3322 (gain 0). After it **all three are exactly 0.6303 ± 0.0318**. The
  refined C changes **zero** of the 500 test predictions.
- **Test, full set, all three**: accuracy 0.9440, balanced acc 0.6185, macro F1 0.5978, and
  **recall `DI_HT` = 0.0000** (not one of the 21 cases detected).
- **Full vs reduced** (C retuned in two phases per set and per penalty):
  29 vars → 0.6303 (sd 0.0318) · **15 vars → 0.6472, i.e. +0.0169 BETTER, and sd drops to 0.0247** ·
  11 vars → 0.6201 (−0.0102, sd jumps to 0.0635). Cutting to ~half the variables *improves* CV macro F1
  here; the gain is still below the between-fold sd, so state it as evidence, not proof.
  Cut points C = 0.0658 (11 vars) and C = 0.1874 (15 vars), justified in the notebook as a **granularity
  judgement, stated as such** (below → 4 vars, above → 23 of 29). The one-standard-error rule would be
  the principled alternative **but is NOT in the course slides** (T5 picks λ by plain CV minimum via
  `RidgeCV`), so it is deliberately not used.
  **The sets are NOT nested**: `age` and `bmi` are in the 11-var set but not in the 15-var one — with
  more budget the Lasso swaps them for `age_level` and `bmi_level`. 11 of the 14 variables dropped at
  the 15-var cut are exactly the EDA's uninformative list.
- **Coefficients** (max |β| over the 3 classes, Lasso): `HbA1c_level` 3.42 ≫ `bmi_level_Underweight`
  1.80 > `triglycerides` 1.02 > `bmi_level_Obese` 0.99. Ridge shrinks the dominant ones more (2.19).
- **Imbalance** (same two-phase C tuning): without weights C* = 1.5199, recall `DI_HT` = 0.0000 (0 of
  21: 17 → `DI_HY`, 4 → `HT`). With `class_weight='balanced'` C* = 0.0137 (coarse 0.0081, gain +0.0028),
  **~100× stronger**: recall 0.5238 (11 of 21) at the cost of 60 `DI_HY` called `DI_HT`, precision
  0.1528. Unlike the seed-42 run, **macro F1 on test IMPROVES** (0.5978 → 0.6585) and `HT` recall rises
  0.8621 → 0.9655, because a zero recall on one class drags the macro average down hard.
- **Why section 9 is L2-only — do not "fix" it.** `class_weight='balanced'` gives weights 0.371 / 5.682
  / 7.812 (ratio 21). Verified on this data: `saga` fails to converge for **all three** penalties at
  C = 1.17 and C = 0.0231, max_iter 10000, tol 1e-4 and 1e-3 (12/12 non-converged), and at 50k/200k too.
  `liblinear` raises `ValueError: The 'liblinear' solver does not support multiclass classification
  (n_classes >= 3)`. `lbfgs` converges but raises `ValueError: Solver lbfgs supports only 'l2' or None
  penalties` for l1_ratio 0.5 and 1.0. A `OneVsRestClassifier(liblinear)` wrapper **does** work, but it
  changes the multiclass strategy, performs worse (bal acc 0.7393 vs 0.7624 on the seed-42 run) and the
  Lasso keeps 39/39 columns — **no sparsity**, so it defeats the section's purpose.
- **VIF now lives in the EDA (added 2026-10-31), §3.2.** Section 10 of the feature-selection notebook
  cites it and the citation is backed by executed output: with the redundant columns present the
  correlation matrix has rank 19 of 20 (an exact linear dependency), so VIF is **not computable**, and
  the condition number is 1.75e+15; pairwise, `age`/`age_normalized` gives VIF ≈ 5.1e+11 and
  `systolic_bp`/`systolic_index` ≈ 4.5e+15, while `ldl`/`ldl_alternative` gives **1.00002** — that one
  is not collinear at all, it is removed for a different reason. After removing the three, the
  condition number drops to 1.29 and max VIF is 1.0441 (`homocysteine_level`).
- **Known limitation, stated in the notebook**: the reduced sets come from a Lasso fitted on the whole
  training set, so their CV scores are mildly optimistic. The test set was never used to decide anything.
