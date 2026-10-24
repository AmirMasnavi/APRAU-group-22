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

- **Our VIF values are all below 1.05** once the three redundant columns are removed. The Hitters
  example has VIF up to 503. **Ridge's main justification does not apply to our data**, so it is
  predictable that Ridge will bring little or no gain. Say so, and back it with the VIF evidence.
- **Lasso is the method that should matter here**, not because of collinearity but because the EDA
  shows most features carry no signal at all — only 6 of 18 numeric features separate the classes.
- A defensible prediction to state *before* running it: Lasso should keep roughly `HbA1c_level`,
  `triglycerides`, and possibly `bmi`/`age`, and zero most of the rest. If it does, the EDA and the
  model agree — that agreement is itself a result worth reporting.

---

## 7. Checklist for Part 5

- [ ] Standardize inside a `Pipeline`, fit on training folds only.
- [ ] Stratified splits (severe class imbalance — see `docs/eda_findings.md`).
- [ ] Tune `C` by cross-validation for both L1 and L2.
- [ ] Report **all features vs reduced set**, for both penalties.
- [ ] Metrics: macro-F1, balanced accuracy, per-class recall, confusion matrix (not accuracy alone).
- [ ] Report how many coefficients survived, and **name them**.
- [ ] Cross-check the surviving set against the EDA ranking and comment on agreement/disagreement.
- [ ] Explain the Ridge result using the VIF evidence.
- [ ] Credit the slides and ISLP ch. 6 in the references.
