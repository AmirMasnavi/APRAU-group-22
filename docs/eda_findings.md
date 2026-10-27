# EDA findings — reference

**Purpose.** Every result and conclusion from the EDA, so later stages can cite the evidence without
re-reading the notebook.

**Source.** `notebooks/EDA_version3.ipynb` (PT) / `notebooks/EDA_version3_en.ipynb` (EN), seed 42,
run on `dataset_grupo_22.csv`. Section numbers below refer to the notebook.
**All figures here were verified against the executed outputs.**

---

## 1. Dataset

- **2000 rows × 35 columns** = 21 numeric features + 12 categorical features + 2 targets.
- **0 missing values, 0 duplicated rows.** No imputation and no row removal required.
- Categorical features stored as text → must be encoded (ordinal or one-hot) before modelling.
- **Working set = 18 numeric features**, after dropping `calcium_baseline`, `age_normalized`,
  `systolic_index` (see §3).

## 2. Targets

### `target_class` — severely imbalanced

| Class     | n    | %     |
| --------- | ---- | ----- |
| `DI_HY` | 1798 | 89.90 |
| `HT`    | 117  | 5.85  |
| `DI_HT` | 85   | 4.25  |

Imbalance ratio 21.2 : 1. **Trivial classifier that always predicts `DI_HY`**: accuracy 0.8990,
balanced accuracy 0.3333, minority recall 0. → accuracy alone is useless; stratification mandatory.

### `target_regression` — continuous, weakly related to everything

Range 70–300 · mean 135.99 · median 134 · skew 0.52 · kurtosis 0.42 · 137 distinct values.
Shapiro-Wilk p ≈ 4.18e-10 (not normal). IQR is nearly symmetric (median−Q1 = 33, Q3−median = 32) — the
skew lives entirely in the right tail.
**13 observations exactly at 70 and 7 exactly at 300** → consistent with **truncation** at both ends.

Mean by class: `DI_HT` 168.19 · `DI_HY` 134.68 · `HT` 132.81.

## 3. Constant and redundant features (§1.3, §3.2)

| Feature                                                                         | Evidence                                                                                                                  | Decision                                           |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `calcium_baseline`                                                            | single value 9.42 across 2000 rows, sd = 0                                                                                | **drop**                                     |
| `systolic_index`                                                              | identical to `systolic_bp` on all 2000 rows                                                                             | **drop**                                     |
| `age_normalized`                                                              | `(age − 2) / 87`, max residual 9.89e-07; Pearson 1.0, Spearman 0.999951                                                | **drop**, keep `age`                       |
| `ldl_alternative`                                                             | same value multiset as `ldl`, but matches on only **18 of 2000** rows, r = −0.0049 → a row permutation          | **suspicious**; let Lasso decide             |
| `age_level`, `bmi_level`, `low_hdl_cholesterol`, `high_ldl_cholesterol` | discretizations of `age`, `bmi`, `hdl`, `ldl`; the two flags do **not** match the usual clinical thresholds | partially redundant; prefer the continuous version |

Spearman for `age`/`age_normalized` is 0.999951 rather than exactly 1 because of the decimal-precision
artefact in §6.

## 4. Multicollinearity — essentially absent

- Correlation matrix: only **two** pairs above |r| = 0.3, both the redundant pairs above. All other
  inter-feature correlations are below 0.05 in absolute value.
- **Condition number** with the redundant columns included: **2.986e+17** (numerically singular).
  After removing them: **1.667**.
- **All VIF values ≤ 1.044** on the 18-feature working set.

> ⚠️ The VIF/condition-number cell was **deleted from the current notebook**, although §4.2 and §4.3
> still quote these numbers. Either restore the cell or remove the claims.

## 5. Signal available

### For regression — essentially none (§3.4, §3.5)

| Feature         | Pearson r | p      |
| --------------- | --------- | ------ |
| `bmi`         | 0.0695    | 0.0019 |
| `salt_intake` | −0.0622  | 0.0054 |
| `HbA1c_level` | 0.0554    | 0.0132 |
| `age`         | 0.0536    | 0.0166 |

Everything else is closer to zero. Spearman ≈ Pearson throughout → **no hidden monotone non-linear
relationship**. Best single feature explains **r² ≈ 0.48%** of the variance.

Categorical vs `target_regression`: only `target_class` has any effect (**η² = 0.027**, F = 27.881,
p_anova 1.14e-12, p_kruskal 5.31e-06). It is the target of the *other* task, not of this one, so it
**can be used as a predictor** in the regression (as binary indicators). All other categoricals have
η² ≤ 0.006.
ANOVA and Kruskal-Wallis disagree on borderline cases (`physical_activity`: p_anova 0.047 vs
p_kruskal 0.333). Since `target_regression` is not normal, **trust Kruskal-Wallis**.

**Diagnostic (not in the notebook):** a linear model with all 40 one-hot columns reaches R² = 3.51%
in-sample, 1.54% adjusted. On held-out data it would be near zero.

### For classification — real, but concentrated in two features (§3.6, §3.7)

| Feature                | η²            | Kruskal H | p                                 |
| ---------------------- | --------------- | --------- | --------------------------------- |
| `HbA1c_level`        | **0.410** | 307.15    | 2.01e-67                          |
| `triglycerides`      | **0.105** | 170.84    | 7.99e-38                          |
| `age`                | 0.013           | 23.97     | 6.23e-06                          |
| `crp_level`          | 0.010           | 2.42      | **0.298 (not significant)** |
| `blood_pressure`     | 0.009           | 48.96     | 2.33e-11                          |
| `bmi`                | 0.006           | 11.40     | 0.0033                            |
| `homocysteine_level` | 0.004           | 20.81     | 3.03e-05                          |

**Only 6 of the 18 numeric features are significant.** `crp_level` has a middling η² but a
non-significant p — η² is mean-based and sensitive to its long tail; the rank test wins.

Class means — `HbA1c_level`: DI_HY 6.936, **HT 5.374**, DI_HT 6.979 · `triglycerides`: DI_HY 151.16,
HT 217.04, DI_HT 227.07 · `bmi`: 27.62 / 29.05 / 29.99 · `age`: 54.42 / 63.06 / 60.71.

Categorical vs `target_class` (Cramér's V): `low_hdl_cholesterol` 0.125 · `age_level` 0.113 ·
`bmi_level` 0.108 · `physical_activity` 0.070 · `education_level` 0.062 · `sugar_consumption` 0.059 ·
`employment_status` 0.059 · `smoking` 0.058 · `family_history` 0.055 · `high_ldl_cholesterol` 0.055 ·
**`gender` 0.039 (p = 0.227)** · **`stress_level` 0.018 (p = 0.852)**. All weak (V < 0.13); the last
two have no association at all.

`DI_HT` and `DI_HY` overlap heavily in the pairplot → expect `DI_HT` to have the worst recall.

## 6. The decimal-precision artefact (§3.8-artefact) — most important finding

Proportion of **non-rounded** values, by class:

| Feature                                                                                                  | DI_HY           | HT              | DI_HT           |
| -------------------------------------------------------------------------------------------------------- | --------------- | --------------- | --------------- |
| `hdl`, `heart_rate`, `alcohol_intake`, `ldl`, `salt_intake`, `diastolic_bp`, `sleep_hours` | ~0.056          | **1.000** | **1.000** |
| `systolic_bp`                                                                                          | 0.053           | 1.000           | 0.941           |
| `bmi`                                                                                                  | 0.049           | 0.923           | 0.953           |
| `HbA1c_level`                                                                                          | **0.855** | 0.188           | 0.294           |
| `crp_level`, `plasma_value`, `homocysteine_level`                                                  | 1.000           | 1.000           | 1.000           |
| `age`                                                                                                  | 0.000           | 0.000           | 0.000           |

Minority-class records were generated or imputed by a **different process** (consistent with synthetic
oversampling). Consequences:

1. **Potential data leakage** — a flexible model could learn the class from the *shape of the number*
   rather than its clinical value.
2. **Mutual information is unusable as a selection criterion here.** Its top entries
   (`alcohol_intake` 0.154, `sleep_hours` 0.144, `diastolic_bp` 0.142, `hdl` 0.131) are exactly the
   artefact-affected features, while `HbA1c_level` — by far the most discriminative — gets a low score
   and `bmi` only 0.073. **Use Kruskal-Wallis/η², χ²/Cramér's V and correlations instead.**
3. Explains why `low_hdl_cholesterol` carries signal while `hdl` does not.
4. Linear models (Logistic Regression, LDA) use values, not precision, so they are little affected —
   but **never build features derived from decimal precision**.

Other signs of synthetic data: near-uniform distributions for 10 numeric features, near-perfect balance
in most categoricals, and zero correlation between clinically related variables.

## 7. LDA assumptions (§3.8)

- **Normality rejected** by Shapiro-Wilk in almost every (feature, class) pair; the only exception is
  `bmi` in `DI_HT` (p = 0.613, n = 85 → low power).
- **Equal covariance questionable**: sd of `HbA1c_level` is 0.359 in `DI_HY` vs 0.907 in `DI_HT`,
  ratio **2.53**.
- Arguments both ways: logistic regression assumes neither; LDA is favoured by K = 3 classes and small
  minority classes.

## 8. What this means for feature selection (Part 5)

1. **Ridge should bring little or no gain** — its purpose is to stabilize collinear predictors, and
   all VIF ≤ 1.05 here. Use that as the explanation, not as an excuse.
2. **Lasso is the method that matters**, because most features carry no signal, not because of
   collinearity.
3. **Expected surviving set**: `HbA1c_level` and `triglycerides` almost certainly; possibly `bmi`,
   `age`, `blood_pressure`, `homocysteine_level`. Everything else is a candidate for elimination.
4. **Expected to be dropped**: `gender`, `stress_level`, `systolic_bp`, `diastolic_bp`, `heart_rate`,
   `hdl`, `ldl`, `alcohol_intake`, `salt_intake`, `sleep_hours`, `cholesterol`, `plasma_value`,
   `crp_level`, and `ldl_alternative`.
5. **If Lasso keeps an artefact-affected feature** (e.g. `hdl`, `alcohol_intake`), treat it as a
   leakage warning, not as a discovery — §6 explains why.
6. **Standardize before penalizing**; scales run from ~1 (`plasma_value`) to ~400 (`triglycerides`).
7. Reduced-set performance that **matches** the full-set performance is a success: it means the dropped
   features really were noise, which is exactly what the EDA predicted.
