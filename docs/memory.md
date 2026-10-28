# Project memory

Last updated: 2026-10-06 (Europe/Lisbon)

## Current status

Documentation bootstrap committed. Draft EDA notebook and builder created, but execution failed before any analysis cells ran: sandbox disallowed the local Jupyter kernel socket. No reports generated. User then requested creating a PUBLIC repository and pushing current files immediately. Original source files preserved. Public repository created and visibility verified: https://github.com/AmirMasnavi/APRAU-group-22. Draft committed as c04e790; publishing the current main branch next.

## User preferences and scope

Casual initial EDA, concise chat, thorough checks. Notebook code sandwiched between intent and findings markdown; final consolidated conclusion. Create docs first and commit before EDA. Latest instruction overrides the original private preference: create a PUBLIC repository and push current work immediately. EDA execution and independent review are pending; draft publication is not completed EDA.

## Confirmed facts and decisions

- Course assignment is APRAU 2026/2027, group dataset 22; workspace spelling APREU does not change assignment naming.
- 2,000 × 35 dataset; no missing values or exact duplicate rows; severe class imbalance.
- Targets already exist: target_regression and target_class. Literal class meanings and biomarker units remain unknown.
- Preserve raw source files; perform cleaning in copies; exclude both targets from predictors.
- Use the user's hockey repository as structural inspiration only.
- GitHub CLI works with system/keychain access; sandbox-only check incorrectly reported invalid authentication.

## Pending questions

Teacher's code key, regression biomarker name/units, feature provenance/timing, profile independence, group details, PL teacher GitHub username, exact intermediate scope.

## Next steps

1. Complete the current main-branch push to the verified PUBLIC repository: https://github.com/AmirMasnavi/APRAU-group-22.
2. On resuming EDA, execute the notebook with local kernel access and update data.md with generated evidence.
3. Obtain independent review, implement sensible changes, rerun verification and commit completed milestones.

## Validation

Initial CSV SHA-256 recorded in data.md; assignment text read in full. Draft notebook format and builder syntax validated; no notebook cells executed and no models trained. Independent review remains pending.

## 2026-10-19 — Regression notebook (Parte 2)

- Created `notebooks/02_regression.ipynb` via `scripts/build_regression.py` (+ `scripts/regression_findings.py` with findings texts written from executed outputs). Executed with no errors; every code cell sandwiched by intent/findings markdown. Based on EDA_version2, slides T2 (ISLP ch.3) and the Boston Housing class example.
- Protocol: drop calcium_baseline/systolic_index/age_normalized; ordinal/binary/one-hot encoding (31 features); holdout 70/30 seed 42; selection by 5-fold CV on train only; test used once; mean baseline; both targets excluded.
- Results (test): baseline RMSE 41.02; simple `bmi` R² −0.002, MAE 34.18, RMSE 40.95; multiple (forward, 8 feats: bmi, age, HbA1c_level, blood_pressure, cholesterol, salt_intake, smoking, physical_activity) R² 0.001, MAE 34.00, RMSE 40.89; all 31 feats R² −0.021 (overfit). Bootstrap IC95% of RMSE diff (multiple−baseline) [−0.62, 0.40] includes 0 → no practical predictive value.
- Added scipy, scikit-learn, statsmodels to requirements.txt. Not committed yet.
- Open: ask teacher whether target_class may be a regression predictor. Independent review of this notebook pending.

## 2026-10-21 — Regression notebook reformulated (EDA_version3 style)

- Each step: explanation (what/why) → code → "Conclusões" with justification. Added: R²=r² ceiling from EDA, 500-noise-feature control (TP2), real-vs-predicted plots (PL2), polynomial degree search, incremental build 1→18 features, all 2/3-combos of top-12, clinical groups, all-features overfitting, bmi+bmi_level collinearity demo, target-shape check (97.9% in 70–200, near-uniform; >200 mostly DI_HT).
- Chosen: simple `bmi` (test R² −0.002, MAE 34.18, RMSE 40.95); multiple = incremental 7 feats (bmi, age, smoking, salt_intake, triglycerides, physical_activity, HbA1c_level): test R² 0.005, MAE 34.11, RMSE 40.81; baseline RMSE 41.02. All bootstrap CIs vs baseline include 0.
- Seed kept at 42 (project rule) although EDA_version3 uses 42 — group should align.

## 2026-10-22 — Classification notebook (Parte 3)

- Created `notebooks/03_classification.ipynb` via `scripts/build_classification.py` + `scripts/classification_findings.py` (same explain→code→conclusions style). Based on slides docs/theory_lessons/T3 & T4 and class notebooks TP3/TP4/bootstrap OOB.
- Same prep as regression (31 feats, both targets excluded). Models: LR multinomial and LDA, each default and imbalance-adjusted (class_weight='balanced' / priors=[1/3]*3), StandardScaler in pipeline. Primary metric macro-F1 + balanced accuracy + per-class recall.
- Key results (10-fold strat.): LR default recall DI_HT 0.07, LDA 0.19; LDA default best macro-F1 0.645 (p=0.043 vs LR); weighted: LR bal_acc 0.714 vs LDA 0.686 (n.s.). LDA assumptions violated (Shapiro, Levene). Recommended: LDA (equal priors for screening).
- Variance: holdout dp 0.02–0.036 (100 seeds); k-fold estimate dp 0.004–0.011; LOOCV deterministic; bootstrap OOB pessimistic (~63% distinct). k effect: between-fold dp rises with k, estimate dp falls; recommend 10-fold for Part 4.
- Not committed.

## 2026-10-28 — Regression updated for final EDA (01_EDA)
- `ldl_alternative` added to DROP_EDA in `scripts/build_regression.py` (EDA decided to remove it: values of `ldl` permuted across rows). Now 30 candidate features (17 continuous). Rebuilt `notebooks/02_regression.ipynb`; texts in `scripts/regression_findings.py` updated.
- Chosen models and their test results unchanged (ldl_alternative was never selected). Changed: all-features model (train R² 0.0370, CV RMSE 41.79, test RMSE 41.26), lipid group, and "all continuous" group now marginally beats baseline (41.373 vs 41.385).
- OPEN: docs/agent.md now says seed 42 (commit 0a59270), but regression and classification builders still use SEED = 22. Group must decide; switching regression to 42 changes the chosen simple feature (bmi → HbA1c_level) and requires rewriting texts.

## 2026-10-28 — Classification updated (PT + EN)
- `ldl_alternative` removed (30 features) in `build_classification.py` / `_en.py`; all references to slides/class exercises removed from notebook texts (only ISLP and scikit-learn docs remain as references). Both notebooks rebuilt, no errors; findings texts rewritten with new numbers.
- Key changes: LDA vs Logistic (unadjusted) macro-F1 now +0.047, p = 0.054 (was 0.043 → now borderline, not significant); with adjustment LDA wins macro-F1 in 6/6 methods (incl. LOOCV). Recommendation unchanged: LDA (equal priors for screening). LOOCV timings vary between runs, so texts use approximate values.
