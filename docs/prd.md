# APRAU group 22 — project requirements

## Purpose
Apply machine learning to the supplied multi-disease patient-profile dataset for the ISEP Master's in Informatics Engineering, APRAU 2026/2027. The local folder is named APREUProject; the assignment uses APRAU. Each row represents a patient profile with a diagnostic classification target and a continuous laboratory regression target.

## Current scope
Create persistent documentation and an executed introductory notebook that shows actual records and formats, audits quality, justifies light cleaning, gives descriptive/univariate/bivariate EDA, clarifies the two modeling tasks, and concludes with everything learned. No fitted predictive models in this stage. Use intent → code → findings throughout. Get an independent agent review, improve the work, and push to a new private GitHub repository.

## Full assignment roadmap (not all implemented now)
1. EDA: descriptive statistics, univariate and bivariate analysis; justify constant, redundant, or uninformative features.
2. Regression: simple linear regression with competing single features; multiple linear regression with feature combinations; holdout evaluation using R², MAE, RMSE.
3. Classification: logistic regression and LDA; holdout, 5-fold and 10-fold CV, LOOCV, bootstrap; appropriate metrics and discussion of variability under resampling.
4. Feature selection: Ridge/Lasso regularization and all-feature versus reduced-feature classification. For logistic regression this likely means L2/L1 penalties; confirm the teacher's interpretation rather than regressing numeric class codes.
5. Notebook submission and group defense; every member must understand the work; credit materials and consulted sources.

## Delivery requirements from the assignment
- GitHub version control is mandatory; share the private repository with the PL teacher when their username is provided.
- Moodle intermediate submission: **2 November 2026**, 25% of final grade; final: **20 December 2026**, 45%. Confirm Moodle cut-off times and exact intermediate scope with the teacher.
- ZIP naming: `APRAU_AAA_CCC_Num1_Num2_Num3.zip` (teacher acronym, class, student numbers).
- Normally 3 group members (2 exceptionally); mandatory individual defense. Assignment says individual defense accounts for 70% of the project grade; do not conflate that with the submission weightings.

## Acceptance criteria for this stage
- Docs: prd.md, agent.md, memory.md, data.md; root AGENTS.md routes future agents.
- Supplied files remain byte-identical; dataset integrity is recorded.
- Executed notebook has saved outputs, no cell errors, adjacent intent/findings markdown, and a consolidated conclusion.
- Quality audit includes formats, nulls/sentinels, duplicates, categorical levels, ranges, constants, redundant columns, and flagged inconsistencies.
- EDA includes numeric and categorical distributions, regression associations, class-conditioned numeric/categorical summaries, and class balance.
- Both targets explicitly defined with known semantics and unresolved metadata; separate predictor matrices prepared without either target.
- Cleaning choices are reversible and reproducible; reviewed before final push.

## Open questions
- Clinical meaning of DI_HY / DI_HT / HT; name and units of the regression biomarker.
- How data were sampled/transformed/imputed; original measurement units and feature availability before diagnosis.
- Group members/student numbers, class, teacher acronym and GitHub username; exact intermediate deliverables.
- Are profiles independent patients, with any unreported repeated-patient/site clusters?

## Sources
- Supplied `Trabalho_APRAU_2026_27_step1.pdf`, pages 1–4 (authoritative assignment).
- Supplied `dataset_grupo_22.csv` (empirical dataset).
- [User's hockey-season-prediction repository](https://github.com/AmirMasnavi/hockey-season-prediction/tree/main): structural inspiration (docs, executed notebook, reproducible reports, independent review); no hockey assumptions reused.
