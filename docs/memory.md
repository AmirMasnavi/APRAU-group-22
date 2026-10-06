# Project memory

Last updated: 2026-10-06 (Europe/Lisbon)

## Current status
Documentation bootstrap complete; EDA notebook not yet created. This file is being saved before the initial commit, as requested. Dataset and assignment have been inspected; original files preserved.

## User preferences and scope
Casual initial EDA, concise chat, thorough checks. Notebook code sandwiched between intent and findings markdown; final consolidated conclusion. Create docs first and commit before EDA. New private repository; independent second-agent review and sensible improvements before final push.

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
1. Commit this starting documentation.
2. Build and execute the introductory EDA notebook and derived reports; update data.md.
3. Obtain independent review, improve meaningful findings, rerun verification.
4. Update memory and commit final reviewed work; create/verify private repository and push.

## Validation
Initial CSV SHA-256 recorded in data.md; assignment text read in full. No models trained.
