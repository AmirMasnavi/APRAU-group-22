# Project memory

Last updated: 2026-10-06 (Europe/Lisbon)

## Current status
Documentation bootstrap committed. Draft EDA notebook and builder created, but execution failed before any analysis cells ran: sandbox disallowed the local Jupyter kernel socket. No reports generated. User then requested creating a PUBLIC repository and pushing current files immediately. Original source files preserved.

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
1. Commit the current draft and updated publication preference; create/verify a public repository and push as requested.
2. On resuming EDA, execute the notebook with local kernel access and update data.md with generated evidence.
3. Obtain independent review, implement sensible changes, rerun verification and commit completed milestones.

## Validation
Initial CSV SHA-256 recorded in data.md; assignment text read in full. No models trained.
