# Project rules for agents

- Read this file and memory.md at the start of every session; consult prd.md and data.md as needed.
- Keep chat concise; make the notebook understandable to a student defending the work.
- Preserve the supplied CSV and assignment PDF unchanged. Clean only derived copies and log each decision with evidence.
- Every notebook code cell must have a markdown intent cell immediately before it and a markdown findings cell immediately after it. Findings must agree with executed outputs.
- End each notebook with a consolidated conclusion: findings, decisions, caveats, target definitions, and ordered next steps.
- Keep initial work to data understanding, cleaning decisions, and simple EDA. Do not jump ahead into model selection without user direction.
- Treat target_class and target_regression as separate outcomes. Exclude BOTH from either predictor matrix; never use one target to predict the other without explicit teacher/user approval.
- Do not infer clinical meanings, diagnostic thresholds, biomarker units, or data provenance from column names alone. Distinguish assignment definitions from empirical observations and hypotheses.
- Fit imputation, encoding, scaling, feature selection, and sampling on training folds only when modeling begins. Do not select features or tune on the final test set.
- Account for severe class imbalance; use stratification and report macro-F1, balanced accuracy, per-class recall, and confusion matrices, with accuracy as secondary.
- Use seed 42. Retain unusual records unless a documented reason supports removal. IQR flags are descriptive, not a deletion rule.
- Update memory.md at meaningful milestones and before commits; include decisions, unresolved questions, validation, and next steps. Update data.md when new facts are established.
- Make an initial documentation commit before EDA; commit completed milestones. Stage intended files only; exclude environments, caches, secrets, and temporary files.
- Latest user instruction (2026-10-06) requests a new PUBLIC GitHub repository and an immediate push of current files; this explicitly replaces the earlier private-repository requirement. Verify the requested visibility before pushing.
- Independent review remains required before delivering completed EDA. The latest user instruction requests publishing current unfinished files immediately; do not describe this draft publication as reviewed or completed EDA.
- Credit consulted sources. Preserve group contribution history and prepare work that all group members can explain.
