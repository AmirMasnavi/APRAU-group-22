# Dataset guide

## Source and grain
`dataset_grupo_22.csv` is the supplied group-22 curated subset. Assignment describes one patient profile per row, 2,000 records and three diagnostic classes. The assignment describes a consolidated medical warehouse and screening programs, but no original source URL, collection dates, identifiers, transformation log, or split is provided. Independence of patient profiles is unverified.

## Initial inventory (2026-10-06)
- 2,000 rows × 35 columns; 33 supplied predictors plus 2 targets.
- Comma-delimited CSV; numeric values use decimal points. Pandas infers 22 numeric columns and 13 string columns (including the targets).
- No parser-recognized missing cells, exact duplicate rows, or missing target values.
- `target_class`: DI_HY 1,798 (89.90%), HT 117 (5.85%), DI_HT 85 (4.25%). Code meanings not supplied; retain literal labels.
- `target_regression`: continuous laboratory measurement; name and unit unspecified. Range 70–300; mean 135.9916, median 134, standard deviation 41.2541.
- `calcium_baseline` constant at 9.42; `systolic_index` duplicates systolic_bp; age_normalized appears to duplicate age's information. Investigate ldl_alternative rather than dropping by its matching marginal summary.
- Raw CSV SHA-256: `38bbc572114bf3967a0773771a5792021df5a1621b731da162d33e9c6798d1bd`.

## Target rules
Regression predicts target_regression; classification predicts the existing three-label target_class. Do not manufacture binary classes or rename diagnosis codes without a key. Exclude both targets from either feature matrix. Current EDA uses all supplied rows; there is no untouched test set yet.

Detailed field dictionary, quality evidence, cleaning choices, and target associations will be added after notebook execution. Preserve original files unchanged.
