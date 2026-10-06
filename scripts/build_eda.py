"""Build and execute the introductory notebook; findings come from its executed evidence."""
from pathlib import Path
import json
import textwrap
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
nb.metadata.language_info = dict(name='python')
nb.cells.append(nbformat.v4.new_markdown_cell('''# APRAU — group 22: introductory exploratory data analysis

**Scope:** understand the supplied data, document reversible cleaning, and explore the two existing targets. No predictive models are fitted here. Seed: **22**.

Each code cell has an intent before it and findings after it. All findings below are populated from the executed analysis. All 2,000 profiles are used for exploration, so future model evaluation will require a fresh, fixed split and training-only preprocessing. Column names do not establish clinical definitions or units.

Source: supplied `dataset_grupo_22.csv` and `Trabalho_APRAU_2026_27_step1.pdf`; persistent context is in `docs/`.'''))

def section(key, title, intent, code):
    nb.cells.extend([
        nbformat.v4.new_markdown_cell(f'## {title}\n\n**Intent.** {intent}'),
        nbformat.v4.new_code_cell(textwrap.dedent(code).strip()),
        nbformat.v4.new_markdown_cell(f'FINDINGS::{key}')])

section('setup', '1. Load and preserve the sources',
        'Locate the project, verify the original file fingerprints, load a derived in-memory copy, and record the software environment.', '''
    from pathlib import Path
    import hashlib, json, platform, importlib.metadata
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display
    SEED = 22
    np.random.seed(SEED)
    ROOT = Path.cwd() if (Path.cwd() / 'dataset_grupo_22.csv').exists() else Path.cwd().parent
    OUT = ROOT / 'reports' / 'eda'
    OUT.mkdir(parents=True, exist_ok=True)
    TARGETS = ['target_class', 'target_regression']
    EXPECTED = {'dataset_grupo_22.csv': '38bbc572114bf3967a0773771a5792021df5a1621b731da162d33e9c6798d1bd',
                'Trabalho_APRAU_2026_27_step1.pdf': 'ab3e8d32cc1668018e368e8a9002d50899e066515be2173c9fa5e1cc1366ca11'}
    fingerprints = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in EXPECTED}
    assert fingerprints == EXPECTED, 'A supplied source has changed; investigate before proceeding.'
    raw = pd.read_csv(ROOT / 'dataset_grupo_22.csv')
    df = raw.copy(deep=True)
    num = df.select_dtypes(include='number').columns.tolist()
    cat = df.columns.difference(num, sort=False).tolist()
    findings = {}
    sns.set_theme(style='whitegrid', context='notebook')
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 60)
    environment = {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ['pandas','numpy','matplotlib','seaborn','nbformat','nbclient','ipykernel']}}
    (OUT / 'environment.json').write_text(json.dumps(environment, indent=2) + '\\n')
    (OUT / 'source_fingerprints.json').write_text(json.dumps(fingerprints, indent=2) + '\\n')
    display(df.head(8))
    display(pd.Series(environment, name='version').to_frame())
    findings['setup'] = f'Loaded **{len(df):,} rows × {df.shape[1]} columns**, with {len(num)} numeric and {len(cat)} categorical columns (targets included). Both source SHA-256 fingerprints match. The first records show mixed integer, decimal and literal category values; no clinical units are inferred.'
''')
section('quality', '2. Formats, missing values and repeated records',
        'Audit every field for type, distinct values, blanks, common textual sentinel tokens and numeric nonfinite values. Check full-row and predictor-only duplicates.', '''
    tokens = {'', 'na', 'n/a', 'null', 'none', 'unknown', '?', 'missing', '-999', '-9999'}
    quality = pd.DataFrame(index=df.columns)
    quality['dtype'] = df.dtypes.astype(str)
    quality['missing'] = df.isna().sum()
    quality['unique'] = df.nunique(dropna=False)
    quality['text_sentinel_count'] = [df[c].astype('string').str.strip().str.lower().isin(tokens).sum() for c in df]
    quality['whitespace_count'] = [df[c].astype('string').ne(df[c].astype('string').str.strip()).sum() for c in df]
    quality['nonfinite_count'] = [int((~np.isfinite(df[c])).sum()) if c in num else 0 for c in df]
    quality['most_common_value'] = [str(df[c].mode().iloc[0]) for c in df]
    quality['most_common_count'] = [int(df[c].value_counts().iloc[0]) for c in df]
    quality.to_csv(OUT / 'quality_audit.csv', index_label='feature')
    display(quality)
    duplicates = {'full_rows': int(df.duplicated().sum()), 'predictors_only': int(df.drop(columns=TARGETS).duplicated().sum())}
    display(pd.Series(duplicates, name='duplicate_count').to_frame())
    findings['quality'] = f"There are **{int(quality.missing.sum())} missing cells**, **{int(quality.text_sentinel_count.sum())} matched sentinel tokens**, **{int(quality.whitespace_count.sum())} whitespace anomalies**, **{int(quality.nonfinite_count.sum())} nonfinite numbers**, and **{duplicates['full_rows']} exact duplicate rows** ({duplicates['predictors_only']} predictor-only duplicates). No imputation or record deduplication is needed. This does not prove that the source was never imputed or that undocumented numeric sentinel values are absent. `calcium_baseline` has only one distinct value."
''')
section('dictionary', '3. Empirical field dictionary and category levels',
        'Show observed ranges and literal levels without inventing units, diagnostic meanings or undocumented category order.', '''
    dictionary = quality[['dtype', 'unique']].copy()
    dictionary['role'] = ['classification target' if c == 'target_class' else 'regression target' if c == 'target_regression' else 'supplied predictor' for c in df]
    dictionary['observed_values_or_range'] = [f'{df[c].min():.6g} to {df[c].max():.6g}' if c in num else ' | '.join(sorted(df[c].unique().tolist())) for c in df]
    dictionary['meaning_and_units'] = ['Existing diagnostic label; code key unknown' if c == 'target_class' else 'Continuous laboratory target; biomarker and unit unknown' if c == 'target_regression' else 'Column name supplied; definitions/units unverified' for c in df]
    dictionary.to_csv(OUT / 'field_dictionary.csv', index_label='feature')
    category_counts = pd.concat([df[c].value_counts(dropna=False).rename_axis('level').reset_index(name='count').assign(feature=c, percent=lambda t: t['count'] / len(df) * 100) for c in cat], ignore_index=True)
    category_counts.to_csv(OUT / 'category_counts.csv', index=False)
    display(dictionary)
    display(category_counts)
    findings['dictionary'] = f'The dictionary covers all **{len(dictionary)} fields**; categorical levels are preserved exactly as supplied. No unknown category tokens appeared in the audit. The two target roles are explicit. Measurement definitions, units, pre-diagnosis availability and code meanings still need teacher confirmation. Numeric precision varies across records; decimal precision alone does not establish data provenance.'
''')
section('redundancy', '4. Constants and redundant information',
        'Test row-wise equality and affine relationships. Matching marginal distributions alone are insufficient evidence for dropping a feature.', '''
    constants = [c for c in df if df[c].nunique(dropna=False) == 1]
    exact_pairs = [(a,b) for i,a in enumerate(df.columns) for b in df.columns[i+1:] if df[a].equals(df[b])]
    age_expected = (df['age'] - 2) / 87
    age_residual = float((df['age_normalized'] - age_expected).abs().max())
    age_mapping_counts = df.groupby('age')['age_normalized'].nunique()
    ldl_equal = int(df['ldl'].eq(df['ldl_alternative']).sum())
    ldl_same_multiset = bool(np.array_equal(np.sort(df['ldl'].to_numpy()), np.sort(df['ldl_alternative'].to_numpy())))
    redundancy = {'constant_columns': constants, 'exact_duplicate_pairs': exact_pairs,
                  'age_normalized_formula': '(age - 2) / 87', 'age_max_abs_residual': age_residual,
                  'age_max_distinct_normalized_per_age': int(age_mapping_counts.max()),
                  'ldl_equal_rows': ldl_equal, 'ldl_same_sorted_values': ldl_same_multiset,
                  'ldl_pair_pearson': float(df['ldl'].corr(df['ldl_alternative']))}
    (OUT / 'redundancy.json').write_text(json.dumps(redundancy, indent=2) + '\\n')
    display(pd.Series(redundancy, name='evidence').to_frame())
    findings['redundancy'] = f"**Drop `calcium_baseline`** from the derived feature set because it is constant, and **drop `systolic_index`** because it equals `systolic_bp` on every row. **Drop `age_normalized`** because it is a deterministic representation of age: `(age − 2)/87`, with maximum rounding residual {age_residual:.2g} and one normalized value per age. **Retain `ldl_alternative` provisionally**: its sorted values exactly match `ldl`, but only {ldl_equal} rows match and their Pearson correlation is {redundancy['ldl_pair_pearson']:.4f}. This is consistent with a row permutation, not proof of the construction mechanism; request provenance before deciding whether it is valid."
''')
section('ranges', '5. Ranges, unusual values and category consistency',
        'Flag observations for investigation while retaining them. IQR fences are descriptive, and no medical cutoff or source-generation mechanism is assumed.', '''
    numeric_summary = df[num].describe().T
    numeric_summary['skew'] = df[num].skew()
    numeric_summary['median'] = df[num].median()
    numeric_summary.to_csv(OUT / 'numeric_summary.csv', index_label='feature')
    q1, q3 = df[num].quantile(.25), df[num].quantile(.75)
    iqr = q3-q1
    flags = (df[num].lt(q1-1.5*iqr) | df[num].gt(q3+1.5*iqr))
    flag_summary = pd.DataFrame({'q1':q1, 'q3':q3, 'iqr':iqr, 'lower_fence':q1-1.5*iqr, 'upper_fence':q3+1.5*iqr, 'flag_count':flags.sum()})
    flag_summary.to_csv(OUT / 'iqr_flags.csv', index_label='feature')
    flagged_records = flags.assign(systolic_below_diastolic=df['systolic_bp'].lt(df['diastolic_bp']))
    flagged_records.to_csv(OUT / 'record_flags.csv', index_label='source_row_0based')
    range_groups = pd.concat([df.groupby(category)[value].agg(['count','min','max']).reset_index().rename(columns={category:'level'}).assign(category=category, numeric_feature=value) for category,value in [('age_level','age'),('bmi_level','bmi'),('low_hdl_cholesterol','hdl'),('high_ldl_cholesterol','ldl')]], ignore_index=True)
    range_groups.to_csv(OUT / 'category_numeric_ranges.csv', index=False)
    display(numeric_summary)
    display(flag_summary.sort_values('flag_count', ascending=False))
    display(range_groups)
    reversal_count = int(flagged_records.systolic_below_diastolic.sum())
    display(df.loc[flagged_records.systolic_below_diastolic, ['systolic_bp','diastolic_bp']].head())
    findings['ranges'] = f"**{int(flags.any(axis=1).sum())} profiles** have at least one numeric IQR flag; the largest count is {int(flag_summary.flag_count.max())} in `{flag_summary.flag_count.idxmax()}`. Narrow central ranges can produce many flags, so these are not removal rules. **{reversal_count} profiles** have `systolic_bp < diastolic_bp`; retain them and ask about field definitions or data quality. Age and BMI levels occupy separate observed intervals, while the HDL/LDL indicators must be interpreted using their supplied labels, not assumed thresholds. Zero alcohol intake is not treated as missing. All rows remain in the derived dataset."
''')
section('clean', '6. Reversible cleaning and two predictor matrices',
        'Apply only supported structural changes to a derived copy, preserve every row and both targets, and demonstrate exclusion of both outcomes from either predictor matrix.', '''
    dropped = ['calcium_baseline','systolic_index','age_normalized']
    clean = df.drop(columns=dropped).copy()
    predictors = [c for c in clean if c not in TARGETS]
    X_classification = clean[predictors].copy()
    X_regression = clean[predictors].copy()
    y_classification = clean['target_class'].copy()
    y_regression = clean['target_regression'].copy()
    assert not set(TARGETS) & set(X_classification.columns)
    assert not set(TARGETS) & set(X_regression.columns)
    assert clean.shape == (2000,32) and X_regression.shape == (2000,30)
    pd.testing.assert_frame_equal(raw, pd.read_csv(ROOT / 'dataset_grupo_22.csv'))
    clean.to_csv(OUT / 'cleaned_profiles.csv', index=False)
    decisions = pd.DataFrame([
        ['calcium_baseline','drop from derived copy','constant at 9.42 across 2,000 rows'],
        ['systolic_index','drop from derived copy','exact row-wise copy of systolic_bp'],
        ['age_normalized','drop from derived copy','one value per age; affine transform within rounding tolerance'],
        ['ldl_alternative','retain provisionally','matching multiset, different row alignment; provenance unknown'],
        ['all rows','retain','IQR flags and BP order flags do not justify removal'],
        ['all remaining columns','retain','no missing values, blanks or nonfinite values found'],
        ['target_class; target_regression','exclude both from either X','separate prediction tasks; avoid cross-target leakage']
    ], columns=['field','decision','evidence'])
    decisions.to_csv(OUT / 'cleaning_decisions.csv', index=False)
    display(decisions)
    print('Derived dataset:', clean.shape, '\\nEach X:', X_classification.shape, '\\nTargets:', y_classification.shape, y_regression.shape)
    findings['clean'] = '**All 2,000 rows retained**. The derived CSV has **32 columns** (30 predictors and 2 targets), compared with 35 originally. Both predictor matrices contain the same 30 supplied features and exclude **both targets**. These are unencoded exploratory matrices; do not feed them directly to models. No imputation, scaling, feature selection based on target associations, or sampling has been fitted. `ldl_alternative` remains provisional; age/BMI category redundancy will be considered using training data when modeling starts.'
''')
section('balance', '7. Classification target and imbalance',
        'Count the three literal labels and quantify a majority-class baseline analytically, without training a classifier.', '''
    balance = df.target_class.value_counts().rename_axis('class').to_frame('count')
    balance['percent'] = balance['count']/len(df)*100
    balance.to_csv(OUT / 'class_balance.csv')
    display(balance)
    majority_fraction = float(balance['count'].iloc[0]/len(df))
    baseline = pd.Series({'accuracy':majority_fraction, 'balanced_accuracy':1/len(balance),
                         'macro_f1':(2*majority_fraction/(1+majority_fraction))/len(balance),
                         'majority_recall':1.0, 'minority_recall':0.0}, name='always predict majority')
    display(baseline.to_frame())
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=balance.index, y=balance['count'], ax=ax, color='#3274A1')
    for i, row in enumerate(balance.itertuples()): ax.text(i, row.count+20, f'{row.count:,} ({row.percent:.2f}%)', ha='center')
    ax.set(title='Classification target: strong class imbalance', ylabel='Profiles', xlabel='Literal supplied label', ylim=(0,2000))
    fig.tight_layout(); fig.savefig(OUT / 'class_balance.png', dpi=150); plt.show(); plt.close(fig)
    findings['balance'] = f"`DI_HY`: **{balance.loc['DI_HY','count']:,} (89.90%)**; `HT`: **117 (5.85%)**; `DI_HT`: **85 (4.25%)**. Predicting the majority label always would achieve **{majority_fraction:.2%} accuracy**, but only **33.33% balanced accuracy**, **{baseline['macro_f1']:.4f} macro-F1**, and zero minority recall. Future evaluation must use stratification, macro-F1, balanced accuracy, per-class recall and confusion matrices, with accuracy secondary. Label meanings remain unknown."
''')
section('numeric_plots', '8. Numeric distributions',
        'Inspect all retained numeric predictors and the regression target with histograms and box plots; use original scales without assigning units.', '''
    retained_num = clean.select_dtypes(include='number').columns.tolist()
    fig, axes = plt.subplots(5,4,figsize=(16,17))
    for ax,c in zip(axes.flat,retained_num):
        sns.histplot(data=clean, x=c, bins=30, ax=ax, color='#3274A1')
        ax.set_title(c); ax.set_xlabel('Supplied value (unit unverified)')
    for ax in axes.flat[len(retained_num):]: ax.set_visible(False)
    fig.suptitle('Retained numeric fields: distributions', y=1.01)
    fig.tight_layout(); fig.savefig(OUT / 'numeric_histograms.png', dpi=130, bbox_inches='tight'); plt.show(); plt.close(fig)
    fig, axes = plt.subplots(5,4,figsize=(16,13))
    for ax,c in zip(axes.flat,retained_num):
        sns.boxplot(data=clean, x=c, ax=ax, color='#7CB4C7', fliersize=2)
        ax.set_title(c); ax.set_xlabel('Supplied value')
    for ax in axes.flat[len(retained_num):]: ax.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / 'numeric_boxplots.png', dpi=130); plt.show(); plt.close(fig)
    concentration = quality.loc[['HbA1c_level','blood_pressure','crp_level','homocysteine_level'], ['most_common_value','most_common_count']]
    display(concentration)
    findings['numeric_plots'] = 'Histograms and box plots show sharply concentrated central distributions for `HbA1c_level`, `blood_pressure`, `crp_level` and `homocysteine_level`, with broader tails. Their repeated values and precision patterns warrant a provenance question, but do not establish prior imputation or synthetic generation. `bmi`, `triglycerides` and the regression target have upper tails. Scaling and any robust transformations must be fitted inside future training folds.'
''')
section('categorical_plots', '9. Categorical distributions',
        'Show the count and frequency of every retained categorical predictor, separate from the target.', '''
    cat_predictors = [c for c in cat if c not in TARGETS]
    fig, axes = plt.subplots(4,3,figsize=(16,15))
    for ax,c in zip(axes.flat,cat_predictors):
        counts = df[c].value_counts()
        sns.barplot(x=counts.index,y=counts.values,ax=ax,color='#3274A1')
        ax.set(title=c, ylabel='Profiles', xlabel='Supplied level')
        ax.tick_params(axis='x',rotation=25)
    fig.tight_layout(); fig.savefig(OUT / 'categorical_counts.png', dpi=130); plt.show(); plt.close(fig)
    rarest = category_counts[category_counts.feature.isin(cat_predictors)].sort_values('count').iloc[0]
    findings['categorical_plots'] = f"All **{len(cat_predictors)} categorical predictors** are shown. The smallest observed predictor level is `{rarest['feature']} = {rarest['level']}` with **{int(rarest['count'])} profiles ({rarest['percent']:.2f}%)**. Uneven prevalence is a distribution finding, not an automatic reason to remove a level. Keep original labels; choose encoding within a training pipeline after confirming whether an order is justified."
''')
section('regression', '10. Regression associations with numeric predictors',
        'Rank descriptive Pearson and Spearman associations and plot the four largest absolute Pearson associations. These full-data results are exploratory, not a feature-selection validation.', '''
    numeric_predictors = [c for c in retained_num if c not in TARGETS]
    associations = pd.DataFrame({'pearson':clean[numeric_predictors].corrwith(y_regression,method='pearson'),
                                 'spearman':clean[numeric_predictors].corrwith(y_regression,method='spearman')})
    associations['abs_pearson'] = associations.pearson.abs()
    associations = associations.sort_values('abs_pearson',ascending=False)
    associations.to_csv(OUT / 'regression_numeric_associations.csv',index_label='feature')
    display(associations)
    fig, axes = plt.subplots(2,2,figsize=(12,9))
    for ax,c in zip(axes.flat,associations.head(4).index):
        sns.scatterplot(data=clean,x=c,y='target_regression',alpha=.25,s=12,ax=ax)
        ax.set_title(f'{c}: Pearson r={associations.loc[c,"pearson"]:.3f}')
    fig.tight_layout(); fig.savefig(OUT / 'regression_scatterplots.png',dpi=150); plt.show(); plt.close(fig)
    reg_stats = y_regression.describe()
    findings['regression'] = f"The regression target ranges **{reg_stats['min']:.0f}–{reg_stats['max']:.0f}**, with mean **{reg_stats['mean']:.4f}**, median **{reg_stats['50%']:.0f}** and SD **{reg_stats['std']:.4f}** (unit unknown). The largest absolute Pearson association is `{associations.index[0]}` at **{associations.iloc[0].pearson:.4f}**; all retained numeric predictors have |r| ≤ {associations.abs_pearson.max():.4f}. Individual linear signals are weak. This does not rule out categorical effects, interactions or nonlinear relationships, and is not evidence of held-out model performance."
''')
section('regression_categories', '11. Regression target by categorical predictor',
        'Compare group sizes, means, medians and spread for every categorical predictor, keeping small-group uncertainty visible.', '''
    regression_groups = pd.concat([df.groupby(c,observed=True)['target_regression'].agg(['count','mean','median','std','min','max']).reset_index().rename(columns={c:'level'}).assign(feature=c) for c in cat_predictors],ignore_index=True)
    regression_groups.to_csv(OUT / 'regression_category_summaries.csv',index=False)
    display(regression_groups)
    fig, axes = plt.subplots(4,3,figsize=(17,16))
    for ax,c in zip(axes.flat,cat_predictors):
        sns.boxplot(data=df,x=c,y='target_regression',ax=ax,fliersize=2,color='#7CB4C7')
        ax.set_title(c); ax.tick_params(axis='x',rotation=25)
    fig.tight_layout(); fig.savefig(OUT / 'regression_by_category.png',dpi=130); plt.show(); plt.close(fig)
    spans = regression_groups.groupby('feature')['mean'].agg(lambda s: s.max()-s.min()).sort_values(ascending=False)
    display(spans.rename('observed mean span').to_frame())
    findings['regression_categories'] = f"The largest observed categorical mean span is **{spans.iloc[0]:.3f}** for `{spans.index[0]}`. Group distributions overlap; count, median and SD should accompany means. These comparisons use all rows and many categories, so treat patterns as hypotheses to evaluate on future held-out data. Neither p-values nor clinical effects are claimed."
''')
section('class_numeric', '12. Numeric predictors by diagnostic label',
        'Use class-conditioned summaries and separate box plots so that the majority class does not conceal minority profiles.', '''
    class_numeric = clean.groupby('target_class')[numeric_predictors].agg(['count','mean','median','std'])
    class_numeric.to_csv(OUT / 'class_numeric_summaries.csv')
    display(class_numeric)
    fig, axes = plt.subplots(5,4,figsize=(17,18))
    for ax,c in zip(axes.flat,numeric_predictors):
        sns.boxplot(data=clean,x='target_class',y=c,order=balance.index,ax=ax,fliersize=2,color='#7CB4C7')
        ax.set_title(c); ax.set_xlabel('Literal label')
    for ax in axes.flat[len(numeric_predictors):]: ax.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / 'numeric_by_class.png',dpi=130); plt.show(); plt.close(fig)
    focus = clean.groupby('target_class')[['HbA1c_level','blood_pressure','crp_level','homocysteine_level']].agg(['median','std'])
    display(focus)
    findings['class_numeric'] = 'Class-conditioned tables and box plots are available for all retained numeric predictors. Central concentration and spread differ across labels for several fields, especially `HbA1c_level`, `blood_pressure`, `crp_level` and `homocysteine_level`; ask whether these measurements or transformations used the diagnosis. Such differences may reflect measurement, curation or target construction. They do not prove leakage or causation. Minority summaries use only 85 and 117 profiles and therefore have less support than the 1,798-profile majority.'
''')
section('class_categories', '13. Categorical predictors by diagnostic label',
        'Compare within-class percentages alongside raw counts. Each class is its own denominator, and no significance test or diagnostic rule is inferred.', '''
    class_category_rows = []
    fig,axes = plt.subplots(4,3,figsize=(17,16))
    for ax,c in zip(axes.flat,cat_predictors):
        counts = pd.crosstab(df[c],df.target_class).reindex(columns=balance.index,fill_value=0)
        percentages = counts.div(counts.sum(axis=0),axis=1)*100
        counts_long = counts.rename_axis(index='level',columns='class').stack().rename('count').reset_index()
        percentages_long = percentages.rename_axis(index='level',columns='class').stack().rename('within_class_percent').reset_index()
        class_category_rows.append(counts_long.merge(percentages_long,on=['level','class']).assign(feature=c))
        percentages.plot.bar(ax=ax)
        ax.set(title=c,ylabel='Within-class %',xlabel='Supplied level',ylim=(0,100))
        ax.tick_params(axis='x',rotation=25); ax.legend(fontsize=7)
    class_categories = pd.concat(class_category_rows,ignore_index=True)
    class_categories.to_csv(OUT / 'class_category_summaries.csv',index=False)
    display(class_categories)
    fig.tight_layout(); fig.savefig(OUT / 'categorical_by_class.png',dpi=130); plt.show(); plt.close(fig)
    max_diffs = class_categories.pivot(index=['feature','level'],columns='class',values='within_class_percent').apply(lambda s:s.max()-s.min(),axis=1).sort_values(ascending=False)
    display(max_diffs.head(10).rename('max within-class percentage-point difference').to_frame())
    findings['class_categories'] = f"Counts and within-class percentages are saved for every categorical predictor. The largest observed percentage-point gap is **{max_diffs.iloc[0]:.2f}** for `{max_diffs.index[0][0]} = {max_diffs.index[0][1]}`. Percentage denominators are 1,798, 117 and 85, respectively; minority estimates are less stable. A descriptive gap does not justify automatic feature selection, class relabeling or causal interpretation."
''')
section('correlations', '14. Predictor relationships and target separation',
        'Inspect correlations among retained numeric predictors and summarize the regression outcome by class for context only; never add one target to the other task’s predictors.', '''
    corr = clean[numeric_predictors].corr()
    corr.to_csv(OUT / 'predictor_pearson_correlations.csv',index_label='feature')
    pairs = pd.DataFrame([(a,b,corr.loc[a,b]) for i,a in enumerate(numeric_predictors) for b in numeric_predictors[i+1:]],columns=['feature_a','feature_b','pearson'])
    pairs['absolute_r'] = pairs.pearson.abs()
    pairs = pairs.sort_values('absolute_r',ascending=False)
    pairs.to_csv(OUT / 'predictor_correlation_pairs.csv',index=False)
    display(pairs.head(12))
    fig,ax = plt.subplots(figsize=(13,11))
    sns.heatmap(corr,vmin=-1,vmax=1,center=0,cmap='vlag',ax=ax,square=True)
    ax.set_title('Retained numeric predictors: Pearson correlation')
    fig.tight_layout(); fig.savefig(OUT / 'predictor_correlations.png',dpi=150); plt.show(); plt.close(fig)
    target_by_class = df.groupby('target_class').target_regression.agg(['count','mean','median','std','min','max'])
    target_by_class.to_csv(OUT / 'regression_target_by_class.csv')
    display(target_by_class)
    strongest = pairs.iloc[0]
    findings['correlations'] = f"The strongest retained numeric predictor pair is `{strongest.feature_a}` / `{strongest.feature_b}` with **r = {strongest.pearson:.4f}**. Correlation is descriptive and does not establish a shared mechanism. `target_regression` has mean **{target_by_class.loc['DI_HT','mean']:.2f}** for `DI_HT`, **{target_by_class.loc['DI_HY','mean']:.2f}** for `DI_HY`, and **{target_by_class.loc['HT','mean']:.2f}** for `HT`. This target-to-target comparison is context only: **both outcomes remain excluded from both predictor matrices**."
''')
section('validation', '15. Validate and save the analysis evidence',
        'Confirm row preservation, target exclusion, supported cleaning and source integrity; save the narrative evidence so the notebook can be regenerated.', '''
    assert fingerprints == {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in EXPECTED}
    assert clean.index.equals(raw.index)
    assert clean[TARGETS].equals(raw[TARGETS])
    assert set(raw.columns)-set(clean.columns) == set(dropped)
    assert all(clean[c].equals(raw[c]) for c in clean)
    assert not clean.isna().any().any()
    assert df['systolic_index'].equals(df['systolic_bp'])
    assert age_residual < 1.1e-6 and age_mapping_counts.max() == 1
    assert X_classification.columns.equals(X_regression.columns)
    assert not set(TARGETS).intersection(X_classification.columns)
    findings['validation'] = 'All integrity and consistency checks passed: sources are unchanged, all rows and retained values are preserved, targets are unchanged, only the three documented columns were dropped, and both outcomes are excluded from either predictor matrix. Reports and figures have been saved to `reports/eda/`. No predictive model has been trained.'
    (OUT / 'executed_findings.json').write_text(json.dumps(findings,indent=2,ensure_ascii=False)+'\\n')
    print('Validation passed. Reports:', len(list(OUT.iterdir())))
''')
nb.cells.append(nbformat.v4.new_markdown_cell('''## Consolidated conclusion

**What we learned.** The supplied dataset contains 2,000 profiles and 35 fields, including two existing targets. No missing cells, exact duplicate records, matched sentinel tokens or nonfinite numbers were found. The three class counts are 1,798 / 117 / 85: severe imbalance makes accuracy alone misleading. The regression outcome ranges from 70 to 300, with mean 135.9916 and median 134. Individual numeric Pearson associations with this target are weak (largest absolute correlation about 0.070).

**Decisions.** Preserve the supplied CSV and PDF unchanged. Retain every profile, including IQR flags and 130 records with systolic values below diastolic values. In a reproducible derived copy, remove `calcium_baseline` (constant), `systolic_index` (exact duplicate), and `age_normalized` (deterministic affine representation of age within rounding). This leaves 30 predictors and 2 targets. Keep `ldl_alternative` provisionally despite its identical value multiset to `ldl`: row alignment differs, so it is not an interchangeable row-wise feature. Age/BMI labels also overlap with their numeric representations and need deliberate handling in later training pipelines.

**Targets.** Classification predicts the literal three-label `target_class`; no diagnostic code key is supplied. Regression predicts continuous laboratory `target_regression`; biomarker name and units are unknown. Both targets are excluded from both predictor matrices. Target-to-target summaries are descriptive only.

**Caveats.** Units, transformations, previous imputation, record independence and feature timing are unverified. Concentrated measurements and differing class-conditioned spreads warrant provenance questions. Full-data EDA is exploratory; no held-out results, causal effects, clinical thresholds or diagnosis meanings are claimed. The empirical dictionary is not a validated clinical data dictionary.

**Ordered next steps.**
1. Ask the teacher for label definitions, biomarker units, source transformations, feature availability, patient independence and the status of `ldl_alternative`; clarify the 130 blood-pressure ordering flags.
2. Confirm group details, intermediate deliverables and the PL teacher’s GitHub username for private repository access.
3. When modeling is authorized, create a fixed seed-22 holdout with stratification for classification; account for possible clusters if identifiers become available. Keep the final test set for final evaluation, acknowledging these full-data EDA observations.
4. Fit imputation if later needed, encoding, scaling, feature selection and sampling on training folds only. Evaluate regression with R², MAE and RMSE; classification with macro-F1, balanced accuracy, per-class recall and confusion matrices, with accuracy secondary.
5. Follow the assignment’s later linear-regression, logistic-regression/LDA, resampling and regularization stages. Confirm the interpretation of Ridge/Lasso for classification before implementation.

**Sources and credit.** Assignment: supplied `Trabalho_APRAU_2026_27_step1.pdf`, pages 1–4. All empirical evidence comes from `dataset_grupo_22.csv`. Structure inspired by [the user's hockey project](https://github.com/AmirMasnavi/hockey-season-prediction/tree/main). Analysis uses pandas, NumPy, Matplotlib, seaborn and Jupyter; versions are recorded in `reports/eda/environment.json`. No external medical definitions were applied. Every group member should review and be able to explain these decisions.'''))

path = ROOT / 'notebooks' / '01_initial_eda.ipynb'
nbformat.write(nb,path)
NotebookClient(nb, timeout=240, kernel_name='python3', resources={'metadata':{'path':str(ROOT)}}).execute()
findings = json.loads((ROOT / 'reports/eda/executed_findings.json').read_text())
for cell in nb.cells:
    if cell.cell_type == 'markdown' and cell.source.startswith('FINDINGS::'):
        cell.source = '**Findings.** ' + findings[cell.source.split('::',1)[1]]
nbformat.validate(nb)
nbformat.write(nb,path)
print(f'Saved executed notebook: {path}')
