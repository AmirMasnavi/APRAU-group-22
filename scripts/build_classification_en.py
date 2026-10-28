"""Build and execute notebooks/03_classification_en.ipynb (Part 3: Classification + resampling) — English version.

Same structure as the Portuguese notebook: explanation (what and why) -> code -> conclusions,
with the conclusion texts in scripts/classification_findings_en.py, written from the executed outputs.
"""
from pathlib import Path
import textwrap
import nbformat
from nbclient import NotebookClient
from classification_findings_en import FINDINGS

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / 'notebooks' / '03_classification_en.ipynb'

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
nb.metadata.language_info = dict(name='python')


def md(s):
    nb.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))


def step(explanation, src, key):
    md(explanation)
    nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(src).strip()))
    md(FINDINGS.get(key, f'FINDINGS::{key}'))


md(FINDINGS.get('intro', 'FINDINGS::intro'))

# ================================================================ 1. preparation
md('## 1. Data preparation')

step('''
### 1.1 Loading the data and setting up the environment

We load the original dataset (without modifying it) and fix the random seed at **22** (project rule), so that every
data split, cross-validation and *bootstrap* resample is reproducible.
''', '''
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold, LeaveOneOut,
                                     cross_val_predict, cross_validate)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             precision_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)

SEED = 22
np.random.seed(SEED)
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_columns', 40)
sns.set_theme(style='whitegrid')

dataset_name = 'dataset_grupo_22.csv'
dataset_path = next(p for p in [Path.cwd() / dataset_name, Path.cwd().parent / dataset_name] if p.is_file())
raw = pd.read_csv(dataset_path)
print('Dataset shape:', raw.shape)
raw['target_class'].value_counts().to_frame('n').assign(**{'%': lambda d: d['n'] / len(raw) * 100})
''', 'load')

step('''
### 1.2 Feature removal, encoding and predictor matrix

We apply **the same preparation as in the regression notebook**, for the same reasons (EDA, `01_EDA`):
- we remove `calcium_baseline` (constant), `systolic_index` (copy of `systolic_bp`) and `age_normalized` (rescaling
  of `age`). The last two would make the covariance matrix **singular**, which is a direct problem for LDA, since it
  has to invert that matrix;
- we also remove `ldl_alternative`: it has the same values as `ldl`, but shuffled across patients (it matches on only
  18 of 2000 rows). The value does not belong to the patient in that row, so it is noise;
- we convert the categorical variables into numbers (ordinal → 0, 1, 2…; binary → 0/1; `employment_status` →
  *one-hot*);
- **we remove both targets from the predictors**: `target_class` is what we want to predict and `target_regression`
  is the other outcome of the assignment.

All remaining 30 features are kept. Feature selection is the topic of Part 4 (Ridge/Lasso); here the goal is to
compare classification and resampling methods under the same conditions.
''', '''
TARGET = 'target_class'
CLASSES = ['DI_HY', 'HT', 'DI_HT']          # fixed order: majority -> minorities
DROP_EDA = ['calcium_baseline', 'systolic_index', 'age_normalized', 'ldl_alternative']
ORDINAL = {
    'age_level': ['Young', 'Middle', 'Mature', 'Senior'],
    'bmi_level': ['Underweight', 'Normal', 'Overweight', 'Obese'],
    'smoking': ['Never', 'Former', 'Current'],
    'physical_activity': ['Low', 'Moderate', 'High'],
    'stress_level': ['Low', 'Medium', 'High'],
    'sugar_consumption': ['Low', 'Medium', 'High'],
    'education_level': ['Primary', 'Secondary', 'Tertiary'],
}
BINARY = {'gender': 'Male', 'family_history': 'Yes', 'low_hdl_cholesterol': 'Yes', 'high_ldl_cholesterol': 'Yes'}

data = raw.drop(columns=DROP_EDA).copy()
for col, levels in ORDINAL.items():
    data[col] = data[col].map({lvl: i for i, lvl in enumerate(levels)})
for col, positive in BINARY.items():
    data[col] = (data[col] == positive).astype(int)
data = pd.get_dummies(data, columns=['employment_status'], drop_first=True, dtype=int)

X = data.drop(columns=[TARGET, 'target_regression'])
y = data[TARGET]
assert X.isna().sum().sum() == 0
FEATURES = list(X.columns)
print(f'{X.shape[0]} records x {len(FEATURES)} features')
''', 'prep')

step('''
### 1.3 Evaluation metrics: why *accuracy* is not enough

The EDA showed that `target_class` is **highly imbalanced** (≈ 90% `DI_HY`). Under these conditions *accuracy* is
misleading: a model that always answers `DI_HY` is right ≈ 90% of the time without having learned anything. We
therefore use the following metrics, adapted to 3 classes:

| Metric | What it measures | Why it is useful here |
|---|---|---|
| **Recall per class** | Of the patients in each class, how many were correctly identified | Shows directly whether the rare classes (`HT`, `DI_HT`) are being detected |
| **Precision per class** | Of the predictions for each class, how many are correct | Shows the cost of "inventing" diagnoses |
| **Macro F1** (main metric) | Simple average of the F1 of the 3 classes | Every class weighs the same, so missing a rare class lowers the metric a lot |
| **Balanced accuracy** | Simple average of the 3 class recalls | A model that always predicts the majority class scores 1/3 = 0.333 |
| *Accuracy* | Proportion of correct predictions | Secondary reference only |
| **Confusion matrix** | Which classes are confused with which | Detailed diagnosis of the errors |

To make this concrete, we evaluate a **reference classifier** (*baseline*) that always predicts the majority class.
''', '''
def metrics(y_true, y_pred):
    rec = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    return {'f1_macro': f1_score(y_true, y_pred, labels=CLASSES, average='macro', zero_division=0),
            'bal_acc': balanced_accuracy_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            **{f'recall_{c}': r for c, r in zip(CLASSES, rec)}}

baseline_pred = np.full(len(y), 'DI_HY')
pd.Series(metrics(y, baseline_pred), name='baseline (always DI_HY)').to_frame().T
''', 'metrics')

# ================================================================ 2. models
md(FINDINGS.get('models_intro', 'FINDINGS::models_intro'))

step('''
### 2.1 Do the LDA assumptions hold?

LDA assumes that, **within each class**, the features follow a **normal** distribution and that
**all classes share the same covariance matrix**. Logistic Regression makes no such assumptions. Knowing whether the
assumptions hold helps us understand which method should perform better.

For the most discriminative features from the EDA, we check:
- **normality within each class** (Shapiro-Wilk test and skewness);
- **equality of variances across classes** (Levene's test and the ratio between the largest and smallest standard
  deviation).
''', '''
key = ['HbA1c_level', 'triglycerides', 'age', 'bmi', 'blood_pressure', 'crp_level']
rows = []
for f in key:
    groups = [data.loc[y == c, f] for c in CLASSES]
    stds = [g.std() for g in groups]
    rows.append({'feature': f,
                 **{f'std_{c}': s for c, s in zip(CLASSES, stds)},
                 'std_ratio (max/min)': max(stds) / min(stds),
                 'p_Levene': stats.levene(*groups).pvalue,
                 'skew_DI_HY': stats.skew(groups[0]),
                 'p_Shapiro_DI_HY': stats.shapiro(groups[0].sample(500, random_state=SEED)).pvalue})
assumptions = pd.DataFrame(rows).set_index('feature')

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, f in zip(axes, ['HbA1c_level', 'triglycerides', 'age']):
    sns.kdeplot(data=data, x=f, hue=y, hue_order=CLASSES, common_norm=False, fill=True, alpha=.3, ax=ax)
    ax.set_title(f'{f} by class')
plt.tight_layout(); plt.show()
assumptions
''', 'assumptions')

step('''
### 2.2 First fit: Logistic Regression and LDA "out of the box"

We fit both methods with scikit-learn's default options:
- **Multinomial Logistic Regression** (generalisation of the logistic model to 3 classes, using *softmax*),
  estimated by maximum likelihood. Features are standardised (`StandardScaler`) because scikit-learn
  applies a small default regularisation that depends on scale;
- **LDA**, which estimates the class means, a common covariance and the **prior probabilities** πₖ from the class
  proportions in the training data.

To avoid depending on a single data split, we use **stratified 10-fold cross-validation**: every patient receives a
prediction made by a model that never saw them. This is the default resampling method; section 3 compares it with the
others.

**Hypothesis to test:** since ~90% of patients are `DI_HY`, we expect the unadjusted models to predict `DI_HY` in most
cases and to miss the minority classes.
''', '''
MODELS = {
    'Logistic':          make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'LDA':                make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    'Logistic (weighted)':  make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
    'LDA (equal priors)': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(priors=[1/3] * 3)),
}
CV10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

def show_confusions(names, preds, title):
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.3))
    for ax, n in zip(np.atleast_1d(axes), names):
        ConfusionMatrixDisplay(confusion_matrix(y, preds[n], labels=CLASSES), display_labels=CLASSES).plot(
            ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(n); ax.set_xlabel('predicted'); ax.set_ylabel('actual')
    fig.suptitle(title); plt.tight_layout(); plt.show()

cv_preds = {n: cross_val_predict(m, X, y, cv=CV10) for n, m in MODELS.items()}
show_confusions(['Logistic', 'LDA'], cv_preds, 'Confusion matrices (stratified 10-fold) — no imbalance adjustment')
first = pd.DataFrame({n: metrics(y, cv_preds[n]) for n in ['Logistic', 'LDA']}).T
first.loc['baseline (always DI_HY)'] = metrics(y, baseline_pred)
first
''', 'first_fit')

step('''
### 2.3 Correcting the imbalance

The problem in 2.2 has a known cause for each method:
- in **Logistic Regression**, the likelihood is a sum over all patients. With 1798 `DI_HY` against 85 `DI_HT`,
  misclassifying a `DI_HT` costs almost nothing in total. With `class_weight='balanced'`, each patient is weighted by
  the inverse of their class frequency: a `DI_HT` becomes worth ~21 times a `DI_HY`;
- in **LDA**, Bayes' rule multiplies the likelihood by the prior probability πₖ. With π(`DI_HY`) = 0.90 and
  π(`DI_HT`) = 0.04, a patient is only classified as `DI_HT` if the evidence is very strong. With **equal priors**
  (1/3 each), the decision depends only on the similarity to each class.

We compare the 4 versions using the same cross-validation partitions.
''', '''
show_confusions(['Logistic (weighted)', 'LDA (equal priors)'], cv_preds,
                'Confusion matrices (stratified 10-fold) — with imbalance adjustment')
versions = pd.DataFrame({n: metrics(y, p) for n, p in cv_preds.items()}).T
for n, p in cv_preds.items():
    versions.loc[n, 'precision_DI_HT'] = precision_score(y, p, labels=['DI_HT'], average='macro', zero_division=0)
    versions.loc[n, 'n_predicted_DI_HT'] = (p == 'DI_HT').sum()
versions
''', 'balanced')

step('''
### 2.4 Which version to use? Comparison with statistical significance

The differences between versions may be small. To know whether they are **real** or due to which patients fell into
each fold, we compute macro F1 and *balanced accuracy* on **each of the 10 folds**, using
the **same partitions** for all models, and apply a **paired *t*-test** between models.
''', '''
fold_scores = {}
for n, m in MODELS.items():
    r = cross_validate(m, X, y, cv=CV10, scoring={'f1_macro': 'f1_macro', 'bal_acc': 'balanced_accuracy'})
    fold_scores[n] = pd.DataFrame({'f1_macro': r['test_f1_macro'], 'bal_acc': r['test_bal_acc']})

summary = pd.DataFrame({n: {'F1 macro (mean)': s.f1_macro.mean(), 'F1 macro (sd)': s.f1_macro.std(),
                            'Bal. acc. (mean)': s.bal_acc.mean(), 'Bal. acc. (sd)': s.bal_acc.std()}
                        for n, s in fold_scores.items()}).T
display(summary)

pairs = [('LDA', 'Logistic'), ('LDA (equal priors)', 'Logistic (weighted)'),
         ('Logistic (weighted)', 'Logistic'), ('LDA (equal priors)', 'LDA')]
tests = []
for a, b in pairs:
    for met in ('f1_macro', 'bal_acc'):
        d = fold_scores[a][met] - fold_scores[b][met]
        tests.append({'comparison': f'{a}  vs  {b}', 'metric': met, 'mean difference': d.mean(),
                      'folds where A > B': f'{(d > 0).sum()}/10', 'p-value (paired t)': stats.ttest_rel(
                          fold_scores[a][met], fold_scores[b][met]).pvalue})
pd.DataFrame(tests)
''', 'versions_test')

# ================================================================ 3. resampling
md(FINDINGS.get('resampling_intro', 'FINDINGS::resampling_intro'))

step('''
### 3.1 *Holdout* (validation set approach)

**How it works:** the data are split once into training (70%) and test (30%); the model learns on the
training set and is evaluated on the test set. We use a **stratified** split so that the 3 classes appear in the test
set in the same proportions as in the dataset: with only 85 `DI_HT`, a non-stratified split could leave the test set
with very few of them.

We show the full report (`classification_report`) for the four models.
''', '''
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
print('Test:', y_te.value_counts().to_dict())
holdout = {}
for n, m in MODELS.items():
    pred = clone(m).fit(X_tr, y_tr).predict(X_te)
    holdout[n] = metrics(y_te, pred)
    if n in ('Logistic (weighted)', 'LDA (equal priors)'):
        print(f'\\n=== {n} ===')
        print(classification_report(y_te, pred, labels=CLASSES, digits=3, zero_division=0))
pd.DataFrame(holdout).T
''', 'holdout')

step('''
### 3.2 How much does the *holdout* vary? Repeating the split 100 times

The result of a *holdout* depends on **which** patients ended up in the test set: the estimate has variance, and a
different split gives a different result. To measure this variance, we repeat the
stratified *holdout* with **100 different seeds** and look at how the metrics are distributed.
''', '''
def repeated_holdout(model, n_rep=100, test_size=0.30):
    out = []
    for s in range(n_rep):
        a, b, c, d = train_test_split(X, y, test_size=test_size, stratify=y, random_state=s)
        out.append(metrics(d, clone(model).fit(a, c).predict(b)))
    return pd.DataFrame(out)

rep_holdout = {n: repeated_holdout(MODELS[n]) for n in MODELS}
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=pd.DataFrame({n: r.f1_macro for n, r in rep_holdout.items()}), ax=ax)
ax.set_ylabel('F1 macro'); ax.set_title('F1 macro over 100 different holdouts (70/30 stratified)')
plt.show()
pd.DataFrame({n: {'mean': r.f1_macro.mean(), 'sd': r.f1_macro.std(), 'min': r.f1_macro.min(),
                  'max': r.f1_macro.max(), 'recall DI_HT sd': r.recall_DI_HT.std()}
              for n, r in rep_holdout.items()}).T
''', 'rep_holdout')

step('''
### 3.3 *k-fold* cross-validation (k = 5 and k = 10)

**How it works:** the data are divided into *k* parts (*folds*); each fold serves once as the test
set while the other *k−1* train the model. Every patient is tested **exactly once**. We use the **stratified** version:
with imbalanced classes, each fold keeps the class proportions.

There are **two kinds of variance** to distinguish:
1. **Between folds:** how much the *k* metrics vary within a single cross-validation;
2. **Of the estimate:** how much the final **mean** changes if we repeat the cross-validation with another random
   partition. We repeat it 20 times, with 20 seeds.
''', '''
def repeated_kfold(model, k, n_rep=20):
    fold_level, means = [], []
    for s in range(n_rep):
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=s)
        r = cross_validate(model, X, y, cv=cv, scoring={'f1_macro': 'f1_macro', 'bal_acc': 'balanced_accuracy'})
        fold_level.append(r['test_f1_macro'])
        means.append({'f1_macro': r['test_f1_macro'].mean(), 'bal_acc': r['test_bal_acc'].mean()})
    return np.concatenate(fold_level), pd.DataFrame(means)

kfold = {}
for k in (5, 10):
    for n in MODELS:
        folds, means = repeated_kfold(MODELS[n], k)
        kfold[(k, n)] = {'folds': folds, 'means': means}

kfold_table = pd.DataFrame({f'{k}-fold | {n}': {
    'F1 macro (estimate)': v['means'].f1_macro.mean(),
    'sd between folds': v['folds'].std(),
    'sd of estimate (20 repeats)': v['means'].f1_macro.std(),
    'Bal. acc. (estimate)': v['means'].bal_acc.mean()} for (k, n), v in kfold.items()}).T
kfold_table
''', 'kfold')

step('''
### 3.4 *Leave-One-Out* (LOOCV)

**How it works:** it is the extreme case of *k-fold*, with *k = N* = 2000. The model is trained 2000
times, always with 1999 patients, and tested on the patient left out.

Two important particularities:
- each fold has **a single patient**, so a "per-fold F1" does not exist (with a single prediction you can only be
  right or wrong). The metrics are computed by pooling the 2000 predictions;
- LOOCV is **deterministic**: there is only one way to leave each patient out, so repeating it always gives the same
  result. It also cannot be stratified.
''', '''
loo_preds, loo_time = {}, {}
for n, m in MODELS.items():
    t0 = time.time()
    loo_preds[n] = cross_val_predict(m, X, y, cv=LeaveOneOut(), n_jobs=-1)
    loo_time[n] = time.time() - t0
loo_table = pd.DataFrame({n: {**metrics(y, p), 'time (s)': loo_time[n], 'n models trained': len(y)}
                          for n, p in loo_preds.items()}).T
loo_table
''', 'loocv')

step('''
### 3.5 *Bootstrap* with *out-of-bag* (OOB) evaluation

**How it works:** at each iteration we draw a sample **with replacement**
of the same size as the dataset (2000). Some patients appear repeated and others are left out; the probability of a
patient being left out is $(1 - 1/N)^N \approx e^{-1} \approx 36.8\%$. The model trains on the *bootstrap* sample
and is evaluated on the left-out (*out-of-bag*) patients, which it never saw.

We repeat this **200 times** and obtain the distribution of the metrics, the mean, the
standard deviation and a 95% confidence interval (2.5th and 97.5th percentiles).
''', '''
def bootstrap_oob(model, B=200, seed=SEED):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    out = []
    for _ in range(B):
        inb = rng.choice(idx_all, size=len(y), replace=True)
        oob = np.setdiff1d(idx_all, inb)
        m = clone(model).fit(X.iloc[inb], y.iloc[inb])
        out.append({**metrics(y.iloc[oob], m.predict(X.iloc[oob])), 'oob_frac': len(oob) / len(y),
                    'n_DI_HT_oob': (y.iloc[oob] == 'DI_HT').sum()})
    return pd.DataFrame(out)

boot = {n: bootstrap_oob(MODELS[n]) for n in MODELS}
boot_table = pd.DataFrame({n: {'F1 macro (mean)': b.f1_macro.mean(), 'sd': b.f1_macro.std(),
                               'IC95% inf': b.f1_macro.quantile(.025), 'IC95% sup': b.f1_macro.quantile(.975),
                               'Bal. acc. (mean)': b.bal_acc.mean(), 'mean OOB fraction': b.oob_frac.mean(),
                               'n DI_HT in OOB (mean)': b.n_DI_HT_oob.mean()} for n, b in boot.items()}).T
boot_table
''', 'bootstrap')

# ================================================================ 4. variance
md('''
## 4. How does the resampling method affect the variance?

This is the central question of the assignment: *"indicate how the variance is affected by the resampling methods
used"*. Here we bring together the results of sections 3.1–3.5.
''')

step('''
### 4.1 Side-by-side comparison

For each method and model we show:
- the macro F1 **estimate**;
- the **variability of the individual evaluations** (between splits, folds or iterations);
- the **variability of the final estimate** (how much the estimate changes if the procedure is repeated);
- the **cost** (number of models trained).
''', '''
rows = []
for n in MODELS:
    rows.append({'model': n, 'method': 'Holdout (1 split)', 'F1 macro': holdout[n]['f1_macro'],
                 'sd of evaluations': np.nan, 'sd of estimate': rep_holdout[n].f1_macro.std(), 'n models': 1})
    for k in (5, 10):
        v = kfold[(k, n)]
        rows.append({'model': n, 'method': f'{k}-fold stratified', 'F1 macro': v['means'].f1_macro.mean(),
                     'sd of evaluations': v['folds'].std(), 'sd of estimate': v['means'].f1_macro.std(), 'n models': k})
    rows.append({'model': n, 'method': 'LOOCV', 'F1 macro': loo_table.loc[n, 'f1_macro'],
                 'sd of evaluations': np.nan, 'sd of estimate': 0.0, 'n models': len(y)})
    rows.append({'model': n, 'method': 'Bootstrap OOB (B=200)', 'F1 macro': boot[n].f1_macro.mean(),
                 'sd of evaluations': boot[n].f1_macro.std(),
                 'sd of estimate': boot[n].f1_macro.std() / np.sqrt(len(boot[n])), 'n models': 200})
variance = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
for ax, n in zip(axes, ['Logistic (weighted)', 'LDA (equal priors)']):
    dist = pd.DataFrame({'Holdout x100': rep_holdout[n].f1_macro,
                         '5-fold (folds)': pd.Series(kfold[(5, n)]['folds']),
                         '10-fold (folds)': pd.Series(kfold[(10, n)]['folds']),
                         '5-fold (means)': kfold[(5, n)]['means'].f1_macro,
                         '10-fold (means)': kfold[(10, n)]['means'].f1_macro,
                         'Bootstrap OOB': boot[n].f1_macro})
    sns.boxplot(data=dist, ax=ax); ax.axhline(loo_table.loc[n, 'f1_macro'], color='red', ls='--', label='LOOCV')
    ax.set_title(n); ax.tick_params(axis='x', rotation=25); ax.legend()
axes[0].set_ylabel('F1 macro')
plt.tight_layout(); plt.show()
variance[variance.model.isin(['Logistic (weighted)', 'LDA (equal priors)'])].set_index(['model', 'method'])
''', 'variance_table')

step('''
### 4.2 The effect of *k* in cross-validation

Choosing *k* involves a trade-off:
- **small *k*** → each model trains on less data → **pessimistic** (more biased) estimate, but more stable and
  cheaper;
- **large *k*** (in the limit, LOOCV) → training on almost all the data → **low-bias** estimate, but the models are
  very similar to each other and the cost grows.

For *k* ∈ {2, 3, 5, 10, 20}, we measure the mean estimate, the variability **between folds** and the variability **of
the estimate** (10 repetitions with different seeds), for the weighted version of each model.
''', '''
k_rows = []
for n in ['Logistic (weighted)', 'LDA (equal priors)']:
    for k in (2, 3, 5, 10, 20):
        folds, means = repeated_kfold(MODELS[n], k, n_rep=10)
        k_rows.append({'model': n, 'k': k, 'training size': int(len(y) * (k - 1) / k),
                       'DI_HT per test fold': round(85 / k, 1), 'F1 macro': means.f1_macro.mean(),
                       'sd between folds': folds.std(), 'sd of estimate': means.f1_macro.std()})
k_effect = pd.DataFrame(k_rows)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for n, g in k_effect.groupby('model'):
    axes[0].plot(g.k, g['F1 macro'], marker='o', label=n)
    axes[1].plot(g.k, g['sd between folds'], marker='o', label=n)
    axes[2].plot(g.k, g['sd of estimate'], marker='o', label=n)
for ax, t in zip(axes, ['F1 macro estimate', 'sd between folds', 'sd of estimate (10 repeats)']):
    ax.set_xlabel('k'); ax.set_title(t); ax.set_xscale('log'); ax.set_xticks([2, 3, 5, 10, 20]); ax.set_xticklabels([2, 3, 5, 10, 20]); ax.legend()
for n in ['Logistic (weighted)', 'LDA (equal priors)']:
    axes[0].axhline(loo_table.loc[n, 'f1_macro'], ls=':', color='grey')
plt.tight_layout(); plt.show()
k_effect.set_index(['model', 'k'])
''', 'k_effect')

# ================================================================ 5. decision
step('''
## 5. Which method is most appropriate?

We gather the evidence from all resampling methods. If the ranking between Logistic Regression and LDA is the same
across all resampling methods, the conclusion is robust.

We also show the Logistic Regression **coefficients**, to confirm that the model uses the features the EDA identified
as informative.
''', '''
decision = pd.DataFrame({
    'Holdout':           {n: holdout[n]['bal_acc'] for n in MODELS},
    'Holdout x100 (mean)': {n: rep_holdout[n].bal_acc.mean() for n in MODELS},
    '5-fold':            {n: kfold[(5, n)]['means'].bal_acc.mean() for n in MODELS},
    '10-fold':           {n: kfold[(10, n)]['means'].bal_acc.mean() for n in MODELS},
    'LOOCV':             {n: loo_table.loc[n, 'bal_acc'] for n in MODELS},
    'Bootstrap OOB':     {n: boot[n].bal_acc.mean() for n in MODELS},
})
decision_f1 = pd.DataFrame({
    'Holdout':           {n: holdout[n]['f1_macro'] for n in MODELS},
    'Holdout x100 (mean)': {n: rep_holdout[n].f1_macro.mean() for n in MODELS},
    '5-fold':            {n: kfold[(5, n)]['means'].f1_macro.mean() for n in MODELS},
    '10-fold':           {n: kfold[(10, n)]['means'].f1_macro.mean() for n in MODELS},
    'LOOCV':             {n: loo_table.loc[n, 'f1_macro'] for n in MODELS},
    'Bootstrap OOB':     {n: boot[n].f1_macro.mean() for n in MODELS},
})
print('=== Balanced accuracy by resampling method ===');  display(decision)
print('=== F1 macro by resampling method ===');           display(decision_f1)

lr = clone(MODELS['Logistic (weighted)']).fit(X, y)
coefs = pd.DataFrame(lr[-1].coef_.T, index=FEATURES, columns=lr[-1].classes_)
coefs['|coef| max'] = coefs.abs().max(axis=1)
coefs.sort_values('|coef| max', ascending=False).head(10)
''', 'decision')

md(FINDINGS.get('discussion', 'FINDINGS::discussion'))

client = NotebookClient(nb, timeout=3600, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT / 'notebooks')}})
client.execute()
nbformat.write(nb, OUT_NB)
print('written', OUT_NB)
