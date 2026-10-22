"""Build and execute notebooks/03_classification.ipynb (Parte 3: Classificação + reamostragem).

Mesma estrutura dos notebooks anteriores: explicação (o quê e porquê) -> código -> conclusões,
com os textos de conclusões em scripts/classification_findings.py, escritos a partir dos outputs.
"""
from pathlib import Path
import textwrap
import nbformat
from nbclient import NotebookClient
from classification_findings import FINDINGS

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / 'notebooks' / '03_classification.ipynb'

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

# ================================================================ 1. preparação
md('## 1. Preparação dos dados')

step('''
### 1.1 Carregamento dos dados e preparação do ambiente

Carregamos o dataset original (sem o alterar) e fixamos a semente aleatória em **22** (regra do projeto), para que
todas as divisões dos dados, validações cruzadas e reamostragens *bootstrap* sejam reprodutíveis.
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
print('Dimensao do dataset:', raw.shape)
raw['target_class'].value_counts().to_frame('n').assign(**{'%': lambda d: d['n'] / len(raw) * 100})
''', 'load')

step('''
### 1.2 Remoção de features, codificação e matriz de preditores

Aplicamos **a mesma preparação do notebook da regressão**, pelas mesmas razões (EDA, secção 4.2):
- removemos `calcium_baseline` (constante), `systolic_index` (cópia de `systolic_bp`) e `age_normalized` (reescala
  de `age`). As duas últimas tornariam a matriz de covariâncias **singular**, o que é um problema direto para a LDA,
  que tem de inverter essa matriz;
- convertemos as categóricas em números (ordinais → 0, 1, 2…; binárias → 0/1; `employment_status` → *one-hot*);
- **retiramos os dois targets dos preditores**: `target_class` é o que queremos prever e `target_regression` é o
  outro resultado do trabalho.

As restantes 31 features ficam todas. A seleção de features é o tema da Parte 4 (Ridge/Lasso); aqui o objetivo é
comparar métodos de classificação e de reamostragem nas mesmas condições.
''', '''
TARGET = 'target_class'
CLASSES = ['DI_HY', 'HT', 'DI_HT']          # ordem fixa: maioritária -> minoritárias
DROP_EDA = ['calcium_baseline', 'systolic_index', 'age_normalized']
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
print(f'{X.shape[0]} registos x {len(FEATURES)} features')
''', 'prep')

step('''
### 1.3 Métricas de avaliação: porque não basta a *accuracy*

A EDA mostrou que `target_class` é **muito desequilibrada** (≈ 90% `DI_HY`). Nestas condições, a *accuracy* engana:
um modelo que responda sempre `DI_HY` acerta em ≈ 90% dos casos sem ter aprendido nada. Usamos por isso as métricas
dos slides T4 (11–18), adaptadas a 3 classes:

| Métrica | O que mede | Porque é útil aqui |
|---|---|---|
| **Recall por classe** | Dos pacientes de cada classe, quantos foram bem identificados | Mostra diretamente se as classes raras (`HT`, `DI_HT`) estão a ser detetadas |
| **Precision por classe** | Das previsões de cada classe, quantas estão certas | Mostra o custo de "inventar" diagnósticos |
| **F1 macro** (métrica principal) | Média simples do F1 das 3 classes | Cada classe pesa o mesmo, por isso falhar uma classe rara baixa muito a métrica |
| **Balanced accuracy** | Média simples dos recalls das 3 classes | Um modelo que prevê sempre a maioritária fica em 1/3 = 0.333 |
| *Accuracy* | Proporção de acertos | Só como referência secundária |
| **Matriz de confusão** | Que classes são confundidas com quais | Diagnóstico detalhado dos erros |

Para tornar isto concreto, avaliamos um **classificador de referência** (*baseline*) que prevê sempre a classe
maioritária.
''', '''
def metrics(y_true, y_pred):
    rec = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    return {'f1_macro': f1_score(y_true, y_pred, labels=CLASSES, average='macro', zero_division=0),
            'bal_acc': balanced_accuracy_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            **{f'recall_{c}': r for c, r in zip(CLASSES, rec)}}

baseline_pred = np.full(len(y), 'DI_HY')
pd.Series(metrics(y, baseline_pred), name='baseline (sempre DI_HY)').to_frame().T
''', 'metrics')

# ================================================================ 2. modelos
md(FINDINGS.get('models_intro', 'FINDINGS::models_intro'))

step('''
### 2.1 Os pressupostos da LDA verificam-se?

A LDA (slides T3, 29–31) assume que, **dentro de cada classe**, as features seguem uma distribuição **normal** e que
**todas as classes têm a mesma matriz de covariâncias**. A Regressão Logística não faz estas suposições. Saber se os
pressupostos se verificam ajuda a perceber qual dos métodos deve funcionar melhor.

Verificamos, para as features mais discriminativas da EDA:
- a **normalidade dentro de cada classe** (teste de Shapiro-Wilk e assimetria);
- a **igualdade das variâncias entre classes** (teste de Levene e rácio entre o maior e o menor desvio padrão).
''', '''
key = ['HbA1c_level', 'triglycerides', 'age', 'bmi', 'blood_pressure', 'crp_level']
rows = []
for f in key:
    groups = [data.loc[y == c, f] for c in CLASSES]
    stds = [g.std() for g in groups]
    rows.append({'feature': f,
                 **{f'std_{c}': s for c, s in zip(CLASSES, stds)},
                 'racio_std (max/min)': max(stds) / min(stds),
                 'p_Levene': stats.levene(*groups).pvalue,
                 'skew_DI_HY': stats.skew(groups[0]),
                 'p_Shapiro_DI_HY': stats.shapiro(groups[0].sample(500, random_state=SEED)).pvalue})
assumptions = pd.DataFrame(rows).set_index('feature')

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, f in zip(axes, ['HbA1c_level', 'triglycerides', 'age']):
    sns.kdeplot(data=data, x=f, hue=y, hue_order=CLASSES, common_norm=False, fill=True, alpha=.3, ax=ax)
    ax.set_title(f'{f} por classe')
plt.tight_layout(); plt.show()
assumptions
''', 'assumptions')

step('''
### 2.2 Primeiro ajuste: Regressão Logística e LDA "tal como vêm"

Ajustamos os dois métodos com as opções por defeito do scikit-learn:
- **Regressão Logística multinomial** (generalização da logística para 3 classes, com *softmax*), estimada por máxima
  verosimilhança (slide T3, 13). As features são padronizadas (`StandardScaler`), porque o scikit-learn aplica por
  defeito uma pequena regularização que depende da escala;
- **LDA**, que estima as médias de cada classe, uma covariância comum e as **probabilidades a priori** πₖ a partir
  das proporções das classes no treino (slide T3, 30).

Para não depender de uma única divisão dos dados, usamos **validação cruzada estratificada 10-fold**: cada paciente
recebe uma previsão feita por um modelo que nunca o viu. É o método de reamostragem padrão; a secção 3 compara-o com
os outros.

**Hipótese a testar:** como ~90% dos pacientes são `DI_HY`, esperamos que os modelos sem ajuste prevejam
`DI_HY` na maioria dos casos e falhem as classes minoritárias.
''', '''
MODELS = {
    'Logística':          make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'LDA':                make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    'Logística (pesos)':  make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
    'LDA (priors iguais)': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(priors=[1/3] * 3)),
}
CV10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

def show_confusions(names, preds, title):
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.3))
    for ax, n in zip(np.atleast_1d(axes), names):
        ConfusionMatrixDisplay(confusion_matrix(y, preds[n], labels=CLASSES), display_labels=CLASSES).plot(
            ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(n); ax.set_xlabel('previsto'); ax.set_ylabel('real')
    fig.suptitle(title); plt.tight_layout(); plt.show()

cv_preds = {n: cross_val_predict(m, X, y, cv=CV10) for n, m in MODELS.items()}
show_confusions(['Logística', 'LDA'], cv_preds, 'Matrizes de confusão (10-fold estratificado) — sem ajuste ao desequilíbrio')
first = pd.DataFrame({n: metrics(y, cv_preds[n]) for n in ['Logística', 'LDA']}).T
first.loc['baseline (sempre DI_HY)'] = metrics(y, baseline_pred)
first
''', 'first_fit')

step('''
### 2.3 Corrigir o desequilíbrio

O problema de 2.2 tem uma causa conhecida para cada método:
- na **Regressão Logística**, a função de verosimilhança é uma soma sobre todos os pacientes. Com 1798 `DI_HY` contra
  85 `DI_HT`, errar um `DI_HT` custa quase nada no total. Com `class_weight='balanced'`, cada paciente passa a pesar
  o inverso da frequência da sua classe: um `DI_HT` passa a valer ~21 vezes um `DI_HY`;
- na **LDA**, a regra de Bayes multiplica a verosimilhança pela probabilidade a priori πₖ. Com π(`DI_HY`) = 0.90 e
  π(`DI_HT`) = 0.04, um paciente só é classificado como `DI_HT` se os dados forem muito fortes. Com
  **priors iguais** (1/3 cada), a decisão passa a depender só da semelhança com cada classe.

Comparamos as 4 versões com as mesmas partições de validação cruzada.
''', '''
show_confusions(['Logística (pesos)', 'LDA (priors iguais)'], cv_preds,
                'Matrizes de confusão (10-fold estratificado) — com ajuste ao desequilíbrio')
versions = pd.DataFrame({n: metrics(y, p) for n, p in cv_preds.items()}).T
for n, p in cv_preds.items():
    versions.loc[n, 'precision_DI_HT'] = precision_score(y, p, labels=['DI_HT'], average='macro', zero_division=0)
    versions.loc[n, 'n_previstos_DI_HT'] = (p == 'DI_HT').sum()
versions
''', 'balanced')

step('''
### 2.4 Que versão usar? Comparação com significância estatística

As diferenças entre versões podem ser pequenas. Para saber se são **reais** ou fruto de quais pacientes calharam em
cada fold, seguimos os slides T4 (31–33): obtemos o F1 macro e a *balanced accuracy* em **cada um dos 10 folds**,
com as **mesmas partições** para todos os modelos, e aplicamos um **teste *t* emparelhado** entre modelos.
''', '''
fold_scores = {}
for n, m in MODELS.items():
    r = cross_validate(m, X, y, cv=CV10, scoring={'f1_macro': 'f1_macro', 'bal_acc': 'balanced_accuracy'})
    fold_scores[n] = pd.DataFrame({'f1_macro': r['test_f1_macro'], 'bal_acc': r['test_bal_acc']})

summary = pd.DataFrame({n: {'F1 macro (média)': s.f1_macro.mean(), 'F1 macro (dp)': s.f1_macro.std(),
                            'Bal. acc. (média)': s.bal_acc.mean(), 'Bal. acc. (dp)': s.bal_acc.std()}
                        for n, s in fold_scores.items()}).T
display(summary)

pairs = [('LDA', 'Logística'), ('LDA (priors iguais)', 'Logística (pesos)'),
         ('Logística (pesos)', 'Logística'), ('LDA (priors iguais)', 'LDA')]
tests = []
for a, b in pairs:
    for met in ('f1_macro', 'bal_acc'):
        d = fold_scores[a][met] - fold_scores[b][met]
        tests.append({'comparação': f'{a}  vs  {b}', 'métrica': met, 'diferença média': d.mean(),
                      'folds em que A > B': f'{(d > 0).sum()}/10', 'p-value (t emparelhado)': stats.ttest_rel(
                          fold_scores[a][met], fold_scores[b][met]).pvalue})
pd.DataFrame(tests)
''', 'versions_test')

# ================================================================ 3. reamostragem
md(FINDINGS.get('resampling_intro', 'FINDINGS::resampling_intro'))

step('''
### 3.1 *Holdout* (método do conjunto de validação)

**Como funciona (slide T4, 19):** divide-se os dados uma vez em treino (70%) e teste (30%); o modelo aprende no
treino e é avaliado no teste. Usamos divisão **estratificada**, para que as 3 classes apareçam no teste nas mesmas
proporções que no dataset: com só 85 `DI_HT`, uma divisão não estratificada podia deixar o teste com muito poucos.

Mostramos o relatório completo (`classification_report`, como no slide T3, 37) para os quatro modelos.
''', '''
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
print('Teste:', y_te.value_counts().to_dict())
holdout = {}
for n, m in MODELS.items():
    pred = clone(m).fit(X_tr, y_tr).predict(X_te)
    holdout[n] = metrics(y_te, pred)
    if n in ('Logística (pesos)', 'LDA (priors iguais)'):
        print(f'\\n=== {n} ===')
        print(classification_report(y_te, pred, labels=CLASSES, digits=3, zero_division=0))
pd.DataFrame(holdout).T
''', 'holdout')

step('''
### 3.2 Quanto varia o *holdout*? Repetir a divisão 100 vezes

O resultado de um *holdout* depende de **quais** pacientes calharam no teste. Os slides (T4, 20 e 36–37) mostram
que a estimativa tem variância: com outra divisão, o resultado muda. Para medir essa variância, repetimos o *holdout*
estratificado com **100 sementes diferentes** e vemos como se distribuem as métricas.
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
ax.set_ylabel('F1 macro'); ax.set_title('F1 macro em 100 holdouts diferentes (70/30 estratificado)')
plt.show()
pd.DataFrame({n: {'média': r.f1_macro.mean(), 'dp': r.f1_macro.std(), 'mín': r.f1_macro.min(),
                  'máx': r.f1_macro.max(), 'recall DI_HT dp': r.recall_DI_HT.std()}
              for n, r in rep_holdout.items()}).T
''', 'rep_holdout')

step('''
### 3.3 Validação cruzada *k-fold* (k = 5 e k = 10)

**Como funciona (slides T4, 21–26):** os dados são divididos em *k* partes (*folds*); cada fold serve uma vez de teste
enquanto os outros *k−1* treinam o modelo. Cada paciente é testado **exatamente uma vez**. Usamos a versão
**estratificada** (slide 26): com classes desequilibradas, cada fold mantém as proporções das classes.

Há **duas formas de variância** a distinguir:
1. **Entre folds:** quanto variam as *k* métricas dentro de uma mesma validação cruzada;
2. **Da estimativa:** quanto muda a **média** final se repetirmos a validação cruzada com outra partição aleatória.
   Repetimos 20 vezes, com 20 sementes.
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
    'F1 macro (estimativa)': v['means'].f1_macro.mean(),
    'dp entre folds': v['folds'].std(),
    'dp da estimativa (20 repetições)': v['means'].f1_macro.std(),
    'Bal. acc. (estimativa)': v['means'].bal_acc.mean()} for (k, n), v in kfold.items()}).T
kfold_table
''', 'kfold')

step('''
### 3.4 *Leave-One-Out* (LOOCV)

**Como funciona (slide T4, 27):** é o caso extremo do *k-fold*, com *k = N* = 2000. Treina-se o modelo 2000 vezes,
sempre com 1999 pacientes, e testa-se no paciente que ficou de fora.

Duas particularidades importantes:
- cada fold tem **um só paciente**, por isso o "F1 de cada fold" não existe (com uma previsão só se pode acertar ou
  errar). As métricas calculam-se juntando as 2000 previsões;
- o LOOCV é **determinístico**: só há uma forma de deixar um paciente de fora de cada vez, logo repetir dá sempre o
  mesmo resultado. Também não pode ser estratificado.
''', '''
loo_preds, loo_time = {}, {}
for n, m in MODELS.items():
    t0 = time.time()
    loo_preds[n] = cross_val_predict(m, X, y, cv=LeaveOneOut(), n_jobs=-1)
    loo_time[n] = time.time() - t0
loo_table = pd.DataFrame({n: {**metrics(y, p), 'tempo (s)': loo_time[n], 'nº de modelos treinados': len(y)}
                          for n, p in loo_preds.items()}).T
loo_table
''', 'loocv')

step('''
### 3.5 *Bootstrap* com avaliação *out-of-bag* (OOB)

**Como funciona (slides T4, 41–46, e o exemplo das aulas):** em cada iteração tira-se uma amostra **com reposição**
do mesmo tamanho do dataset (2000). Alguns pacientes aparecem repetidos e outros ficam de fora; a probabilidade de um
paciente ficar de fora é $(1 - 1/N)^N \\approx e^{-1} \\approx 36.8\\%$. O modelo treina com a amostra *bootstrap* e é
avaliado nos pacientes que ficaram de fora (*out-of-bag*), que nunca viu.

Repetimos **200 vezes** (como no exemplo das aulas) e obtemos a distribuição das métricas, a média, o desvio padrão e
um intervalo de confiança a 95% (percentis 2.5 e 97.5).
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
boot_table = pd.DataFrame({n: {'F1 macro (média)': b.f1_macro.mean(), 'dp': b.f1_macro.std(),
                               'IC95% inf': b.f1_macro.quantile(.025), 'IC95% sup': b.f1_macro.quantile(.975),
                               'Bal. acc. (média)': b.bal_acc.mean(), 'fração OOB média': b.oob_frac.mean(),
                               'nº DI_HT no OOB (média)': b.n_DI_HT_oob.mean()} for n, b in boot.items()}).T
boot_table
''', 'bootstrap')

# ================================================================ 4. variância
md('''
## 4. Como é que o método de reamostragem afeta a variância?

Esta é a pergunta central do enunciado: *"indicate how the variance is affected by the resampling methods used"*.
Juntamos aqui os resultados das secções 3.1–3.5.
''')

step('''
### 4.1 Comparação lado a lado

Para cada método e modelo mostramos:
- a **estimativa** do F1 macro;
- a **variabilidade das avaliações individuais** (entre divisões, folds ou iterações);
- a **variabilidade da estimativa final** (quanto muda a estimativa se o procedimento for repetido);
- o **custo** (nº de modelos treinados).
''', '''
rows = []
for n in MODELS:
    rows.append({'modelo': n, 'método': 'Holdout (1 divisão)', 'F1 macro': holdout[n]['f1_macro'],
                 'dp das avaliações': np.nan, 'dp da estimativa': rep_holdout[n].f1_macro.std(), 'nº modelos': 1})
    for k in (5, 10):
        v = kfold[(k, n)]
        rows.append({'modelo': n, 'método': f'{k}-fold estratificado', 'F1 macro': v['means'].f1_macro.mean(),
                     'dp das avaliações': v['folds'].std(), 'dp da estimativa': v['means'].f1_macro.std(), 'nº modelos': k})
    rows.append({'modelo': n, 'método': 'LOOCV', 'F1 macro': loo_table.loc[n, 'f1_macro'],
                 'dp das avaliações': np.nan, 'dp da estimativa': 0.0, 'nº modelos': len(y)})
    rows.append({'modelo': n, 'método': 'Bootstrap OOB (B=200)', 'F1 macro': boot[n].f1_macro.mean(),
                 'dp das avaliações': boot[n].f1_macro.std(),
                 'dp da estimativa': boot[n].f1_macro.std() / np.sqrt(len(boot[n])), 'nº modelos': 200})
variance = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
for ax, n in zip(axes, ['Logística (pesos)', 'LDA (priors iguais)']):
    dist = pd.DataFrame({'Holdout x100': rep_holdout[n].f1_macro,
                         '5-fold (folds)': pd.Series(kfold[(5, n)]['folds']),
                         '10-fold (folds)': pd.Series(kfold[(10, n)]['folds']),
                         '5-fold (médias)': kfold[(5, n)]['means'].f1_macro,
                         '10-fold (médias)': kfold[(10, n)]['means'].f1_macro,
                         'Bootstrap OOB': boot[n].f1_macro})
    sns.boxplot(data=dist, ax=ax); ax.axhline(loo_table.loc[n, 'f1_macro'], color='red', ls='--', label='LOOCV')
    ax.set_title(n); ax.tick_params(axis='x', rotation=25); ax.legend()
axes[0].set_ylabel('F1 macro')
plt.tight_layout(); plt.show()
variance[variance.modelo.isin(['Logística (pesos)', 'LDA (priors iguais)'])].set_index(['modelo', 'método'])
''', 'variance_table')

step('''
### 4.2 O efeito de *k* na validação cruzada

Os slides (T4, 40) resumem o compromisso na escolha de *k*:
- ***k* pequeno** → cada modelo treina com menos dados → estimativa **pessimista** (mais enviesada), mas mais
  estável e mais barata;
- ***k* grande** (no limite, LOOCV) → treino quase com todos os dados → estimativa **pouco enviesada**, mas os
  modelos são muito parecidos entre si e o custo cresce.

Medimos, para *k* ∈ {2, 3, 5, 10, 20}, a estimativa média, a variabilidade **entre folds** e a variabilidade **da
estimativa** (10 repetições com sementes diferentes), para a versão ponderada de cada modelo.
''', '''
k_rows = []
for n in ['Logística (pesos)', 'LDA (priors iguais)']:
    for k in (2, 3, 5, 10, 20):
        folds, means = repeated_kfold(MODELS[n], k, n_rep=10)
        k_rows.append({'modelo': n, 'k': k, 'tamanho do treino': int(len(y) * (k - 1) / k),
                       'DI_HT por fold de teste': round(85 / k, 1), 'F1 macro': means.f1_macro.mean(),
                       'dp entre folds': folds.std(), 'dp da estimativa': means.f1_macro.std()})
k_effect = pd.DataFrame(k_rows)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for n, g in k_effect.groupby('modelo'):
    axes[0].plot(g.k, g['F1 macro'], marker='o', label=n)
    axes[1].plot(g.k, g['dp entre folds'], marker='o', label=n)
    axes[2].plot(g.k, g['dp da estimativa'], marker='o', label=n)
for ax, t in zip(axes, ['Estimativa do F1 macro', 'dp entre folds', 'dp da estimativa (10 repetições)']):
    ax.set_xlabel('k'); ax.set_title(t); ax.set_xscale('log'); ax.set_xticks([2, 3, 5, 10, 20]); ax.set_xticklabels([2, 3, 5, 10, 20]); ax.legend()
for n in ['Logística (pesos)', 'LDA (priors iguais)']:
    axes[0].axhline(loo_table.loc[n, 'f1_macro'], ls=':', color='grey')
plt.tight_layout(); plt.show()
k_effect.set_index(['modelo', 'k'])
''', 'k_effect')

# ================================================================ 5. decisão
step('''
## 5. Qual é o método mais adequado?

Juntamos as evidências de todos os métodos de reamostragem para as duas versões ponderadas (que, como se viu em 2.3,
são as que detetam as classes minoritárias). Se a ordem entre Regressão Logística e LDA for a mesma em todos os
métodos de reamostragem, a conclusão é robusta.

Mostramos também os **coeficientes** da Regressão Logística, para confirmar que o modelo usa as features que a EDA
identificou como informativas.
''', '''
decision = pd.DataFrame({
    'Holdout':           {n: holdout[n]['bal_acc'] for n in MODELS},
    'Holdout x100 (média)': {n: rep_holdout[n].bal_acc.mean() for n in MODELS},
    '5-fold':            {n: kfold[(5, n)]['means'].bal_acc.mean() for n in MODELS},
    '10-fold':           {n: kfold[(10, n)]['means'].bal_acc.mean() for n in MODELS},
    'LOOCV':             {n: loo_table.loc[n, 'bal_acc'] for n in MODELS},
    'Bootstrap OOB':     {n: boot[n].bal_acc.mean() for n in MODELS},
})
decision_f1 = pd.DataFrame({
    'Holdout':           {n: holdout[n]['f1_macro'] for n in MODELS},
    'Holdout x100 (média)': {n: rep_holdout[n].f1_macro.mean() for n in MODELS},
    '5-fold':            {n: kfold[(5, n)]['means'].f1_macro.mean() for n in MODELS},
    '10-fold':           {n: kfold[(10, n)]['means'].f1_macro.mean() for n in MODELS},
    'LOOCV':             {n: loo_table.loc[n, 'f1_macro'] for n in MODELS},
    'Bootstrap OOB':     {n: boot[n].f1_macro.mean() for n in MODELS},
})
print('=== Balanced accuracy por método de reamostragem ===');  display(decision)
print('=== F1 macro por método de reamostragem ===');           display(decision_f1)

lr = clone(MODELS['Logística (pesos)']).fit(X, y)
coefs = pd.DataFrame(lr[-1].coef_.T, index=FEATURES, columns=lr[-1].classes_)
coefs['|coef| máx'] = coefs.abs().max(axis=1)
coefs.sort_values('|coef| máx', ascending=False).head(10)
''', 'decision')

md(FINDINGS.get('discussion', 'FINDINGS::discussion'))

client = NotebookClient(nb, timeout=3600, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT / 'notebooks')}})
client.execute()
nbformat.write(nb, OUT_NB)
print('written', OUT_NB)
