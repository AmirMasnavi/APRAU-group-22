"""Build and execute notebooks/02_regression.ipynb (Parte 2: Regressão Linear).

Estrutura igual à da EDA_version3: cada passo tem (1) uma explicação do que se vai fazer e porquê,
(2) o código e (3) as conclusões justificadas, escritas a partir dos outputs executados
(ver scripts/regression_findings.py).
"""
from pathlib import Path
import textwrap
import nbformat
from nbclient import NotebookClient
from regression_findings import FINDINGS

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / 'notebooks' / '02_regression.ipynb'

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
nb.metadata.language_info = dict(name='python')


def md(s):
    nb.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))


def step(explanation, src, key):
    """Explicação (o quê e porquê) -> código -> conclusões."""
    md(explanation)
    nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(src).strip()))
    md(FINDINGS.get(key, f'FINDINGS::{key}'))


# =============================================================== introdução
md(FINDINGS['intro'])

# =============================================================== 1. preparação
md('## 1. Preparação dos dados')

step('''
### 1.1 Carregamento dos dados e preparação do ambiente

Carregamos as bibliotecas e o dataset original. O ficheiro CSV **nunca é alterado**: todas as transformações
são feitas numa cópia em memória.

Fixamos a semente aleatória em **22** (regra do projeto, `docs/agent.md`). Assim, a divisão treino/teste e a
validação cruzada dão sempre os mesmos resultados, e qualquer membro do grupo pode reproduzir o notebook.

Usamos duas bibliotecas de regressão, como nas aulas:
- **scikit-learn** (`LinearRegression`), para treinar, prever e validar;
- **statsmodels** (`OLS`), para obter a tabela estatística (erros padrão, testes *t* e *F*, p-values), como
  nos slides 19 e 25.
''', '''
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SEED = 22
np.random.seed(SEED)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 40)
sns.set_theme(style='whitegrid')

dataset_name = 'dataset_grupo_22.csv'
candidates = [Path.cwd() / dataset_name, Path.cwd().parent / dataset_name]
dataset_path = next(p for p in candidates if p.is_file())
raw = pd.read_csv(dataset_path)
print('Dimensao do dataset:', raw.shape)
raw[['age', 'bmi', 'HbA1c_level', 'smoking', 'employment_status', 'target_regression']].head()
''', 'load')

step('''
### 1.2 Remoção de features e codificação das categóricas

**O que fazemos e porquê:**

1. **Removemos as 3 features que a EDA (secção 4.2) mostrou não terem informação nova:**
   - `calcium_baseline`: é constante (9.42 em todas as linhas). Uma variável que não varia não consegue explicar
     a variação do target. Além disso, numa regressão ficaria colinear com o intercepto β₀.
   - `systolic_index`: é uma cópia exata de `systolic_bp`. Com as duas no modelo, a matriz $X^TX$ fica singular e
     $\\hat\\beta = (X^TX)^{-1}X^Ty$ deixa de ter solução única (slide 21).
   - `age_normalized`: é uma reescala de `age`, com o mesmo problema.

2. **Convertemos as categóricas em números**, porque a regressão linear só trabalha com valores numéricos:
   - **ordinais** (têm uma ordem natural, ex.: Low < Medium < High) → códigos 0, 1, 2…, que preservam a ordem
     com uma única coluna;
   - **binárias** (Yes/No, Male/Female) → 0/1;
   - **nominais** sem ordem (`employment_status`) → *one-hot* com `drop_first=True`, tal como `pd.get_dummies` no
     exercício PL2 (Insurance). Se não retirássemos uma coluna, as *dummies* somariam sempre 1 e ficariam
     colineares com o intercepto ("armadilha das *dummies*").

3. **Retiramos os dois targets da matriz de preditores.** `target_class` é o outro resultado a prever no trabalho;
   usá-lo para prever `target_regression` seria usar um resultado para prever outro.

As features "pouco informativas" da EDA **não** são removidas aqui. A ideia desta parte é precisamente testá-las
e deixar que os resultados confirmem (ou não) o que a EDA sugeriu.
''', '''
TARGET = 'target_regression'
DROP_EDA = ['calcium_baseline', 'systolic_index', 'age_normalized']

ORDINAL = {
    'age_level':         ['Young', 'Middle', 'Mature', 'Senior'],
    'bmi_level':         ['Underweight', 'Normal', 'Overweight', 'Obese'],
    'smoking':           ['Never', 'Former', 'Current'],
    'physical_activity': ['Low', 'Moderate', 'High'],
    'stress_level':      ['Low', 'Medium', 'High'],
    'sugar_consumption': ['Low', 'Medium', 'High'],
    'education_level':   ['Primary', 'Secondary', 'Tertiary'],
}
BINARY = {'gender': 'Male', 'family_history': 'Yes', 'low_hdl_cholesterol': 'Yes', 'high_ldl_cholesterol': 'Yes'}

data = raw.drop(columns=DROP_EDA).copy()
for col, levels in ORDINAL.items():
    data[col] = data[col].map({lvl: i for i, lvl in enumerate(levels)})
for col, positive in BINARY.items():
    data[col] = (data[col] == positive).astype(int)
data = pd.get_dummies(data, columns=['employment_status'], drop_first=True, dtype=int)

X = data.drop(columns=[TARGET, 'target_class'])
y = data[TARGET]
assert X.isna().sum().sum() == 0, 'codificacao gerou valores em falta'
FEATURES = list(X.columns)
CONTINUOUS = [c for c in FEATURES if c in raw.columns and pd.api.types.is_numeric_dtype(raw[c])]
print(f'{len(FEATURES)} features candidatas ({len(CONTINUOUS)} numericas continuas):')
print(FEATURES)
''', 'prep')

step('''
### 1.3 Divisão em treino e teste (*holdout*)

O enunciado permite avaliar com o método *holdout*: separar os dados em treino e teste.
- O modelo **aprende** os coeficientes β apenas com o conjunto de **treino**.
- O conjunto de **teste** simula "pacientes novos" e mede a capacidade de **generalização**.

Usamos **70% / 30%**, a mesma proporção do exemplo Boston Housing das aulas. Ficam 1400 registos para treinar,
o que é suficiente para estimar poucos coeficientes, e 600 para testar, o que dá uma estimativa razoável do erro.

**Regra importante:** todas as escolhas (que feature, que combinação) são feitas **só com o treino**. O teste é
usado uma única vez, no fim (secção 4). Se escolhêssemos a combinação que dá menos erro no teste, o próprio teste
passaria a fazer parte da escolha, e o erro reportado seria otimista.

Verificamos também se o target tem distribuição semelhante nos dois conjuntos. Se não tivesse, o teste não seria
representativo.
''', '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED)
print('treino:', X_train.shape, '| teste:', X_test.shape)
print(pd.DataFrame({'treino': y_train.describe(), 'teste': y_test.describe()}).T)
print(f'\\nTeste de Kolmogorov-Smirnov (mesma distribuicao?): p = {stats.ks_2samp(y_train, y_test).pvalue:.3f}')

fig, ax = plt.subplots(figsize=(7, 3.5))
sns.kdeplot(y_train, ax=ax, label='treino', fill=True, alpha=.3)
sns.kdeplot(y_test, ax=ax, label='teste', fill=True, alpha=.3)
ax.set_title('target_regression em treino e teste'); ax.legend(); plt.show()
''', 'split')

step('''
### 1.4 Métricas de avaliação, validação cruzada e modelo de referência (*baseline*)

**Métricas pedidas no enunciado:**

| Métrica | Fórmula | O que diz |
|---|---|---|
| **R²** | $1 - RSS/TSS$ | Proporção da variância do target explicada pelo modelo (slide 11). 1 = perfeito; **0 = tão bom como prever sempre a média**; negativo = pior que a média. |
| **MAE** | $\\frac{1}{n}\\sum \\lvert y_i - \\hat y_i \\rvert$ | Erro médio absoluto, nas unidades do target. Fácil de interpretar e pouco sensível a erros grandes isolados. |
| **RMSE** | $\\sqrt{\\frac{1}{n}\\sum (y_i - \\hat y_i)^2}$ | Raiz do erro quadrático médio, também nas unidades do target. Penaliza mais os erros grandes, porque eleva-os ao quadrado. |

**Porquê validação cruzada (5-fold) dentro do treino?** Para escolher entre features/combinações precisamos de
medir o erro em dados que o modelo não viu, mas sem tocar no teste. A validação cruzada divide o treino em 5
partes: treina com 4, avalia na 5.ª, e repete 5 vezes. A média das 5 avaliações é mais estável do que uma única
divisão.

**Porquê uma *baseline*?** Um modelo sem nenhuma feature prevê sempre a média do treino. Um R² ou um RMSE só têm
significado comparados com isto: **se um modelo não erra menos que a média, as features não acrescentam
informação.** É o equivalente, na regressão, a um classificador que prevê sempre a classe maioritária.

A padronização (`StandardScaler`) não altera as previsões de uma regressão linear, apenas a escala dos
coeficientes. Incluímo-la para os coeficientes serem comparáveis entre si e para preparar a Parte 4 (Ridge/Lasso),
onde é obrigatória.
''', '''
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}

def make_model(degree=1):
    """Regressao linear (opcionalmente com termos polinomiais) com padronizacao."""
    steps = [PolynomialFeatures(degree, include_bias=False)] if degree > 1 else []
    return make_pipeline(*steps, StandardScaler(), LinearRegression())

def cv_scores(model, cols, Xd=None, yd=None):
    Xd = X_train if Xd is None else Xd
    yd = y_train if yd is None else yd
    r = cross_validate(model, Xd[cols], yd, cv=CV, scoring=SCORING, return_train_score=True)
    return {'train_r2': r['train_r2'].mean(), 'cv_r2': r['test_r2'].mean(), 'cv_r2_std': r['test_r2'].std(),
            'cv_mae': -r['test_mae'].mean(), 'cv_rmse': -r['test_rmse'].mean()}

def metrics(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred), 'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))}

baseline_cv = cv_scores(DummyRegressor(strategy='mean'), FEATURES)
print(f'Desvio padrao do target no treino: {y_train.std():.3f}')
pd.Series(baseline_cv, name='baseline (media)').to_frame().T
''', 'metrics')

# =============================================================== 2. simples
md('''
## 2. Regressão Linear Simples

$$\\hat y = \\hat\\beta_0 + \\hat\\beta_1 x$$

Uma só feature tenta explicar o target. Os coeficientes são escolhidos por **mínimos quadrados**, isto é, de
forma a minimizar $RSS = \\sum (y_i - \\hat y_i)^2$ (slides 8–9).

**Estratégia:**
1. Calcular, a partir da EDA, **qual é o melhor R² possível** (2.1).
2. Testar **todas** as features, uma a uma (2.2).
3. Comparar com uma **feature aleatória**, para saber se as melhores são diferentes de ruído (2.3).
4. Analisar em detalhe o modelo escolhido, com visualizações e resíduos (2.4–2.6).
5. Verificar se uma relação **não-linear** (polinómio) seria melhor (2.7).
''')

step('''
### 2.1 O que esperar? O limite que a EDA já nos dá

Na regressão linear simples há uma relação direta entre a correlação de Pearson e o R²:

$$R^2_{\\text{treino}} = r^2$$

Ou seja, **a correlação calculada na EDA diz-nos, antes de treinar qualquer modelo, qual é o R² máximo**
que cada feature pode atingir no treino. Na EDA, a correlação mais alta com o target foi a do `bmi` (r ≈ 0.07).
Calculamos r e r² para todas as features no conjunto de treino.
''', '''
r_train = X_train.corrwith(y_train)
expected = pd.DataFrame({'r (treino)': r_train, 'r2 = R2 maximo esperado': r_train ** 2,
                         '% variancia explicada': r_train ** 2 * 100})
expected.sort_values('r2 = R2 maximo esperado', ascending=False).head(10)
''', 'expected')

step('''
### 2.2 Testar todas as features, uma de cada vez

Para cada uma das 31 features ajustamos uma regressão simples e registamos:
- **R² de treino** (deve coincidir com r² da tabela anterior);
- **β₁ e o seu p-value** (teste *t* para $H_0: \\beta_1 = 0$, slide 18). Um p-value < 0.05 indica que a relação
  é estatisticamente significativa;
- **R², MAE e RMSE em validação cruzada**: este é o **critério de escolha**, porque mede o desempenho em dados que
  não foram usados para estimar os coeficientes.

A tabela fica ordenada pelo RMSE de CV (menor = melhor). Mostramos a *baseline* para comparação.
''', '''
rows = []
for f in FEATURES:
    ols = sm.OLS(y_train, sm.add_constant(X_train[[f]])).fit()
    cv = cv_scores(make_model(), [f])
    rows.append({'feature': f, 'R2_treino': ols.rsquared, 'beta1': ols.params[f], 'p_value': ols.pvalues[f],
                 'cv_r2': cv['cv_r2'], 'cv_mae': cv['cv_mae'], 'cv_rmse': cv['cv_rmse'],
                 'melhor que baseline?': cv['cv_rmse'] < baseline_cv['cv_rmse']})
simple = pd.DataFrame(rows).set_index('feature').sort_values('cv_rmse')
print('Baseline: CV R2 = {cv_r2:.4f} | CV MAE = {cv_mae:.3f} | CV RMSE = {cv_rmse:.3f}'.format(**baseline_cv))
print(f"Features com p < 0.05: {(simple['p_value'] < 0.05).sum()} de {len(simple)}")
print(f"Features com CV RMSE menor que a baseline: {simple['melhor que baseline?'].sum()} de {len(simple)}")
simple
''', 'simple_table')

step('''
### 2.3 As melhores features são diferentes de ruído?

No exercício TP2 (ponto 3) vimos que, quando a relação entre X e Y é **aleatória**, o R² fica perto de 0 e o
coeficiente não é significativo. Usamos essa ideia como **controlo**: criamos 500 features de **ruído puro**
(números aleatórios sem qualquer relação com o target), ajustamos uma regressão simples com cada uma e vemos que
R² de treino obtemos **só por acaso**.

Se o R² do `bmi` ficar claramente acima do que o ruído consegue, a relação é real (ainda que fraca). Se ficar no
meio da distribuição do ruído, o `bmi` não é melhor que um número aleatório.
''', '''
rng = np.random.default_rng(SEED)
noise_r2 = np.array([sm.OLS(y_train, sm.add_constant(rng.normal(size=len(y_train)))).fit().rsquared
                     for _ in range(500)])
p95 = np.percentile(noise_r2, 95)

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(noise_r2, bins=40, ax=ax, color='lightgrey', label='500 features de ruído')
ax.axvline(p95, color='k', ls='--', label=f'percentil 95 do ruído ({p95:.4f})')
for f, c in zip(simple.index[:4], ['red', 'orange', 'green', 'purple']):
    ax.axvline(simple.loc[f, 'R2_treino'], color=c, lw=2, label=f'{f} ({simple.loc[f, "R2_treino"]:.4f})')
ax.set_xlabel('R² de treino'); ax.set_title('R² das melhores features vs R² obtido por ruído aleatório'); ax.legend()
plt.show()

real = (simple['R2_treino'] > p95)
print(f'R2 medio do ruido: {noise_r2.mean():.4f} | percentil 95: {p95:.4f}')
print(f'Features acima do percentil 95 do ruido ({real.sum()}):', list(simple.index[real]))
''', 'noise')

step('''
### 2.4 Modelo simples escolhido: análise estatística (OLS)

Escolhemos a feature com **menor RMSE em validação cruzada** (tabela 2.2). Analisamo-la com o `statsmodels`,
como no slide 19, para obter:
- os coeficientes $\\hat\\beta_0$ e $\\hat\\beta_1$ e a respetiva **interpretação**;
- o **erro padrão** e o **intervalo de confiança a 95%** de $\\hat\\beta_1$;
- o **teste *t*** e o p-value.

À semelhança do exercício PL2, calculamos também previsões para pacientes concretos, para dar significado
prático ao coeficiente.
''', '''
BEST_SIMPLE = simple.index[0]
ols_simple = sm.OLS(y_train, sm.add_constant(X_train[[BEST_SIMPLE]])).fit()
print(f'Feature escolhida: {BEST_SIMPLE}\\n')
print(ols_simple.summary().tables[1])
b0, b1 = ols_simple.params
print(f'\\nEquacao: target_regression = {b0:.3f} + {b1:.3f} x {BEST_SIMPLE}')
print(f'R2 treino = {ols_simple.rsquared:.4f}\\n')
for v in (18.5, 25, 30, 40):
    print(f'  {BEST_SIMPLE} = {v:>4}  ->  previsao = {b0 + b1 * v:.1f}')
print(f'\\nAmplitude real do target: {y_train.min():.0f} a {y_train.max():.0f} (desvio padrao {y_train.std():.1f})')
''', 'simple_ols')

step('''
### 2.5 Visualização: valores reais vs previstos

Seguindo o exercício PL2 (*"Visualize Predictions Of Simple Linear Regression"*), usamos três gráficos no
conjunto de treino:
1. **Dispersão com a reta ajustada**: valores reais (pontos) e a reta $\\hat y = \\hat\\beta_0 + \\hat\\beta_1 x$;
2. **Distribuição dos valores reais vs previstos**: um bom modelo produz previsões com uma distribuição
   parecida com a dos valores reais;
3. **Erro de previsão** (resíduo) de cada registo.
''', '''
pred_simple_train = ols_simple.fittedvalues

def plot_real_vs_pred(x, y_real, y_pred, xlabel, title, line=True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    axes[0].scatter(x, y_real, s=8, alpha=.35, label='valores reais')
    if line:
        order = np.argsort(np.asarray(x))
        axes[0].plot(np.asarray(x)[order], np.asarray(y_pred)[order], color='red', lw=2, label='previsões')
    else:
        axes[0].scatter(x, y_pred, s=8, alpha=.5, color='red', label='previsões')
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel('target_regression'); axes[0].legend()
    axes[0].set_title('Reais vs previstos')
    sns.kdeplot(y_real, ax=axes[1], fill=True, alpha=.3, label='reais')
    sns.kdeplot(y_pred, ax=axes[1], fill=True, alpha=.3, color='red', label='previstos')
    axes[1].set_title('Distribuição: reais vs previstos'); axes[1].legend()
    axes[2].scatter(range(len(y_real)), np.asarray(y_real) - np.asarray(y_pred), s=6, alpha=.4, color='indianred')
    axes[2].axhline(0, color='k'); axes[2].set_title('Erro de previsão (real − previsto)')
    fig.suptitle(title); plt.tight_layout(); plt.show()

plot_real_vs_pred(X_train[BEST_SIMPLE], y_train, pred_simple_train, BEST_SIMPLE,
                  f'Regressão simples com {BEST_SIMPLE} (treino)')
print(f'Previsoes: min {pred_simple_train.min():.1f}, max {pred_simple_train.max():.1f}, '
      f'desvio padrao {pred_simple_train.std():.2f}  |  valores reais: desvio padrao {y_train.std():.2f}')
''', 'simple_viz')

step('''
### 2.6 Diagnóstico dos resíduos

Os slides (26) listam os problemas a verificar numa regressão linear. Com os resíduos $e_i = y_i - \\hat y_i$:
- **Resíduos vs valores previstos**: uma curva indicaria **não-linearidade**; um "funil" indicaria **variância não
  constante** dos erros;
- **Q-Q plot**: compara os resíduos com uma normal. A normalidade dos erros é um pressuposto dos testes *t* e *F*.
''', '''
def residual_plots(fitted, resid, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(fitted, resid, s=6, alpha=.4); axes[0].axhline(0, color='red')
    axes[0].set_xlabel('valor previsto'); axes[0].set_ylabel('resíduo'); axes[0].set_title('Resíduos vs previstos')
    stats.probplot(resid, plot=axes[1]); axes[1].set_title('Q-Q plot dos resíduos')
    fig.suptitle(title); plt.tight_layout(); plt.show()
    print(f'media dos residuos = {np.mean(resid):.3f} | skew = {stats.skew(resid):.2f}')

residual_plots(ols_simple.fittedvalues, ols_simple.resid, f'Resíduos: regressão simples ({BEST_SIMPLE})')
print(f'Registos com o target no limite inferior (70): {(y_train == 70).sum()} | no limite superior (300): {(y_train == 300).sum()}')
''', 'simple_resid')

step('''
### 2.7 Uma relação não-linear seria melhor?

Uma reta pode ser demasiado simples se a relação for curva. Os slides (31–33) e o exercício PL2
(*"Find Optimum Value of degree"*) mostram como acrescentar termos polinomiais:
$\\hat y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\dots$. Continua a ser regressão **linear** nos coeficientes.

Testamos os graus 1 a 5 com a feature escolhida. Um grau maior aumenta **sempre** o R² de treino, porque o modelo
fica mais flexível. Por isso a escolha tem de ser feita pela **validação cruzada**.
''', '''
poly = pd.DataFrame({d: cv_scores(make_model(d), [BEST_SIMPLE]) for d in range(1, 6)}).T
poly.index.name = 'grau'

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(poly.index, poly['train_r2'], marker='o', label='R² treino')
ax.plot(poly.index, poly['cv_r2'], marker='o', label='R² validação cruzada')
ax.fill_between(poly.index, poly['cv_r2'] - poly['cv_r2_std'], poly['cv_r2'] + poly['cv_r2_std'], alpha=.15)
ax.axhline(0, color='k', lw=.8); ax.set_xlabel('grau do polinómio'); ax.set_title(f'Polinómio em {BEST_SIMPLE}')
ax.legend(); plt.show()
poly
''', 'simple_poly')

md(FINDINGS['simple_summary'])

# =============================================================== 3. múltipla
md('''
## 3. Regressão Linear Múltipla

$$\\hat y = \\hat\\beta_0 + \\hat\\beta_1 x_1 + \\hat\\beta_2 x_2 + \\dots + \\hat\\beta_p x_p$$

Cada $\\beta_j$ mede o efeito de $x_j$ **mantendo as outras features fixas** (slide 22). A ideia é combinar
várias features fracas na esperança de que, juntas, expliquem mais do que qualquer uma sozinha.

**Estratégia (construção incremental):**
1. Verificar a **colinearidade**, para saber que features não devem entrar juntas (3.1).
2. Construir o modelo **passo a passo**: 1 feature → 2 → 3 → …, e explicar cada ganho ou perda (3.2).
3. Testar **todas as combinações** de 2 e 3 features, para confirmar que a ordem escolhida não deixou nada de
   fora (3.3).
4. Testar **grupos clínicos** de features (3.4).
5. Mostrar o que acontece quando se usam **todas** as features: *overfitting* (3.5).
6. Mostrar o efeito da **colinearidade** nos coeficientes (3.6).
7. Testar variantes (transformação do target, polinómio) e escolher o modelo final (3.7–3.8).
''')

step('''
### 3.1 Colinearidade entre features (VIF)

Os slides (26–27, 35) alertam: quando duas features estão muito correlacionadas, o modelo não consegue separar o
efeito de cada uma. Os coeficientes ficam instáveis e podem perder significância.

O **VIF** (*Variance Inflation Factor*) de uma feature é $1/(1-R_j^2)$, onde $R_j^2$ é o R² obtido ao prever essa
feature a partir de todas as outras. Valores de referência: VIF = 1 significa que a feature é independente das
restantes; VIF > 5–10 indica colinearidade problemática.
''', '''
Xc = sm.add_constant(X_train.astype(float))
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
                index=FEATURES, name='VIF').sort_values(ascending=False)
print(f'corr(age, age_level) = {X_train.age.corr(X_train.age_level):.3f} | '
      f'corr(bmi, bmi_level) = {X_train.bmi.corr(X_train.bmi_level):.3f}')
vif.to_frame().head(10)
''', 'vif')

step('''
### 3.2 Construção incremental: 1, 2, 3, … features

Construímos o modelo acrescentando **uma feature de cada vez**. A ordem segue a **força da relação com o target**
(|r| no treino, secção 2.1), mas usamos só uma versão de cada variável: a contínua (`age`, `bmi`), e não a
discretizada (`age_level`, `bmi_level`), por causa da colinearidade vista em 3.1.

Em cada passo registamos:
- **R² de treino**: sobe *sempre* que se acrescenta uma feature, mesmo que seja ruído, porque o modelo tem mais
  liberdade para se ajustar aos dados;
- **R² ajustado**: $1 - (1-R^2)\\frac{n-1}{n-p-1}$. Penaliza o número de features *p* e só sobe se a nova feature
  acrescentar mais do que o esperado por acaso;
- **R², MAE e RMSE em validação cruzada**: dizem se o modelo melhora em dados novos.

**O que procuramos:** o ponto a partir do qual acrescentar features deixa de reduzir o erro de validação.
''', '''
order = [f for f in r_train.abs().sort_values(ascending=False).index
         if f not in ('age_level', 'bmi_level', 'low_hdl_cholesterol', 'high_ldl_cholesterol')]
rows = []
for k in range(1, len(order) + 1):
    cols = order[:k]
    ols = sm.OLS(y_train, sm.add_constant(X_train[cols].astype(float))).fit()
    cv = cv_scores(make_model(), cols)
    rows.append({'n': k, 'feature adicionada': cols[-1], 'r da feature': r_train[cols[-1]],
                 'R2_treino': ols.rsquared, 'R2_ajustado': ols.rsquared_adj,
                 'cv_r2': cv['cv_r2'], 'cv_mae': cv['cv_mae'], 'cv_rmse': cv['cv_rmse']})
incremental = pd.DataFrame(rows).set_index('n')

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
axes[0].plot(incremental.index, incremental['R2_treino'], marker='o', label='R² treino')
axes[0].plot(incremental.index, incremental['R2_ajustado'], marker='o', label='R² ajustado')
axes[0].plot(incremental.index, incremental['cv_r2'], marker='o', label='R² validação cruzada')
axes[0].axhline(0, color='k', lw=.8); axes[0].set_xlabel('nº de features'); axes[0].legend()
axes[0].set_title('R² à medida que se acrescentam features')
axes[1].plot(incremental.index, incremental['cv_rmse'], marker='o', color='darkorange', label='modelo')
axes[1].axhline(baseline_cv['cv_rmse'], color='grey', ls='--', label='baseline (média)')
axes[1].set_xlabel('nº de features'); axes[1].set_ylabel('RMSE (CV)'); axes[1].legend()
axes[1].set_title('Erro de validação à medida que se acrescentam features')
plt.tight_layout(); plt.show()
incremental.head(12)
''', 'incremental')

step('''
### 3.3 Todas as combinações de 2 e de 3 features

A construção incremental seguiu uma ordem fixa (a do |r|), mas pode acontecer que duas features individualmente
fracas funcionem bem em conjunto. Para confirmar, testamos **todas** as combinações de 2 e de 3 features entre as
12 melhores features contínuas/ordinais (66 pares e 220 trios), avaliadas por validação cruzada.

Mostramos as 5 melhores e as 3 piores de cada tamanho, para perceber **o que distingue uma boa combinação de uma
má.**
''', '''
pool = order[:12]
combo_rows = []
for size in (2, 3):
    for cols in combinations(pool, size):
        cv = cv_scores(make_model(), list(cols))
        combo_rows.append({'n': size, 'features': ' + '.join(cols), 'cv_r2': cv['cv_r2'],
                           'cv_mae': cv['cv_mae'], 'cv_rmse': cv['cv_rmse']})
combos_df = pd.DataFrame(combo_rows)
for size in (2, 3):
    sub = combos_df[combos_df.n == size].sort_values('cv_rmse')
    print(f'\\n=== {size} features: {len(sub)} combinacoes | {(sub.cv_rmse < baseline_cv["cv_rmse"]).sum()} melhores que a baseline ===')
    display(pd.concat([sub.head(5), sub.tail(3)]).set_index('features'))
print(f'corr(bmi, age) no treino = {X_train.bmi.corr(X_train.age):.3f}')
''', 'combos')

step('''
### 3.4 Grupos clínicos de features

Outra forma de escolher combinações é usar **conhecimento do domínio**: agrupar as features pelo tipo de
informação clínica que representam e ver se algum grupo, no seu conjunto, explica o target. Se o target fosse, por
exemplo, um marcador lipídico, o grupo lipídico deveria destacar-se.
''', '''
GROUPS = {
    'Antropométrico (age, bmi)':              ['age', 'bmi'],
    'Antropométrico + glicémico':             ['age', 'bmi', 'HbA1c_level'],
    'Glicémico (HbA1c)':                      ['HbA1c_level'],
    'Lipídico':                               ['cholesterol', 'triglycerides', 'hdl', 'ldl', 'ldl_alternative'],
    'Pressão arterial / cardíaco':            ['systolic_bp', 'diastolic_bp', 'blood_pressure', 'heart_rate'],
    'Inflamatório':                           ['crp_level', 'homocysteine_level', 'plasma_value'],
    'Estilo de vida':                         ['smoking', 'physical_activity', 'stress_level', 'sugar_consumption',
                                               'sleep_hours', 'alcohol_intake', 'salt_intake'],
    'Sociodemográfico':                       ['gender', 'education_level', 'family_history',
                                               'employment_status_Retired', 'employment_status_Unemployed'],
    'Todas as numéricas contínuas':           CONTINUOUS,
}
groups = pd.DataFrame({g: {'n_features': len(c), **cv_scores(make_model(), c)} for g, c in GROUPS.items()}).T
groups['melhor que baseline?'] = groups['cv_rmse'] < baseline_cv['cv_rmse']
groups.sort_values('cv_rmse')
''', 'groups')

step('''
### 3.5 E se usarmos todas as features? *Overfitting*

A tentação natural é "pôr tudo no modelo". Ajustamos a regressão com as 31 features e comparamos o desempenho no
**treino** com o desempenho em **validação cruzada**. Uma diferença grande entre os dois é o sinal clássico de
***overfitting***: o modelo ajusta-se ao ruído específico dos dados de treino, que não se repete em dados novos.
''', '''
ols_all = sm.OLS(y_train, sm.add_constant(X_train.astype(float))).fit()
cv_all = cv_scores(make_model(), FEATURES)
print(f'R2 treino = {ols_all.rsquared:.4f} | R2 ajustado = {ols_all.rsquared_adj:.4f} | '
      f'teste F: F = {ols_all.fvalue:.2f}, p = {ols_all.f_pvalue:.4f}')
print(f'CV: R2 = {cv_all["cv_r2"]:.4f} | MAE = {cv_all["cv_mae"]:.3f} | RMSE = {cv_all["cv_rmse"]:.3f}')
print(f'Baseline CV RMSE = {baseline_cv["cv_rmse"]:.3f}')
sig = ols_all.pvalues.drop('const')
print(f'\\nCoeficientes com p < 0.05: {list(sig[sig < 0.05].index)}')
''', 'all_features')

step('''
### 3.6 O efeito da colinearidade nos coeficientes

Para mostrar na prática o problema identificado em 3.1, comparamos o coeficiente de `bmi` em dois modelos:
- só com `bmi`;
- com `bmi` **e** `bmi_level` (a sua versão discretizada, VIF ≈ 8).

Se a colinearidade for um problema, o erro padrão do coeficiente aumenta e o p-value piora, **sem que o modelo
preveja melhor**.
''', '''
m1 = sm.OLS(y_train, sm.add_constant(X_train[['bmi']])).fit()
m2 = sm.OLS(y_train, sm.add_constant(X_train[['bmi', 'bmi_level']])).fit()
colin = pd.DataFrame({
    'so bmi':            {'coef bmi': m1.params['bmi'], 'erro padrao bmi': m1.bse['bmi'], 'p-value bmi': m1.pvalues['bmi'],
                          'R2 treino': m1.rsquared, 'CV RMSE': cv_scores(make_model(), ['bmi'])['cv_rmse']},
    'bmi + bmi_level':   {'coef bmi': m2.params['bmi'], 'erro padrao bmi': m2.bse['bmi'], 'p-value bmi': m2.pvalues['bmi'],
                          'R2 treino': m2.rsquared, 'CV RMSE': cv_scores(make_model(), ['bmi', 'bmi_level'])['cv_rmse']},
}).T
colin
''', 'collinearity')

step('''
### 3.7 Variantes: transformação do target e termos polinomiais

Testamos duas ideias vistas nas aulas, aplicadas à melhor combinação encontrada até aqui:
1. **`log1p(target)`**, como no exemplo Boston Housing: reduz a assimetria do target. As métricas são calculadas
   de volta na escala original, para serem comparáveis;
2. **Polinómio de grau 2**, como no exercício PL2: acrescenta os quadrados e os produtos entre features
   (interações).
''', '''
best_combo_row = combos_df.sort_values('cv_rmse').iloc[0]
best_inc_n = incremental['cv_rmse'].idxmin()
candidates_multi = {
    f'Incremental (melhor n = {best_inc_n})': order[:best_inc_n],
    f'Melhor combinação ({best_combo_row.features})': best_combo_row.features.split(' + '),
}
BEST_MULTI_NAME = min(candidates_multi, key=lambda k: cv_scores(make_model(), candidates_multi[k])['cv_rmse'])
BEST_MULTI = candidates_multi[BEST_MULTI_NAME]
print('Candidatos:'); [print(f'  {k}: {v}') for k, v in candidates_multi.items()]
print(f'Base para as variantes: {BEST_MULTI_NAME}\\n')

log_model = TransformedTargetRegressor(regressor=make_model(), func=np.log1p, inverse_func=np.expm1)
variants = pd.DataFrame({
    'linear':                cv_scores(make_model(), BEST_MULTI),
    'log1p(target)':         cv_scores(log_model, BEST_MULTI),
    'polinómio grau 2':      cv_scores(make_model(2), BEST_MULTI),
}).T
print(f'skew do target: {y_train.skew():.2f} -> skew de log1p(target): {np.log1p(y_train).skew():.2f}')
variants
''', 'variants')

step('''
### 3.8 Modelo múltiplo escolhido: análise estatística e visualização

Analisamos o modelo escolhido com o `statsmodels` (slide 25):
- **teste *F*** do modelo global, $H_0: \\beta_1 = \\dots = \\beta_p = 0$, que responde à questão 1 do slide 30
  (alguma feature é útil?);
- **teste *t*** de cada coeficiente;
- os mesmos gráficos de valores reais vs previstos e de resíduos usados na regressão simples.
''', '''
ols_multi = sm.OLS(y_train, sm.add_constant(X_train[BEST_MULTI].astype(float))).fit()
print(f'Features: {BEST_MULTI}\\n')
print(ols_multi.summary().tables[1])
print(f'\\nR2 treino = {ols_multi.rsquared:.4f} | R2 ajustado = {ols_multi.rsquared_adj:.4f}')
print(f'Teste F global: F = {ols_multi.fvalue:.3f}, p = {ols_multi.f_pvalue:.2e}')
eq = ' '.join(f'{v:+.3f} x {k}' for k, v in ols_multi.params.drop('const').items())
print(f'\\nEquacao: target_regression = {ols_multi.params["const"]:.3f} {eq}')

plot_real_vs_pred(X_train['bmi'], y_train, ols_multi.fittedvalues, 'bmi (eixo x apenas para visualizar)',
                  'Regressão múltipla (treino)', line=False)
residual_plots(ols_multi.fittedvalues, ols_multi.resid, 'Resíduos: regressão múltipla')
print(f'Previsoes: min {ols_multi.fittedvalues.min():.1f}, max {ols_multi.fittedvalues.max():.1f}, '
      f'desvio padrao {ols_multi.fittedvalues.std():.2f}')
''', 'multi_ols')

md(FINDINGS['multi_summary'])

# =============================================================== 4. teste
md('''
## 4. Avaliação final no conjunto de teste

Com todas as escolhas fechadas **usando apenas o treino**, avaliamos agora os modelos no teste (600 registos
nunca usados). É esta avaliação que responde ao enunciado: *"Evaluate both models using appropriate metrics
(R², MAE, RMSE)"*.

Comparamos quatro modelos:
- **Baseline** (média do treino), a referência mínima;
- **Regressão simples** escolhida na secção 2;
- **Regressão múltipla** escolhida na secção 3;
- **Regressão com todas as features**, apenas para confirmar o *overfitting* (não é um modelo escolhido).
''')

step('''
### 4.1 Métricas no teste

Cada modelo é treinado em todo o conjunto de treino e avaliado no teste. Mostramos as métricas de treino ao lado
das de teste: a diferença entre as duas mostra quanto cada modelo perde ao passar para dados novos.
''', '''
final_models = {
    'Baseline (média)':                        (DummyRegressor(strategy='mean'), FEATURES),
    f'Simples ({BEST_SIMPLE})':                 (make_model(), [BEST_SIMPLE]),
    f'Múltipla ({" + ".join(BEST_MULTI)})':     (make_model(), BEST_MULTI),
    f'Todas as features ({len(FEATURES)})':     (make_model(), FEATURES),
}
preds, rows = {}, {}
for name, (model, cols) in final_models.items():
    model.fit(X_train[cols], y_train)
    preds[name] = model.predict(X_test[cols])
    rows[name] = {**{f'treino_{k}': v for k, v in metrics(y_train, model.predict(X_train[cols])).items()},
                  **{f'teste_{k}': v for k, v in metrics(y_test, preds[name]).items()}}
results = pd.DataFrame(rows).T
names = list(preds)
results
''', 'test_table')

step('''
### 4.2 As diferenças entre modelos são reais? (*bootstrap*)

As diferenças da tabela anterior podem ser pequenas. Para saber se são reais ou fruto do acaso (de quais pacientes
calharam no teste), usamos ***bootstrap***: reamostramos o conjunto de teste 1000 vezes, com reposição, e
recalculamos as métricas em cada reamostra. Obtemos assim um **intervalo de confiança a 95%** para cada métrica e
para a diferença entre modelos.

Se o intervalo da diferença entre um modelo e a baseline incluir o zero, **não podemos afirmar que o modelo é
melhor que prever a média**.
''', '''
rng = np.random.default_rng(SEED)
yt = y_test.to_numpy()
idx = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))

boot = {}
for name, p in preds.items():
    r2s = [r2_score(yt[i], p[i]) for i in idx]; rm = [rmse(yt[i], p[i]) for i in idx]
    boot[name] = {'R2 IC95%': f'[{np.percentile(r2s, 2.5):.3f}, {np.percentile(r2s, 97.5):.3f}]',
                  'RMSE IC95%': f'[{np.percentile(rm, 2.5):.2f}, {np.percentile(rm, 97.5):.2f}]'}
for name in names[1:]:
    d = [rmse(yt[i], preds[name][i]) - rmse(yt[i], preds[names[0]][i]) for i in idx]
    boot[name]['dif. RMSE vs baseline (IC95%)'] = f'{np.mean(d):+.3f}  [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]'
pd.DataFrame(boot).T.fillna('—')
''', 'bootstrap')

step('''
### 4.3 Visualização da comparação

Dois tipos de gráfico:
1. **Barras com R², MAE e RMSE no teste**, para comparar os modelos diretamente;
2. **Valores previstos vs reais no teste.** Num modelo perfeito, todos os pontos ficariam sobre a diagonal
   vermelha. Num modelo que não aprende nada, os pontos formam uma linha horizontal na média.
''', '''
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
short = ['Baseline', 'Simples', 'Múltipla', 'Todas']
for ax, m in zip(axes, ['R2', 'MAE', 'RMSE']):
    vals = results[f'teste_{m}'].to_numpy()
    ax.bar(short, vals, color=['grey', 'steelblue', 'seagreen', 'darkorange'])
    ax.set_title(f'{m} no teste'); ax.axhline(0, color='k', lw=.8)
    for i, v in enumerate(vals): ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    if m != 'R2': ax.set_ylim(vals.min() * 0.97, vals.max() * 1.01)
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, name, s in zip(axes, names[1:], short[1:]):
    ax.scatter(y_test, preds[name], s=8, alpha=.4)
    lims = [y_test.min(), y_test.max()]; ax.plot(lims, lims, 'r--', label='previsão perfeita')
    ax.axhline(y_train.mean(), color='grey', ls=':', label='média do treino')
    ax.set_xlabel('valor real'); ax.set_ylabel('valor previsto'); ax.set_title(s); ax.legend()
plt.tight_layout(); plt.show()
''', 'test_plots')

step('''
### 4.4 Porque é que nenhuma feature explica o target? Um olhar final sobre `target_regression`

Antes da discussão, voltamos ao próprio target para perceber **de onde vem** a sua variação. Os gráficos de
valores reais vs previstos (2.5) sugerem que quase todos os valores estão espalhados de forma regular entre 70 e
200, com poucos valores acima. Verificamos:
- se, entre 70 e 200, a distribuição é compatível com uma **uniforme** (teste de Kolmogorov-Smirnov);
- quantos valores ficam acima de 200 e em que classe diagnóstica estão (uso apenas descritivo de `target_class`,
  como na EDA; não entra em nenhum modelo).
''', '''
t = raw[TARGET]
inside = t[t <= 200]
ks_u = stats.kstest(inside, stats.uniform(loc=70, scale=130).cdf)
print(f'Valores entre 70 e 200: {len(inside)} ({len(inside)/len(t):.1%}) | KS vs uniforme(70, 200): '
      f'D = {ks_u.statistic:.3f}, p = {ks_u.pvalue:.3g}')
high = raw[t > 200]
print(f'Valores acima de 200: {len(high)} | por classe: {high.target_class.value_counts().to_dict()}')
print(f'Proporcao de cada classe com target > 200: '
      f'{(high.target_class.value_counts() / raw.target_class.value_counts()).fillna(0).round(3).to_dict()}')

fig, axes = plt.subplots(1, 2, figsize=(15, 4))
sns.histplot(t, bins=46, ax=axes[0]); axes[0].axvline(200, color='red', ls='--')
axes[0].set_title('target_regression (todos os registos)')
sns.boxplot(data=raw, x='target_class', y=TARGET, ax=axes[1]); axes[1].axhline(200, color='red', ls='--')
axes[1].set_title('target_regression por classe (descritivo)')
plt.tight_layout(); plt.show()
''', 'target_shape')

# =============================================================== 5. discussão
md(FINDINGS['discussion'])

client = NotebookClient(nb, timeout=1800, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT / 'notebooks')}})
client.execute()
nbformat.write(nb, OUT_NB)
print('written', OUT_NB)
