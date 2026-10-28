"""Textos do notebook 02_regression (intro, conclusões e discussão).

Todos os números foram retirados dos outputs executados (seed 22). Se o notebook for alterado,
os textos têm de ser revistos para continuarem a corresponder aos resultados.
"""

FINDINGS = {
# ------------------------------------------------------------------ introdução
'intro': '''
# APRAU 2026/27 — Trabalho 1 · Grupo 22
## Parte 2: Aplicação de métodos — Regressão Linear

**Dataset:** `dataset_grupo_22.csv` — 2000 perfis de pacientes. **Variável a prever:** `target_regression`
(medição laboratorial contínua; nome e unidades não indicados no enunciado).

### Objetivo (enunciado, secção 3)
1. **Regressão Linear Simples:** ajustar um modelo com uma só feature, testar várias features e escolher a mais
   adequada.
2. **Regressão Linear Múltipla:** ajustar um modelo com várias features clínicas, experimentar combinações e
   determinar o melhor grupo.
3. **Comparação e discussão:** avaliar os dois modelos com R², MAE e RMSE e discutir os resultados.

### Ponto de partida: o que a EDA já nos disse
A análise exploratória (`01_EDA.ipynb`) deixou três indicações que orientam tudo o que se segue:

| Resultado da EDA | Consequência para a regressão |
|---|---|
| A correlação mais alta com o target é a do `bmi` (r ≈ 0.07). Todas as outras são ainda mais próximas de 0. | Na regressão simples, R² = r². Mesmo a melhor feature explicará **menos de 1%** da variância. **Não esperamos um bom modelo.** |
| As features são praticamente independentes entre si, exceto pares contínua/discretizada (`age`/`age_level`, `bmi`/`bmi_level`). | Juntar features pode acumular pequenos contributos, mas só faz sentido usar **uma versão** de cada variável. |
| `calcium_baseline` é constante; `systolic_index` e `age_normalized` são cópias de outras features; `ldl_alternative` tem os valores de `ldl` trocados entre pacientes (não pertence ao paciente da linha). | São removidas antes de modelar. |

O objetivo desta parte é, por isso, **testar de forma rigorosa** se alguma feature ou combinação consegue prever
o target melhor do que um modelo sem informação, e **explicar porquê** umas combinações funcionam melhor que
outras.

### Base teórica (slides T2 — *Linear Regression*, Elsa Ferreira Gomes; ISLP cap. 3)
- Regressão simples: $Y = \\beta_0 + \\beta_1 X + \\varepsilon$. Múltipla: $Y = \\beta_0 + \\beta_1 X_1 + \\dots + \\beta_p X_p + \\varepsilon$.
- Coeficientes estimados por **mínimos quadrados**, minimizando $RSS = \\sum (y_i - \\hat y_i)^2$.
- $R^2 = 1 - RSS/TSS$: proporção da variância explicada. Testes *t* (cada coeficiente) e *F* (modelo global)
  avaliam a significância estatística.
- Problemas a vigiar: não-linearidade, variância não constante, *outliers*, **colinearidade**.

### Decisões de método
| Decisão | Porquê |
|---|---|
| *Holdout* 70% treino / 30% teste, seed 22 | Permitido pelo enunciado; proporção do exemplo Boston Housing; seed definida nas regras do projeto. |
| Escolhas feitas **só no treino**, com validação cruzada 5-fold | Escolher a melhor combinação olhando para o teste tornaria o erro do teste otimista. |
| Teste usado **uma única vez** (secção 4) | Estimativa honesta do erro em pacientes novos. |
| Comparação sempre com uma ***baseline*** (prever a média) | Um R² ou um RMSE só têm significado comparados com "não usar nenhuma feature". |
| `target_class` **não** é preditor | É o outro target do trabalho (regra do projeto). |
''',

# ------------------------------------------------------------------ 1. preparação
'load': '''
**Conclusões:** o dataset tem **2000 linhas e 35 colunas**, como na EDA. Há features numéricas (`age`, `bmi`,
`HbA1c_level`), categóricas em texto (`smoking`, `employment_status`) e o target, que varia entre valores como 90
e 220 nas primeiras linhas.
''',

'prep': '''
**Conclusões:** ficam **30 features candidatas** (35 colunas − 4 removidas − 2 targets), das quais 17 são
numéricas contínuas e 13 vêm de categóricas codificadas. `employment_status` deu origem a 2 colunas
(`Retired` e `Unemployed`; `Employed` é a categoria de referência, representada por 0 nas duas). A codificação não
gerou valores em falta.
''',

'split': '''
**Conclusões:** o treino tem **1400** registos e o teste **600**. As distribuições do target são semelhantes nos
dois conjuntos:
- médias de 136.9 e 133.8, desvios padrão de 41.4 e 40.9, e o mesmo intervalo (70–300);
- o teste de Kolmogorov-Smirnov não encontra diferença (p = 0.288).

**Justificação:** a divisão aleatória produziu um teste representativo. As diferenças que encontrarmos entre
treino e teste serão do modelo, não de uma divisão desequilibrada.
''',

'metrics': '''
**Conclusões:** a *baseline* erra, em média, **34.0 unidades (MAE)**, com RMSE de **41.4**. Este RMSE é
praticamente igual ao desvio padrão do target (41.37), o que é esperado: prever sempre a média produz um erro
igual à dispersão natural dos dados.

**Este é o valor a bater.** Uma feature só é útil se fizer o RMSE descer claramente abaixo de 41.4.

**Nota sobre o R² da baseline:** em validação cruzada não é exatamente 0, mas −0.006. Em cada fold a média é
calculada com os dados de treino desse fold e avaliada noutros dados, onde a média real é ligeiramente diferente.
Por isso, ao longo do notebook, **comparamos os modelos com a baseline (−0.006) e não com o zero.**
''',

# ------------------------------------------------------------------ 2. simples
'expected': '''
**Conclusões:** antes de treinar qualquer modelo, já sabemos o **melhor R² de treino possível** para uma
regressão simples:
- **`bmi`: r = 0.074 → R² máximo = 0.0055**, ou seja, **0.55%** da variância do target;
- `age` vem a seguir com 0.49%, depois `age_level` (0.37%) e `smoking` (0.36%).

**Justificação:** com uma só feature, a reta de mínimos quadrados explica exatamente r² da variância. Como nenhuma
feature tem |r| acima de 0.075, **nenhuma regressão simples pode explicar mais de ~0.5% da variância**, nem no
treino. Este resultado confirma numericamente o que a EDA indicava: mesmo a melhor feature não dará um modelo
bom.
''',

'simple_table': '''
**Conclusões:**
- O R² de treino de cada feature **coincide com r²** da tabela anterior (`bmi`: 0.0055), o que confirma a relação
  R² = r².
- **`bmi` é a melhor feature** em todos os critérios: maior R² de treino, menor RMSE de CV (41.309) e o p-value
  mais baixo (0.0055). Seguem-se `age` (RMSE 41.323) e `age_level` (41.339).
- **6 das 30 features têm β₁ significativo** (p < 0.05): `bmi`, `age`, `age_level`, `smoking`, `salt_intake` e
  `triglycerides`.
- Apenas **11 das 30** têm RMSE de CV abaixo da baseline, e **nenhuma tem R² de CV positivo** (o melhor é −0.0024,
  do `bmi`).

**Justificação — porque é que estas features ficam à frente?** São as que têm maior |r| com o target (secção
2.1). `bmi` e `age` aparecem sempre no topo: na tabela de correlações da EDA, nos p-values e agora no erro de
validação.

**Porque é que o ganho é tão pequeno?** O `bmi` baixa o RMSE de 41.385 (baseline) para 41.309: **menos de 0.1
unidades**, num target com desvio padrão de 41. Isto acontece porque o `bmi` explica apenas 0.55% da variância:
os restantes 99.45% continuam por explicar.

**Significância ≠ capacidade de previsão.** O β₁ do `bmi` é significativo (p = 0.006) porque, com 1400 registos,
até efeitos minúsculos se distinguem de zero. O p-value diz que o efeito provavelmente **existe**; não diz que é
**grande** ou útil para prever.
''',

'noise': '''
**Conclusões:** as 500 features de ruído atingem, só por acaso, um R² médio de **0.0006**, e 5% delas passam os
**0.0023**. **8 features ficam acima deste limiar**: `bmi` (0.0055), `age` (0.0049), `age_level`, `bmi_level`,
`smoking`, `salt_intake`, `physical_activity` e `triglycerides`.

**Justificação:** o `bmi` explica cerca de **9 vezes** mais variância do que uma feature aleatória típica, e
mais do dobro do percentil 95 do ruído. **A relação é real**, e não um acaso da amostra. Ao mesmo tempo, o
gráfico mostra que todas as features vivem na mesma escala minúscula que o ruído (0 a 0.006). A relação existe,
mas é demasiado fraca para ter utilidade prática.

As restantes 22 features ficam dentro da distribuição do ruído: **não se distinguem de números aleatórios**, o
que coincide com as features "pouco informativas" identificadas na EDA.
''',

'simple_ols': '''
**Conclusões — modelo simples escolhido: `bmi`.**

$$\\widehat{target} = 125.06 + 0.425 \\times bmi$$

- **Interpretação de β₁:** por cada unidade a mais de BMI, o target aumenta em média **0.43 unidades**. O
  intervalo de confiança a 95% é [0.125, 0.726], que não inclui o 0, e o teste *t* dá p = 0.006.
- **Interpretação de β₀:** 125.06 seria o valor previsto para BMI = 0. Não tem significado clínico (ninguém tem
  BMI 0); serve apenas para posicionar a reta.
- **Exemplos de previsão:** um paciente magro (BMI 18.5) tem uma previsão de 132.9 e um paciente com obesidade
  severa (BMI 40) de 142.1. **Uma diferença de 21.5 unidades de BMI muda a previsão em apenas 9 unidades**, num
  target que varia entre 70 e 300 (desvio padrão 41.4).

**Justificação da escolha:** o `bmi` é o melhor em todos os critérios: R² de treino (0.0055), RMSE de CV
(41.309), p-value (0.006), e o único destes com relação claramente acima do ruído (2.3). É também uma medição
contínua, mais informativa do que a sua versão categórica `bmi_level`. A escolha é coerente com a EDA, onde o `bmi`
já era a feature mais correlacionada com o target.
''',

'simple_viz': '''
**Conclusões:**
1. **Reais vs previstos:** a reta é quase **horizontal**. Os pontos reais estão espalhados por toda a faixa 70–200,
   para qualquer valor de BMI. Conhecer o BMI de um paciente praticamente não ajuda a adivinhar o seu target.
2. **Distribuição:** o contraste é o resultado mais elucidativo. Os valores reais espalham-se de 70 a 300 (desvio
   padrão **41.4**); as previsões concentram-se num pico estreito entre 131 e 149 (desvio padrão **3.1**). **O
   modelo prevê praticamente o mesmo valor (~137) para todos os pacientes.**
3. **Erro de previsão:** os erros vão de cerca de −65 a +160 e não têm nenhum padrão ao longo dos registos.

**Justificação:** isto é exatamente o que um R² de 0.0055 significa na prática. Quando uma feature quase não
explica o target, a melhor estratégia de mínimos quadrados é prever um valor perto da média para todos. É o mesmo
fenómeno que na classificação: um modelo sem boas features tende a prever sempre a classe maioritária; aqui
tende a prever sempre a média.
''',

'simple_resid': '''
**Conclusões:**
- **Resíduos vs previstos:** a nuvem é uma faixa horizontal, sem curvatura nem forma de "funil". **Não há sinais
  de não-linearidade nem de variância não constante.**
- **Q-Q plot:** os resíduos afastam-se da normal nas caudas e têm **assimetria positiva** (skew = 0.50). Isto
  deve-se aos poucos valores muito altos do target (até 300), que geram resíduos positivos grandes. Com n = 1400,
  os testes *t* e *F* continuam aproximadamente válidos (teorema do limite central).
- Há 10 registos no valor mínimo (70) e 5 no máximo (300) do treino. A EDA tinha assinalado esta acumulação nos
  extremos como possível truncatura, mas afeta poucos registos.

**Justificação:** os resíduos não indicam que o modelo tenha a **forma errada**. Indicam que a feature **não tem
informação** suficiente. Mudar a forma do modelo (por exemplo, usar uma curva) não deve resolver o problema; é isso
que testamos a seguir.
''',

'simple_poly': '''
**Conclusões:**
- O **R² de treino sobe sempre** com o grau (0.0057 → 0.0234), como esperado: um polinómio mais flexível ajusta-se
  melhor aos dados com que foi treinado.
- O **R² de CV** também sobe ligeiramente (de −0.002 no grau 1 para ~0.012 nos graus 3–5), e o RMSE de CV desce de
  41.31 para 40.99.
- No entanto, o **desvio padrão entre folds (0.03)** é 2–3 vezes maior que o ganho. Em alguns folds o polinómio é
  melhor, noutros pior.

**Justificação — devemos usar o polinómio?** Não. Há três razões:
1. o ganho não é **robusto**: está dentro da variabilidade normal da validação cruzada;
2. o gráfico 2.5 mostra que quase todos os pacientes têm BMI entre 15 e 40 e só alguns têm valores até ~55. Um
   polinómio de grau 3 ou mais pode "dobrar-se" para acompanhar esses poucos pontos extremos, o que é
   *overfitting* e não uma relação clínica;
3. o enunciado pede uma regressão **linear simples**, e a reta é mais interpretável.

Mantemos o modelo de grau 1.
''',

'simple_summary': '''
### Resumo da Regressão Linear Simples

| Pergunta | Resposta | Evidência |
|---|---|---|
| Qual é a melhor feature? | **`bmi`** | Menor RMSE de CV (41.309), maior R² (0.0055), menor p-value (0.006) |
| A relação é real? | **Sim, mas fraquíssima** | 9× o R² médio do ruído; β₁ significativo |
| O modelo é útil para prever? | **Não** | Explica 0.55% da variância; o RMSE desce menos de 0.1 face à baseline; as previsões ficam todas entre 131 e 149 |
| Uma curva seria melhor? | **Não de forma robusta** | Ganho menor que a variabilidade entre folds |
| Faz sentido? | **Sim** | A EDA mostrou r ≈ 0.07 para o `bmi`, o máximo do dataset, e R² = r². O resultado era previsível. |
''',

# ------------------------------------------------------------------ 3. múltipla
'vif': '''
**Conclusões:** há **dois pares de features colineares**, os mesmos que a EDA identificou:
- `age` ↔ `age_level`: VIF ≈ 11.3, correlação 0.954;
- `bmi` ↔ `bmi_level`: VIF ≈ 8.4, correlação 0.936.

Há também dois pares fracos: `ldl`/`high_ldl_cholesterol` (VIF ≈ 2.9) e `hdl`/`low_hdl_cholesterol` (≈ 1.4).
**Todas as outras features têm VIF próximo de 1**, ou seja, são praticamente independentes entre si.

**Justificação e decisão:** `age_level` e `bmi_level` são discretizações das variáveis contínuas e contêm a mesma
informação, mas com menos detalhe. Nas combinações seguintes usamos **apenas a versão contínua** (`age`, `bmi`) e
as variáveis originais `hdl`/`ldl` em vez das *flags*. Na secção 3.6 mostramos o que acontece quando esta regra não
é seguida.
''',

'incremental': '''
**Conclusões passo a passo:**

| Passo | O que acontece | Porquê |
|---|---|---|
| **1 → 2** (`bmi` + `age`) | O R² de CV passa de −0.002 para **0.001** (primeiro valor positivo) e o RMSE desce de 41.31 para 41.25. **É o maior ganho de toda a construção.** | `age` é a 2.ª feature mais correlacionada com o target (r = 0.070) e é **independente** do `bmi` (correlação −0.007). Por isso traz informação nova: os contributos somam-se (R² de treino 0.0055 + 0.0049 ≈ 0.0104). |
| **2 → 4** (+ `smoking`, + `salt_intake`) | Pequenas melhorias: RMSE 41.23 → 41.22. | Features com r ≈ −0.06, também independentes das anteriores. Cada uma acrescenta ~0.3% de R² de treino. |
| **5** (+ `triglycerides`) | O RMSE de CV **sobe** (41.25). | A feature ajuda no treino, mas não generaliza: o contributo que dá no treino é, em parte, ruído. |
| **7** (+ `HbA1c_level`) | **Melhor RMSE de CV: 41.191.** | `HbA1c_level` tem r baixo no treino (0.044), mas combinado com as restantes melhora a validação. |
| **8 → 12** | O RMSE de CV oscila entre 41.20 e 41.23, sem melhorar. | As features novas têm r < 0.04: estão ao nível do ruído (secção 2.3). |

**O padrão geral (gráfico):**
- o **R² de treino sobe sempre** (0.0055 → 0.0298 com 12 features), porque cada feature dá ao modelo mais
  liberdade para se ajustar aos dados de treino, mesmo que seja ruído;
- o **R² ajustado** sobe cada vez mais devagar, porque penaliza o número de features;
- o **R² de CV estabiliza à volta de 0.002–0.003** a partir de 4 features e depois oscila. O RMSE de CV nunca desce
  mais do que ~0.2 abaixo da baseline.

**Justificação:** a diferença crescente entre o R² de treino e o R² de CV é a definição de *overfitting*. As
primeiras features (`bmi`, `age`) trazem informação real; as seguintes trazem sobretudo ruído, que melhora o treino
mas não os dados novos.
''',

'combos': '''
**Conclusões:**
- **Melhor par: `bmi` + `age`** (RMSE de CV 41.248). **Melhor trio: `bmi` + `age` + `HbA1c_level`** (41.208),
  seguido de `bmi` + `age` + `blood_pressure` (41.214).
- **Os 5 melhores pares e os 5 melhores trios contêm todos o `bmi`**, e todos os melhores trios contêm `bmi` e
  `age`.
- **As piores combinações** juntam `hdl`, `alcohol_intake`, `family_history` e `triglycerides` (RMSE ≈ 41.41–41.43,
  **pior que a baseline**).
- Ainda assim, 53 dos 66 pares e 188 dos 220 trios ficam (muito ligeiramente) abaixo da baseline.

**Justificação — porque é que umas combinações dão menos erro que outras?**
1. **Uma combinação é tão boa quanto as features que a compõem.** `bmi` e `age` são as duas features com maior |r|
   e as únicas claramente acima do ruído em 2.3. Qualquer combinação sem elas perde.
2. **As features devem trazer informação diferente.** `bmi` e `age` têm correlação −0.007 entre si: cada uma
   explica uma parte diferente da variância do target, e os efeitos somam-se. Juntar duas features correlacionadas
   (como `bmi` e `bmi_level`) não acrescenta nada, porque repetem a mesma informação.
3. **Juntar features sem relação com o target piora o modelo.** As piores combinações são formadas por features ao
   nível do ruído. O modelo estima coeficientes para elas a partir de padrões aleatórios do treino, que não se
   repetem nos dados de validação, e o erro aumenta.

A pesquisa exaustiva confirma a ordem da construção incremental: não há nenhum par ou trio "escondido" que
funcione melhor do que as combinações com `bmi` e `age`.
''',

'groups': '''
**Conclusões:**
- Só **4 grupos** ficam abaixo da baseline: **antropométrico + glicémico** (RMSE 41.208), antropométrico (41.248),
  glicémico (41.353) e, por muito pouco, todas as numéricas contínuas (41.373 contra 41.385).
- Os grupos **lipídico, pressão arterial, inflamatório, estilo de vida e sociodemográfico ficam todos acima da
  baseline**: prevêem pior do que a média.
- O grupo com **todas as 17 numéricas contínuas** tem o maior R² de treino (0.029), mas em validação cruzada fica
  praticamente igual à baseline (41.373 contra 41.385). O pouco sinal de `age` e `bmi` é quase todo anulado pelo
  ruído das outras 15 features.

**Justificação:** um grupo funciona se contiver features relacionadas com o target. Os grupos vencedores são os
que incluem `age` e `bmi`, as únicas com sinal real. Os outros grupos são formados por features com r ≈ 0 (EDA,
secção 3.4) e, sem nada para aprender, só acrescentam ruído.

O mau desempenho do grupo lipídico é informativo: **não há evidência de que o target seja um marcador lipídico**,
porque colesterol, triglicéridos, HDL e LDL não o explicam. O mesmo vale para a pressão arterial e a inflamação. O
nome e as unidades do target continuam desconhecidos.
''',

'all_features': '''
**Conclusões:** com as 30 features:
- o **R² de treino é o mais alto de todos (0.0370)** e o teste *F* é significativo (p = 0.007);
- mas o **R² de CV é −0.026** e o RMSE de CV é **41.79, claramente pior que a baseline (41.38)**;
- o R² ajustado (0.0159) é mais baixo que o de modelos com menos features (secção 3.2);
- 4 coeficientes são significativos: `bmi_level`, `bmi`, `triglycerides` e `salt_intake`. Dois deles são o par
  colinear `bmi`/`bmi_level`, cujos coeficientes ficam distorcidos (ver 3.6).

**Justificação:** isto é ***overfitting***. Com 30 coeficientes para estimar e quase nenhum sinal real, o modelo
usa a sua flexibilidade para se ajustar a padrões aleatórios do treino. Esses padrões não existem nos dados de
validação, por isso **o modelo com mais features é o que pior generaliza**.

É também um bom exemplo de que um teste *F* significativo não garante um modelo útil: o teste diz que pelo menos
um coeficiente é diferente de zero, não que o modelo prevê bem.
''',

'collinearity': '''
**Conclusões:**
- **Só `bmi`:** coeficiente 0.425, erro padrão 0.153.
- **`bmi` + `bmi_level`:** o coeficiente do `bmi` **quase triplica** (1.245) e o erro padrão também (0.434).
  O R² de treino sobe um pouco (0.0055 → 0.0084), mas o **RMSE de CV piora** (41.31 → 41.34).

**Justificação:** como `bmi` e `bmi_level` medem o mesmo (correlação 0.936), o modelo não consegue separar o efeito
de cada uma. O coeficiente do `bmi` deixa de significar "efeito de uma unidade de BMI", porque passa a ser
compensado pelo coeficiente de `bmi_level` em sentido contrário. Os coeficientes ficam **instáveis e
inflacionados**, sem nenhum ganho de previsão. É exatamente o problema dos slides 27 e 35, e confirma a decisão de
usar apenas as variáveis contínuas.
''',

'variants': '''
**Conclusões:** as duas candidatas a modelo múltiplo eram:
- **incremental com 7 features** (RMSE de CV 41.191);
- **o melhor trio** `bmi` + `age` + `HbA1c_level` (41.208).

A de 7 features tem o menor erro e foi usada como base para as variantes:
- **`log1p(target)`:** reduz a assimetria do target (0.53 → −0.16), mas **piora** o modelo (RMSE de CV 41.62, R² de
  CV −0.018);
- **polinómio de grau 2:** sobe o R² de CV para 0.012 e baixa o RMSE para 41.01, mas **piora o MAE** (34.09 vs
  33.93). Com 7 features, passa de 7 para 35 termos e o R² de treino triplica (0.077): sinal de *overfitting*.

**Justificação:**
- No exemplo Boston Housing, o `log1p` ajudava porque havia relações fortes, que ficavam mais lineares na escala
  logarítmica. Aqui o problema não é a forma da distribuição: é a falta de informação nas features. O log
  comprime os valores altos, e o modelo passa a subestimá-los ainda mais.
- O ganho do polinómio é menor que o desvio padrão entre folds (0.022) e não se confirma no MAE. Não é robusto.

**Decisão:** manter a regressão **linear**.
''',

'multi_ols': '''
**Conclusões — modelo múltiplo escolhido: construção incremental com 7 features.**

$$\\widehat{target} = 93.26 + 0.398\\,bmi + 0.139\\,age - 2.44\\,smoking - 0.672\\,salt + 0.034\\,trig + 2.11\\,phys.act + 3.71\\,HbA1c$$

- **Teste *F* global:** F = 4.74, p = 2.8×10⁻⁵. Rejeitamos $H_0$ (todos os β = 0): as features, em conjunto, têm
  relação estatística com o target.
- **Coeficientes significativos (p < 0.05):** `bmi` (+0.40 por unidade), `age` (+0.14 por ano), `salt_intake`
  (−0.67) e `triglycerides` (+0.034). `HbA1c_level` (p = 0.055) e `smoking` (p = 0.068) ficam na fronteira;
  `physical_activity` não é significativo (p = 0.12).
- Os coeficientes de `bmi` (0.398) e `age` (0.139) são praticamente iguais aos dos modelos simples (0.425 e 0.141).
  Isto confirma que as features são independentes (VIF ≈ 1): juntá-las não altera o efeito de cada uma.
- **R² de treino = 0.023 (ajustado 0.018):** o modelo explica cerca de **2%** da variância, 4 vezes mais que a
  regressão simples, mas continua muito baixo.

**Visualização:** as previsões espalham-se um pouco mais que as do modelo simples (114–158, desvio padrão 6.3, contra
3.1), mas continuam muito concentradas face aos valores reais (desvio padrão 41.4). Os resíduos mantêm o mesmo
comportamento: sem padrão, com assimetria positiva (skew = 0.43).

**Interpretação clínica: cautela.** Alguns sinais não são os esperados clinicamente (por exemplo, mais sal associado
a um target *menor*). Como os efeitos são minúsculos e o target é desconhecido, **não devemos tirar conclusões
clínicas** destes coeficientes.

**Justificação da escolha:** é a combinação com menor RMSE de CV (41.191). Em alternativa, o trio `bmi` + `age` +
`HbA1c_level` tem um erro quase igual (41.208), com menos de metade das features. A diferença entre os dois (0.017)
é muito menor que a variabilidade entre folds, por isso o trio seria uma escolha igualmente defensável pelo
princípio da parcimónia (preferir o modelo mais simples quando o desempenho é equivalente).
''',

'multi_summary': '''
### Resumo da Regressão Linear Múltipla

| Pergunta | Resposta | Evidência |
|---|---|---|
| Juntar features ajuda? | **Um pouco, até ~4–7 features** | RMSE de CV 41.31 (1 feature) → 41.19 (7) |
| Qual é o melhor grupo? | **`bmi`, `age`, `smoking`, `salt_intake`, `triglycerides`, `physical_activity`, `HbA1c_level`** | Menor RMSE de CV entre construção incremental, 286 combinações e 9 grupos clínicos |
| Porque é que `bmi` + `age` é o melhor par? | São as 2 features com mais sinal **e** são independentes entre si | r = 0.074 e 0.070; correlação entre ambas −0.007 |
| E com todas as features? | **Pior que a baseline** | *Overfitting*: R² de treino 0.0370, R² de CV −0.026 |
| Usar contínua + discretizada? | **Não** | Coeficientes instáveis e inflacionados, sem ganho de previsão |
| Faz sentido? | **Sim** | Combinar features fracas e independentes soma R² pequenos (0.0055 + 0.0049 + …), mas o total fica em ~2% |
''',

# ------------------------------------------------------------------ 4. teste
'test_table': '''
**Conclusões — resultados no teste (600 pacientes nunca vistos):**

| Modelo | R² | MAE | RMSE |
|---|---|---|---|
| Baseline (média) | −0.006 | 34.29 | 41.02 |
| Simples (`bmi`) | −0.002 | 34.18 | 40.95 |
| **Múltipla (7 features)** | **0.005** | **34.11** | **40.81** |
| Todas as features (30) | −0.018 | 34.31 | 41.26 |

- **A regressão múltipla é a melhor nas três métricas**, seguida da simples e da baseline. **A ordem é a mesma que
  em validação cruzada**: as escolhas feitas só com o treino generalizaram de forma coerente.
- O modelo com todas as features é **o pior**, abaixo da baseline. Tem o maior R² de treino (0.0370) e o menor de
  teste (−0.018), o que confirma o *overfitting* detetado em 3.5.
- As melhorias são **muito pequenas**: a múltipla erra em média **34.11** contra **34.29** da baseline, menos 0.18
  unidades (**0.5%**).
- Em todos os modelos o R² de teste é menor que o de treino, como esperado.
''',

'bootstrap': '''
**Conclusões:**
- Os intervalos de confiança do RMSE **sobrepõem-se quase totalmente**: todos os modelos ficam entre ~38.5 e ~43.5.
- **Diferença de RMSE face à baseline:**
  - simples: −0.065, IC95% **[−0.29, +0.15]**;
  - múltipla: −0.194, IC95% **[−0.74, +0.32]**;
  - todas as features: +0.256, IC95% [−0.37, +0.87].
- **Todos os intervalos incluem o zero.**

**Justificação:** como o zero está dentro de todos os intervalos, **não podemos afirmar, com 95% de confiança,
que algum modelo é melhor do que prever a média**. As pequenas vantagens da tabela 4.1 podem resultar de quais
pacientes calharam no conjunto de teste. Com outra divisão treino/teste, a ordem poderia mudar.
''',

'test_plots': '''
**Conclusões:**
- **Barras:** os quatro modelos têm métricas praticamente iguais (MAE entre 34.1 e 34.4; RMSE entre 40.8 e 41.3).
  As escalas dos gráficos de MAE e RMSE foram ampliadas para as diferenças serem visíveis; numa escala a partir de
  zero, as barras seriam indistinguíveis.
- **Previstos vs reais:** em vez de seguirem a diagonal vermelha, as previsões formam uma **faixa horizontal** à
  volta da média do treino (~137). Os pacientes com target 70 e os pacientes com target 300 recebem previsões
  semelhantes.

**Justificação:** é a imagem de um modelo que não encontrou nas features informação para distinguir pacientes. O
modelo múltiplo "abre" um pouco mais a faixa do que o simples, mas continua longe da diagonal.
''',

'target_shape': '''
**Conclusões:**
- **97.9%** dos valores do target estão entre 70 e 200, com uma distribuição **quase uniforme** (histograma plano).
  O teste KS rejeita a uniformidade perfeita (p = 0.007), mas a distância é pequena (D = 0.038). Com quase 2000
  valores, até desvios mínimos ficam "significativos".
- Apenas **43 valores (2.1%) passam os 200**, e concentram-se na classe `DI_HT`: **24.7% dos pacientes `DI_HT`**
  têm target > 200, contra 1.2% de `DI_HY` e 0% de `HT`.

**Justificação — o que isto explica:**
1. **Um target uniforme, sem relação com as features, é o pior cenário para uma regressão.** Num intervalo de 70 a
   200, todos os valores são igualmente prováveis para qualquer paciente, seja qual for o seu BMI, idade ou
   colesterol. Não há nada que o modelo possa aprender.
2. **A pouca estrutura que existe está ligada à classe diagnóstica** (valores altos em `DI_HT`), que não podemos
   usar como preditor. Isto explica também porque a EDA encontrou em `target_class` a variável mais associada ao
   target (η² ≈ 0.027).
3. Distribuições "planas" já tinham sido observadas em muitas features na EDA (secção 2.2). Reforça a hipótese de
   dados **sintéticos**, em que o target terá sido gerado de forma praticamente independente das features.
''',

# ------------------------------------------------------------------ 5. discussão
'discussion': '''
## 5. Comparação, discussão e conclusões

### 5.1 Comparação dos dois modelos

| | **Regressão Linear Simples** | **Regressão Linear Múltipla** |
|---|---|---|
| Features | `bmi` | `bmi`, `age`, `smoking`, `salt_intake`, `triglycerides`, `physical_activity`, `HbA1c_level` |
| Como foram escolhidas | Melhor de 30 features, testadas uma a uma | Construção incremental, confirmada por 286 combinações e 9 grupos clínicos |
| R² treino | 0.0055 | 0.0233 |
| R² teste | −0.002 | **0.005** |
| MAE teste | 34.18 | **34.11** |
| RMSE teste | 40.95 | **40.81** |
| Diferença de RMSE face à baseline (IC95%) | −0.07 [−0.29, +0.15] | −0.19 [−0.74, +0.32] |
| Significância | *t*: p = 0.006 | *F*: p = 2.8×10⁻⁵ |

**A regressão múltipla é ligeiramente melhor** nas três métricas, em validação cruzada e no teste. Ao juntar várias
features fracas e independentes, acumula pequenos contributos: o R² de treino é 4 vezes maior que o da simples.

**Mas nenhum dos dois modelos é útil na prática.** Ambos erram, em média, cerca de 34 unidades, praticamente o
mesmo que prever sempre a média (34.29). As diferenças não são estatisticamente robustas.

### 5.2 Discussão: o resultado faz sentido?

**Sim, e era previsível a partir da EDA.** A cadeia de raciocínio é a seguinte:

1. **A EDA mostrou que a melhor correlação com o target é r ≈ 0.07 (`bmi`).** Na regressão simples, R² = r², por isso
   o limite era 0.0055 (0.55%). O modelo atingiu exatamente esse valor (secções 2.1 e 2.2).
2. **As features são independentes entre si**, por isso, na regressão múltipla, os R² individuais somam-se
   aproximadamente. Somando os contributos das ~7 melhores features (todos abaixo de 0.6%) obtém-se ~2%, que foi o
   resultado obtido (3.8).
3. **As features que mais ajudam são as que têm maior |r| e estão acima do ruído** (`bmi`, `age`). As que pioram o
   modelo são as que, na EDA e no teste do ruído (2.3), não se distinguem de números aleatórios (`hdl`,
   `alcohol_intake`, `heart_rate`…).
4. **Mais features não é melhor.** Com 30 features o modelo memoriza ruído do treino (*overfitting*) e fica pior do
   que a média em dados novos (3.5 e 4.1).
5. **A forma do modelo não é o problema.** Os resíduos não mostram curvatura; polinómios e `log1p` não trazem
   ganhos robustos (2.7 e 3.7).
6. **O próprio target explica o resultado.** É quase uniforme entre 70 e 200 e só os valores raros acima de 200 têm
   alguma ligação, à classe `DI_HT` (4.4). Não há estrutura nas features para o modelo aprender.

**Paralelo com a classificação:** na EDA vimos que `target_class` é dominada por `DI_HY` (90%), e um classificador
sem boas features tenderá a prever quase sempre essa classe. Aqui acontece o equivalente na regressão: **sem
features informativas, o modelo de mínimos quadrados converge para a média** e prevê ~137 para quase todos os
pacientes (2.5, 4.3). Em ambos os casos, a comparação com a *baseline* (classe maioritária / média) é obrigatória
para não sobrestimar o modelo.

**Significância estatística ≠ capacidade de previsão.** Vários coeficientes têm p < 0.05 e o teste *F* do modelo
múltiplo é altamente significativo (p = 2.8×10⁻⁵). Com 1400 registos, até efeitos minúsculos se distinguem de
zero. O p-value diz que o efeito provavelmente **existe**; o R², o MAE e o RMSE dizem que é **demasiado pequeno**
para prever.

### 5.3 Conclusões
- **Regressão simples:** `target_regression = 125.06 + 0.425 × bmi`. O `bmi` é a melhor feature isolada em todos
  os critérios, mas explica apenas 0.55% da variância.
- **Regressão múltipla:** 7 features (`bmi`, `age`, `smoking`, `salt_intake`, `triglycerides`, `physical_activity`,
  `HbA1c_level`), que explicam cerca de 2% da variância. O trio `bmi` + `age` + `HbA1c_level` é uma alternativa
  quase equivalente e mais simples.
- **Desempenho no teste:** R² ≈ 0, MAE ≈ 34, RMSE ≈ 41. **Nenhum modelo prevê melhor do que a média** de forma
  estatisticamente robusta.
- **É um resultado negativo, mas válido:** foi obtido com uma metodologia correta (escolhas só no treino, validação
  cruzada, teste usado uma vez, *baseline*, controlo com ruído, intervalos de confiança) e é coerente com a EDA.

### 5.4 Limitações e próximos passos
- Um único *holdout* de 600 registos tem variância considerável (o IC95% do RMSE tem ~4.5 unidades de largura).
  A validação cruzada no treino reduziu este problema na escolha dos modelos.
- O nome, as unidades e a origem do target são desconhecidos, por isso os coeficientes não podem ser
  interpretados clinicamente.
- **Próximos passos:**
  1. perguntar ao docente se `target_class` pode ser usada como preditor da regressão;
  2. avançar para a classificação (Regressão Logística e LDA), onde a EDA encontrou features realmente
     informativas (`HbA1c_level`, `triglycerides`);
  3. na Parte 4 (Ridge/Lasso), verificar se o Lasso coloca a zero as mesmas features que aqui se mostraram ao nível
     do ruído.

### Referências
- Slides T2 — *Linear Regression: Simple & Multiple Linear Regression*, Elsa Ferreira Gomes, APRAU 2026/27.
- James, Witten, Hastie & Tibshirani — *An Introduction to Statistical Learning with Python* (ISLP), cap. 3.
- Exercícios das aulas: *Ex. Regression — Boston Housing* (*holdout* 70/30, `log1p`), *PL2 — Linear Regression
  Models (Insurance)* (visualização de reais vs previstos, *dummies*, grau do polinómio) e *TP2* (efeito do ruído
  e de features aleatórias no R² e nos p-values).
- Documentação scikit-learn (`LinearRegression`, `cross_validate`, `DummyRegressor`) e statsmodels (`OLS`,
  `variance_inflation_factor`).
''',
}
