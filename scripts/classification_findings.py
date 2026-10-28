"""Textos do notebook 03_classification (intro, conclusões e discussão).

Todos os números vêm dos outputs executados (seed 22, 30 features, sem `ldl_alternative`).
Se o notebook mudar, os textos têm de ser revistos.
"""

FINDINGS = {
# ------------------------------------------------------------------ introdução
'intro': '''
# APRAU 2026/27 — Trabalho 1 · Grupo 22
## Parte 3: Aplicação de métodos — Classificação

**Dataset:** `dataset_grupo_22.csv`, com 2000 perfis de pacientes. **Variável a prever:** `target_class`, a categoria
diagnóstica (`DI_HY`, `HT`, `DI_HT`). O significado clínico dos códigos não é dado no enunciado.

### Objetivo (enunciado, secção 4)
1. Aplicar **Regressão Logística** e **Análise Discriminante Linear (LDA)** e **decidir qual é o método mais
   adequado**, justificando.
2. Avaliar com os métodos de reamostragem: ***holdout***, **validação cruzada (k = 5 e k = 10)**, **LOOCV** e
   ***bootstrap***.
3. Escolher métricas adequadas e **explicar como a variância é afetada pelo método de reamostragem**.

### Ponto de partida: o que a EDA já nos disse
| Resultado da EDA | Consequência para a classificação |
|---|---|
| `target_class` é **muito desequilibrada**: 89.9% `DI_HY`, 5.85% `HT`, 4.25% `DI_HT` | Um modelo sem cuidados tende a prever quase sempre `DI_HY`. A *accuracy* engana; são precisas métricas por classe e **estratificação** na reamostragem. |
| `HbA1c_level` separa fortemente `HT` das outras classes (η² = 0.41); `triglycerides` separa `DI_HY` das minoritárias (η² = 0.105) | Esperamos que `HT` seja relativamente fácil de identificar e que estas sejam as features com mais peso. |
| `DI_HT` e `DI_HY` sobrepõem-se muito | `DI_HT` deve ser a classe mais difícil. |
| As distribuições são pouco "normais" (uniformes, assimétricas, caudas longas) | Os pressupostos da LDA podem não se verificar (verificamos em 2.1). |
| Artefacto de precisão decimal ligado à classe | Logística e LDA usam o valor numérico de forma linear e não exploram as casas decimais; não criamos features a partir delas. |
''',

# ------------------------------------------------------------------ 1. preparação
'load': '''
**Conclusões:** o dataset tem **2000 registos**: 1798 `DI_HY` (89.9%), 117 `HT` (5.85%) e **apenas 85 `DI_HT`
(4.25%)**. A classe maioritária é 21 vezes maior que a mais pequena. Este número vai condicionar tudo o que se segue:
a escolha das métricas, a necessidade de estratificar e o comportamento dos modelos.
''',

'prep': '''
**Conclusões:** ficam **30 features** para os 2000 registos, sem valores em falta. Com cerca de 67 registos por
feature, há dados suficientes para estimar uma Regressão Logística ou uma LDA com covariância comum.

Ainda assim, a classe `DI_HT` tem só 85 pacientes. Isto inviabiliza, por exemplo, uma **QDA**: teria de estimar uma
matriz de covariâncias 30×30 (465 parâmetros) só com esses 85 pacientes, e a estimativa ficaria muito instável. É
uma das razões para nos ficarmos pelos dois métodos pedidos.
''',

'metrics': '''
**Conclusões:** o classificador que responde sempre `DI_HY` tem **accuracy de 0.899**, o que parece um bom
resultado. Mas não identifica **nenhum** paciente `HT` nem `DI_HT` (recall = 0), e por isso tem:
- **F1 macro = 0.316**;
- **balanced accuracy = 0.333** (= 1/3).

**Justificação:** é a prova numérica de que a *accuracy* não serve como métrica principal neste problema. Daqui em
diante, a métrica principal é o **F1 macro**, complementado pela **balanced accuracy** e pelo **recall de cada
classe**. Qualquer modelo útil tem de ficar claramente acima de 0.316 e 0.333.
''',

# ------------------------------------------------------------------ 2. modelos
'models_intro': '''
## 2. Regressão Logística vs LDA

**Regressão Logística:** modela diretamente a probabilidade de cada classe,
$P(Y=k \\mid X) = \\frac{e^{\\beta_{k0} + \\beta_k^T x}}{\\sum_j e^{\\beta_{j0} + \\beta_j^T x}}$ (versão multinomial),
com os coeficientes estimados por **máxima verosimilhança**. **Não assume nada** sobre a distribuição das features.

**LDA:** modela a distribuição das features **dentro de cada classe** como normal multivariada, com uma **matriz de
covariâncias comum**, e usa o teorema de Bayes, $P(Y=k \\mid X) \\propto \\pi_k \\, f_k(x)$, onde $\\pi_k$ é a
probabilidade a priori da classe. As fronteiras de decisão são lineares.

**Estratégia:**
1. verificar se os pressupostos da LDA se verificam nos nossos dados (2.1);
2. ajustar os dois métodos sem ajustes e ver o efeito do desequilíbrio (2.2);
3. corrigir o desequilíbrio e comparar (2.3);
4. testar se as diferenças são estatisticamente significativas (2.4).
''',

'assumptions': '''
**Conclusões:** **os pressupostos da LDA não se verificam.**
- **Normalidade:** o teste de Shapiro-Wilk rejeita a normalidade (p < 0.001) em todas as features testadas. Os
  gráficos mostram distribuições claramente não normais, por exemplo `HbA1c_level` muito concentrada em `DI_HY`
  (assimetria 1.37) e `age` aproximadamente uniforme.
- **Covariâncias iguais:** o teste de Levene rejeita variâncias iguais entre classes em 5 das 6 features. Em
  `HbA1c_level`, o desvio padrão das classes minoritárias (≈ 0.87–0.91) é **2.5 vezes** maior que o de `DI_HY`
  (0.36); em `blood_pressure` e `crp_level` o rácio também passa de 2.6.

**O que isto implica:** à partida, estas violações favorecem a Regressão Logística, que não depende destes
pressupostos. No entanto, a LDA é conhecida por ser **robusta** a violações moderadas: as suas fronteiras de decisão
continuam a ser lineares e podem funcionar bem, mesmo que as probabilidades estimadas não sejam exatas. Por isso, a
decisão entre os dois métodos tem de ser feita pelos **resultados**, e não apenas pela teoria.

Os gráficos confirmam também a EDA: `HbA1c_level` separa bem `HT` (valores mais baixos), `triglycerides` desloca
`HT` e `DI_HT` para valores mais altos, e as curvas de `DI_HT` e `DI_HY` sobrepõem-se muito.
''',

'first_fit': '''
**Conclusões: a hipótese confirma-se.** Sem ajustes, os dois modelos preveem quase sempre `DI_HY`.

| | F1 macro | Bal. acc. | Accuracy | Recall `DI_HY` | Recall `HT` | Recall `DI_HT` |
|---|---|---|---|---|---|---|
| Baseline (sempre `DI_HY`) | 0.316 | 0.333 | 0.899 | 1.000 | 0.000 | 0.000 |
| Logística | 0.607 | 0.584 | 0.934 | 0.991 | 0.692 | **0.071** |
| LDA | **0.656** | **0.617** | 0.934 | 0.986 | 0.675 | **0.188** |

- **`DI_HY`** é reconhecida quase sempre (recall ≈ 0.99).
- **`HT`** é razoavelmente bem detetada (recall ≈ 0.68–0.69 nos dois modelos), graças a `HbA1c_level`, que separa
  bem esta classe.
- **`DI_HT`** quase nunca é detetada. A Logística acerta em apenas 6 dos 85 pacientes (7%); a LDA em 16 (19%). Na
  matriz de confusão, a maioria dos `DI_HT` aparece como `DI_HY`.
- A *accuracy* (0.93) parece excelente, mas está a medir sobretudo a classe maioritária.

**Justificação — porque é que isto acontece?** Os dois métodos procuram minimizar o erro **total**. Com 1798 `DI_HY`
contra 85 `DI_HT`, e com `DI_HT` muito sobreposta a `DI_HY` (EDA), a forma mais "barata" de errar pouco é classificar
os casos duvidosos como `DI_HY`. Na LDA isto é explícito: a probabilidade a priori de `DI_HY` é 0.90 contra 0.04 de
`DI_HT`, e a regra de Bayes favorece a classe mais frequente.

**Porque é que a LDA deteta mais `DI_HT` que a Logística?** A LDA usa a estrutura de covariâncias de todas as
classes ao mesmo tempo (as médias de cada classe e uma covariância comum). Quando uma classe tem poucos exemplos, isto
dá-lhe estimativas mais estáveis do que a Logística, que ajusta coeficientes separados para cada classe a partir dos
poucos casos disponíveis.
''',

'balanced': '''
**Conclusões:** as versões ajustadas detetam muito mais pacientes das classes minoritárias, mas pagam um preço.

| | F1 macro | Bal. acc. | Accuracy | Recall `DI_HY` | Recall `HT` | Recall `DI_HT` | Precision `DI_HT` | Nº previstos `DI_HT` |
|---|---|---|---|---|---|---|---|---|
| Logística | 0.607 | 0.584 | 0.934 | 0.991 | 0.692 | 0.071 | 0.429 | 14 |
| LDA | **0.656** | 0.617 | **0.934** | 0.986 | 0.675 | 0.188 | **0.552** | 29 |
| Logística (pesos) | 0.578 | **0.711** | 0.754 | 0.759 | **0.821** | **0.553** | 0.105 | 447 |
| LDA (priors iguais) | 0.590 | 0.689 | 0.786 | 0.801 | 0.761 | 0.506 | 0.113 | 382 |

- **Ganho:** o recall de `DI_HT` passa de 0.07 para **0.55** (Logística) e de 0.19 para **0.51** (LDA). O de `HT`
  também sobe. A *balanced accuracy* melhora muito (0.58 → 0.71 e 0.62 → 0.69).
- **Custo:** para encontrar mais `DI_HT`, os modelos passam a classificar como `DI_HT` muitos pacientes que são
  `DI_HY`. A Logística prevê 447 `DI_HT` quando só existem 85. A **precision de `DI_HT` cai para ≈ 0.11** (só 1 em
  cada 9 previsões `DI_HT` está certa) e a *accuracy* desce para 0.75–0.79.
- O **F1 macro desce ligeiramente**, porque combina recall e precision: o ganho no recall das minoritárias é
  anulado pela queda da precision de `DI_HT` e do recall de `DI_HY`.

**Justificação — porque é que não há "almoços grátis"?** As classes `DI_HT` e `DI_HY` sobrepõem-se muito (EDA).
Mudar os pesos ou as priors **não cria informação nova**: apenas desloca a fronteira de decisão. Na zona de
sobreposição, ou se perdem os `DI_HT` (versões sem ajuste) ou se apanham juntamente com muitos `DI_HY` (versões
ajustadas). É o mesmo compromisso que se obtém ao baixar o limiar de decisão de um classificador.

**Qual versão é melhor?** Depende do que custa mais errar:
- se o objetivo for **não deixar escapar** pacientes `DI_HT`/`HT` (rastreio), as versões ajustadas são preferíveis;
- se o objetivo for **não dar falsos diagnósticos**, as versões sem ajuste são preferíveis.

Como o desequilíbrio é a característica central deste problema, e o enunciado pede métricas adequadas, **comparamos
sempre os dois métodos nas duas versões** nas secções seguintes.
''',

'versions_test': '''
**Conclusões:**

| Comparação | F1 macro | Balanced accuracy |
|---|---|---|
| **LDA vs Logística** (sem ajuste) | LDA +0.047, melhor em 7/10 folds, **p = 0.054** | LDA +0.031, 7/10 folds, p = 0.119 |
| **LDA (priors iguais) vs Logística (pesos)** | LDA +0.013, 7/10 folds, p = 0.369 | Logística +0.023, 7/10 folds, p = 0.297 |
| Logística (pesos) vs Logística | −0.025, p = 0.183 | **+0.128, 10/10 folds, p < 0.001** |
| LDA (priors iguais) vs LDA | **−0.059, p = 0.043** | **+0.074, p = 0.017** |

- **Sem ajuste, a LDA é melhor que a Logística no F1 macro** em 7 dos 10 folds (+0.047), mas a diferença fica **no
  limite da significância** (p = 0.054, ligeiramente acima de 0.05).
- **Com ajuste, os dois métodos ficam estatisticamente empatados**: a LDA tem um F1 macro ligeiramente superior e a
  Logística uma *balanced accuracy* ligeiramente superior, mas nenhuma das diferenças é significativa (p > 0.29).
- **O ajuste ao desequilíbrio** melhora sempre de forma significativa a *balanced accuracy* (a Logística melhora nos 10
  folds), à custa de uma pequena descida do F1 macro (significativa na LDA).

**Justificação:** um teste emparelhado compara os dois modelos **nos mesmos folds**, eliminando a variabilidade que
vem de quais pacientes calharam em cada fold. Por isso consegue detetar diferenças que as médias sozinhas não
mostram. Os desvios padrão entre folds (0.037–0.062) são da mesma ordem das diferenças entre modelos, e é por isso que
a maioria das diferenças LDA vs Logística não é significativa. Como os 10 folds partilham dados de treino, não são
totalmente independentes, e os p-values devem ser lidos como indicativos.
''',

# ------------------------------------------------------------------ 3. reamostragem
'resampling_intro': '''
## 3. Métodos de reamostragem

O enunciado pede quatro formas de estimar o desempenho em pacientes novos. Todas respondem à mesma pergunta (*"que
desempenho terá o modelo em dados que nunca viu?"*), mas de maneiras diferentes, com **custo, enviesamento e
variância** diferentes.

| Método | Treino em cada repetição | Teste | Nº de modelos |
|---|---|---|---|
| *Holdout* | 70% (1400) | 30% (600), uma vez | 1 |
| *k-fold* (k = 5 / 10) | (k−1)/k dos dados (1600 / 1800) | cada fold uma vez | 5 / 10 |
| LOOCV | 1999 | 1 paciente, 2000 vezes | 2000 |
| *Bootstrap* OOB | 2000 sorteados com reposição (~63% distintos) | os ~37% que ficaram de fora | 200 |

**Cuidados comuns a todos os métodos:**
- a padronização (`StandardScaler`) está dentro de um *pipeline*, por isso é ajustada **apenas com os dados de treino
  de cada repetição**. Ajustá-la com todos os dados deixaria passar informação do teste para o treino;
- sempre que possível usamos **estratificação**, para que todas as classes estejam representadas no teste;
- avaliamos os **quatro modelos** da secção 2 com todos os métodos.
''',

'holdout': '''
**Conclusões:** o conjunto de teste tem 539 `DI_HY`, 35 `HT` e só **26 `DI_HT`**.

| | F1 macro | Bal. acc. | Recall `HT` | Recall `DI_HT` |
|---|---|---|---|---|
| Logística | 0.581 | 0.606 | 0.829 | **0.000** |
| LDA | 0.609 | 0.602 | 0.743 | 0.077 |
| Logística (pesos) | 0.538 | 0.629 | 0.857 | 0.231 |
| LDA (priors iguais) | **0.592** | **0.675** | 0.857 | **0.308** |

- Os relatórios mostram o mesmo padrão da secção 2: `DI_HY` e `HT` são bem identificadas (F1 de 0.88–0.91 e
  0.64–0.70), e **`DI_HT` é a classe problemática** (F1 0.10–0.17, precision 0.06–0.12).
- A Logística sem ajuste **não identifica nenhum dos 26 `DI_HT`** do teste.
- Neste *holdout* em particular, a LDA com priors iguais é a melhor nas duas métricas principais.

**Justificação — porquê desconfiar de um único *holdout*?** Os resultados diferem dos da validação cruzada (por
exemplo, o recall de `DI_HT` da Logística (pesos) é 0.23 aqui contra 0.55 em 10-fold). Com só 26 `DI_HT` no teste,
**cada paciente `DI_HT` acertado ou falhado muda o recall em ~4 pontos percentuais**. A estimativa depende muito de
quais pacientes calharam no teste, o que medimos a seguir.
''',

'rep_holdout': '''
**Conclusões:** repetir o *holdout* 100 vezes mostra que **a mesma metodologia dá resultados muito diferentes
consoante a divisão**:

| | F1 macro médio | Desvio padrão | Mínimo | Máximo | dp do recall `DI_HT` |
|---|---|---|---|---|---|
| Logística | 0.604 | 0.032 | 0.509 | 0.671 | 0.048 |
| LDA | 0.638 | 0.035 | 0.516 | 0.708 | 0.064 |
| Logística (pesos) | 0.567 | 0.024 | 0.503 | 0.617 | **0.104** |
| LDA (priors iguais) | 0.590 | **0.021** | 0.546 | 0.637 | **0.102** |

- O F1 macro da LDA varia entre **0.52 e 0.71** consoante a divisão: uma diferença de 0.19 só por acaso.
- O recall de `DI_HT` tem um desvio padrão de **0.10**: numa divisão pode ser 0.3 e noutra 0.6.
- O *holdout* único da secção 3.1 (0.538 para a Logística (pesos)) está **abaixo da média** das 100 repetições
  (0.567). Se só tivéssemos feito essa divisão, teríamos subestimado o modelo.

**Justificação — de onde vem esta variância?**
1. **Teste pequeno para as classes raras:** só ~26 `DI_HT` e ~35 `HT` em cada teste, por isso poucos pacientes mudam
   muito as métricas por classe.
2. **Treino reduzido:** o modelo aprende com 70% dos dados, ou seja, só ~60 `DI_HT`. Divisões diferentes produzem
   modelos diferentes.
3. **Uma única medição:** não há média de várias avaliações para "suavizar" o acaso.

Curiosamente, as versões ajustadas têm um F1 macro **mais estável** (dp 0.021–0.024) do que as não ajustadas
(0.032–0.035). Sem ajuste, o F1 de `DI_HT` depende de acertar em poucos casos, e oscila entre perto de 0 e valores
razoáveis de divisão para divisão.
''',

'kfold': '''
**Conclusões:**

| | Estimativa F1 macro | dp entre folds | dp da estimativa (20 rep.) | Bal. acc. |
|---|---|---|---|---|
| 5-fold · Logística (pesos) | 0.573 | 0.025 | 0.007 | 0.707 |
| 5-fold · LDA (priors iguais) | 0.586 | 0.027 | 0.006 | 0.682 |
| 10-fold · Logística (pesos) | 0.575 | 0.038 | **0.004** | 0.709 |
| 10-fold · LDA (priors iguais) | 0.586 | 0.040 | 0.006 | 0.683 |
| 10-fold · LDA (sem ajuste) | **0.647** | 0.068 | 0.009 | 0.613 |

1. **A estimativa da validação cruzada é muito mais estável que a do *holdout*.** Repetindo com outra partição, a
   média do 10-fold muda apenas 0.004–0.010 (desvio padrão), contra 0.021–0.035 no *holdout*: uma redução de
   **3 a 6 vezes**. Isto acontece porque a média de *k* avaliações cancela grande parte do acaso, e porque **todos
   os pacientes são usados como teste** uma vez.
2. **A variabilidade entre folds é maior com k = 10 do que com k = 5** (por exemplo 0.038 vs 0.025). Com k = 10, cada
   fold de teste tem só ~200 pacientes e ~8–9 `DI_HT`, por isso cada avaliação individual é mais "ruidosa".
3. **Ainda assim, a estimativa final é igual ou mais estável com k = 10**, porque é a média de mais avaliações.
4. **A estimativa sobe ligeiramente de k = 5 para k = 10** (0.573 → 0.575 na Logística (pesos)). Com k = 10 o modelo
   treina com 1800 pacientes em vez de 1600, e aprende um pouco melhor as classes raras: o k = 5 é ligeiramente mais
   **pessimista**.
5. **A ordem dos modelos mantém-se:** LDA > Logística no F1 macro, Logística (pesos) > LDA (priors iguais) na
   *balanced accuracy*.
''',

'loocv': '''
**Conclusões:**

| | F1 macro | Bal. acc. | Recall `DI_HT` | Tempo |
|---|---|---|---|---|
| Logística | 0.628 | 0.596 | 0.106 | ~5.5 s |
| LDA | **0.662** | 0.621 | 0.200 | ~2.5 s |
| Logística (pesos) | 0.575 | **0.706** | **0.565** | ~4.5 s |
| LDA (priors iguais) | 0.585 | 0.681 | 0.494 | ~2.5 s |

- Nas versões sem ajuste, o LOOCV dá as estimativas **mais altas** de todos os métodos (por exemplo LDA 0.662 vs
  0.647 em 10-fold e 0.638 no *holdout* ×100). Cada modelo treina com 1999 pacientes, quase o dataset completo, por
  isso é o método **menos pessimista (menos enviesado)**. Nas versões ajustadas, o LOOCV fica praticamente igual ao
  10-fold (0.575 e 0.585 contra 0.575 e 0.586).
- As conclusões são as mesmas: sem ajuste a LDA é melhor; com ajuste a LDA tem melhor F1 macro e a Logística melhor
  *balanced accuracy*.
- O custo foi baixo (poucos segundos, com processamento paralelo; os tempos variam ligeiramente entre execuções), porque estes modelos são rápidos de treinar. Com modelos
  mais pesados, 2000 ajustes por modelo seriam proibitivos.

**Variância do LOOCV:**
- **Repetir o LOOCV dá sempre o mesmo resultado** (variância de partição = 0): só há uma forma de deixar cada
  paciente de fora.
- No entanto, **isto não significa que a estimativa seja precisa**. Os 2000 modelos são quase idênticos (diferem em
  um paciente), por isso os seus erros estão muito correlacionados. Segundo a teoria (ISLP, 5.1.4), a estimativa do
  LOOCV tem **variância elevada em relação ao verdadeiro erro**: se tivéssemos recolhido outros 2000 pacientes, o
  resultado do LOOCV podia mudar bastante.
- Também **não é possível estratificar** e não há "métrica por fold" (cada fold tem um único paciente), por isso não
  há forma de medir a dispersão a partir dos próprios resultados.
''',

'bootstrap': '''
**Conclusões:**

| | F1 macro médio | dp | IC95% | Bal. acc. |
|---|---|---|---|---|
| Logística | 0.615 | 0.030 | [0.563, 0.689] | 0.592 |
| LDA | **0.640** | 0.029 | [0.580, 0.701] | 0.605 |
| Logística (pesos) | 0.561 | 0.022 | [0.521, 0.606] | **0.695** |
| LDA (priors iguais) | 0.592 | **0.021** | [0.546, 0.629] | 0.687 |

- A fração média de pacientes *out-of-bag* é **0.367**, exatamente o valor teórico $e^{-1} \\approx 0.368$. Cada
  conjunto OOB tem em média ~31 `DI_HT`.
- O *bootstrap* devolve uma **distribuição completa** e um **intervalo de confiança** para o desempenho. Os
  intervalos da LDA e da Logística sobrepõem-se bastante, o que confirma que as diferenças entre os dois métodos são
  modestas.
- Para a Logística (pesos), a estimativa é **mais baixa** do que a do 10-fold e do LOOCV: 0.561 contra 0.575.

**Justificação — porque é que o *bootstrap* é pessimista?** Cada amostra *bootstrap* tem 2000 pacientes, mas só ~63%
são **distintos** (os outros são repetições). O modelo aprende, na prática, com ~1264 pacientes diferentes, menos que
no 5-fold (1600). Isto afeta mais a Logística, que precisa de exemplos variados das classes raras para estimar os
seus coeficientes. A LDA, que só precisa de médias e de uma covariância comum, é menos afetada (0.592 vs 0.586 no
10-fold).
''',

# ------------------------------------------------------------------ 4. variância
'variance_table': '''
**Conclusões (versões ajustadas; as versões sem ajuste seguem o mesmo padrão):**

| Método | Estimativa F1 (Log. pesos / LDA priors) | dp das avaliações | **dp da estimativa** | Nº modelos |
|---|---|---|---|---|
| *Holdout* (1 divisão) | 0.538 / 0.592 | — | **0.024 / 0.021** | 1 |
| 5-fold estratificado | 0.573 / 0.586 | 0.025 / 0.027 | 0.007 / 0.006 | 5 |
| 10-fold estratificado | 0.575 / 0.586 | 0.038 / 0.040 | 0.004 / 0.006 | 10 |
| LOOCV | 0.575 / 0.585 | — | 0 (determinístico) | 2000 |
| *Bootstrap* OOB (B = 200) | 0.561 / 0.592 | 0.022 / 0.021 | 0.002 / 0.001 | 200 |

*O "dp da estimativa" do bootstrap é o erro padrão da média das 200 iterações ($dp/\\sqrt{200}$).*

**Como é que o método de reamostragem afeta a variância?**

1. ***Holdout*: a estimativa com mais variância.** É uma única medição num teste de 600 pacientes com ~26 `DI_HT`. A
   divisão sozinha pode mudar o F1 macro em ±0.04–0.05 (dois desvios padrão). Como vimos, o *holdout* único ficou
   abaixo da média das repetições.
2. ***k-fold*: variância muito menor**, porque cada paciente é testado uma vez e o resultado é a média de *k*
   avaliações. A estimativa é **3 a 6 vezes mais estável** do que a do *holdout*. Cada fold individual varia mais
   com *k* maior (folds de teste mais pequenos), mas a média fica mais estável.
3. **LOOCV: sem variância de partição** (é determinístico), mas os 2000 modelos treinados são quase iguais e os seus
   erros estão correlacionados. A estimativa tem pouco enviesamento, mas **variância elevada em relação ao verdadeiro
   erro**, e não pode ser medida a partir dos próprios resultados.
4. ***Bootstrap*: a variância de cada iteração é semelhante à de um *holdout*** (dp 0.021–0.030), mas ao fazer a média
   de 200 iterações a estimativa final fica muito estável. É o único método que dá diretamente um **intervalo de
   confiança**. O custo é ser **pessimista**, porque cada modelo treina com só ~63% de pacientes distintos.

**O gráfico** mostra isto visualmente: as caixas das avaliações individuais (*holdout* ×100, folds, *bootstrap*) são
largas, e as das **médias** do 5-fold e do 10-fold são muito estreitas.

**Enviesamento:** quanto menos dados o modelo usa para treinar, mais pessimista é a estimativa. A ordem das
estimativas da Logística (pesos) segue essa lógica: *bootstrap* (~63% de pacientes distintos) 0.561 < 5-fold (80%)
0.573 < 10-fold (90%) 0.575 ≈ LOOCV (99.95%) 0.575. O *holdout* (70%) ficou ainda mais abaixo (0.538) por azar da
divisão. A partir de ~90% dos dados de treino, a estimativa praticamente estabiliza.
''',

'k_effect': '''
**Conclusões:**

| k | Treino | `DI_HT` por fold | F1 Log. (pesos) | F1 LDA (priors) | dp entre folds | dp da estimativa |
|---|---|---|---|---|---|---|
| 2 | 1000 | 42.5 | 0.556 | 0.594 | 0.019 / 0.014 | **0.010 / 0.007** |
| 5 | 1600 | 17.0 | 0.574 | 0.586 | 0.025 / 0.027 | 0.009 / 0.006 |
| 10 | 1800 | 8.5 | 0.575 | 0.586 | 0.035 / 0.037 | 0.004 / 0.005 |
| 20 | 1900 | 4.2 | 0.578 | 0.586 | **0.058 / 0.062** | **0.003 / 0.004** |

1. **Enviesamento:** a estimativa da Logística (pesos) **sobe com *k*** (0.556 → 0.578): com k = 2 o modelo treina
   só com metade dos dados, e o desempenho é subestimado. A LDA é praticamente insensível a *k* (0.586–0.594): estima
   poucos parâmetros (médias e uma covariância comum), por isso aprende quase o mesmo com 1000 ou com 1900 pacientes.
2. **Variância entre folds:** **aumenta** com *k* (0.014–0.019 → 0.058–0.062). Com k = 20, cada fold de teste tem
   apenas ~4 `DI_HT`, e um único paciente acertado ou falhado muda muito o F1 desse fold.
3. **Variância da estimativa:** **tende a diminuir** com *k* (0.010 → 0.003 na Logística), com alguma oscilação
   entre valores intermédios. A média de muitos folds é mais estável do que a média de poucos, e os modelos treinados
   com mais dados são mais parecidos entre si.

**Justificação — porquê k = 5 ou k = 10?** *k* pequeno é pessimista; *k* = N (LOOCV) é pouco enviesado mas caro e com
variância elevada em relação ao verdadeiro erro. Neste dataset, **k = 10 é um bom equilíbrio**: a estimativa coincide
praticamente com o LOOCV, é estável entre repetições (dp ≈ 0.005) e custa 10 ajustes em vez de 2000. Com classes tão
pequenas, **não convém usar *k* muito maior que 10**, porque os folds de teste ficariam com pouquíssimos pacientes
das classes raras.
''',

# ------------------------------------------------------------------ 5. decisão
'decision': '''
**Conclusões:**

**F1 macro (métrica principal):**

| | Holdout | Holdout ×100 | 5-fold | 10-fold | LOOCV | Bootstrap |
|---|---|---|---|---|---|---|
| Logística | 0.581 | 0.604 | 0.608 | 0.612 | 0.628 | 0.615 |
| **LDA** | **0.609** | **0.638** | **0.642** | **0.647** | **0.662** | **0.640** |
| Logística (pesos) | 0.538 | 0.567 | 0.573 | 0.575 | 0.575 | 0.561 |
| LDA (priors iguais) | 0.592 | 0.590 | 0.586 | 0.586 | 0.585 | 0.592 |

**Balanced accuracy:**

| | Holdout | Holdout ×100 | 5-fold | 10-fold | LOOCV | Bootstrap |
|---|---|---|---|---|---|---|
| Logística | 0.606 | 0.586 | 0.586 | 0.591 | 0.596 | 0.592 |
| LDA | 0.602 | 0.603 | 0.608 | 0.613 | 0.621 | 0.605 |
| **Logística (pesos)** | 0.629 | **0.699** | **0.707** | **0.709** | **0.706** | **0.695** |
| LDA (priors iguais) | **0.675** | 0.688 | 0.682 | 0.683 | 0.681 | 0.687 |

- **Sem ajuste, a LDA ganha à Logística em todos os métodos de reamostragem** no F1 macro (6/6), e em 5/6 na
  *balanced accuracy* (a exceção é o *holdout* único, por 0.004).
- **Com ajuste**, a LDA tem melhor F1 macro em todos os métodos (6/6), e a Logística tem melhor *balanced accuracy*
  em todos exceto o *holdout* único (5/6). As diferenças são pequenas e não significativas (2.4).
- A **ordem dos modelos é a mesma em todos os métodos de reamostragem**: o que muda é o nível da estimativa (o
  enviesamento), não a conclusão. Isto torna a decisão robusta.

**Coeficientes da Logística:** as features com mais peso são **`HbA1c_level`** (coeficiente −1.61 para `HT`: quanto
mais baixa, mais provável `HT`) e **`triglycerides`** (+0.52 para `DI_HT`, −0.70 para `DI_HY`), seguidas de
`bmi_level` e `high_ldl_cholesterol`. São, sobretudo, as features que a EDA identificou como informativas. O modelo
está a aprender padrões com sentido clínico, e não ruído. Os coeficientes de features como `employment_status` e
`education_level` são mais pequenos e serão avaliados com Lasso na parte de seleção de features.
''',

# ------------------------------------------------------------------ 6. discussão
'discussion': '''
## 6. Conclusões e escolha do método

### 6.1 Qual é o método mais adequado? → **LDA**

| Critério | Logística | LDA | Vantagem |
|---|---|---|---|
| F1 macro sem ajuste (todos os métodos) | 0.58–0.63 | **0.61–0.66** | LDA, no limite da significância (p = 0.054) |
| F1 macro com ajuste (todos os métodos) | 0.54–0.58 | **0.59** | LDA, 6/6 métodos (não significativo) |
| Balanced accuracy com ajuste | **0.63–0.71** | 0.68–0.69 | Logística, 5/6 métodos (não significativo) |
| Deteção de `DI_HT` sem ajuste (recall) | 0.07–0.11 | **0.19–0.20** | LDA (cerca de 2× mais casos) |
| Falsos `DI_HT` com ajuste (10-fold) | 447 previstos | **382 previstos** | LDA (menos falsos alarmes) |
| Estabilidade (dp em 100 *holdouts*, com ajuste) | 0.024 | **0.021** | LDA |
| Sensibilidade a ter menos dados de treino (k = 2 → 20; *bootstrap*) | sensível (0.556 → 0.578) | **estável** (≈ 0.59) | LDA |
| Pressupostos | **nenhum** | normalidade e covariâncias iguais (violados) | Logística |
| Custo computacional (LOOCV) | ~4.5–5.5 s | **~2.5 s** | LDA |

**Justificação:**
1. **Melhor desempenho na métrica principal**, em todos os métodos de reamostragem e nas duas versões (com e sem
   ajuste). Na versão sem ajuste a diferença fica no limite da significância (p = 0.054).
2. **Melhor com classes pequenas.** Com só 85 `DI_HT`, a LDA beneficia de estimar parâmetros partilhados (uma
   covariância comum para todas as classes), o que lhe dá estimativas mais estáveis. A Logística tem de estimar 30
   coeficientes por classe a partir de poucos exemplos. A LDA é especialmente útil quando há poucos exemplos e as
   classes estão bem separadas, como `HT` está por `HbA1c_level`.
3. **Mais estável:** menor variância entre divisões e quase insensível à quantidade de dados de treino.
4. **Apesar dos pressupostos violados**, as fronteiras lineares da LDA funcionam bem, o que confirma a sua robustez na
   prática. A vantagem teórica da Logística (não ter pressupostos) não se traduziu num melhor desempenho.

**Ressalva honesta:** se o critério fosse **só** a *balanced accuracy* com ajuste ao desequilíbrio, a Logística
(pesos) seria ligeiramente melhor (0.709 vs 0.683 em 10-fold). Como esta diferença não é significativa (p = 0.297),
e a LDA ganha em quase todos os outros critérios, a recomendação mantém-se.

**Que versão da LDA?** A escolha depende do custo dos erros:
- **LDA com priors iguais**, se o objetivo for **detetar** os pacientes `HT`/`DI_HT` (rastreio). O recall de `DI_HT`
  sobe de ~0.19 para ~0.51, mas com muitos falsos alarmes (precision ≈ 0.11);
- **LDA sem ajuste**, se o objetivo for maximizar o F1 macro e **evitar falsos diagnósticos**, aceitando que a
  maioria dos `DI_HT` passa despercebida.

Num contexto clínico, deixar escapar um diagnóstico costuma ser mais grave do que um falso alarme que leva a mais
exames. Por isso, **recomendamos a LDA com priors iguais**.

### 6.2 Como a variância é afetada pelos métodos de reamostragem
| Método | Enviesamento | Variância da estimativa | Custo | Quando usar |
|---|---|---|---|---|
| *Holdout* | pessimista (treina com 70%) | **a mais alta** (dp 0.02–0.035; F1 entre 0.50 e 0.71) | 1 ajuste | conjunto de teste final, dados muito grandes |
| 5-fold | ligeiramente pessimista | baixa (dp ≈ 0.006–0.010) | 5 ajustes | bom compromisso, dados pequenos ou modelos caros |
| **10-fold** | pequeno | **baixa** (dp ≈ 0.004–0.010) | 10 ajustes | **recomendado neste dataset** |
| LOOCV | o menor | 0 entre repetições, mas alta em relação ao verdadeiro erro (modelos correlacionados) | 2000 ajustes | dados muito pequenos, modelos baratos |
| *Bootstrap* OOB | pessimista (~63% de pacientes distintos) | baixa na média; dá intervalos de confiança | 200 ajustes | quando se quer quantificar a incerteza |

**Ideias principais:**
- **Fazer a média de várias avaliações reduz a variância.** Por isso o *k-fold* e o *bootstrap* são muito mais
  estáveis do que um único *holdout*.
- **Avaliações individuais com poucos dados de teste são ruidosas**, sobretudo nas classes raras: com k = 20, cada
  fold tem ~4 `DI_HT`.
- **Treinar com menos dados torna a estimativa pessimista** (*bootstrap*, k pequeno, *holdout*). Treinar com quase
  todos (LOOCV) reduz o enviesamento, mas aumenta o custo e a correlação entre os modelos.
- **A estratificação é essencial** com este desequilíbrio: garante que cada teste tem `HT` e `DI_HT` suficientes.
- **A conclusão (LDA vs Logística) foi a mesma com todos os métodos.** O método de reamostragem muda a confiança na
  estimativa, mas não a ordem dos modelos.

### 6.3 Limitações e próximos passos
- **`DI_HT` continua difícil:** o F1 desta classe nunca passa de ~0.2. A sobreposição com `DI_HY` limita qualquer
  classificador linear, e não há informação nas features para separar melhor estas classes.
- **Probabilidades da LDA:** com os pressupostos violados, as probabilidades estimadas pela LDA podem não estar bem
  calibradas. A classificação continua válida, mas as probabilidades devem ser interpretadas com cautela.
- **O artefacto de precisão decimal** identificado na EDA não é explorado por estes modelos lineares, mas convém
  confirmar com o docente a origem dos dados.
- **Próximos passos (seleção de features):** aplicar regularização Ridge (L2) e Lasso (L1) à Regressão Logística,
  comparar todas as features com um conjunto reduzido, usar **10-fold estratificado** como método de avaliação
  (justificado na secção 4), e verificar se o Lasso mantém sobretudo `HbA1c_level`, `triglycerides`, `age` e `bmi`.

### Referências
- James, Witten, Hastie & Tibshirani — *An Introduction to Statistical Learning with Python* (ISLP), cap. 4
  (classificação) e cap. 5 (métodos de reamostragem).
- Documentação scikit-learn (`LogisticRegression`, `LinearDiscriminantAnalysis`, `StratifiedKFold`, `LeaveOneOut`).
''',
}
