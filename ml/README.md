# Previsão do Resultado de Partidas de Futebol via Mineração de Dados

> **Rascunho do artigo** (Trabalho Final — Mineração de Dados Aplicada).
> Segue a estrutura sugerida no enunciado (template SBC). Os blocos marcados com
> `> TODO` precisam ser completados/escritos no artigo final.
> Processo conduzido segundo o **KDD** (Knowledge Discovery in Databases).

> TODO: autores (grupo de 2), instituição, e-mails (template SBC).

---

## 1. Introdução

**Contextualização.** Prever o resultado de partidas de futebol é um problema
clássico de análise esportiva, com aplicações em jornalismo, clubes e mercado de
apostas. O resultado, porém, tem forte componente de aleatoriedade, o que o torna
um bom estudo de caso para mineração de dados.

**Motivação.** Estimar, *antes* do jogo, o resultado mais provável a partir de
informações objetivas (força das equipes, mando de campo, competição).

**Objetivo de negócio.** Auxiliar analistas/apostadores a estimar o resultado
mais provável de uma partida antes de ela acontecer.

**Objetivo de mineração.** Treinar um modelo de **classificação** que, a partir de
atributos conhecidos antes do jogo, prevê a classe do resultado:
*vitória do mandante (0)*, *empate (1)* ou *vitória do visitante (2)*.

**Organização.** A Seção 2 traz o referencial teórico; a Seção 3 detalha a
metodologia (dados, preparação e modelagem); a Seção 4 apresenta resultados e
discussão; a Seção 5 conclui e aponta trabalhos futuros.

---

## 2. Referencial Teórico e Trabalhos Correlatos

- **KDD**: processo de descoberta de conhecimento em bases de dados (seleção,
  pré-processamento, transformação, mineração e interpretação).
- **Classificação**: tarefa de prever uma categoria. Algoritmos usados: Regressão
  Logística, Random Forest, Naïve Bayes e KNN.
- **Sistema de rating Elo**: medida contínua de força de uma equipe, usada aqui
  como principal atributo preditivo.

---

## 3. Metodologia

### 3.1 Compreensão dos Dados

- **Fonte.** Dados extraídos do sistema do **[futmetricas.com.br](https://futmetricas.com.br)**.
- **Volume.** 6.063 partidas e 27 atributos no arquivo bruto (`matches.csv`).
- **Tipos de variáveis.** Numéricas (placar, posse, chutes, elo...), categóricas
  (times, liga, árbitro) e temporais (data, temporada).
- **Concentração temporal.** A base é majoritariamente de 2022 a 2026 (2010 e
  2018 aparecem com 1–2 jogos, descartáveis como ruído).

**Análise exploratória — problemas identificados:**

| Achado | Decisão |
|---|---|
| Dataset auxiliar `teams.csv` com +92% de valores ausentes | Descartado |
| Atributos pós-jogo (posse, chutes, cartões, escanteios, faltas) | Removidos (vazamento) |
| Identificadores e nomes redundantes (`match_id`, `*_team_name`, `league_name`) | Removidos |
| Identidade dos times redundante com o elo | Substituída pelo elo |
| **Desbalanceamento de classes**: mandante vence ~46% das vezes | Considerado na avaliação |

### 3.2 Preparação dos Dados

1. **Limpeza** — remoção de partidas sem placar (alvo).
2. **Criação do alvo** — `result` (0/1/2) derivado da comparação dos placares; os
   placares são então descartados (seriam vazamento).
3. **Valores ausentes** — preenchimento neutro em colunas não-alvo
   (`referee` → "Unknown", `round_number` → 0).
4. **Codificação de categóricas** (por cardinalidade):
   - times: **descartados** (redundantes com o elo);
   - `referee` (alta cardinalidade, ~619 valores): **frequency encoding** (1 coluna);
   - `league_id` (7 ligas): **one-hot encoding**.
5. **Padronização** — `StandardScaler` (média 0, desvio 1) nos modelos sensíveis a
   escala, para o elo (~1500) não dominar as demais features.
6. **Divisão treino/teste** — **por temporada** (não aleatória): treino até 2024,
   teste em 2025–2026. Evita vazamento temporal (treinar no passado, testar no
   futuro). Resultado: 4.061 jogos de treino e 1.963 de teste, com 13 atributos.

> Resultado da preparação: de ~1.550 colunas (com one-hot em tudo) para **13**,
> sem perda relevante de performance — confirmando a redundância dos times.

### 3.3 Modelagem

- **Ferramentas/bibliotecas.** Python, pandas, scikit-learn, matplotlib.
- **Algoritmos** (classificação) e justificativa:
  - **Regressão Logística** — baseline linear; estima a probabilidade de cada classe.
  - **Random Forest** — comitê de árvores, robusto a ruído.
  - **Naïve Bayes** — probabilístico, rápido (assume independência das features).
  - **KNN** — classifica pelos jogos mais parecidos.
  - **DummyClassifier** (classe mais comum) — baseline ingênuo / régua mínima.
- **Estratégia de validação.** Avaliação em conjunto de teste temporal separado
  (out-of-time). Métrica principal: **acurácia**; análise complementar com
  **matriz de confusão** e **precision/recall/F1** por classe.
- **Ajuste de hiperparâmetros.** Cada modelo *com* hiperparâmetros (Regressão
  Logística, Random Forest, Naïve Bayes e KNN) é ajustado via **GridSearchCV** —
  busca em grade que testa todas as combinações de uma lista de valores. O
  baseline (DummyClassifier) fica de fora, por não ter hiperparâmetros: ele é a
  régua e não deve ser otimizado.
- **Validação cruzada temporal.** A busca **não** usa a validação cruzada aleatória
  padrão (que embaralharia os jogos e treinaria no futuro para prever o passado —
  vazamento). Em vez disso usa **`TimeSeriesSplit`**: cada dobra treina no passado
  e valida no futuro, na mesma filosofia da divisão treino/teste. Isso depende de
  os dados estarem ordenados por data, o que o CSV bruto já garante.
- **Toggle "com vs sem ajuste".** Um interruptor (`TUNE_HYPERPARAMS` em
  `src/main.py`, e o parâmetro `tune` de cada modelo) permite rodar todos os
  modelos com hiperparâmetros fixos (sem busca) para comparação direta — o
  experimento controlado reportado na Seção 4.

---

## 4. Resultados e Discussão

**Acurácia no conjunto de teste — com vs sem ajuste de hiperparâmetros:**

| Modelo | Sem ajuste | Com ajuste (GridSearchCV) | Melhor configuração encontrada |
|---|---|---|---|
| **Regressão Logística** | 0.483 | **0.484** | `C=0.01` (mais regularizada da grade) |
| Random Forest | 0.476 | 0.478 | `max_depth=5, min_samples_leaf=30` (árvore rasa) |
| Baseline (classe mais comum) | 0.463 | 0.463 | — (não ajustável) |
| KNN | 0.452 | **0.473** | `n_neighbors=101, weights=uniform` |
| Naïve Bayes | 0.418 | 0.418 | `var_smoothing=1e-3` (efeito nulo) |

**Efeito do ajuste.** Em 3 dos 4 modelos o GridSearchCV praticamente não move a
agulha (Logística +0.001, Random Forest +0.002, Naïve Bayes 0.000). A exceção é o
**KNN (+0.021)**: o K fixo inicial (15) estava mal escolhido, e a busca revelou que
este problema ruidoso pede uma vizinhança bem maior (K=101). Ou seja, o ajuste
serviu mais para **corrigir uma má escolha inicial** (KNN) do que para extrair
desempenho extra — o que, em si, é um achado.

**Um padrão revelador.** Todos os modelos com botão de complexidade escolheram a
configuração **mais simples/contida** da grade: a Logística pegou o `C` menor (mais
regularização), o Random Forest a árvore mais rasa e o KNN a maior vizinhança.
Isso é evidência numérica independente da tese central: o sinal nos dados é fraco e
essencialmente linear, e o **teto está nos atributos (o elo), não no algoritmo** —
dar mais flexibilidade ao modelo só o faz decorar ruído.

Mesmo após o ajuste, apenas **Regressão Logística**, **Random Forest** e **KNN**
superam o baseline ingênuo (0.463). O **Naïve Bayes** fica abaixo — coerente: é
penalizado por assumir independência entre features correlacionadas (elos de
mandante e visitante).

**Matriz de confusão (Regressão Logística):**

![Matriz de confusão da Regressão Logística](reports/confusion_logistic.png)

| Real ↓ / Previsto → | Mandante | Empate | Visitante |
|---|---|---|---|
| **Mandante** | 819 | 1 | 89 |
| **Empate** | 442 | 0 | 105 |
| **Visitante** | 376 | 0 | 131 |

**Análise crítica.** A acurácia de ~48% esconde o achado mais importante: o modelo
**praticamente nunca prevê empate** (recall do empate = 0%, 0 de 547 acertos). Ele
concentra os palpites em "mandante vence" (recall 90%). Esse comportamento se
**acentuou** após a regularização escolhida pela busca (`C=0.01`): ao se conter
mais, o modelo abandona de vez a classe mais difícil. Isso reflete o
**desbalanceamento** e a natureza notoriamente imprevisível do empate no futebol.
A acurácia, sozinha, mascararia esse comportamento — daí a importância da matriz de
confusão.

**Limitações.** Conjunto pequeno de atributos pré-jogo (essencialmente o elo);
ausência de variáveis como escalações, lesões, sequência de resultados e mando
real (viagem/altitude). Placar exato havia se mostrado ainda mais difícil, o que
motivou a reformulação do problema como classificação de resultado.

---

## 5. Conclusões e Trabalhos Futuros

**Conclusões.** Foi possível conduzir o processo completo de KDD e construir
modelos que superam (modestamente) o baseline. O sinal preditivo é dominado por um
componente linear simples (elo + mando de campo), e modelos mais complexos não
trouxeram ganho — indicando que o gargalo está nos **atributos**, não no algoritmo.
O ajuste de hiperparâmetros (GridSearchCV com validação cruzada temporal) reforça
essa conclusão: todos os modelos preferiram a configuração mais simples da grade, e
o ganho foi desprezível, exceto no KNN — onde a busca apenas corrigiu uma escolha
inicial ruim de `K`.

**Atendimento aos objetivos.** O objetivo de mineração (classificar o resultado)
foi atingido; o de negócio é parcialmente atendido, dado o teto de previsibilidade
do futebol.

**Lições aprendidas.** Importância de evitar vazamento de dados; valor dos
baselines como régua; e que acurácia isolada pode enganar em classes
desbalanceadas.

**Trabalhos futuros.** Engenharia de novos atributos (diferença de elo, forma
recente, distância de viagem); tratamento do desbalanceamento (reamostragem,
pesos por classe); ampliação da grade de busca (ex.: confirmar o melhor `K` do KNN,
que tocou o limite da grade atual); e uso de outras métricas de seleção no
GridSearch (ex.: F1 macro) para penalizar o abandono da classe "empate".

---

## Como rodar (reprodutibilidade)

A partir da raiz do projeto (`ml/`):

```bash
pip install -r requirements.txt
python -m src.main
```

O script trata os dados (`datasets/raw/matches.csv`), salva a versão processada em
`datasets/processed/`, treina/avalia os modelos e gera a matriz de confusão em
`reports/`.

Para comparar **com** e **sem** ajuste de hiperparâmetros, basta alternar a flag
`TUNE_HYPERPARAMS` no topo de `src/main.py` (`True` = GridSearchCV; `False` =
hiperparâmetros fixos) e rodar de novo.

### Estrutura do projeto

```
ml/
├── datasets/raw/            # dados originais
├── datasets/processed/      # saída tratada (ignorada no git)
├── reports/                 # gráficos (matriz de confusão)
├── src/
│   ├── data/                # preparação dos dados
│   │   ├── pipeline.py      # limpeza, criação do alvo, encoding, X/y
│   │   └── split.py         # divisão treino/teste por temporada
│   ├── models/              # algoritmos de classificação
│   │   ├── baseline.py      # DummyClassifier (régua)
│   │   ├── logistic.py      # Regressão Logística
│   │   ├── forest.py        # Random Forest
│   │   ├── naive_bayes.py   # Naïve Bayes
│   │   └── knn.py           # KNN
│   ├── evaluation.py        # matriz de confusão + relatório por classe
│   └── main.py              # orquestra todo o fluxo
└── requirements.txt
```

---

> **TODO (resolver depois do merge).** O merge trouxe duas abordagens de ajuste de
> hiperparâmetros convivendo no código: (1) o módulo central `src/models/tuning.py`
> (saída "DEFAULT vs. TUNED" + `reports/confusion_best_tuned.png`), vindo do remoto,
> e (2) o toggle `TUNE_HYPERPARAMS` em `main.py` + parâmetro `tune` em cada modelo
> (`logistic.py`, `forest.py`, `knn.py`, `naive_bayes.py`), feito localmente.
> Decidir qual manter, remover a duplicação e alinhar o `main.py` com a abordagem
> escolhida.
