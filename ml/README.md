# Previsão de Placar — Futebol

Projeto de Machine Learning para **prever o placar de partidas de futebol** a partir de
informações conhecidas *antes* do jogo (força dos times via elo, competição, contexto
temporal). Inclui o pipeline de tratamento dos dados e uma escada de modelos baseline.

## Fonte dos dados

O dataset (`matches.csv`) foi extraído do sistema do **[futmetricas.com.br](https://futmetricas.com.br)**.

## Estrutura do projeto

```
ml/
├── datasets/
│   ├── raw/                  # dados originais, não editar
│   │   └── matches.csv
│   └── processed/            # saída tratada (gerada pelo pipeline, ignorada no git)
│       └── matches_clean.csv
├── src/
│   ├── __init__.py
│   ├── pipeline.py           # DataPipeline: limpeza, encoding, X/y
│   ├── split.py              # separação treino/teste por temporada
│   ├── baseline.py           # nível 0: DummyRegressor (sempre a média)
│   ├── poisson.py            # nível 1: regressão de Poisson + StandardScaler
│   ├── xgb_model.py          # nível 2: XGBoost
│   └── main.py               # ponto de entrada (orquestra tudo)
├── requirements.txt
└── README.md
```

## Como rodar

A partir da raiz do projeto (`ml/`):

```bash
pip install -r requirements.txt
python -m src.main
```

O script lê `datasets/raw/matches.csv`, trata os dados, grava a versão limpa em
`datasets/processed/` e treina/avalia os modelos, imprimindo o MAE de cada um.

## O que o pipeline faz

As etapas em [`src/pipeline.py`](src/pipeline.py) (classe `DataPipeline`) são encadeadas nesta ordem:

| Etapa | Método | Descrição |
|---|---|---|
| 1 | `clean_targets()` | Remove partidas sem placar (sem alvo não há o que treinar). |
| 2 | `format_types()` | Converte `match_date` de texto para datetime. |
| 3 | `fill_missing()` | Preenche nulos em colunas não-alvo com valores neutros. |
| 4 | `select_features()` | Descarta colunas de vazamento, redundantes e a identidade dos times. |
| 5 | `encode_frequencies()` | Frequency encoding nas categóricas de alta cardinalidade (`referee`). |
| 6 | `encode_categoricals()` | One-hot apenas nas de baixa cardinalidade (`league_id`). |
| 7 | `save()` | Grava o resultado em `datasets/processed/`. |
| — | `split_features_target()` | Separa X (features) de y (alvos `home_score`/`away_score`). |

## Decisões de modelagem

### Vazamento de dados (data leakage)

Estatísticas que só existem **depois** da partida (posse de bola, chutes a gol, cartões,
escanteios, faltas) são **removidas**. Usá-las para prever o placar seria trapaça: no
momento real da previsão, antes do jogo, esses valores não existem.

### Colunas redundantes / sem sinal

São descartados identificadores puros (`match_id`) e nomes redundantes com seus códigos
(`home_team_name`, `away_team_name`, `league_name`).

### Encoding de categóricas (por cardinalidade)

A estratégia depende de quantos valores distintos a coluna tem:

- **Identidade dos times** (`home_team_id`, `away_team_id`): **descartada**. O sinal que
  ela carrega (a força do time) já está nas colunas de **elo**, de forma contínua e numa
  única coluna. Fazer one-hot geraria ~920 colunas esparsas e redundantes.
- **Alta cardinalidade** (`referee`, ~619 valores): **frequency encoding** — cada valor
  vira o número de vezes que aparece, condensando tudo em uma coluna.
- **Baixa cardinalidade** (`league_id`, 7 ligas): **one-hot encoding**, barato neste caso.

> Resultado: o dataset sai de ~1550 colunas (com one-hot em tudo) para ~15, com
> praticamente a mesma performance — confirmando que o one-hot dos times era redundante.

### Separação treino/teste por tempo

Como os dados são temporais, o corte é **por temporada** (`src/split.py`), não aleatório:
treino com as temporadas mais antigas, teste com as mais recentes (`2025`, `2026`). Isso
simula o uso real (prever o futuro a partir do passado) e evita vazamento temporal.

### Modelos (escada de baselines)

| Nível | Modelo | Papel |
|---|---|---|
| 0 | `DummyRegressor` (média) | régua mínima — qualquer modelo precisa superá-la |
| 1 | `PoissonRegressor` + `StandardScaler` | baseline sério (gol é contagem → Poisson) |
| 2 | `XGBoost` (`count:poisson`) | modelo forte, candidato a campeão |

Métrica: **MAE** (erro médio em gols). Resultado atual no teste:

| Modelo | MAE geral |
|---|---|
| Nível 0 (média) | 0.862 |
| **Nível 1 (Poisson)** | **0.850** ✅ |
| Nível 2 (XGBoost) | 0.872 (overfitting) |

O Poisson é o melhor por ora. O XGBoost ficou pior até que o baseline — sinal de
sobreajuste com poucas features, esperado num alvo tão ruidoso quanto placar de futebol.

### Pontos em aberto (TODO)

- Avaliar a remoção de `referee` (sinal fraco) ou outras formas de encoding.
- Domar o XGBoost (regularização) e reavaliar quando houver mais features.
- O dataset `teams.csv` foi descartado por estar +92% vazio (sem valor para o treino).
