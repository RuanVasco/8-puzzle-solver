# Previsão de Placar — Pipeline de Dados

Projeto de Machine Learning para **prever o placar de partidas de futebol** a partir de
informações conhecidas *antes* do jogo (força dos times, competição, contexto temporal).

## Fonte dos dados

O dataset (`matches.csv`) foi extraído do sistema do **[futmetricas.com.br](https://futmetricas.com.br)**.

## Estrutura do projeto

```
ml/
├── datasets/
│   ├── raw/                  # dados originais, não editar
│   │   └── matches.csv
│   └── processed/            # saída tratada (gerada pelo pipeline)
│       └── matches_clean.csv
├── src/
│   ├── __init__.py
│   ├── pipeline.py           # classe DataPipeline (lógica de tratamento)
│   └── main.py               # ponto de entrada (orquestra o pipeline)
├── requirements.txt
└── README.md
```

## Como rodar

A partir da raiz do projeto (`ml/`):

```bash
pip install -r requirements.txt
python -m src.main
```

O script lê `datasets/raw/matches.csv`, aplica o tratamento e grava o resultado em
`datasets/processed/matches_clean.csv`.

## O que o pipeline faz

As etapas em [`src/pipeline.py`](src/pipeline.py) (classe `DataPipeline`) são encadeadas nesta ordem:

| Etapa | Método | Descrição |
|---|---|---|
| 1 | `clean_targets()` | Remove partidas sem placar (sem alvo não há o que treinar). |
| 2 | `format_types()` | Converte `match_date` de texto para datetime. |
| 3 | `fill_missing()` | Preenche nulos em colunas não-alvo com valores neutros. |
| 4 | `select_features()` | Descarta colunas de vazamento e identificadores redundantes (ver abaixo). |
| 5 | `encode_categoricals()` | Aplica one-hot encoding nas colunas categóricas. |
| 6 | `save()` | Grava o resultado em `datasets/processed/`. |

## Decisões de modelagem

### Vazamento de dados (data leakage)

Estatísticas que só existem **depois** da partida (posse de bola, chutes a gol, cartões,
escanteios, faltas) são **removidas**. Usá-las para prever o placar seria trapaça: no
momento real da previsão, antes do jogo, esses valores não existem.

### Colunas redundantes / sem sinal

São descartados identificadores puros (`match_id`) e nomes redundantes com seus códigos
(`home_team_name`, `away_team_name`, `league_name`).

### Encoding de categóricas

`league_id`, `home_team_id`, `away_team_id` e `referee` são **rótulos**, não quantidades.
Recebem one-hot encoding para o modelo não interpretá-los como números ordenáveis.

> Observação: o one-hot gera muitas colunas (alta cardinalidade em times e juízes).
> É o ponto de partida didático; técnicas mais enxutas podem ser avaliadas depois.

### Pontos em aberto (TODO)

- Avaliar a remoção de `referee` (619 valores distintos — cardinalidade muito alta).
- O dataset `teams.csv` foi descartado por estar +92% vazio (sem valor para o treino).
