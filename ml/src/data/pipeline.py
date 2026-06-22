import numpy as np
import pandas as pd
from pathlib import Path

# Estatísticas que só existem DEPOIS do jogo. Usá-las para prever o placar
# é vazamento de dados (data leakage): na hora real da previsão elas não existem.
LEAKAGE_COLS = [
    'home_ball_possession', 'away_ball_possession',
    'home_shots_on_target', 'away_shots_on_target',
    'home_yellow_cards', 'away_yellow_cards',
    'home_red_cards', 'away_red_cards',
    'home_corners', 'away_corners',
    'home_fouls', 'away_fouls',
]

# Identificadores sem sinal preditivo ou redundantes com as versões *_id.
REDUNDANT_COLS = [
    'match_id',
    'home_team_name', 'away_team_name',
    'league_name',
]

# Identidade dos times: descartada porque o sinal que ela carrega (a força do
# time) já está nas colunas de elo, de forma contínua e numa única coluna. Fazer
# one-hot disso geraria ~920 colunas esparsas e redundantes com o elo.
TEAM_ID_COLS = ['home_team_id', 'away_team_id']

# Categóricas de BAIXA cardinalidade: poucas categorias, então one-hot é barato.
ONEHOT_COLS = ['league_id']  # apenas 7 ligas

# Categóricas de ALTA cardinalidade: viram UMA coluna via frequency encoding
# (cada valor é trocado pelo nº de vezes que aparece). Evita a explosão do
# one-hot. 'referee' tem ~619 valores distintos.
FREQUENCY_COLS = ['referee']

# Placares brutos: usados para DERIVAR o alvo de resultado e depois descartados
# (mantê-los como feature seria vazamento total — eles são a própria resposta).
SCORE_COLS = ['home_score', 'away_score']

# Alvo: o RESULTADO da partida (classificação), codificado como:
#   0 = vitória do mandante | 1 = empate | 2 = vitória do visitante
TARGET_COL = 'result'


class DataPipeline:
    """Carrega e prepara os dados de partidas para o treinamento de previsão de resultado."""

    def __init__(self, matches_path: Path):
        """Lê o CSV de partidas para um DataFrame ao criar o objeto.

        Args:
            matches_path: caminho para o arquivo de partidas (matches.csv).
        """
        self.matches_df = pd.read_csv(matches_path)

    def verify_integrity(self):
        """Imprime um diagnóstico do DataFrame: tipos, contagens e nulos por coluna."""
        print(self.matches_df.info())
        print(self.matches_df.isnull().sum())

    def clean_targets(self):
        """Remove partidas sem placar, pois sem alvo não há o que treinar.

        Returns:
            self, para permitir encadeamento (method chaining).
        """
        self.matches_df = self.matches_df.dropna(subset=SCORE_COLS)
        return self

    def make_target(self):
        """Cria o alvo de RESULTADO (0/1/2) a partir do placar e descarta os gols.

        Comparando home_score x away_score, classificamos a partida em três
        categorias: vitória do mandante (0), empate (1) ou vitória do visitante (2).
        Os placares são removidos logo em seguida — usá-los como feature seria
        vazamento total, já que são a própria resposta.

        Returns:
            self, para permitir encadeamento.
        """
        home, away = self.matches_df['home_score'], self.matches_df['away_score']
        self.matches_df[TARGET_COL] = np.where(
            home > away, 0,            # mandante vence
            np.where(home == away, 1,  # empate
                     2),               # visitante vence
        )
        self.matches_df = self.matches_df.drop(columns=SCORE_COLS)
        return self

    def format_types(self):
        """Converte a data de texto para datetime, habilitando operações temporais.

        Returns:
            self, para permitir encadeamento.
        """
        self.matches_df['match_date'] = pd.to_datetime(self.matches_df['match_date'])
        return self

    def fill_missing(self):
        """Preenche nulos em colunas não-alvo com valores neutros.

        Evita que NaN quebre etapas seguintes do pipeline.

        Returns:
            self, para permitir encadeamento.
        """
        self.matches_df['referee'] = self.matches_df['referee'].fillna('Unknown')
        self.matches_df['round_number'] = self.matches_df['round_number'].fillna(0)
        return self

    def select_features(self):
        """Descarta colunas de vazamento (pós-jogo) e identificadores redundantes.

        Mantém apenas informação conhecida ANTES da partida (elos, contexto
        temporal/competição, juiz) junto com os alvos home_score/away_score.
        A identidade dos times (TEAM_ID_COLS) também sai: sua informação de força
        já está no elo. Usa errors='ignore' para não quebrar se a coluna não existir.

        Returns:
            self, para permitir encadeamento.
        """
        self.matches_df = self.matches_df.drop(
            columns=LEAKAGE_COLS + REDUNDANT_COLS + TEAM_ID_COLS,
            errors='ignore',
        )
        return self

    def encode_frequencies(self):
        """Codifica categóricas de alta cardinalidade pela sua frequência.

        Cada valor (ex.: um juiz) é trocado pelo número de vezes que ele aparece
        no dataset. Assim 619 juízes viram UMA coluna numérica, em vez de 619
        colunas one-hot. O sinal capturado é indireto (juízes mais "rodados"
        aparecem mais), mas é barato e não explode a dimensionalidade.

        Observação: usa a contagem sobre todos os dados. Como é só contagem (não
        envolve o alvo), o risco de vazamento é mínimo.

        Returns:
            self, para permitir encadeamento.
        """
        for col in FREQUENCY_COLS:
            counts = self.matches_df[col].value_counts()
            self.matches_df[col] = self.matches_df[col].map(counts)
        return self

    def encode_categoricals(self):
        """Aplica one-hot apenas nas categóricas de BAIXA cardinalidade (ONEHOT_COLS).

        Para cada valor distinto cria uma coluna 0/1 ("é esse valor? sim/não"), o
        que evita o modelo tratar códigos como números ordenáveis. Só usamos aqui
        em colunas com poucas categorias (ex.: as 7 ligas), onde o custo é baixo.

        Returns:
            self, para permitir encadeamento.
        """
        self.matches_df = pd.get_dummies(
            self.matches_df,
            columns=ONEHOT_COLS,
            dtype=int,  # gera 0/1 inteiros em vez de True/False
        )
        return self

    def get_clean_data(self):
        """Devolve o DataFrame já tratado para o consumidor (treino/análise)."""
        return self.matches_df

    def split_features_target(self):
        """Separa o DataFrame em X (features) e y (alvo).

        X recebe todas as colunas exceto o alvo; y recebe apenas a coluna 'result'
        (0/1/2). É a divisão "matéria" vs "resposta" que o modelo usa para aprender.

        Returns:
            (X, y): DataFrame de features e Series do alvo.
        """
        X = self.matches_df.drop(columns=[TARGET_COL])
        y = self.matches_df[TARGET_COL]
        return X, y

    def save(self, output_path: Path):
        """Grava o DataFrame tratado em CSV, criando a pasta de destino se preciso.

        Args:
            output_path: caminho do arquivo de saída (ex.: processed/matches_clean.csv).

        Returns:
            self, para permitir encadeamento.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.matches_df.to_csv(output_path, index=False)  # index=False: não grava a coluna de índice
        return self
