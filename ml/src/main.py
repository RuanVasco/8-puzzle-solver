import pandas as pd
from pathlib import Path

# Caminhos ancorados no próprio arquivo, não no diretório de execução,
# pra funcionar de qualquer lugar que o script seja rodado.
BASE_DIR = Path(__file__).resolve().parent.parent   # raiz do projeto (.../ml)
RAW_DIR = BASE_DIR / "datasets" / "raw"              # CSVs originais (não mexer)
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"  # saída já tratada

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
    # TODO: experimentar mover 'referee' para cá. São 619 juízes diferentes
    # (cardinalidade muito alta), o que pode atrapalhar mais que ajudar.
    # Por ora mantemos como feature para testar se agrega algo.
]

# Colunas categóricas: são rótulos (códigos de time/liga ou nome do juiz), não
# quantidades. Precisam de encoding para o modelo não interpretá-las como números
# ordenáveis (ex.: achar que o time 481 é "maior" que o time 50).
CATEGORICAL_COLS = [
    'league_id',
    'home_team_id', 'away_team_id',
    'referee',
]


class DataPipeline:
    """Carrega e prepara os dados de partidas para o treinamento de previsão de placar."""

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
        self.matches_df = self.matches_df.dropna(subset=['home_score', 'away_score'])
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

        Mantém apenas informação conhecida ANTES da partida (elos, times,
        contexto temporal/competição) junto com os alvos home_score/away_score.
        Usa errors='ignore' para não quebrar caso alguma coluna já não exista.

        Returns:
            self, para permitir encadeamento.
        """
        self.matches_df = self.matches_df.drop(
            columns=LEAKAGE_COLS + REDUNDANT_COLS,
            errors='ignore',
        )
        return self

    def encode_categoricals(self):
        """Converte as colunas categóricas em colunas numéricas via one-hot encoding.

        Para cada valor distinto (cada time, liga ou juiz) o pandas cria uma coluna
        0/1 do tipo "é esse valor? sim(1)/não(0)". Assim o modelo deixa de tratar os
        códigos como quantidades ordenáveis e passa a vê-los como categorias.

        Atenção: gera muitas colunas (alta cardinalidade em times e juízes). É o ponto
        de partida didático; depois dá para comparar com técnicas mais enxutas.

        Returns:
            self, para permitir encadeamento.
        """
        self.matches_df = pd.get_dummies(
            self.matches_df,
            columns=CATEGORICAL_COLS,
            dtype=int,  # gera 0/1 inteiros em vez de True/False
        )
        return self

    def get_clean_data(self):
        """Devolve o DataFrame já tratado para o consumidor (treino/análise)."""
        return self.matches_df

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


if __name__ == "__main__":
    # Monta o pipeline, aplica as etapas de limpeza e salva o resultado tratado.
    pipeline = DataPipeline(RAW_DIR / "matches.csv")
    pipeline.clean_targets().format_types().fill_missing().select_features().encode_categoricals()

    output_path = PROCESSED_DIR / "matches_clean.csv"
    pipeline.save(output_path)
    print(f"Salvo em {output_path} | shape (linhas, colunas):", pipeline.get_clean_data().shape)
