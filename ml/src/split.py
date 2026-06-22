import pandas as pd

# Temporadas reservadas para teste (as mais recentes). Tudo antes vira treino.
# Manter como teste os anos mais novos simula o uso real: prever o futuro a
# partir do passado. Ajuste aqui se quiser um conjunto de teste maior/menor.
DEFAULT_TEST_SEASONS = [2025, 2026]


def train_test_split_by_season(X, y, test_seasons=DEFAULT_TEST_SEASONS):
    """Separa treino/teste por temporada, em vez de aleatoriamente.

    Como os dados são temporais, um corte aleatório deixaria o modelo "ver o
    futuro" (treinar com jogos recentes e testar com antigos). Aqui o teste fica
    com as temporadas mais novas e o treino com todo o resto.

    Args:
        X: DataFrame de features (precisa conter a coluna 'season').
        y: DataFrame/Series de alvos, alinhado linha a linha com X.
        test_seasons: lista de temporadas que vão para o conjunto de teste.

    Returns:
        (X_train, X_test, y_train, y_test).
    """
    is_test = X['season'].isin(test_seasons)
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]
    return X_train, X_test, y_train, y_test
