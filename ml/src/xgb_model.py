"""Nível 2: XGBoost — o modelo "forte", candidato a superar os baselines.

XGBoost é um conjunto de árvores de decisão treinadas em sequência, cada uma
corrigindo os erros da anterior (gradient boosting). Vantagens sobre o Poisson
linear neste caso:

- Captura relações NÃO-lineares e interações (ex.: a vantagem de jogar em casa
  pode pesar diferente conforme a diferença de elo).
- Lida bem com escalas diferentes sozinho: NÃO precisa de StandardScaler.

Mantemos objective='count:poisson' porque o alvo continua sendo contagem de gols
— assim o XGBoost também respeita que gol é um número inteiro não-negativo.
"""

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_xgb(X_train, y_train):
    """Treina um XGBoost (um por alvo) para prever os gols.

    Como o XGBRegressor prevê um alvo por vez, usamos MultiOutputRegressor para
    treinar um modelo para o mandante e outro para o visitante.

    Args:
        X_train: features de treino.
        y_train: alvos de treino (home_score, away_score).

    Returns:
        O modelo já treinado.
    """
    base = XGBRegressor(
        objective="count:poisson",  # alvo é contagem de gols
        n_estimators=300,           # nº de árvores
        learning_rate=0.05,         # passo de aprendizado (menor = mais cauteloso)
        max_depth=4,                # profundidade de cada árvore (controla complexidade)
        subsample=0.8,              # usa 80% das linhas por árvore (reduz overfitting)
        colsample_bytree=0.8,       # usa 80% das colunas por árvore
        random_state=42,            # reprodutibilidade
    )
    model = MultiOutputRegressor(base)
    model.fit(_numeric_features(X_train), y_train)
    return model


def evaluate_xgb(model, X_test, y_test):
    """Mede o erro do XGBoost no teste usando MAE (mesma métrica dos baselines).

    Args:
        model: modelo XGBoost já treinado.
        X_test: features de teste.
        y_test: placares reais de teste.

    Returns:
        (mae_home, mae_away, mae_geral): erro de cada alvo e a média dos dois.
    """
    y_pred = model.predict(_numeric_features(X_test))
    mae_home, mae_away = mean_absolute_error(
        y_test, y_pred, multioutput="raw_values"
    )
    mae_geral = (mae_home + mae_away) / 2
    return mae_home, mae_away, mae_geral
