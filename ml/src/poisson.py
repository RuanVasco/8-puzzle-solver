"""Baseline nível 1: regressão de Poisson — o primeiro modelo que de fato aprende.

Diferente do nível 0 (que ignora tudo e chuta a média), este modelo OLHA as
features (força dos times, competição, etc.) e aprende uma relação com o número
de gols. Usamos Poisson porque gol é uma CONTAGEM (0, 1, 2, 3...): a distribuição
de Poisson é a matemática natural para "quantas vezes um evento acontece", então
o modelo nunca prevê coisas sem sentido como -0.4 gol.

Este é o baseline "sério": é o número que o XGBoost terá que bater para justificar
a sua complexidade.
"""

from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _numeric_features(X):
    """Mantém apenas as colunas numéricas que o modelo consegue usar.

    Um modelo só faz conta com números. A coluna 'match_date' é uma data
    (datetime), que o Poisson não sabe processar crua — e o ano dela já está
    representado pela coluna 'season'. Então a removemos aqui.

    Args:
        X: DataFrame de features.

    Returns:
        DataFrame só com colunas numéricas.
    """
    return X.select_dtypes(include="number")


def train_poisson(X_train, y_train):
    """Treina uma regressão de Poisson para prever os gols.

    O StandardScaler padroniza as features (média 0, desvio 1) para que o elo,
    numa escala de ~1500, não seja abafado pelas demais colunas. Importante: o
    scaler aprende média/desvio só no fit() (ou seja, só no treino), e o mesmo
    ajuste é reaplicado no teste — sem vazamento.

    O PoissonRegressor prevê UM alvo por vez; como temos dois (mandante e
    visitante), envolvemos com MultiOutputRegressor, que treina um pipeline
    (scaler + Poisson) para cada lado automaticamente.

    Args:
        X_train: features de treino.
        y_train: alvos de treino (home_score, away_score).

    Returns:
        O modelo já treinado.
    """
    # Pipeline: padroniza e então ajusta o Poisson. max_iter alto p/ convergência.
    base = make_pipeline(StandardScaler(), PoissonRegressor(max_iter=1000))
    model = MultiOutputRegressor(base)
    model.fit(_numeric_features(X_train), y_train)
    return model


def evaluate_poisson(model, X_test, y_test):
    """Mede o erro do Poisson no teste usando MAE (erro médio em gols).

    Mesma métrica do baseline nível 0, de propósito: assim a comparação é justa
    (maçã com maçã). Quanto menor o MAE, melhor o modelo previu os placares.

    Args:
        model: modelo Poisson já treinado.
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
