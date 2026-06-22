"""Baseline nível 0: o modelo "burro" que serve de régua mínima.

A ideia deste nível é deliberadamente NÃO aprender nada inteligente. O
DummyRegressor ignora completamente as features (X) e sempre prevê o mesmo
valor: a média dos placares vistos no treino. Qualquer modelo de verdade
(Poisson, XGBoost...) precisa ERRAR MENOS que isto para se justificar.
"""

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error


def train_baseline(X_train, y_train):
    """Treina o baseline 'sempre a média'.

    O strategy='mean' faz o modelo memorizar apenas a média de cada alvo
    (home_score e away_score) no treino. Ele nem olha para X — por isso é o
    piso: representa "chutar a média sem pensar".

    Args:
        X_train: features de treino (ignoradas pelo modelo, exigidas pela API).
        y_train: alvos de treino (home_score, away_score).

    Returns:
        O modelo já treinado.
    """
    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(model, X_test, y_test):
    """Mede o erro do baseline no conjunto de teste usando MAE.

    MAE (Mean Absolute Error) = erro médio em gols. Ex.: MAE de 1.1 significa
    que, em média, a previsão errou o placar por ~1.1 gol. Quanto menor, melhor.

    Args:
        model: baseline já treinado.
        X_test: features de teste.
        y_test: placares reais de teste, para comparar com a previsão.

    Returns:
        (mae_home, mae_away, mae_geral): erro de cada alvo e a média dos dois.
    """
    # Como o modelo sempre prevê a média, todas as linhas recebem o mesmo palpite.
    y_pred = model.predict(X_test)

    # multioutput='raw_values' devolve o MAE separado por coluna (mandante/visitante).
    mae_home, mae_away = mean_absolute_error(
        y_test, y_pred, multioutput="raw_values"
    )
    mae_geral = (mae_home + mae_away) / 2
    return mae_home, mae_away, mae_geral
