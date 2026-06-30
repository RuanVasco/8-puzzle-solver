"""Baseline nível 1: regressão logística — o primeiro modelo que de fato aprende.

A regressão logística é o equivalente linear da Poisson, só que para
CLASSIFICAÇÃO: em vez de prever um número, ela estima a PROBABILIDADE de cada
classe (vitória mandante / empate / vitória visitante) e escolhe a mais provável.
Com 3 classes, usa a versão "softmax" (multinomial).

É o baseline sério da classificação: o número que o XGBoost terá que bater.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_logistic(X_train, y_train):
    """Treina a regressão logística com hiperparâmetros padrão (sem ajuste).

    Serve como versão "default" da comparação DEFAULT vs. TUNED — a busca de
    hiperparâmetros fica centralizada em models/tuning.py.

    O StandardScaler padroniza as features (média 0, desvio 1) para que o elo,
    numa escala de ~1500, não seja abafado pelas demais colunas.

    Args:
        X_train: features de treino (ordenadas por data).
        y_train: alvo de treino (result).

    Returns:
        O pipeline treinado (expõe predict).
    """
    # Pipeline: padroniza e então ajusta a logística. max_iter alto p/ convergência.
    # class_weight='balanced' compensa o desbalanceamento: faz errar empate (classe
    # minoritária) "doer" mais, evitando que o modelo simplesmente o ignore.
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipe.fit(_numeric_features(X_train), y_train)
    return pipe


def evaluate_logistic(model, X_test, y_test):
    """Mede a acurácia da logística no teste (mesma métrica do baseline nível 0).

    Args:
        model: modelo já treinado.
        X_test: features de teste.
        y_test: resultados reais de teste.

    Returns:
        A acurácia (float entre 0 e 1).
    """
    y_pred = model.predict(_numeric_features(X_test))
    return accuracy_score(y_test, y_pred)
