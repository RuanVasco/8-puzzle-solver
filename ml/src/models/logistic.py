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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Grade de hiperparâmetros. C é o inverso da força de regularização: C pequeno
# regulariza mais (modelo mais "contido", evita decorar); C grande deixa o modelo
# mais livre. O prefixo 'logisticregression__' aponta para a etapa do pipeline.
PARAM_GRID = {
    'logisticregression__C': [0.01, 0.1, 1, 10],
}

# Nº de dobras da validação cruzada temporal.
N_SPLITS = 5


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_logistic(X_train, y_train, tune=True):
    """Treina a regressão logística, com ou sem busca de hiperparâmetros.

    O StandardScaler padroniza as features (média 0, desvio 1) para que o elo,
    numa escala de ~1500, não seja abafado pelas demais colunas; ele é reaplicado
    a cada dobra sem vazamento.

    Com tune=True: varre o C via GridSearchCV usando TimeSeriesSplit (passado
    treina, futuro valida), o que depende de X_train estar ordenado por data —
    garantido pela ordem cronológica do CSV. Com tune=False: usa o C padrão (=1),
    servindo de comparação "sem ajuste de hiperparâmetros".

    Args:
        X_train: features de treino (ordenadas por data).
        y_train: alvo de treino (result).
        tune: se True, faz GridSearch; se False, usa hiperparâmetros fixos.

    Returns:
        O modelo treinado (GridSearchCV se tune=True, senão o pipeline direto);
        ambos expõem predict.
    """
    # Pipeline: padroniza e então ajusta a logística. max_iter alto p/ convergência.
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if not tune:
        pipe.fit(_numeric_features(X_train), y_train)
        return pipe

    search = GridSearchCV(
        pipe,
        param_grid=PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring='accuracy',
        n_jobs=-1,
    )
    search.fit(_numeric_features(X_train), y_train)
    print(f"[Regressão Logística] melhor C: {search.best_params_}")
    print(f"[Regressão Logística] acurácia média na validação temporal: {search.best_score_:.3f}")
    return search


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
