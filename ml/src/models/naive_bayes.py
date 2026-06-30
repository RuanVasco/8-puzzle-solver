"""Naïve Bayes — classificação baseada em probabilidade (Teorema de Bayes).

A ideia: dado os atributos de um jogo, qual a classe MAIS PROVÁVEL? Ele combina
a probabilidade de cada feature sob cada classe usando o Teorema de Bayes. O
"naïve" (ingênuo) vem da suposição simplificadora de que as features são
independentes entre si — quase nunca é verdade, mas funciona surpreendentemente
bem e é muito rápido.

Usamos o GaussianNB, que assume que as features numéricas seguem uma distribuição
normal. Por isso a padronização ajuda — incluímos um StandardScaler no pipeline.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Grade de hiperparâmetros. O GaussianNB quase não tem o que ajustar: o único
# botão é var_smoothing (uma "almofada" somada à variância para estabilizar o
# cálculo). Buscamos por consistência com os demais modelos, mas o efeito aqui
# costuma ser pequeno. O prefixo 'gaussiannb__' aponta para a etapa do pipeline.
PARAM_GRID = {
    'gaussiannb__var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3],
}

# Nº de dobras da validação cruzada temporal.
N_SPLITS = 5


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_naive_bayes(X_train, y_train, tune=True):
    """Treina o Naïve Bayes gaussiano, com ou sem busca de hiperparâmetros.

    Com tune=True: varre a PARAM_GRID validando com TimeSeriesSplit (passado
    treina, futuro valida), o que depende de X_train estar ordenado por data —
    garantido pela ordem cronológica do CSV. Com tune=False: usa o var_smoothing
    padrão, servindo de comparação "sem ajuste de hiperparâmetros".

    Args:
        X_train: features de treino (ordenadas por data).
        y_train: alvo de treino (result).
        tune: se True, faz GridSearch; se False, usa hiperparâmetros fixos.

    Returns:
        O modelo treinado (GridSearchCV se tune=True, senão o pipeline direto);
        ambos expõem predict.
    """
    pipe = make_pipeline(StandardScaler(), GaussianNB())
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
    print(f"[Naïve Bayes] melhores hiperparâmetros: {search.best_params_}")
    print(f"[Naïve Bayes] acurácia média na validação temporal: {search.best_score_:.3f}")
    return search


def evaluate_naive_bayes(model, X_test, y_test):
    """Mede a acurácia do Naïve Bayes no teste (mesma métrica dos demais).

    Args:
        model: modelo já treinado.
        X_test: features de teste.
        y_test: resultados reais de teste.

    Returns:
        A acurácia (float entre 0 e 1).
    """
    y_pred = model.predict(_numeric_features(X_test))
    return accuracy_score(y_test, y_pred)
