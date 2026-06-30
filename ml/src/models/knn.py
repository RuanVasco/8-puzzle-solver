"""KNN (K-Nearest Neighbors) — classifica pelos vizinhos mais parecidos.

A ideia é intuitiva: para prever um jogo novo, o KNN procura os K jogos passados
MAIS PARECIDOS (em termos de elo, liga, etc.) e devolve o resultado que mais
apareceu entre eles. "Me diga com quem andas..." aplicado a partidas.

A padronização é ESSENCIAL aqui: como o KNN mede distância entre os jogos, uma
feature em escala grande (elo ~1500) dominaria a conta. O StandardScaler coloca
todas na mesma escala antes de medir a proximidade.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Grade de hiperparâmetros do KNN. n_neighbors (o "K") é o botão principal:
# K pequeno decora o ruído; K grande suaviza demais. weights='distance' faz
# vizinhos mais próximos pesarem mais. Os prefixos 'kneighborsclassifier__'
# apontam para a etapa do pipeline (make_pipeline nomeia pela classe).
PARAM_GRID = {
    'kneighborsclassifier__n_neighbors': [5, 15, 25, 51, 101],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
}

# Nº de dobras da validação cruzada temporal.
N_SPLITS = 5


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_knn(X_train, y_train, tune=True):
    """Treina o KNN, com ou sem busca de hiperparâmetros.

    O StandardScaler segue dentro do pipeline: é essencial no KNN porque ele mede
    distância, e é reaplicado a cada dobra sem vazamento.

    Com tune=True: varre a PARAM_GRID validando com TimeSeriesSplit (passado
    treina, futuro valida) — depende de X_train estar ordenado por data, o que é
    garantido pela ordem cronológica do CSV. Com tune=False: usa K fixo (=15),
    servindo de comparação "sem ajuste de hiperparâmetros".

    Args:
        X_train: features de treino (ordenadas por data).
        y_train: alvo de treino (result).
        tune: se True, faz GridSearch; se False, usa hiperparâmetros fixos.

    Returns:
        O modelo treinado (GridSearchCV se tune=True, senão o pipeline direto);
        ambos expõem predict.
    """
    if not tune:
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
        pipe.fit(_numeric_features(X_train), y_train)
        return pipe

    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    search = GridSearchCV(
        pipe,
        param_grid=PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring='accuracy',
        n_jobs=-1,
    )
    search.fit(_numeric_features(X_train), y_train)
    print(f"[KNN] melhores hiperparâmetros: {search.best_params_}")
    print(f"[KNN] acurácia média na validação temporal: {search.best_score_:.3f}")
    return search


def evaluate_knn(model, X_test, y_test):
    """Mede a acurácia do KNN no teste (mesma métrica dos demais).

    Args:
        model: modelo já treinado.
        X_test: features de teste.
        y_test: resultados reais de teste.

    Returns:
        A acurácia (float entre 0 e 1).
    """
    y_pred = model.predict(_numeric_features(X_test))
    return accuracy_score(y_test, y_pred)
