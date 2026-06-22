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


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_knn(X_train, y_train, n_neighbors=15):
    """Treina um KNN para classificar o resultado (0/1/2).

    Args:
        X_train: features de treino.
        y_train: alvo de treino (result).
        n_neighbors: quantos vizinhos considerar (o "K"). Valores maiores suavizam.

    Returns:
        O modelo já treinado.
    """
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors),
    )
    model.fit(_numeric_features(X_train), y_train)
    return model


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
