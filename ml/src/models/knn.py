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


def train_knn(X_train, y_train):
    """Treina o KNN com hiperparâmetros padrão (sem ajuste).

    Serve como versão "default" da comparação DEFAULT vs. TUNED — a busca de
    hiperparâmetros fica centralizada em models/tuning.py.

    O StandardScaler segue dentro do pipeline: é essencial no KNN porque ele mede
    distância entre os jogos.

    Args:
        X_train: features de treino (ordenadas por data).
        y_train: alvo de treino (result).

    Returns:
        O pipeline treinado (expõe predict).
    """
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    pipe.fit(_numeric_features(X_train), y_train)
    return pipe


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
