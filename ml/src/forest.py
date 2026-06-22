"""Random Forest — um "comitê" de árvores de decisão votando.

Uma árvore de decisão sozinha é uma sequência de perguntas sim/não
("elo do mandante > 1600?") até chegar a um palpite. O problema é que uma
árvore só decora demais. O Random Forest treina MUITAS árvores, cada uma
vendo uma parte aleatória dos dados/colunas, e faz elas VOTAREM. O voto da
maioria costuma ser bem mais robusto que uma árvore isolada.

Não precisa de padronização (árvores não se importam com escala).
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_forest(X_train, y_train):
    """Treina um Random Forest para classificar o resultado (0/1/2).

    Args:
        X_train: features de treino.
        y_train: alvo de treino (result).

    Returns:
        O modelo já treinado.
    """
    model = RandomForestClassifier(
        n_estimators=300,      # nº de árvores no comitê
        max_depth=8,           # profundidade máxima de cada árvore (controla overfitting)
        random_state=42,       # reprodutibilidade
        n_jobs=-1,             # usa todos os núcleos da CPU
    )
    model.fit(_numeric_features(X_train), y_train)
    return model


def evaluate_forest(model, X_test, y_test):
    """Mede a acurácia do Random Forest no teste (mesma métrica dos demais).

    Args:
        model: modelo já treinado.
        X_test: features de teste.
        y_test: resultados reais de teste.

    Returns:
        A acurácia (float entre 0 e 1).
    """
    y_pred = model.predict(_numeric_features(X_test))
    return accuracy_score(y_test, y_pred)
