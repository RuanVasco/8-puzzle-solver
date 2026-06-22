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


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_naive_bayes(X_train, y_train):
    """Treina um Naïve Bayes gaussiano para classificar o resultado (0/1/2).

    Args:
        X_train: features de treino.
        y_train: alvo de treino (result).

    Returns:
        O modelo já treinado.
    """
    model = make_pipeline(StandardScaler(), GaussianNB())
    model.fit(_numeric_features(X_train), y_train)
    return model


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
