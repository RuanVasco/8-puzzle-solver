"""Baseline nível 0: o classificador "burro" que serve de régua mínima.

Para classificação, o palpite trivial é sempre prever a classe mais comum. No
futebol isso costuma ser "vitória do mandante" (mando de campo). O DummyClassifier
faz exatamente isso: ignora as features e sempre responde a classe majoritária do
treino. Qualquer modelo de verdade precisa ACERTAR MAIS que isto para se justificar.
"""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


def train_baseline(X_train, y_train):
    """Treina o baseline 'sempre a classe mais comum'.

    O strategy='most_frequent' faz o modelo memorizar qual resultado mais apareceu
    no treino e repetir esse palpite para todo jogo. Ele nem olha para X.

    Args:
        X_train: features de treino (ignoradas pelo modelo, exigidas pela API).
        y_train: alvo de treino (result: 0/1/2).

    Returns:
        O modelo já treinado.
    """
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(model, X_test, y_test):
    """Mede a acurácia do baseline no teste.

    Acurácia = fração de jogos cujo resultado foi previsto corretamente
    (de 0 a 1; 0.50 = acertou metade). Quanto maior, melhor.

    Args:
        model: baseline já treinado.
        X_test: features de teste.
        y_test: resultados reais de teste.

    Returns:
        A acurácia (float entre 0 e 1).
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
