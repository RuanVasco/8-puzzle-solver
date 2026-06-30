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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Grade de hiperparâmetros que o GridSearch vai varrer. Ele testa TODAS as
# combinações (2 x 3 x 3 = 18) e, para cada uma, mede a média da acurácia na
# validação cruzada — ficando com a melhor.
PARAM_GRID = {
    'n_estimators': [200, 300],          # nº de árvores no comitê
    'max_depth': [5, 8, 12],             # profundidade máxima (controla overfitting)
    'min_samples_leaf': [1, 10, 30],     # mínimo de jogos por folha (folha maior = menos decoreba)
}

# Nº de dobras da validação cruzada temporal.
N_SPLITS = 5


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def train_forest(X_train, y_train, tune=True):
    """Treina o Random Forest, com ou sem busca de hiperparâmetros.

    Com tune=True: varre a PARAM_GRID com GridSearchCV usando TimeSeriesSplit
    (não o corte aleatório padrão): cada dobra treina no passado e valida no
    futuro, respeitando a ordem do tempo e evitando vazamento. Isso depende de
    X_train estar ordenado por data — garantido porque o CSV bruto já vem em
    ordem cronológica e a divisão por temporada preserva essa ordem. Com
    tune=False: usa hiperparâmetros fixos (n_estimators=300, max_depth=8),
    servindo de comparação "sem ajuste de hiperparâmetros".

    Args:
        X_train: features de treino (ordenadas por data).
        y_train: alvo de treino (result).
        tune: se True, faz GridSearch; se False, usa hiperparâmetros fixos.

    Returns:
        O modelo treinado (GridSearchCV se tune=True, senão o RandomForest direto);
        ambos expõem predict.
    """
    if not tune:
        model = RandomForestClassifier(
            n_estimators=300,      # nº de árvores no comitê
            max_depth=8,           # profundidade máxima de cada árvore (controla overfitting)
            random_state=42,       # reprodutibilidade
            n_jobs=-1,             # usa todos os núcleos da CPU
        )
        model.fit(_numeric_features(X_train), y_train)
        return model

    search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),  # dobras temporais (passado treina, futuro valida)
        scoring='accuracy',                     # mesma métrica do resto do projeto
        n_jobs=-1,                              # paraleliza as combinações em todos os núcleos
    )
    search.fit(_numeric_features(X_train), y_train)
    print(f"[Random Forest] melhores hiperparâmetros: {search.best_params_}")
    print(f"[Random Forest] acurácia média na validação temporal: {search.best_score_:.3f}")
    return search


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
