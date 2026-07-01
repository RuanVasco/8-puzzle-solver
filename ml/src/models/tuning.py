"""Ajuste de hiperparâmetros via GridSearchCV com validação cruzada temporal.

Este módulo centraliza a busca dos melhores hiperparâmetros para cada modelo.
Usa GridSearchCV (busca exaustiva) porque o espaço de busca é pequeno e
viável computacionalmente com ~4 k amostras de treino.

A validação cruzada interna é feita com TimeSeriesSplit para respeitar a
ordem temporal dos dados — o mesmo princípio da divisão treino/teste por
temporada usada no restante do projeto.
"""

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Grids de hiperparâmetros por modelo
# ---------------------------------------------------------------------------
# Cada entrada mapeia o nome do modelo para:
#   - "pipeline": o estimador (com scaler se necessário)
#   - "param_grid": o dicionário de hiperparâmetros a testar
#
# Para modelos dentro de um Pipeline, os nomes dos parâmetros seguem o formato
# "<nome_do_step>__<parâmetro>" (ex.: "logisticregression__C").

MODEL_CONFIGS = {
    "Logistic Regression": {
        "pipeline": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "param_grid": {
            "logisticregression__C": [0.01, 0.1, 1, 10, 100],
            # 'liblinear' não suporta classificação multiclasse (3 classes);
            # mantemos só solvers multinomiais.
            "logisticregression__solver": ["lbfgs", "newton-cg"],
        },
    },
    "Random Forest": {
        # Árvores não precisam de padronização.
        # n_jobs=1 aqui de propósito: quem paraleliza é o GridSearchCV (n_jobs=-1).
        # Deixar a floresta também com n_jobs=-1 cria paralelismo aninhado — gera
        # oversubscription (mais threads que núcleos) e enche a tela de UserWarning
        # do joblib sobre config não propagada.
        "pipeline": RandomForestClassifier(
            random_state=42, n_jobs=1, class_weight="balanced",
        ),
        "param_grid": {
            "n_estimators": [100, 300, 500],
            "max_depth": [5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
        },
    },
    "Naïve Bayes": {
        "pipeline": make_pipeline(StandardScaler(), GaussianNB()),
        "param_grid": {
            "gaussiannb__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
        },
    },
    "KNN": {
        "pipeline": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(),
        ),
        "param_grid": {
            "kneighborsclassifier__n_neighbors": [5, 9, 15, 21, 31],
            "kneighborsclassifier__weights": ["uniform", "distance"],
            "kneighborsclassifier__metric": ["euclidean", "manhattan"],
        },
    },
}


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def tune_model(name, X_train, y_train, n_splits=5, scoring="f1_macro"):
    """Roda GridSearchCV para um modelo e retorna o melhor estimador.

    Args:
        name: nome do modelo (chave em MODEL_CONFIGS).
        X_train: features de treino.
        y_train: alvo de treino.
        n_splits: quantidade de folds no TimeSeriesSplit.
        scoring: métrica usada para escolher o melhor conjunto de parâmetros.

    Returns:
        Tupla (best_estimator, best_params, best_cv_score).
    """
    config = MODEL_CONFIGS[name]
    cv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["param_grid"],
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,          # re-treina com os melhores params em todo o treino
    )
    grid.fit(_numeric_features(X_train), y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_all_models(X_train, y_train):
    """Aplica tune_model para cada modelo registrado.

    Returns:
        Dicionário {nome: (best_estimator, best_params, best_cv_score)}.
    """
    results = {}
    for name in MODEL_CONFIGS:
        print(f"  Tuning {name}...")
        best_est, best_params, best_cv = tune_model(name, X_train, y_train)
        results[name] = (best_est, best_params, best_cv)
        print(f"    Melhor CV score: {best_cv:.4f}")
        print(f"    Params: {best_params}")
    return results


def evaluate_tuned(results, X_test, y_test):
    """Avalia todos os modelos tuned no conjunto de teste.

    Args:
        results: dicionário retornado por tune_all_models.
        X_test: features de teste.
        y_test: alvo de teste.

    Returns:
        Dicionário {nome: accuracy_no_teste}.
    """
    X_num = _numeric_features(X_test)
    accuracies = {}
    for name, (model, _, _) in results.items():
        y_pred = model.predict(X_num)
        accuracies[name] = accuracy_score(y_test, y_pred)
    return accuracies


def print_comparison(default_accs, tuned_accs, tuned_results):
    """Imprime tabela comparativa: acurácia default vs tuned + melhores params.

    Args:
        default_accs: dict {nome: acurácia_default}.
        tuned_accs: dict {nome: acurácia_tuned}.
        tuned_results: dict retornado por tune_all_models (para pegar params).
    """
    print("\n" + "=" * 78)
    print("COMPARAÇÃO: DEFAULT vs. TUNED (acurácia no teste)")
    print("=" * 78)
    print(f"{'Modelo':<25} {'Default':>8} {'Tuned':>8} {'Δ':>8}  Melhores parâmetros")
    print("-" * 78)

    for name in tuned_accs:
        default = default_accs.get(name, float("nan"))
        tuned = tuned_accs[name]
        delta = tuned - default
        _, best_params, _ = tuned_results[name]
        # Formata params de forma legível (remove prefixo do pipeline)
        short_params = {k.split("__")[-1]: v for k, v in best_params.items()}
        print(f"{name:<25} {default:>8.3f} {tuned:>8.3f} {delta:>+8.3f}  {short_params}")

    print("=" * 78)
