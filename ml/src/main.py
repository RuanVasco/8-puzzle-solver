from pathlib import Path

from .data.pipeline import DataPipeline
from .data.split import train_test_split_by_season
from .models.baseline import train_baseline, evaluate_baseline
from .models.logistic import train_logistic, evaluate_logistic
from .models.forest import train_forest, evaluate_forest
from .models.naive_bayes import train_naive_bayes, evaluate_naive_bayes
from .models.knn import train_knn, evaluate_knn
from .evaluation import report_and_plot

# Caminhos ancorados no próprio arquivo, não no diretório de execução,
# pra funcionar de qualquer lugar que o script seja rodado.
BASE_DIR = Path(__file__).resolve().parent.parent   # raiz do projeto (.../ml)
RAW_DIR = BASE_DIR / "datasets" / "raw"              # CSVs originais (não mexer)
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"  # saída já tratada
REPORTS_DIR = BASE_DIR / "reports"                   # gráficos/relatórios


if __name__ == "__main__":
    # Monta o pipeline, trata os dados, cria o alvo de resultado e salva.
    pipeline = DataPipeline(RAW_DIR / "matches.csv")
    pipeline.clean_targets().make_target().format_types().fill_missing()
    pipeline.select_features().encode_frequencies().encode_categoricals()

    output_path = PROCESSED_DIR / "matches_clean.csv"
    pipeline.save(output_path)
    print(f"Salvo em {output_path} | shape (linhas, colunas):", pipeline.get_clean_data().shape)

    # Passo 1 da modelagem: separar features (X) do alvo (y = resultado 0/1/2).
    X, y = pipeline.split_features_target()
    print("X (features):", X.shape, "| y (alvo):", y.shape)

    # Passo 2: separar treino/teste por temporada (não aleatório).
    X_train, X_test, y_train, y_test = train_test_split_by_season(X, y)
    print("Treino:", X_train.shape, "| Teste:", X_test.shape)

    # Passo 3: baseline nível 0 (sempre a classe mais comum) — a régua mínima.
    baseline = train_baseline(X_train, y_train)
    acc_base = evaluate_baseline(baseline, X_test, y_test)
    print(f"\n[Nível 0 - classe mais comum]  acurácia: {acc_base:.3f}")

    # Passo 4: baseline nível 1 (regressão logística) — primeiro modelo que aprende.
    logistic = train_logistic(X_train, y_train)
    acc_log = evaluate_logistic(logistic, X_test, y_test)
    print(f"[Nível 1 - regressão logística]  acurácia: {acc_log:.3f}  (régua: {acc_base:.3f})")

    # Passo 5: outros classificadores da lista do enunciado, para comparação.
    forest = train_forest(X_train, y_train)
    acc_rf = evaluate_forest(forest, X_test, y_test)
    print(f"[Random Forest]                  acurácia: {acc_rf:.3f}")

    nb = train_naive_bayes(X_train, y_train)
    acc_nb = evaluate_naive_bayes(nb, X_test, y_test)
    print(f"[Naïve Bayes]                    acurácia: {acc_nb:.3f}")

    knn = train_knn(X_train, y_train)
    acc_knn = evaluate_knn(knn, X_test, y_test)
    print(f"[KNN]                            acurácia: {acc_knn:.3f}")

    # Passo 6: avaliação detalhada do modelo campeão (Logística): relatório por
    # classe e matriz de confusão salva como imagem para o artigo.
    report_and_plot(
        logistic, X_test, y_test,
        title="Matriz de Confusão - Regressão Logística",
        save_path=REPORTS_DIR / "confusion_logistic.png",
    )
