import sys
from pathlib import Path

# Saída em UTF-8 para não estourar com acentos e símbolos (ex.: 'Δ') em
# consoles cujo codepage padrão (cp1252 no Windows) não os suporta.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .data.pipeline import DataPipeline
from .data.split import train_test_split_by_season
from .models.baseline import train_baseline, evaluate_baseline
from .models.logistic import train_logistic, evaluate_logistic
from .models.forest import train_forest, evaluate_forest
from .models.naive_bayes import train_naive_bayes, evaluate_naive_bayes
from .models.knn import train_knn, evaluate_knn
from .models.tuning import tune_all_models, evaluate_tuned, print_comparison
from .evaluation import report_and_plot, compute_metrics, save_metrics_csv

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
    pipeline.select_features().add_elo_features().encode_frequencies().encode_categoricals()

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

    print("\n>>> Modelos DEFAULT (hiperparâmetros padrão, sem ajuste) <<<")

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

    # Passo 6: avaliação detalhada do modelo campeão default (Logística):
    # relatório por classe e matriz de confusão salva como imagem para o artigo.
    report_and_plot(
        logistic, X_test, y_test,
        title="Matriz de Confusão - Regressão Logística (Default)",
        save_path=REPORTS_DIR / "confusion_logistic.png",
    )

    # ------------------------------------------------------------------
    # Passo 7: AJUSTE DE HIPERPARÂMETROS (GridSearchCV + TimeSeriesSplit)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AJUSTE DE HIPERPARÂMETROS (GridSearchCV + TimeSeriesSplit)")
    print("=" * 60)

    tuned_results = tune_all_models(X_train, y_train)

    # Passo 8: avaliar modelos tuned no conjunto de teste.
    tuned_accs = evaluate_tuned(tuned_results, X_test, y_test)

    # Acurácias default para comparação (mesma ordem dos nomes em tuning.py).
    default_accs = {
        "Logistic Regression": acc_log,
        "Random Forest": acc_rf,
        "Naïve Bayes": acc_nb,
        "KNN": acc_knn,
    }

    print_comparison(default_accs, tuned_accs, tuned_results)

    # Passo 9: matriz de confusão do melhor modelo tuned.
    # Campeão escolhido por F1-macro (média do F1 por classe), não por acurácia
    # pura — coerente com o scoring do tuning. A acurácia pura elegeria quem
    # ignora o empate; o F1-macro premia quem equilibra precision e recall nas
    # 3 classes, punindo tanto deixar passar quanto chutar à toa.
    tuned_f1 = {
        name: compute_metrics(model, X_test, y_test)["f1_macro"]
        for name, (model, _, _) in tuned_results.items()
    }
    best_name = max(tuned_f1, key=tuned_f1.get)
    best_model = tuned_results[best_name][0]
    print(f"\nMelhor modelo tuned (F1-macro): {best_name} "
          f"({tuned_f1[best_name]:.3f})")

    report_and_plot(
        best_model, X_test, y_test,
        title=f"Matriz de Confusão - {best_name} (Tuned)",
        save_path=REPORTS_DIR / "confusion_best_tuned.png",
    )

    # ------------------------------------------------------------------
    # Passo 10: registrar todas as métricas (accuracy/precision/recall/F1)
    # em CSV — uma linha por modelo, separando baseline, default e tuned.
    # ------------------------------------------------------------------
    default_models = {
        "Logistic Regression": logistic,
        "Random Forest": forest,
        "Naïve Bayes": nb,
        "KNN": knn,
    }

    metric_rows = []
    metric_rows.append({"modelo": "Baseline (classe mais comum)", "variante": "baseline",
                        **compute_metrics(baseline, X_test, y_test)})
    for name, model in default_models.items():
        metric_rows.append({"modelo": name, "variante": "default",
                            **compute_metrics(model, X_test, y_test)})
    for name, (model, _, _) in tuned_results.items():
        metric_rows.append({"modelo": name, "variante": "tuned",
                            **compute_metrics(model, X_test, y_test)})

    save_metrics_csv(metric_rows, REPORTS_DIR / "metrics.csv")
