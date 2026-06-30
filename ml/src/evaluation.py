"""Avaliação detalhada: matriz de confusão e relatório por classe.

A acurácia é um número só e esconde detalhes. Aqui geramos:
- a matriz de confusão (acertos/erros separados por classe), e
- o relatório de precision/recall/F1 de cada classe,
para enxergar ONDE o modelo erra (em especial nos empates).
"""

import csv

import matplotlib
matplotlib.use("Agg")  # backend sem janela: só salva arquivo
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

# Nomes das classes na ordem do alvo: 0, 1, 2.
LABELS = ["Mandante", "Empate", "Visitante"]


def _numeric_features(X):
    """Mantém apenas colunas numéricas (remove 'match_date', que é datetime)."""
    return X.select_dtypes(include="number")


def report_and_plot(model, X_test, y_test, title, save_path):
    """Imprime o relatório por classe e salva a matriz de confusão como imagem.

    Args:
        model: modelo já treinado.
        X_test: features de teste.
        y_test: resultados reais de teste.
        title: título usado na imagem e no cabeçalho do relatório.
        save_path: caminho (Path) onde salvar o PNG da matriz.

    Returns:
        A matriz de confusão (array NxN).
    """
    y_pred = model.predict(_numeric_features(X_test))

    print(f"\n=== {title} ===")
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {save_path}")
    return cm


# Colunas do CSV de métricas, nesta ordem.
METRIC_FIELDS = ["modelo", "variante", "accuracy", "precision_macro", "recall_macro", "f1_macro"]


def compute_metrics(model, X_test, y_test):
    """Calcula acurácia e precision/recall/F1 (média macro) de um modelo no teste.

    Usamos a média macro (cada classe pesa igual) porque as classes são
    desbalanceadas — o empate é raro — e a média macro penaliza quem ignora as
    classes minoritárias, ao contrário da acurácia pura.

    Args:
        model: modelo já treinado (expõe predict).
        X_test: features de teste.
        y_test: resultados reais de teste.

    Returns:
        Dict com accuracy, precision_macro, recall_macro, f1_macro.
    """
    y_pred = model.predict(_numeric_features(X_test))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def save_metrics_csv(rows, save_path):
    """Salva a tabela de métricas (uma linha por modelo) em CSV.

    Args:
        rows: lista de dicts, cada um com as chaves de METRIC_FIELDS. Os valores
            numéricos são arredondados para 4 casas na escrita.
        save_path: caminho (Path) do CSV de saída.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in row.items()
            })
    print(f"Métricas salvas em: {save_path}")
