"""Avaliação detalhada: matriz de confusão e relatório por classe.

A acurácia é um número só e esconde detalhes. Aqui geramos:
- a matriz de confusão (acertos/erros separados por classe), e
- o relatório de precision/recall/F1 de cada classe,
para enxergar ONDE o modelo erra (em especial nos empates).
"""

import matplotlib
matplotlib.use("Agg")  # backend sem janela: só salva arquivo
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
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
