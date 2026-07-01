"""Script avulso: treina o Random Forest e exporta UMA arvore de decisao
como imagem (visao rasa, ate profundidade 3, pra ficar legivel) e como texto.
Nao sobrescreve nada em reports/ — salva na raiz do ml/.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

from src.data.pipeline import DataPipeline
from src.data.split import train_test_split_by_season
from src.models.forest import train_forest, _numeric_features

BASE_DIR = Path(__file__).resolve().parent
RAW = BASE_DIR / "datasets" / "raw" / "matches.csv"

# Mesmo preparo do main.py (sem salvar CSV).
pipe = DataPipeline(RAW)
pipe.clean_targets().make_target().format_types().fill_missing()
pipe.select_features().add_elo_features().encode_frequencies().encode_categoricals()

X, y = pipe.split_features_target()
X_train, X_test, y_train, y_test = train_test_split_by_season(X, y)

forest = train_forest(X_train, y_train)

# Pega a PRIMEIRA das 300 arvores do comite.
arvore = forest.estimators_[0]
features = _numeric_features(X_train).columns.tolist()
classes = ["visitante(0)", "empate(1)", "mandante(2)"]  # ajuste se o mapeamento diferir

# --- imagem (so ate profundidade 3 pra caber na tela) ---
plt.figure(figsize=(22, 11))
plot_tree(
    arvore,
    max_depth=3,
    feature_names=features,
    class_names=classes,
    filled=True,        # cor por classe majoritaria
    rounded=True,
    fontsize=9,
    impurity=True,      # mostra o gini de cada no
    proportion=True,
)
plt.title("Uma arvore do Random Forest (visao ate profundidade 3)")
plt.tight_layout()
out_img = BASE_DIR / "arvore_exemplo.png"
plt.savefig(out_img, dpi=90)
print("Imagem salva em:", out_img)
print("Total de arvores no comite:", len(forest.estimators_))
print("Profundidade real desta arvore:", arvore.get_depth(), "| nos:", arvore.tree_.node_count)

# --- versao texto (as primeiras perguntas) ---
print("\n----- TOP DA ARVORE (texto, ate profundidade 3) -----")
print(export_text(arvore, feature_names=features, max_depth=3))
