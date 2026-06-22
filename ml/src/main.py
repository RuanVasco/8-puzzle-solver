from pathlib import Path

from .pipeline import DataPipeline

# Caminhos ancorados no próprio arquivo, não no diretório de execução,
# pra funcionar de qualquer lugar que o script seja rodado.
BASE_DIR = Path(__file__).resolve().parent.parent   # raiz do projeto (.../ml)
RAW_DIR = BASE_DIR / "datasets" / "raw"              # CSVs originais (não mexer)
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"  # saída já tratada


if __name__ == "__main__":
    # Monta o pipeline, aplica as etapas de limpeza e salva o resultado tratado.
    pipeline = DataPipeline(RAW_DIR / "matches.csv")
    pipeline.clean_targets().format_types().fill_missing().select_features().encode_categoricals()

    output_path = PROCESSED_DIR / "matches_clean.csv"
    pipeline.save(output_path)
    print(f"Salvo em {output_path} | shape (linhas, colunas):", pipeline.get_clean_data().shape)

    # Passo 1 da modelagem: separar features (X) do alvo (y).
    X, y = pipeline.split_features_target()
    print("X (features):", X.shape, "| y (alvo):", y.shape)
