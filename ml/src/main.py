from pathlib import Path

from .pipeline import DataPipeline
from .split import train_test_split_by_season
from .baseline import train_baseline, evaluate_baseline
from .poisson import train_poisson, evaluate_poisson
from .xgb_model import train_xgb, evaluate_xgb

# Caminhos ancorados no próprio arquivo, não no diretório de execução,
# pra funcionar de qualquer lugar que o script seja rodado.
BASE_DIR = Path(__file__).resolve().parent.parent   # raiz do projeto (.../ml)
RAW_DIR = BASE_DIR / "datasets" / "raw"              # CSVs originais (não mexer)
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"  # saída já tratada


if __name__ == "__main__":
    # Monta o pipeline, aplica as etapas de limpeza e salva o resultado tratado.
    pipeline = DataPipeline(RAW_DIR / "matches.csv")
    pipeline.clean_targets().format_types().fill_missing().select_features().encode_frequencies().encode_categoricals()

    output_path = PROCESSED_DIR / "matches_clean.csv"
    pipeline.save(output_path)
    print(f"Salvo em {output_path} | shape (linhas, colunas):", pipeline.get_clean_data().shape)

    # Passo 1 da modelagem: separar features (X) do alvo (y).
    X, y = pipeline.split_features_target()
    print("X (features):", X.shape, "| y (alvo):", y.shape)

    # Passo 2: separar treino/teste por temporada (não aleatório).
    X_train, X_test, y_train, y_test = train_test_split_by_season(X, y)
    print("Treino:", X_train.shape, "| Teste:", X_test.shape)

    # Passo 3: baseline nível 0 (sempre a média) — a régua mínima a ser superada.
    baseline = train_baseline(X_train, y_train)
    mae_home, mae_away, mae_geral = evaluate_baseline(baseline, X_test, y_test)
    print("\n[Baseline nível 0 - sempre a média]")
    print(f"  MAE mandante:  {mae_home:.3f} gols")
    print(f"  MAE visitante: {mae_away:.3f} gols")
    print(f"  MAE geral:     {mae_geral:.3f} gols")

    # Passo 4: baseline nível 1 (regressão de Poisson) — primeiro modelo que aprende.
    poisson = train_poisson(X_train, y_train)
    p_home, p_away, p_geral = evaluate_poisson(poisson, X_test, y_test)
    print("\n[Baseline nível 1 - regressão de Poisson]")
    print(f"  MAE mandante:  {p_home:.3f} gols")
    print(f"  MAE visitante: {p_away:.3f} gols")
    print(f"  MAE geral:     {p_geral:.3f} gols  (régua a bater: {mae_geral:.3f})")

    # Passo 5: nível 2 (XGBoost) — modelo forte, comparado com os baselines.
    xgb = train_xgb(X_train, y_train)
    x_home, x_away, x_geral = evaluate_xgb(xgb, X_test, y_test)
    print("\n[Nível 2 - XGBoost]")
    print(f"  MAE mandante:  {x_home:.3f} gols")
    print(f"  MAE visitante: {x_away:.3f} gols")
    print(f"  MAE geral:     {x_geral:.3f} gols  (régua Poisson: {p_geral:.3f})")
