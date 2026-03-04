from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

SCORES = "data/gold/afe/efa/efa_factor_scores.parquet"
LABELS = "data/gold/clustering/kmeans_labels_winner.parquet"
TAB = "data/gold/tabular/loan_features.parquet"
FEAT_LIST_22 = Path("data/gold/afe/afe_feature_list_22.txt")

OUT_DIR = Path("data/gold/clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_FACTORS = ["F1", "F2", "F3", "F4", "F5"]  # ganador global

def main():
    # --- 1) cargar labels ---
    df_labels = pd.read_parquet(LABELS)  # loan_id, cluster

    # --- 2) perfil con factores (F1..F5) ---
    df_scores = pd.read_parquet(SCORES, columns=["loan_id"] + USE_FACTORS)
    df = df_scores.merge(df_labels, on="loan_id", how="inner")

    sizes = df["cluster"].value_counts().sort_index().reset_index()
    sizes.columns = ["cluster", "n_loans"]
    sizes.to_csv(OUT_DIR / "winner_cluster_sizes.csv", index=False)

    prof_f = df.groupby("cluster")[USE_FACTORS].mean().reset_index()
    prof_f.to_csv(OUT_DIR / "winner_cluster_profile_factors.csv", index=False)

    # --- 3) perfil con variables originales (las 22 mean_num_*) ---
    feats22 = [l.strip() for l in FEAT_LIST_22.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Filtrar solo columnas que existan en loan_features
    schema_cols = set(pq.ParquetFile(TAB).schema.names)
    keep_feats = [c for c in feats22 if c in schema_cols]

    df_tab = pd.read_parquet(TAB, columns=["loan_id"] + keep_feats)
    df2 = df_labels.merge(df_tab, on="loan_id", how="inner")

    prof_x = df2.groupby("cluster")[keep_feats].mean().reset_index()
    prof_x.to_csv(OUT_DIR / "winner_cluster_profile_means22.csv", index=False)

    print("OK ->", OUT_DIR / "winner_cluster_sizes.csv")
    print("OK ->", OUT_DIR / "winner_cluster_profile_factors.csv")
    print("OK ->", OUT_DIR / "winner_cluster_profile_means22.csv")

if __name__ == "__main__":
    main()