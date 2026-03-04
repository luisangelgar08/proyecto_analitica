from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

EMB = "data/gold/vae_ld5_gpu/vae_embeddings.parquet"
LABELS = "data/gold/vae_ld5_gpu/clustering/vae_kmeans_labels.parquet"
TAB = "data/gold/tabular/loan_features.parquet"
FEAT_LIST_22 = Path("data/gold/afe/afe_feature_list_22.txt")

OUT_DIR = Path("data/gold/vae_ld5_gpu/clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading embeddings:", EMB)
    df_emb = pd.read_parquet(EMB)
    z_cols = [c for c in df_emb.columns if c.startswith("z")]
    print("Embeddings rows:", len(df_emb), "| z_cols:", z_cols)

    print("Loading labels:", LABELS)
    df_lab = pd.read_parquet(LABELS)
    print("Labels rows:", len(df_lab), "| clusters:", df_lab["cluster"].nunique())

    df = df_emb.merge(df_lab, on="loan_id", how="inner")
    print("Merged rows:", len(df))

    # tamaños
    sizes = df["cluster"].value_counts().sort_index().reset_index()
    sizes.columns = ["cluster", "n_loans"]
    out_sizes = OUT_DIR / "vae_cluster_sizes.csv"
    sizes.to_csv(out_sizes, index=False)

    # perfil embeddings
    prof_z = df.groupby("cluster")[z_cols].mean().reset_index()
    out_profz = OUT_DIR / "vae_cluster_profile_embeddings.csv"
    prof_z.to_csv(out_profz, index=False)

    # perfil variables originales (22 mean)
    feats22 = [l.strip() for l in FEAT_LIST_22.read_text(encoding="utf-8").splitlines() if l.strip()]
    schema_cols = set(pq.ParquetFile(TAB).schema.names)
    keep_feats = [c for c in feats22 if c in schema_cols]
    print("Mean features available:", len(keep_feats))

    df_tab = pd.read_parquet(TAB, columns=["loan_id"] + keep_feats)
    df2 = df_lab.merge(df_tab, on="loan_id", how="inner")
    prof_x = df2.groupby("cluster")[keep_feats].mean().reset_index()
    out_profx = OUT_DIR / "vae_cluster_profile_means22.csv"
    prof_x.to_csv(out_profx, index=False)

    print("OK ->", out_sizes)
    print("OK ->", out_profz)
    print("OK ->", out_profx)

if __name__ == "__main__":
    main()