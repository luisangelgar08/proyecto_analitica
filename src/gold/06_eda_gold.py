from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TAB = "data/gold/tabular/loan_features.parquet"
FEAT_LIST = Path("data/gold/afe/afe_feature_list_22.txt")

OUT_DIR = Path("data/gold/eda/features_22")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLE = 50000

def main():
    feats = [l.strip() for l in FEAT_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not feats:
        raise ValueError("La lista de features 22 está vacía.")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")
    con.execute(f"CREATE VIEW gf AS SELECT * FROM read_parquet('{TAB}', hive_partitioning=false);")

    # ---- stats por feature (missingness + std + p01/p99) ----
    rows = []
    for c in feats:
        null_ratio = con.execute(f"SELECT avg(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) FROM gf;").fetchone()[0]
        sd = con.execute(f"SELECT stddev_samp({c}) FROM gf;").fetchone()[0]
        p01 = con.execute(f"SELECT approx_quantile({c}, 0.01) FROM gf;").fetchone()[0]
        p99 = con.execute(f"SELECT approx_quantile({c}, 0.99) FROM gf;").fetchone()[0]
        rows.append((c, float(null_ratio), float(sd) if sd is not None else np.nan,
                     float(p01) if p01 is not None else np.nan,
                     float(p99) if p99 is not None else np.nan))

    df_stats = pd.DataFrame(rows, columns=["feature", "null_ratio", "std", "p01", "p99"])
    df_stats.to_csv(OUT_DIR / "01_feature_stats_22.csv", index=False)

    # ---- sample para correlación y PCA ----
    df = con.execute("SELECT " + ", ".join(feats) + f" FROM gf USING SAMPLE {N_SAMPLE} ROWS;").df()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(df.mean(numeric_only=True))

    corr = df.corr()
    corr.to_csv(OUT_DIR / "02_corr_matrix_22.csv")

    plt.figure(figsize=(9, 7))
    plt.imshow(corr.to_numpy(), aspect="auto")
    plt.title("Correlación (22 mean_num_*)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_corr_heatmap_22.png", dpi=200)
    plt.close()

    # ---- PCA loadings ----
    Z = StandardScaler().fit_transform(df.to_numpy(dtype=np.float64))
    pca = PCA(n_components=min(10, Z.shape[1]))
    pca.fit(Z)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feats,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    loadings["loading_norm_top10"] = np.sqrt((loadings**2).sum(axis=1))
    loadings = loadings.sort_values("loading_norm_top10", ascending=False)
    loadings.to_csv(OUT_DIR / "03_pca_loading_norm_22.csv")

    top = loadings.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top["loading_norm_top10"])
    plt.title("Top 15 contribución PCA (22 features)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_top15_pca_loading_22.png", dpi=200)
    plt.close()

    print("OK ->", OUT_DIR)

if __name__ == "__main__":
    main()