from __future__ import annotations
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("data/gold/eda/features_28")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAB = "data/gold/tabular/loan_features.parquet"
SEL = Path("data/logs/selected_columns.json")

N_SAMPLE = 50000   # muestra para correlación/PCA (rápido)

def main():
    sel = json.loads(SEL.read_text(encoding="utf-8"))
    bases = sel["numeric"]  # tus 28 cXXX
    mean_cols = [f"mean_num_{b}" for b in bases]

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")
    con.execute(f"CREATE VIEW gf AS SELECT * FROM read_parquet('{TAB}', hive_partitioning=false);")

    # --- 1) Tabla por feature: missingness + varianza + percentiles (aprox) ---
    rows = []
    total = con.execute("SELECT COUNT(*) FROM gf;").fetchone()[0]
    for c in mean_cols:
        null_ratio = con.execute(f"SELECT avg(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) FROM gf;").fetchone()[0]
        mu = con.execute(f"SELECT avg({c}) FROM gf;").fetchone()[0]
        sd = con.execute(f"SELECT stddev_samp({c}) FROM gf;").fetchone()[0]
        # percentiles aproximados (más rápido que quantile_cont)
        p01 = con.execute(f"SELECT approx_quantile({c}, 0.01) FROM gf;").fetchone()[0]
        p99 = con.execute(f"SELECT approx_quantile({c}, 0.99) FROM gf;").fetchone()[0]
        rows.append((c, float(null_ratio), float(mu) if mu is not None else np.nan,
                     float(sd) if sd is not None else np.nan,
                     float(p01) if p01 is not None else np.nan,
                     float(p99) if p99 is not None else np.nan))

    df_feat = pd.DataFrame(rows, columns=["feature", "null_ratio", "mean", "std", "p01", "p99"])
    df_feat["base"] = df_feat["feature"].str.extract(r"(c\d{3})", expand=False)
    df_feat.to_csv(OUT_DIR / "01_feature_stats_28.csv", index=False)

    # --- 2) Correlación entre las 28 (para ver redundancia / bloques) ---
    df_sample = con.execute(
        "SELECT " + ", ".join(mean_cols) + f" FROM gf USING SAMPLE {N_SAMPLE} ROWS;"
    ).df()
    df_sample = df_sample.apply(pd.to_numeric, errors="coerce")
    df_sample = df_sample.fillna(df_sample.mean(numeric_only=True))

    corr = df_sample.corr()
    corr.to_csv(OUT_DIR / "02_corr_matrix_28.csv")

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.to_numpy(), aspect="auto")
    plt.title("Matriz de correlación (28 mean_num_*)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_corr_heatmap_28.png", dpi=200)
    plt.close()

    # --- 3) PCA loadings: qué features aportan más a la estructura latente ---
    Z = StandardScaler().fit_transform(df_sample.to_numpy(dtype=np.float64))
    pca = PCA(n_components=min(10, Z.shape[1]))
    pca.fit(Z)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=mean_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    loadings["loading_norm_top10"] = np.sqrt((loadings**2).sum(axis=1))
    loadings = loadings.sort_values("loading_norm_top10", ascending=False)

    loadings.to_csv(OUT_DIR / "03_pca_loadings_norm_28.csv")

    top = loadings.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top["loading_norm_top10"])
    plt.title("Top 15 features por contribución PCA (norma loadings)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_top15_pca_loading_norm.png", dpi=200)
    plt.close()

    # --- 4) Ranking simple (para justificar “individualmente”) ---
    # Score: prioriza baja nulidad + varianza + contribución PCA
    merged = df_feat.merge(loadings[["loading_norm_top10"]], left_on="feature", right_index=True)
    merged["score"] = (1 - merged["null_ratio"]) * merged["std"].fillna(0) * merged["loading_norm_top10"].fillna(0)
    merged = merged.sort_values("score", ascending=False)
    merged.to_csv(OUT_DIR / "04_feature_ranking_28.csv", index=False)

    print("OK ->", OUT_DIR)

if __name__ == "__main__":
    main()