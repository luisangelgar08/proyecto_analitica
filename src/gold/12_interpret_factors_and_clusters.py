from pathlib import Path
import pandas as pd
import numpy as np

LOADINGS = "data/gold/afe/efa/efa_pc_varimax_loadings.csv"
SIZES = "data/gold/clustering/winner_cluster_sizes.csv"
PROF_F = "data/gold/clustering/winner_cluster_profile_factors.csv"
PROF_X = "data/gold/clustering/winner_cluster_profile_means22.csv"

OUT_DIR = Path("data/gold/clustering/interpretation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_FACTORS = ["F1","F2","F3","F4","F5"]

def main():
    # --- 1) Top loadings por factor (para poner “qué significa” cada factor) ---
    L = pd.read_csv(LOADINGS, index_col=0)  # filas = mean_num_cXXX, cols = F1..F10
    L5 = L[USE_FACTORS].copy()

    rows = []
    for f in USE_FACTORS:
        top = L5[f].abs().sort_values(ascending=False).head(6).index.tolist()
        for feat in top:
            rows.append({"factor": f, "feature": feat, "loading": float(L5.loc[feat, f]), "abs_loading": float(abs(L5.loc[feat, f]))})

    df_top = pd.DataFrame(rows).sort_values(["factor","abs_loading"], ascending=[True, False])
    df_top.to_csv(OUT_DIR / "factor_top_loadings_F1toF5.csv", index=False)

    # --- 2) Qué factores distinguen cada cluster (z-score de medias por cluster) ---
    prof = pd.read_csv(PROF_F)
    # prof: cluster, F1..F5
    mu = prof[USE_FACTORS].mean(axis=0)
    sd = prof[USE_FACTORS].std(axis=0).replace(0, np.nan)

    z = prof.copy()
    for f in USE_FACTORS:
        z[f] = (prof[f] - mu[f]) / sd[f]

    z.to_csv(OUT_DIR / "cluster_factor_zscores.csv", index=False)

    # --- 3) Juntar tamaños + factores + variables originales (para perfilar) ---
    sizes = pd.read_csv(SIZES)
    out = sizes.merge(prof, on="cluster", how="left")
    out.to_csv(OUT_DIR / "cluster_profile_factors_with_sizes.csv", index=False)

    if Path(PROF_X).exists():
        x = pd.read_csv(PROF_X)
        out2 = out.merge(x, on="cluster", how="left")
        out2.to_csv(OUT_DIR / "cluster_profile_full.csv", index=False)

    print("OK ->", OUT_DIR / "factor_top_loadings_F1toF5.csv")
    print("OK ->", OUT_DIR / "cluster_factor_zscores.csv")
    print("OK ->", OUT_DIR / "cluster_profile_factors_with_sizes.csv")
    if Path(PROF_X).exists():
        print("OK ->", OUT_DIR / "cluster_profile_full.csv")

if __name__ == "__main__":
    main()