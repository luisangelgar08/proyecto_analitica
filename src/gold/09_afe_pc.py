from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from factor_analyzer.rotator import Rotator

PCA_DIR = Path("data/gold/afe/pca")
OUT_DIR = Path("data/gold/afe/efa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PCA_SCORES = PCA_DIR / "pca_scores.parquet"
PCA_LOADINGS = PCA_DIR / "pca_loadings.csv"

M = 10  # elegimos m=10 por varianza acumulada ~81%

def main():
    # 1) cargar scores y loadings del PCA ya hecho
    scores = pd.read_parquet(PCA_SCORES)
    loadings = pd.read_csv(PCA_LOADINGS, index_col=0)

    pc_cols = [f"PC{i}" for i in range(1, M+1)]
    S = scores[pc_cols].to_numpy(dtype=np.float64)            # (n_loans, M)
    L = loadings[pc_cols].to_numpy(dtype=np.float64)          # (n_features, M)

    # 2) Varimax sobre loadings
    rot = Rotator(method="varimax")
    L_rot = rot.fit_transform(L)                              # (n_features, M)
    R = rot.rotation_                                         # (M, M) matriz de rotación

    # 3) Scores rotados (ortogonal): S_rot = S @ R
    S_rot = S @ R

    # 4) Guardar outputs
    out_load = OUT_DIR / "efa_pc_varimax_loadings.csv"
    out_scores = OUT_DIR / "efa_factor_scores.parquet"
    out_var = OUT_DIR / "efa_variance_by_factor.csv"

    df_load = pd.DataFrame(L_rot, index=loadings.index, columns=[f"F{i}" for i in range(1, M+1)])
    df_load.to_csv(out_load)

    df_scores = pd.DataFrame(S_rot, columns=[f"F{i}" for i in range(1, M+1)])
    df_scores.insert(0, "loan_id", scores["loan_id"])
    df_scores.to_parquet(out_scores, index=False)

    # Varianza por factor (SS loadings y proporción) — útil para el informe
    ss = (df_load**2).sum(axis=0).to_numpy()
    prop = ss / ss.sum()
    cum = np.cumsum(prop)

    pd.DataFrame({
        "factor": [f"F{i}" for i in range(1, M+1)],
        "ss_loadings": ss,
        "proportion": prop,
        "cumulative": cum
    }).to_csv(out_var, index=False)

    print("OK ->", out_load)
    print("OK ->", out_scores)
    print("OK ->", out_var)
    print("m (factores) =", M)

if __name__ == "__main__":
    main()