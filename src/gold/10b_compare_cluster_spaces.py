from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SCORES_EFA = "data/gold/afe/efa/efa_factor_scores.parquet"
SCORES_PCA = "data/gold/afe/pca/pca_scores.parquet"

OUT_DIR = Path("data/gold/clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_MIN, K_MAX = 2, 12

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

K_MIN, K_MAX = 2, 12

def best_silhouette_fast(df, cols, tag, fit_n=30000, sil_n=10000, n_init=3):
    """
    fit_n: tamaño muestra para entrenar
    sil_n: tamaño muestra para silhouette (mucho más rápido)
    """
    # 1) muestreo para fit
    if len(df) > fit_n:
        df_fit = df.sample(fit_n, random_state=42).reset_index(drop=True)
    else:
        df_fit = df.reset_index(drop=True)

    X = df_fit[cols].to_numpy(dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)

    # 2) pre-sample para silhouette
    if len(df_fit) > sil_n:
        idx_sil = np.random.RandomState(42).choice(len(df_fit), size=sil_n, replace=False)
    else:
        idx_sil = np.arange(len(df_fit))

    Xs_sil = Xs[idx_sil]

    rows = []
    for k in range(K_MIN, K_MAX + 1):
        km = MiniBatchKMeans(
            n_clusters=k,
            n_init=n_init,
            batch_size=4096,
            random_state=42
        )
        labels = km.fit_predict(Xs)

        # silhouette en sub-muestra (rápido)
        sil = silhouette_score(Xs_sil, labels[idx_sil])
        rows.append((tag, k, float(sil)))
    return rows

def main():
    efa = pd.read_parquet(SCORES_EFA)
    pca = pd.read_parquet(SCORES_PCA)

    efa_f10 = [f"F{i}" for i in range(1, 11)]
    efa_f5  = [f"F{i}" for i in range(1, 6)]
    pca_pc10 = [f"PC{i}" for i in range(1, 11)]
    pca_pc5  = [f"PC{i}" for i in range(1, 6)]

    results = []
    results += best_silhouette_fast(efa, efa_f10, "EFA_F1-10")
    results += best_silhouette_fast(efa, efa_f5,  "EFA_F1-5")
    results += best_silhouette_fast(pca, pca_pc10, "PCA_PC1-10")
    results += best_silhouette_fast(pca, pca_pc5,  "PCA_PC1-5")

    df_res = pd.DataFrame(results, columns=["space", "k", "silhouette"])
    out_csv = OUT_DIR / "compare_spaces_silhouette.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8")

    # resumen: mejor k por espacio
    best = (df_res.sort_values(["space", "silhouette"], ascending=[True, False])
                 .groupby("space", as_index=False)
                 .head(1)
                 .sort_values("silhouette", ascending=False))
    out_best = OUT_DIR / "compare_spaces_best.csv"
    best.to_csv(out_best, index=False, encoding="utf-8")

    print("OK ->", out_csv)
    print("OK ->", out_best)
    print("\n=== BEST por espacio ===")
    print(best.to_string(index=False))

    print("\n=== GANADOR GLOBAL ===")
    print(best.iloc[0].to_string(index=False))

if __name__ == "__main__":
    main()