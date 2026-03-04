from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IN_SCORES = "data/gold/afe/efa/efa_factor_scores.parquet"
OUT_DIR = Path("data/gold/clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GANADOR GLOBAL del comparador:
K = 3
USE_FACTORS = ["F1", "F2", "F3", "F4", "F5"]

def main():
    df = pd.read_parquet(IN_SCORES)

    missing = [c for c in USE_FACTORS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en efa_factor_scores: {missing}. Columnas disponibles: {list(df.columns)}")

    X = df[USE_FACTORS].to_numpy(dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)

    # KMeans final robusto
    km = KMeans(n_clusters=K, n_init=50, random_state=42)
    labels = km.fit_predict(Xs)

    sil = silhouette_score(Xs, labels)
    print(f"Final KMeans (winner): k={K}, silhouette={sil:.6f}")

    # Guardar labels
    out_labels = OUT_DIR / "kmeans_labels_winner.parquet"
    pd.DataFrame({"loan_id": df["loan_id"], "cluster": labels}).to_parquet(out_labels, index=False)

    # Guardar centroides (en espacio estandarizado)
    cent = pd.DataFrame(km.cluster_centers_, columns=USE_FACTORS)
    cent.insert(0, "cluster", np.arange(K))
    out_cent = OUT_DIR / "kmeans_centroids_winner.csv"
    cent.to_csv(out_cent, index=False, encoding="utf-8")

    print("OK ->", out_labels)
    print("OK ->", out_cent)

if __name__ == "__main__":
    main()