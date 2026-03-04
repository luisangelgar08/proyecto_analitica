from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram

IN_SCORES = Path("data/gold/afe/efa/efa_factor_scores.parquet")
OUT_DIR = Path("data/gold/clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# rango de k para evaluar
K_MIN, K_MAX = 2, 12

def main():
    df = pd.read_parquet(IN_SCORES)
    factor_cols = [c for c in df.columns if c.startswith("F")]
    X = df[factor_cols].to_numpy(dtype=np.float64)

    # estandarizar para clustering
    Xs = StandardScaler().fit_transform(X)

    # ===== A) KMeans + Silhouette =====
    ks = []
    sils = []

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        ks.append(k)
        sils.append(sil)
        print(f"k={k} silhouette={sil:.4f}")

    df_sil = pd.DataFrame({"k": ks, "silhouette": sils})
    df_sil.to_csv(OUT_DIR / "kmeans_silhouette.csv", index=False)

    plt.figure()
    plt.plot(df_sil["k"], df_sil["silhouette"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("Selección de k - KMeans (Silhouette) sobre factor scores")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_kmeans_silhouette.png", dpi=200)
    plt.close()

    best_k = int(df_sil.sort_values("silhouette", ascending=False).iloc[0]["k"])
    print("BEST k (silhouette):", best_k)

    # entrenar KMeans final
    km_final = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    df["cluster_kmeans"] = km_final.fit_predict(Xs)
    df[["loan_id", "cluster_kmeans"]].to_parquet(OUT_DIR / "kmeans_labels.parquet", index=False)

    # ===== B) Jerárquico Ward + Dendrograma =====
    # OJO: dendrograma con 206k puntos es imposible.
    # Hacemos muestra para dendrograma (válido para seleccionar estructura).
    n = len(df)
    sample_n = min(5000, n)
    samp = df.sample(sample_n, random_state=42)
    Xs_s = StandardScaler().fit_transform(samp[factor_cols].to_numpy(dtype=np.float64))

    Z = linkage(Xs_s, method="ward")
    plt.figure(figsize=(10, 6))
    dendrogram(Z, no_labels=True, count_sort="descending")
    plt.title(f"Dendrograma (Ward) sobre muestra n={sample_n}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_ward_dendrogram.png", dpi=200)
    plt.close()

    print("OK ->", OUT_DIR / "kmeans_silhouette.csv")
    print("OK ->", OUT_DIR / "fig_kmeans_silhouette.png")
    print("OK ->", OUT_DIR / "kmeans_labels.parquet")
    print("OK ->", OUT_DIR / "fig_ward_dendrogram.png")

if __name__ == "__main__":
    main()