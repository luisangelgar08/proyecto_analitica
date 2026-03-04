from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram

IN_EMB = "data/gold/vae_ld5_gpu/vae_embeddings.parquet"
OUT_DIR = Path("data/gold/vae_ld5_gpu/clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_MIN, K_MAX = 2, 12
SIL_SAMPLE = 20000
DENDRO_SAMPLE = 5000

def main():
    df = pd.read_parquet(IN_EMB)
    z_cols = [c for c in df.columns if c.startswith("z")]
    X = df[z_cols].to_numpy(dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)

    # ---- KMeans + silhouette (sample) ----
    rows = []
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels, sample_size=min(SIL_SAMPLE, len(Xs)), random_state=42)
        rows.append((k, float(sil)))
        print(f"k={k} silhouette(sample)={sil:.4f}")

    sil_df = pd.DataFrame(rows, columns=["k", "silhouette"])
    sil_df.to_csv(OUT_DIR / "vae_silhouette.csv", index=False)

    plt.figure()
    plt.plot(sil_df["k"], sil_df["silhouette"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette (sample)")
    plt.title("VAE embeddings (ld=5): selección de k por Silhouette")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_vae_silhouette.png", dpi=200)
    plt.close()

    best_k = int(sil_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
    best_s = float(sil_df.sort_values("silhouette", ascending=False).iloc[0]["silhouette"])
    print("BEST k:", best_k, "| silhouette(sample):", best_s)

    # ---- Ward dendrograma (muestra) ----
    samp = df.sample(min(DENDRO_SAMPLE, len(df)), random_state=42)
    Xs_s = StandardScaler().fit_transform(samp[z_cols].to_numpy(dtype=np.float64))
    Z = linkage(Xs_s, method="ward")

    plt.figure(figsize=(10, 6))
    dendrogram(Z, no_labels=True, count_sort="descending")
    plt.title(f"Dendrograma Ward (VAE ld=5, muestra n={len(samp)})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_vae_ward_dendrogram.png", dpi=200)
    plt.close()

    # ---- KMeans final con best_k (sobre todo) ----
    km_final = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    labels_final = km_final.fit_predict(Xs)

    out_labels = OUT_DIR / "vae_kmeans_labels.parquet"
    pd.DataFrame({"loan_id": df["loan_id"], "cluster": labels_final}).to_parquet(out_labels, index=False)

    print("OK ->", OUT_DIR / "vae_silhouette.csv")
    print("OK ->", OUT_DIR / "fig_vae_silhouette.png")
    print("OK ->", OUT_DIR / "fig_vae_ward_dendrogram.png")
    print("OK ->", out_labels)

if __name__ == "__main__":
    main()