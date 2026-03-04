from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

EMB = "data/gold/vae_ld5_gpu/vae_embeddings.parquet"
LABELS = "data/gold/vae_ld5_gpu/clustering/vae_kmeans_labels.parquet"
OUT_DIR = Path("data/gold/vae_ld5_gpu/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_N = 30000  # para que el scatter no pese (206k puntos se vuelve denso)

def main():
    df_emb = pd.read_parquet(EMB)
    df_lab = pd.read_parquet(LABELS)
    df = df_emb.merge(df_lab, on="loan_id", how="inner")

    z_cols = [c for c in df.columns if c.startswith("z")]
    if len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)

    X = df[z_cols].to_numpy(dtype=np.float64)
    y = df["cluster"].to_numpy()

    # PCA 2D
    pca2 = PCA(n_components=2, random_state=42)
    X2 = pca2.fit_transform(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], s=3, alpha=0.4, c=y)
    plt.xlabel("PC1 (embeddings)")
    plt.ylabel("PC2 (embeddings)")
    plt.title(f"VAE latent space (ld=5) - PCA 2D | n={len(df)}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "vae_latent_pca2d.png", dpi=200)
    plt.close()

    # PCA 3D (opcional)
    pca3 = PCA(n_components=3, random_state=42)
    X3 = pca3.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=2, alpha=0.35, c=y)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"VAE latent space (ld=5) - PCA 3D | n={len(df)}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "vae_latent_pca3d.png", dpi=200)
    plt.close()

    # Guardar varianza explicada (para texto en el informe)
    ev = pd.DataFrame({
        "component": [1, 2, 3],
        "explained_var_ratio": np.r_[pca2.explained_variance_ratio_, pca3.explained_variance_ratio_[2]]
    })
    ev.to_csv(OUT_DIR / "vae_latent_pca_explained.csv", index=False)

    print("OK ->", OUT_DIR / "vae_latent_pca2d.png")
    print("OK ->", OUT_DIR / "vae_latent_pca3d.png")
    print("OK ->", OUT_DIR / "vae_latent_pca_explained.csv")

if __name__ == "__main__":
    main()