from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ERR = "data/gold/vae_ld5_gpu/recon_error.parquet"
LAB = "data/gold/vae_ld5_gpu/clustering/vae_kmeans_labels.parquet"
OUT_DIR = Path("data/gold/vae_ld5_gpu/decoder_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    err = pd.read_parquet(ERR)
    lab = pd.read_parquet(LAB)
    df = err.merge(lab, on="loan_id", how="inner")

    # resumen numérico
    summ = (df.groupby("cluster")["recon_mse"]
              .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
              .reset_index())
    summ.to_csv(OUT_DIR / "recon_mse_by_cluster_summary.csv", index=False)

    # hist global
    plt.figure()
    plt.hist(df["recon_mse"], bins=60)
    plt.xlabel("recon_mse")
    plt.ylabel("count")
    plt.title("Distribución del error de reconstrucción (VAE)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_recon_hist.png", dpi=200)
    plt.close()

    # boxplot por cluster
    clusters = sorted(df["cluster"].unique())
    data = [df.loc[df["cluster"] == c, "recon_mse"].to_numpy() for c in clusters]
    plt.figure()
    plt.boxplot(data, labels=[str(c) for c in clusters], showfliers=False)
    plt.xlabel("cluster")
    plt.ylabel("recon_mse")
    plt.title("Error de reconstrucción por cluster (VAE)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_recon_boxplot.png", dpi=200)
    plt.close()

    print("OK ->", OUT_DIR / "recon_mse_by_cluster_summary.csv")
    print("OK ->", OUT_DIR / "fig_recon_hist.png")
    print("OK ->", OUT_DIR / "fig_recon_boxplot.png")

if __name__ == "__main__":
    main()