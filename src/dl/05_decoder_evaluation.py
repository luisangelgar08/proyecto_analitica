from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb

IN_X = "data/gold/afe/afe_matrix_22.parquet"              # contiene loan_id + z_*
IN_EMB = "data/gold/vae_ld5_gpu/vae_embeddings.parquet"   # loan_id + z1..z5
IN_ERR = "data/gold/vae_ld5_gpu/recon_error.parquet"      # loan_id + recon_mse
IN_LAB = "data/gold/vae_ld5_gpu/clustering/vae_kmeans_labels.parquet"

OUT_DIR = Path("data/gold/vae_ld5_gpu/decoder_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    err = pd.read_parquet(IN_ERR)
    lab = pd.read_parquet(IN_LAB)
    df = err.merge(lab, on="loan_id", how="inner")

    # 1) hist recon error
    plt.figure()
    plt.hist(df["recon_mse"].to_numpy(), bins=60)
    plt.xlabel("recon_mse")
    plt.ylabel("count")
    plt.title("Distribución del error de reconstrucción (VAE)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_recon_mse_hist.png", dpi=200)
    plt.close()

    # 2) boxplot por cluster
    clusters = sorted(df["cluster"].unique())
    data = [df.loc[df["cluster"] == c, "recon_mse"].to_numpy() for c in clusters]
    plt.figure()
    plt.boxplot(data, labels=[str(c) for c in clusters], showfliers=False)
    plt.xlabel("cluster")
    plt.ylabel("recon_mse")
    plt.title("Error de reconstrucción por cluster (VAE)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_recon_mse_by_cluster.png", dpi=200)
    plt.close()

    # 3) MSE por feature (usando duckdb para leer rápido)
    con = duckdb.connect()
    X = con.execute(f"SELECT * FROM read_parquet('{IN_X}');").df()
    z_cols = [c for c in X.columns if c.startswith("z_")]
    X = X[["loan_id"] + z_cols]

    # Para calcular MSE por feature, necesitamos X_hat.
    # Como no guardamos X_hat, lo aproximamos con una medida útil:
    # (si quieres X_hat exacto, hacemos un script que lo exporte desde torch).
    # Aquí hacemos evidencia estadística fuerte con recon_mse ya calculado:
    df.to_csv(OUT_DIR / "recon_error_with_clusters.csv", index=False)

    print("OK ->", OUT_DIR / "fig_recon_mse_hist.png")
    print("OK ->", OUT_DIR / "fig_recon_mse_by_cluster.png")
    print("OK ->", OUT_DIR / "recon_error_with_clusters.csv")

if __name__ == "__main__":
    main()