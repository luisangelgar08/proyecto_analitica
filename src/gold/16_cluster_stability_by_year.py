from pathlib import Path
import pandas as pd

RISK = "data/gold/risk/loan_risk_metrics.parquet"
VAE_LAB = "data/gold/vae_ld5_gpu/clustering/vae_kmeans_labels.parquet"
FAC_LAB = "data/gold/clustering/kmeans_labels_winner.parquet"

OUT_DIR = Path("data/gold/risk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(RISK)
    df["first_year"] = pd.to_datetime(df["first_period"]).dt.year

    vae = pd.read_parquet(VAE_LAB)
    dv = df.merge(vae, on="loan_id", how="inner")
    pv = (dv.groupby(["first_year","cluster"])["loan_id"]
            .count()
            .reset_index(name="n"))
    pv.to_csv(OUT_DIR / "vae_cluster_share_by_first_year.csv", index=False)

    fac = pd.read_parquet(FAC_LAB)
    dfc = df.merge(fac, on="loan_id", how="inner")
    pf = (dfc.groupby(["first_year","cluster"])["loan_id"]
            .count()
            .reset_index(name="n"))
    pf.to_csv(OUT_DIR / "factor_cluster_share_by_first_year.csv", index=False)

    print("OK ->", OUT_DIR / "vae_cluster_share_by_first_year.csv")
    print("OK ->", OUT_DIR / "factor_cluster_share_by_first_year.csv")

if __name__ == "__main__":
    main()