from pathlib import Path
import json
import duckdb
import pandas as pd

MAP_PATH = Path("data/gold/risk/risk_mapping.json")
OUT_DIR = Path("data/gold/risk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAE_LAB = "data/gold/vae_ld5_gpu/clustering/vae_kmeans_labels.parquet"
FAC_LAB = "data/gold/clustering/kmeans_labels_winner.parquet"

def main():
    mp = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    glob = mp["temporal_glob"]

    loan_id = mp.get("loan_id") or "loan_id"
    period = mp["period"]
    dq = mp["dq"]
    upb = mp["upb"]
    rate = mp["rate"]

    con = duckdb.connect()
    t = f"read_parquet('{glob}', hive_partitioning=true)"

    # Métricas por préstamo usando DQ + proxy terminación: UPB==0 en algún momento
    q = f"""
    WITH base AS (
      SELECT
        {loan_id} AS loan_id,
        CAST({period} AS TIMESTAMP) AS period_dt,
        try_cast({dq} AS INTEGER) AS dq_int,
        try_cast({upb} AS DOUBLE) AS upb_num,
        try_cast({rate} AS DOUBLE) AS rate_num
      FROM {t}
      WHERE {loan_id} IS NOT NULL
    ),
    agg AS (
      SELECT
        loan_id,
        min(period_dt) AS first_period,
        max(period_dt) AS last_period,
        max(coalesce(dq_int, 0)) AS max_dq,
        max(CASE WHEN coalesce(dq_int,0) >= 1 THEN 1 ELSE 0 END) AS ever_30,
        max(CASE WHEN coalesce(dq_int,0) >= 2 THEN 1 ELSE 0 END) AS ever_60,
        max(CASE WHEN coalesce(dq_int,0) >= 3 THEN 1 ELSE 0 END) AS ever_90,
        max(CASE WHEN coalesce(dq_int,0) >= 6 THEN 1 ELSE 0 END) AS ever_180,
        avg(rate_num) AS avg_rate,
        max(upb_num) AS max_upb,
        min(upb_num) AS min_upb,
        max(CASE WHEN upb_num = 0 THEN 1 ELSE 0 END) AS terminated_upb0
      FROM base
      GROUP BY loan_id
    )
    SELECT
      *,
      CASE WHEN max_upb > 0 THEN (max_upb - min_upb) / max_upb ELSE NULL END AS amortization_ratio
    FROM agg;
    """

    df = con.execute(q).fetchdf()
    out_loan = OUT_DIR / "loan_risk_metrics.parquet"
    df.to_parquet(out_loan, index=False)
    print("OK ->", out_loan)

    # --- Tasas por cluster (VAE) ---
    vae = pd.read_parquet(VAE_LAB)
    dfv = df.merge(vae, on="loan_id", how="inner")
    vae_rates = dfv.groupby("cluster").agg(
        n_loans=("loan_id","count"),
        ever_30_rate=("ever_30","mean"),
        ever_60_rate=("ever_60","mean"),
        ever_90_rate=("ever_90","mean"),
        ever_180_rate=("ever_180","mean"),
        terminated_rate=("terminated_upb0","mean"),
        avg_rate=("avg_rate","mean"),
        avg_max_upb=("max_upb","mean"),
        avg_amort_ratio=("amortization_ratio","mean"),
    ).reset_index()
    vae_rates.to_csv(OUT_DIR / "vae_cluster_risk_rates.csv", index=False)

    # --- Tasas por cluster (Factor scores) ---
    fac = pd.read_parquet(FAC_LAB)
    dff = df.merge(fac, on="loan_id", how="inner")
    fac_rates = dff.groupby("cluster").agg(
        n_loans=("loan_id","count"),
        ever_30_rate=("ever_30","mean"),
        ever_60_rate=("ever_60","mean"),
        ever_90_rate=("ever_90","mean"),
        ever_180_rate=("ever_180","mean"),
        terminated_rate=("terminated_upb0","mean"),
        avg_rate=("avg_rate","mean"),
        avg_max_upb=("max_upb","mean"),
        avg_amort_ratio=("amortization_ratio","mean"),
    ).reset_index()
    fac_rates.to_csv(OUT_DIR / "factor_cluster_risk_rates.csv", index=False)

    print("OK ->", OUT_DIR / "vae_cluster_risk_rates.csv")
    print("OK ->", OUT_DIR / "factor_cluster_risk_rates.csv")

if __name__ == "__main__":
    main()