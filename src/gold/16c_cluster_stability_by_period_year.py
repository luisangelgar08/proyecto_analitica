from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

TEMP_GLOB = "data/gold/temporal/**/*.parquet"
PERIOD_COL = "period_date"  # <- el que quieres usar

VAE_LAB = "data/gold/vae_ld5_gpu/clustering/vae_kmeans_labels.parquet"
FAC_LAB = "data/gold/clustering/kmeans_labels_winner.parquet"

OUT_DIR = Path("data/gold/risk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def _make_share(df: pd.DataFrame) -> pd.DataFrame:
    # df: year, cluster, n_loans
    tot = df.groupby("year")["n_loans"].sum().reset_index(name="total_year")
    out = df.merge(tot, on="year", how="left")
    out["share"] = out["n_loans"] / out["total_year"]
    return out

def _plot_share(df_share: pd.DataFrame, title: str, out_png: Path):
    piv = df_share.pivot_table(index="year", columns="cluster", values="share", aggfunc="sum", fill_value=0)
    plt.figure()
    for c in piv.columns:
        plt.plot(piv.index, piv[c], marker="o", label=f"cluster {c}")
    plt.xlabel("Año (period_date)")
    plt.ylabel("Proporción (loans únicos)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    con = duckdb.connect()

    t = f"read_parquet('{TEMP_GLOB}', hive_partitioning=true)"

    # --- VAE ---
    q_vae = f"""
    WITH base AS (
      SELECT
        year(CAST({PERIOD_COL} AS TIMESTAMP)) AS year,
        loan_id
      FROM {t}
      WHERE {PERIOD_COL} IS NOT NULL AND loan_id IS NOT NULL
    )
    SELECT
      b.year,
      l.cluster,
      count(DISTINCT b.loan_id) AS n_loans
    FROM base b
    JOIN read_parquet('{VAE_LAB}') l
      ON b.loan_id = l.loan_id
    GROUP BY 1,2
    ORDER BY 1,2;
    """
    df_vae = con.execute(q_vae).fetchdf()
    out_vae = OUT_DIR / "vae_cluster_share_by_period_year_counts.csv"
    df_vae.to_csv(out_vae, index=False)

    df_vae_share = _make_share(df_vae)
    out_vae_share = OUT_DIR / "vae_cluster_share_by_period_year.csv"
    df_vae_share.to_csv(out_vae_share, index=False)

    _plot_share(df_vae_share, "VAE: share por año (period_date)", PLOT_DIR / "vae_share_by_period_year.png")

    # --- Factor ---
    q_fac = f"""
    WITH base AS (
      SELECT
        year(CAST({PERIOD_COL} AS TIMESTAMP)) AS year,
        loan_id
      FROM {t}
      WHERE {PERIOD_COL} IS NOT NULL AND loan_id IS NOT NULL
    )
    SELECT
      b.year,
      l.cluster,
      count(DISTINCT b.loan_id) AS n_loans
    FROM base b
    JOIN read_parquet('{FAC_LAB}') l
      ON b.loan_id = l.loan_id
    GROUP BY 1,2
    ORDER BY 1,2;
    """
    df_fac = con.execute(q_fac).fetchdf()
    out_fac = OUT_DIR / "factor_cluster_share_by_period_year_counts.csv"
    df_fac.to_csv(out_fac, index=False)

    df_fac_share = _make_share(df_fac)
    out_fac_share = OUT_DIR / "factor_cluster_share_by_period_year.csv"
    df_fac_share.to_csv(out_fac_share, index=False)

    _plot_share(df_fac_share, "Factor: share por año (period_date)", PLOT_DIR / "factor_share_by_period_year.png")

    print("OK ->", out_vae)
    print("OK ->", out_vae_share)
    print("OK ->", out_fac)
    print("OK ->", out_fac_share)
    print("OK ->", PLOT_DIR / "vae_share_by_period_year.png")
    print("OK ->", PLOT_DIR / "factor_share_by_period_year.png")

if __name__ == "__main__":
    main()