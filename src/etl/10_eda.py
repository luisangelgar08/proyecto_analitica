from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("data/gold/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_TEMP = "data/gold/temporal/gold_temporal/year=*/quarter=*/*.parquet"
GOLD_TAB = "data/gold/tabular/loan_features.parquet"

def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")

    # --- Temporal ---
    con.execute(f"CREATE VIEW gt AS SELECT * FROM read_parquet('{GOLD_TEMP}', hive_partitioning=false);")

    # 1) rango + tamaño
    df_range = con.execute("""
        SELECT MIN(period_date) AS min_period,
               MAX(period_date) AS max_period,
               COUNT(*) AS n_rows,
               approx_count_distinct(loan_id) AS n_loans_approx
        FROM gt;
    """).df()
    df_range.to_csv(OUT_DIR / "01_temporal_range.csv", index=False)

    # 2) filas por mes
    df_by_month = con.execute("""
        SELECT date_trunc('month', period_date) AS month,
               COUNT(*) AS n_rows
        FROM gt
        GROUP BY 1
        ORDER BY 1;
    """).df()
    df_by_month.to_csv(OUT_DIR / "02_rows_by_month.csv", index=False)

    plt.figure()
    plt.plot(df_by_month["month"], df_by_month["n_rows"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Gold temporal: filas por mes")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_rows_by_month.png", dpi=200)
    plt.close()

    # --- Tabular ---
    con.execute(f"CREATE VIEW gf AS SELECT * FROM read_parquet('{GOLD_TAB}', hive_partitioning=false);")

    # 3) distribución de longitud de serie (n_months)
    df_nmonths = con.execute("SELECT n_months FROM gf WHERE n_months IS NOT NULL;").df()
    df_nmonths.describe().to_csv(OUT_DIR / "03_n_months_desc.csv")

    plt.figure()
    plt.hist(df_nmonths["n_months"], bins=60)
    plt.title("Distribución de duración por préstamo (n_months)")
    plt.xlabel("n_months")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_n_months_hist.png", dpi=200)
    plt.close()

    # 4) Missingness de features numéricas tabulares (top 20)
    cols = con.execute("DESCRIBE gf").df()["column_name"].tolist()
    num_cols = [c for c in cols if c.startswith(("mean_num_", "std_num_", "min_num_", "max_num_"))]

    rows = []
    for c in num_cols:
        null_ratio = con.execute(f"SELECT avg(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) FROM gf;").fetchone()[0]
        rows.append((c, float(null_ratio)))

    df_miss = pd.DataFrame(rows, columns=["feature", "null_ratio"]).sort_values("null_ratio", ascending=False)
    df_miss.to_csv(OUT_DIR / "04_missingness_numeric.csv", index=False)

    top = df_miss.head(20)
    plt.figure()
    plt.barh(top["feature"], top["null_ratio"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 missingness (features numéricas tabulares)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_missingness_top20.png", dpi=200)
    plt.close()

    print("OK ->", OUT_DIR)

if __name__ == "__main__":
    main()