from pathlib import Path
import duckdb
import pandas as pd

IN_GLOB = "data/parquet_silver_v2/performance/year=*/quarter=*/*.parquet"
OUT_DIR = Path("data/logs/fase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    con = duckdb.connect()

    # Lee todos los parquets SILVER v2
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{IN_GLOB.replace('\\', '/')}');")

    # 1) Resumen por partición: filas, loans approx, cobertura temporal, nulos en llave
    q1 = """
    SELECT
      year,
      quarter,
      count(*) AS n_rows,
      approx_count_distinct(c001) AS n_loans_approx,
      min(try_strptime(c002, '%m%Y')) AS min_period,
      max(try_strptime(c002, '%m%Y')) AS max_period,
      sum(CASE WHEN c001 IS NULL OR c001='' THEN 1 ELSE 0 END) AS null_loan_id,
      sum(CASE WHEN c002 IS NULL OR c002='' THEN 1 ELSE 0 END) AS null_period
    FROM t
    GROUP BY 1,2
    ORDER BY 1,2;
    """
    df_part = con.execute(q1).df()
    df_part.to_csv(OUT_DIR / "01_partition_summary.csv", index=False)

    # 2) Cobertura global (min/max periodo + total filas + loans approx)
    q2 = """
    SELECT
      count(*) AS total_rows,
      approx_count_distinct(c001) AS total_loans_approx,
      min(try_strptime(c002, '%m%Y')) AS global_min_period,
      max(try_strptime(c002, '%m%Y')) AS global_max_period
    FROM t;
    """
    df_global = con.execute(q2).df()
    df_global.to_csv(OUT_DIR / "02_global_summary.csv", index=False)

    # 3) Duplicados (estimado) por partición: usa approx distinct del par (loan_id, period)
    # Nota: es estimación (HLL), suficiente para diagnóstico en Fase I.
    q3 = """
    SELECT
      year,
      quarter,
      count(*) AS n_rows,
      approx_count_distinct(c001 || '|' || c002) AS n_pairs_approx,
      (count(*) - approx_count_distinct(c001 || '|' || c002)) AS dup_pairs_approx
    FROM t
    GROUP BY 1,2
    ORDER BY 1,2;
    """
    df_dups = con.execute(q3).df()
    df_dups.to_csv(OUT_DIR / "03_duplicates_approx.csv", index=False)

    # 4) Missingness (muestra de 1M filas para no tardar demasiado)
    # Genera tabla col / null_ratio
    cols = con.execute("DESCRIBE SELECT * FROM t").df()["column_name"].tolist()

    # muestreo determinístico “rápido”: LIMIT 1e6
    con.execute("CREATE TEMP VIEW s AS SELECT * FROM t LIMIT 1000000;")

    rows = []
    for c in cols:
        if c in ("year","quarter"):
            # ya sabemos que deberían ser completas
            continue
        q = f"""
        SELECT
          '{c}' AS col,
          avg(CASE WHEN {c} IS NULL OR {c}='' THEN 1 ELSE 0 END) AS null_ratio
        FROM s;
        """
        null_ratio = con.execute(q).fetchone()[1]
        rows.append((c, float(null_ratio)))

    df_nulls = pd.DataFrame(rows, columns=["col", "null_ratio"]).sort_values("null_ratio", ascending=False)
    df_nulls.to_csv(OUT_DIR / "04_missingness_sample_1M.csv", index=False)

    print("OK ->", OUT_DIR)

if __name__ == "__main__":
    main()