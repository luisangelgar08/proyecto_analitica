from __future__ import annotations
from pathlib import Path
import json
import duckdb

SILVER_GLOB = "data/parquet_silver_v2/performance/year=*/quarter=*/*.parquet"
SAMPLE_IDS = "data/gold/sample_loan_ids.parquet"
SELECT_JSON = Path("data/logs/selected_columns.json")

OUT_DIR = Path("data/gold/temporal/gold_temporal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    sel = json.loads(SELECT_JSON.read_text(encoding="utf-8"))
    numeric_cols = sel["numeric"]
    code_cols = sel["codes"]
    text_cols = sel["text"]

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")

    con.execute(f"CREATE VIEW s AS SELECT loan_id FROM read_parquet('{SAMPLE_IDS}');")
    con.execute(
        f"CREATE VIEW t AS SELECT * "
        f"FROM read_parquet('{SILVER_GLOB}', hive_partitioning=false);"
    )

    select_parts = [
        "t.c001 AS loan_id",
        "t.c002 AS period_raw",
        "try_strptime(t.c002, '%m%Y') AS period_date",
        "try_cast(t.year AS INTEGER) AS year",
        "t.quarter AS quarter",
    ]

    for c in code_cols:
        select_parts.append(f"t.{c} AS cat_{c}")
    for c in text_cols:
        select_parts.append(f"t.{c} AS txt_{c}")
    for c in numeric_cols:
        select_parts.append(f"try_cast(t.{c} AS DOUBLE) AS num_{c}")

    select_sql = ",\n  ".join(select_parts)

    con.execute("DROP VIEW IF EXISTS gold_temporal_view;")
    con.execute(f"""
        CREATE VIEW gold_temporal_view AS
        SELECT
          {select_sql}
        FROM t
        INNER JOIN s ON t.c001 = s.loan_id
        WHERE try_strptime(t.c002, '%m%Y') IS NOT NULL
          AND try_cast(t.year AS INTEGER) IS NOT NULL;
    """)

    out_path = str(OUT_DIR).replace("\\", "/")
    con.execute(f"""
        COPY (SELECT * FROM gold_temporal_view)
        TO '{out_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (year, quarter));
    """)

    rows = con.execute("SELECT COUNT(*) FROM gold_temporal_view;").fetchone()[0]
    loans = con.execute("SELECT approx_count_distinct(loan_id) FROM gold_temporal_view;").fetchone()[0]
    print("OK ->", OUT_DIR)
    print("Rows:", rows)
    print("Loans approx:", loans)

if __name__ == "__main__":
    main()