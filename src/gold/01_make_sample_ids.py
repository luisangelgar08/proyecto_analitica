from __future__ import annotations
from pathlib import Path
import json
import duckdb

SILVER_GLOB = "data/parquet_silver_v2/performance/year=*/quarter=*/*.parquet"
OUT_DIR = Path("data/gold")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ajusta tamaño de muestra:
MOD = 10000
THRESH = 150  # ~1.5%

def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")

    con.execute(
        f"CREATE VIEW t AS "
        f"SELECT c001 AS loan_id "
        f"FROM read_parquet('{SILVER_GLOB}', hive_partitioning=false);"
    )

    con.execute("DROP TABLE IF EXISTS sample_ids;")
    con.execute(f"""
        CREATE TABLE sample_ids AS
        SELECT DISTINCT loan_id
        FROM t
        WHERE abs(hash(loan_id)) % {MOD} < {THRESH};
    """)

    n = con.execute("SELECT COUNT(*) FROM sample_ids;").fetchone()[0]
    print("Sample loan_id count:", n)

    con.execute("COPY sample_ids TO 'data/gold/sample_loan_ids.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);")
    con.execute("COPY sample_ids TO 'data/gold/sample_loan_ids.csv' (HEADER, DELIMITER ',');")

    params = {"mod": MOD, "thresh": THRESH, "expected_fraction": THRESH / MOD}
    (OUT_DIR / "sample_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    print("OK -> data/gold/sample_loan_ids.parquet")
    print("OK -> data/gold/sample_loan_ids.csv")
    print("OK -> data/gold/sample_params.json")

if __name__ == "__main__":
    main()