from __future__ import annotations
from pathlib import Path
import json
import duckdb

SELECT_JSON = Path("data/logs/selected_columns.json")
GOLD_TEMPORAL_GLOB = "data/gold/temporal/gold_temporal/year=*/quarter=*/*.parquet"
OUT_PATH = Path("data/gold/tabular/loan_features.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    sel = json.loads(SELECT_JSON.read_text(encoding="utf-8"))
    numeric_cols = sel["numeric"]
    code_cols = sel["codes"]
    text_cols = sel["text"]

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")

    con.execute(
        f"CREATE VIEW gt AS "
        f"SELECT * FROM read_parquet('{GOLD_TEMPORAL_GLOB}', hive_partitioning=false);"
    )

    agg_parts = [
        "loan_id",
        "COUNT(*) AS n_months",
        "MIN(period_date) AS min_period",
        "MAX(period_date) AS max_period",
    ]

    for c in numeric_cols:
        nc = f"num_{c}"
        agg_parts += [
            f"AVG({nc}) AS mean_{nc}",
            f"STDDEV_SAMP({nc}) AS std_{nc}",
            f"MIN({nc}) AS min_{nc}",
            f"MAX({nc}) AS max_{nc}",
        ]

    for c in code_cols:
        cc = f"cat_{c}"
        agg_parts += [
            f"arg_min({cc}, period_date) AS first_{cc}",
            f"arg_max({cc}, period_date) AS last_{cc}",
            f"approx_count_distinct({cc}) AS nunique_{cc}",
        ]

    for c in text_cols:
        tc = f"txt_{c}"
        agg_parts += [
            f"arg_min({tc}, period_date) AS first_{tc}",
            f"arg_max({tc}, period_date) AS last_{tc}",
            f"approx_count_distinct({tc}) AS nunique_{tc}",
        ]

    agg_sql = ",\n  ".join(agg_parts)

    con.execute("DROP VIEW IF EXISTS loan_features_view;")
    con.execute(f"""
        CREATE VIEW loan_features_view AS
        SELECT
          {agg_sql}
        FROM gt
        GROUP BY loan_id;
    """)

    con.execute(f"""
        COPY (SELECT * FROM loan_features_view)
        TO '{str(OUT_PATH).replace("\\\\","/")}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    n = con.execute("SELECT COUNT(*) FROM loan_features_view;").fetchone()[0]
    print("OK ->", OUT_PATH)
    print("Loans (rows):", n)

if __name__ == "__main__":
    main()