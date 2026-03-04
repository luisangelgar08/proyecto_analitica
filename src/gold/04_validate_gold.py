import duckdb

def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")

    con.execute(
        "CREATE VIEW gt AS "
        "SELECT * FROM read_parquet('data/gold/temporal/gold_temporal/year=*/quarter=*/*.parquet', hive_partitioning=false);"
    )
    con.execute(
        "CREATE VIEW gf AS "
        "SELECT * FROM read_parquet('data/gold/tabular/loan_features.parquet', hive_partitioning=false);"
    )

    print("Gold temporal rows:", con.execute("SELECT COUNT(*) FROM gt;").fetchone()[0])
    print("Gold temporal loans approx:", con.execute("SELECT approx_count_distinct(loan_id) FROM gt;").fetchone()[0])
    print("Gold tabular rows (loans):", con.execute("SELECT COUNT(*) FROM gf;").fetchone()[0])
    print("Temporal min/max period:", con.execute("SELECT MIN(period_date), MAX(period_date) FROM gt;").fetchone())

if __name__ == "__main__":
    main()