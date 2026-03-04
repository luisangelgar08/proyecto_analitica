import duckdb

con = duckdb.connect()
t = "read_parquet('data/gold/temporal/**/*.parquet', hive_partitioning=true)"
df = con.execute(f"select min(num_c007) as minv, max(num_c007) as maxv, approx_count_distinct(num_c007) as uniq from {t}").fetchdf()
print(df)