from pathlib import Path
import pyarrow.parquet as pq

TEST_FILES = [
    Path("data/parquet_silver_v2/performance/year=2025/quarter=Q1/2025Q1.parquet"),
    Path("data/parquet_silver_v2/performance/year=2020/quarter=Q4/2020Q4.parquet"),
]

def main():
    for p in TEST_FILES:
        pf = pq.ParquetFile(p)
        print("\nFile:", p)
        print("Rows:", pf.metadata.num_rows)
        print("Cols:", len(pf.schema_arrow.names))
        print("First cols:", pf.schema_arrow.names[:12])

        rb = next(pf.iter_batches(batch_size=5))
        df = rb.to_pandas()
        print(df[["c001","c002","year","quarter"]].head(5))

if __name__ == "__main__":
    main()