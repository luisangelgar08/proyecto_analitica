from pathlib import Path
import pyarrow.parquet as pq

# Cambia a un archivo pequeño si quieres (ej 2025Q1)
P = Path("data/parquet/performance/year=2020/quarter=Q4/2020Q4.parquet")

FIRST32 = [f"c{i:03d}" for i in range(32)]

def main():
    pf = pq.ParquetFile(P)
    print("File:", P)
    print("Rows (metadata):", pf.metadata.num_rows)
    print("Cols:", pf.schema_arrow.names[:10], "... total:", len(pf.schema_arrow.names))

    # lee poquitas filas (primer row group) solo 32 cols
    rb = next(pf.iter_batches(batch_size=20, columns=FIRST32))
    tbl = rb.to_pandas()

    print("\n--- Sample first 5 rows (first 8 cols) ---")
    print(tbl[FIRST32[:8]].head(5))

    # checks típicos:
    # c000 = loan seq (string alfanum)
    # c001 = monthly_reporting_period (YYYYMM)
    # c003 = delinquency status (0,1,2,... o RA etc)
    print("\n--- Quick sanity ---")
    print("c001 examples:", tbl["c001"].head(10).tolist())
    print("c003 examples:", tbl["c003"].head(10).tolist())

if __name__ == "__main__":
    main()