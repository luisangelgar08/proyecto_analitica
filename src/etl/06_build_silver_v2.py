from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

IN_ROOT = Path("data/parquet/performance")
OUT_ROOT = Path("data/parquet_silver_v2/performance")
SELECT_PATH = Path("data/logs/selected_columns.json")

BATCH_SIZE = 250_000

def parse_hive_partitions(rel_path: Path):
    # rel: year=2020/quarter=Q4/2020Q4.parquet
    parts = rel_path.parts
    year = None
    quarter = None
    for p in parts:
        if p.startswith("year="):
            year = p.split("=", 1)[1]
        elif p.startswith("quarter="):
            quarter = p.split("=", 1)[1]
    return year, quarter

def main():
    sel = json.loads(SELECT_PATH.read_text(encoding="utf-8"))

    loan_id = sel["loan_id"]     # c001
    period = sel["period"]       # c002
    drop = set(sel["drop"])      # c000

    # IMPORTANT: evitamos 'year'/'quarter' si vinieron del perfilado (DuckDB hive partitioning)
    drop |= {"year", "quarter"}

    keep_core = [loan_id, period] + sel["codes"] + sel["numeric"] + sel["text"]
    keep_core = [c for c in keep_core if c not in drop]

    # En SILVER v2 agregamos year/quarter como columnas reales (derivadas de la ruta)
    out_cols = keep_core + ["year", "quarter"]
    schema = pa.schema([(c, pa.string()) for c in out_cols])

    in_files = sorted(IN_ROOT.rglob("*.parquet"))
    print("Parquets encontrados:", len(in_files))
    print("Columnas SILVER v2 (incluye year/quarter):", len(out_cols))

    for f in tqdm(in_files, desc="Building SILVER v2"):
        rel = f.relative_to(IN_ROOT)
        year_val, quarter_val = parse_hive_partitions(rel)

        out_f = OUT_ROOT / rel
        out_f.parent.mkdir(parents=True, exist_ok=True)

        pf = pq.ParquetFile(f)
        available = set(pf.schema_arrow.names)

        present_cols = [c for c in keep_core if c in available]
        missing_cols = [c for c in keep_core if c not in available]

        writer = pq.ParquetWriter(out_f, schema=schema, compression="zstd")

        rows = 0
        try:
            for rb in pf.iter_batches(batch_size=BATCH_SIZE, columns=present_cols):
                tbl = pa.Table.from_batches([rb])

                # completar columnas faltantes con nulls (por robustez)
                for c in missing_cols:
                    tbl = tbl.append_column(c, pa.nulls(tbl.num_rows, type=pa.string()))

                # agregar particiones como columnas reales
                y_arr = pa.array([year_val] * tbl.num_rows, type=pa.string())
                q_arr = pa.array([quarter_val] * tbl.num_rows, type=pa.string())
                tbl = tbl.append_column("year", y_arr)
                tbl = tbl.append_column("quarter", q_arr)

                # asegurar orden exacto del schema
                tbl = tbl.select(out_cols)

                writer.write_table(tbl)
                rows += tbl.num_rows
        finally:
            writer.close()

        tqdm.write(f"OK {rel} | rows={rows:,}")

if __name__ == "__main__":
    main()