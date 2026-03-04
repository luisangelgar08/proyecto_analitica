from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from distributed import LocalCluster, Client
import dask
import dask.bag as db
import dask.dataframe as dd
import pyarrow.parquet as pq

from dask import delayed
import dask
import numpy as np
# ---- paths ----
SILVER_ROOT = Path("data/parquet_silver_v2/performance")
OUT_DIR = Path("data/logs/fase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# glob de todos los parquets SILVER v2
ALL_GLOB = str(SILVER_ROOT / "year=*/quarter=*/*.parquet")

# partición pequeña (para evidencia de “procesamiento real”)
SMALL_FILE = SILVER_ROOT / "year=2025" / "quarter=Q1" / "2025Q1.parquet"

# ---- helpers ----
def parse_partitions(p: str):
    # .../year=2020/quarter=Q4/xxxx.parquet
    m_year = re.search(r"year=(\d{4})", p)
    m_q = re.search(r"quarter=(Q[1-4])", p)
    return (m_year.group(1) if m_year else None,
            m_q.group(1) if m_q else None)

def parquet_num_rows(path: str):
    # lee SOLO metadata (rápido)
    year, quarter = parse_partitions(path)
    n = pq.ParquetFile(path).metadata.num_rows
    return {"year": year, "quarter": quarter, "file": path, "n_rows": n}

def main():
    # ---- Dask “cluster” local ----
    cluster = LocalCluster(
        n_workers=4,          # ajusta si quieres (2–6 suele ir bien)
        threads_per_worker=1, # mejor para I/O + pyarrow
        memory_limit="3GB",   # para no reventar tu RAM
        dashboard_address=":8787",
    )
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    # =========================
    # A) SUMMARY a escala (metadata)
    # =========================
    files = [str(p).replace("\\", "/") for p in SILVER_ROOT.rglob("*.parquet")]
    b = db.from_sequence(files, npartitions=min(64, max(8, len(files))))
    meta_rows = b.map(parquet_num_rows).compute()

    df_meta = pd.DataFrame(meta_rows)
    # resumen por year/quarter
    part = (df_meta.groupby(["year","quarter"], as_index=False)["n_rows"]
                  .sum()
                  .sort_values(["year","quarter"]))

    out_a = OUT_DIR / "06_dask_metadata_partition_summary.csv"
    part.to_csv(out_a, index=False, encoding="utf-8")
    print("OK ->", out_a)

    #     # =========================
    # B) Procesamiento real (en memoria) sobre 2025Q1 (pequeño)
    #    SIN dd.read_parquet para evitar conflicto year string vs year int (hive)
    # =========================
    if SMALL_FILE.exists():
        small_path = str(SMALL_FILE).replace("\\", "/")
        pf_small = pq.ParquetFile(small_path)
        nrg = pf_small.num_row_groups
        print("Row groups en 2025Q1:", nrg)

        @delayed
        def rg_counts(path: str, rg_idx: int):
            pf = pq.ParquetFile(path)
            tbl = pf.read_row_group(rg_idx, columns=["c003"])
            s = tbl.column(0).to_pandas()
            return s.value_counts(dropna=False)

        tasks = [rg_counts(small_path, i) for i in range(nrg)]
        parts = dask.compute(*tasks)

        # combinar conteos de todos los row-groups
        if len(parts) == 1:
            total = parts[0]
        else:
            dfc = pd.concat([p.rename(f"rg{i}") for i, p in enumerate(parts)], axis=1).fillna(0)
            total = dfc.sum(axis=1).astype("int64").sort_values(ascending=False)

        df_vc = total.head(30).rename("count").reset_index().rename(columns={"index": "c003"})

        out_b = OUT_DIR / "07_dask_2025Q1_c003_counts.csv"
        df_vc.to_csv(out_b, index=False, encoding="utf-8")
        print("OK ->", out_b)
    else:
        print("WARN: No encuentro", SMALL_FILE)
    input("Dashboard activo. Abre el link y presiona Enter para cerrar...")

    client.close()
    cluster.close()
if __name__ == "__main__":
    main()