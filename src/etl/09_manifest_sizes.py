from pathlib import Path
import json
import pyarrow.parquet as pq

ROOT = Path("data/parquet_silver_v2/performance")
OUT = Path("data/logs/fase1/05_manifest.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    files = sorted(ROOT.rglob("*.parquet"))
    if not files:
        raise SystemExit("No hay archivos parquet en silver_v2")

    # schema desde el primero
    pf0 = pq.ParquetFile(files[0])
    schema = [{"name": n, "type": str(pf0.schema_arrow.field(n).type)} for n in pf0.schema_arrow.names]

    parts = []
    total_bytes = 0
    total_files = 0
    for f in files:
        st = f.stat()
        total_bytes += st.st_size
        total_files += 1
        # year=YYYY/quarter=Qn
        rel = f.relative_to(ROOT)
        year = None
        quarter = None
        for p in rel.parts:
            if p.startswith("year="): year = p.split("=",1)[1]
            if p.startswith("quarter="): quarter = p.split("=",1)[1]
        parts.append({
            "file": str(rel).replace("\\","/"),
            "year": year,
            "quarter": quarter,
            "bytes": st.st_size
        })

    OUT.write_text(json.dumps({
        "silver_v2_root": str(ROOT).replace("\\","/"),
        "n_files": total_files,
        "total_gb": round(total_bytes / 1024**3, 2),
        "columns": schema,
        "files": parts[:200]  # no lo hagas infinito en el JSON
    }, indent=2), encoding="utf-8")

    print("OK manifest:", OUT)

if __name__ == "__main__":
    main()