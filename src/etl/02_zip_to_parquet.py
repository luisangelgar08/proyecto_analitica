from __future__ import annotations

from pathlib import Path
import zipfile
import re
import time
import csv
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

try:
    import psutil
except ImportError:
    psutil = None


ZIP_PATH = Path("data/raw/performance.zip")
OUT_ROOT = Path("data/parquet/performance")
LOG_PATH = Path("data/logs/zip_to_parquet_log.csv")

# Detecta cosas tipo 2020Q1, 2003Q3, etc.
NAME_RE = re.compile(r"(?P<year>\d{4})\s*Q(?P<q>[1-4])", re.IGNORECASE)

# Por defecto: solo 2020–2025 (tu prototipo)
DEFAULT_YEARS = set(range(2020, 2026))

# Tamaño de bloque de lectura CSV (bytes). 64MB suele ir bien.
BLOCK_SIZE = 64 * 1024 * 1024

def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def mem_gb():
    if psutil is None:
        return None
    return psutil.virtual_memory().available / (1024**3)

def parse_year_quarter(filename: str):
    m = NAME_RE.search(filename)
    if not m:
        return None
    return int(m.group("year")), f"Q{int(m.group('q'))}"

def log_row(row: dict):
    file_exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)

## ------------------------------------
def detect_num_columns(z: zipfile.ZipFile, member_name: str, delim: str,
                       sample_bytes: int = 2 * 1024 * 1024) -> int:
    d = delim.encode("utf-8")
    with z.open(member_name, "r") as f:
        sample = f.read(sample_bytes)

    lines = sample.splitlines()
    # busca una línea "buena": no vacía y con muchos delimitadores
    best = 0
    for line in lines:
        if not line:
            continue
        c = line.count(d)
        if c > best:
            best = c
    # columnas = delimitadores + 1
    return best + 1
## ------------------------------------

def detect_delimiter_from_zip(z: zipfile.ZipFile, member_name: str, sample_bytes: int = 256 * 1024) -> str:
    # lee un pedazo pequeño para contar delimitadores
    with z.open(member_name, "r") as f:
        sample = f.read(sample_bytes)

    candidates = {
        ",": sample.count(b","),
        "|": sample.count(b"|"),
        "\t": sample.count(b"\t"),
        ";": sample.count(b";"),
    }
    # escoge el que más aparece
    return max(candidates, key=candidates.get)


def convert_one_csv(z: zipfile.ZipFile, member_name: str, year: int, quarter: str,
                    overwrite: bool = False, compression: str = "snappy"):
    out_dir = OUT_ROOT / f"year={year}" / f"quarter={quarter}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(member_name).stem
    out_file = out_dir / f"{stem}.parquet"
    tmp_file = out_dir / f"{stem}.parquet.tmp"

    if out_file.exists() and not overwrite:
        print(f"SKIP (ya existe): {out_file}")
        return

    delim = detect_delimiter_from_zip(z, member_name)
    ncols = detect_num_columns(z, member_name, delim)

    # nombres estables (mejor que f0/f1)
    col_names = [f"c{i:03d}" for i in range(ncols)]
    col_types = {name: pa.string() for name in col_names}

    print(f"\n>>> Convirtiendo: {member_name} (delimiter='{delim}', cols={ncols}) -> {out_file}")
    if psutil:
        print(f"RAM disponible aprox: {mem_gb():.2f} GB")

    t0 = time.time()
    start_ts = datetime.now(timezone.utc).isoformat()

    if tmp_file.exists():
        tmp_file.unlink()

    total_rows = 0
    writer = None

    try:
        with z.open(member_name, "r") as f:
            read_opts = pacsv.ReadOptions(
                block_size=BLOCK_SIZE,
                use_threads=True,
                autogenerate_column_names=False,
                column_names=col_names
            )
            parse_opts = pacsv.ParseOptions(
                delimiter=delim,
                newlines_in_values=False
            )
            convert_opts = pacsv.ConvertOptions(
                column_types=col_types,
                strings_can_be_null=True,
                # opcional: valores tratados como null
                null_values=["", "NA", "NULL", "null"]
            )

            reader = pacsv.open_csv(
                f,
                read_options=read_opts,
                parse_options=parse_opts,
                convert_options=convert_opts
            )

            schema = None
            for batch in reader:
                if schema is None:
                    schema = batch.schema
                    writer = pq.ParquetWriter(
                        tmp_file,
                        schema=schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True
                    )
                table = pa.Table.from_batches([batch], schema=schema)
                writer.write_table(table)
                total_rows += batch.num_rows

        if writer is not None:
            writer.close()
            writer = None

        if out_file.exists():
            out_file.unlink()
        tmp_file.rename(out_file)

        secs = time.time() - t0
        end_ts = datetime.now(timezone.utc).isoformat()

        log_row({
            "member_name": member_name,
            "year": year,
            "quarter": quarter,
            "delimiter": delim,
            "ncols": ncols,
            "out_file": str(out_file),
            "rows_written": total_rows,
            "seconds": round(secs, 2),
            "compression": compression,
            "started_utc": start_ts,
            "ended_utc": end_ts
        })

        print(f"OK: {out_file} | filas: {total_rows:,} | tiempo: {secs/60:.1f} min")

    except Exception as e:
        try:
            if writer is not None:
                writer.close()
        except:
            pass
        if tmp_file.exists():
            tmp_file.unlink()
        print(f"ERROR en {member_name}: {e}")
        raise
def main(years=None, overwrite=False, compression="snappy", process_all=False):
    ensure_dirs()

    if years is None:
        years = DEFAULT_YEARS

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"No encuentro el ZIP en: {ZIP_PATH.resolve()}")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = []
        for info in z.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith(".csv"):
                continue

            parsed = parse_year_quarter(info.filename)
            if not parsed:
                continue

            year, quarter = parsed
            if (not process_all) and (year not in years):
                continue

            members.append((info.filename, info.file_size, year, quarter))

        # grandes primero
        members.sort(key=lambda x: x[1], reverse=True)

        print("Archivos a convertir:", len(members))
        if not members:
            print("No encontré CSVs que cumplan el filtro.")
            return

        for name, _, y, q in members:
            convert_one_csv(z, name, y, q, overwrite=overwrite, compression=compression)

if __name__ == "__main__":
    # Config simple sin argparse (para que sea fácil). Si quieres, lo volvemos CLI luego.
    # 1) Prototipo: 2020–2025
    main(years=DEFAULT_YEARS, overwrite=False, compression="snappy", process_all=False)

    # 2) Para TODO el histórico (muchísimo más pesado), usar:
    # main(overwrite=False, compression="snappy", process_all=True)