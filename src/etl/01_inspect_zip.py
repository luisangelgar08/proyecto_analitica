from pathlib import Path
import zipfile
import re

ZIP_PATH = Path("data/raw/performance.zip")  # <-- cámbialo al nombre real

year_re = re.compile(r"(20(20|21|22|23|24|25))")
# intenta detectar cuatrimestre como C1/C2/C3 o Q1/Q2/Q3 o similar
cuatri_re = re.compile(r"(C[123]|Q[123]|T[123])", re.IGNORECASE)

def guess_period(name: str):
    y = year_re.search(name)
    c = cuatri_re.search(name)
    year = y.group(1) if y else "unknown_year"
    cuatri = c.group(1).upper() if c else "unknown_period"
    return year, cuatri

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"No encuentro el ZIP en: {ZIP_PATH.resolve()}")

    total = 0
    files = []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            total += info.file_size
            year, cuatri = guess_period(info.filename)
            files.append((info.filename, info.file_size, year, cuatri))

    files.sort(key=lambda x: x[1], reverse=True)

    print("\n=== RESUMEN ZIP ===")
    print("ZIP:", ZIP_PATH)
    print("Archivos dentro:", len(files))
    print("Tamaño total (descomprimido aprox):", round(total/1024**3, 2), "GB")

    print("\n=== TOP 20 ARCHIVOS MÁS GRANDES ===")
    for f, sz, y, c in files[:20]:
        print(f"{round(sz/1024**3,2):>8} GB | {y:>10} | {c:>12} | {f}")

    # Conteo por año/periodo
    buckets = {}
    for _, sz, y, c in files:
        buckets.setdefault((y,c), 0)
        buckets[(y,c)] += sz

    print("\n=== TAMAÑO POR (AÑO, PERIODO) ===")
    for (y,c), sz in sorted(buckets.items()):
        print(f"{y:>10} | {c:>12} | {round(sz/1024**3,2):>8} GB")

if __name__ == "__main__":
    main()