from __future__ import annotations

from pathlib import Path
import json
import duckdb
import pandas as pd

OUT_DIR = Path("data/gold/risk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Busca temporal parquets en ubicaciones comunes
CANDIDATE_GLOBS = [
    "data/gold/temporal/**/*.parquet",
    "data/gold/temporal/gold_temporal/**/*.parquet",
    "data/gold/gold_temporal/**/*.parquet",
]

def pick_glob() -> str:
    for g in CANDIDATE_GLOBS:
        if len(list(Path(".").glob(g))) > 0:
            return g
    raise FileNotFoundError(
        "No encontré parquets temporales en data/gold/temporal. "
        "Revisa tu ruta o ajusta CANDIDATE_GLOBS."
    )

def safe_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def safe_int(x):
    try:
        return int(x) if x is not None else 0
    except Exception:
        return 0

def score_row(r: dict) -> dict:
    """
    Heurística para sugerir columnas: period (YYYYMM), delinquency, zero balance code,
    UPB y rate. Funciona con numéricos, strings y fechas/timestamps.
    """
    typ = r["type"]
    mn = r.get("min", None)
    mx = r.get("max", None)
    uq = r.get("approx_unique", 0)
    rx_yyyymm = r.get("regex_yyyymm", 0.0)
    rx_zbc = r.get("regex_zbc", 0.0)

    s = {"period": 0, "dq": 0, "zbc": 0, "upb": 0, "rate": 0}

    # period: YYYYMM numérico
    if typ in ("BIGINT","INTEGER","SMALLINT"):
        if mn is not None and mx is not None:
            if 199000 <= mn <= 202700 and 199000 <= mx <= 202900:
                s["period"] += 6

    # period: string YYYYMM
    if typ == "VARCHAR":
        if rx_yyyymm >= 0.6:
            s["period"] += 6

    # period: timestamp/date -> ya lo convertimos a YYYYMM en stats
    if ("TIMESTAMP" in typ) or (typ == "DATE"):
        # ya viene con min/max tipo YYYYMM
        if mn is not None and mx is not None and 199000 <= mn <= 202700 and 199000 <= mx <= 202900:
            s["period"] += 7

    # delinquency: 0..12 (o algo pequeño) con baja cardinalidad
    if typ in ("BIGINT","INTEGER","SMALLINT"):
        if mn is not None and mx is not None:
            if 0 <= mn and mx <= 12 and uq <= 20:
                s["dq"] += 7

    # zero balance code: numérico 0..99 y cardinalidad pequeña
    if typ in ("BIGINT","INTEGER","SMALLINT"):
        if mn is not None and mx is not None:
            if 0 <= mn and mx <= 99 and uq <= 40:
                s["zbc"] += 5

    # zero balance code: string con regex típico (01,02,03,09,15,16,96)
    if typ == "VARCHAR":
        if rx_zbc >= 0.4:
            s["zbc"] += 7

    # UPB: valores grandes y muchos únicos (saldo)
    if typ in ("DOUBLE","FLOAT","DECIMAL","BIGINT","INTEGER"):
        if mx is not None:
            if mx >= 50000 and uq >= 1000:
                s["upb"] += 5

    # rate: 0..20 aprox y numérico continuo
    if typ in ("DOUBLE","FLOAT","DECIMAL"):
        if mn is not None and mx is not None:
            if 0.0 <= mn and mx <= 20.0:
                s["rate"] += 5

    return {f"score_{k}": v for k, v in s.items()}

def main():
    glob = pick_glob()
    print("Temporal glob:", glob)

    con = duckdb.connect()
    t = f"read_parquet('{glob}', hive_partitioning=true)"

    # schema / columns
    schema = con.execute(f"DESCRIBE SELECT * FROM {t};").fetchdf()
    cols = schema["column_name"].tolist()

    rows = []
    for c in cols:
        typ = con.execute(f"SELECT typeof({c}) FROM {t} LIMIT 1;").fetchone()[0]

        # NUMÉRICOS
        if typ in ("BIGINT","INTEGER","SMALLINT","DOUBLE","FLOAT","DECIMAL"):
            q = f"""
            SELECT
              '{c}' AS column,
              '{typ}' AS type,
              min({c}) AS min,
              max({c}) AS max,
              approx_count_distinct({c}) AS approx_unique
            FROM {t};
            """
            r = con.execute(q).fetchone()
            rows.append({
                "column": r[0],
                "type": r[1],
                "min": safe_float(r[2]),
                "max": safe_float(r[3]),
                "approx_unique": safe_int(r[4]),
                "regex_yyyymm": 0.0,
                "regex_zbc": 0.0
            })

        # TIMESTAMP / DATE (convertimos a YYYYMM por strftime)
        elif ("TIMESTAMP" in typ) or (typ == "DATE"):
            q = f"""
            SELECT
              '{c}' AS column,
              '{typ}' AS type,
              min(CAST(strftime({c}, '%Y%m') AS INTEGER)) AS min,
              max(CAST(strftime({c}, '%Y%m') AS INTEGER)) AS max,
              approx_count_distinct(strftime({c}, '%Y%m')) AS approx_unique
            FROM {t};
            """
            r = con.execute(q).fetchone()
            rows.append({
                "column": r[0],
                "type": r[1],
                "min": safe_float(r[2]),
                "max": safe_float(r[3]),
                "approx_unique": safe_int(r[4]),
                "regex_yyyymm": 1.0,  # ya es YYYYMM
                "regex_zbc": 0.0
            })

        # STRINGS U OTROS (regex sobre CAST a VARCHAR)
        else:
            q = f"""
            SELECT
              '{c}' AS column,
              '{typ}' AS type,
              NULL AS min,
              NULL AS max,
              approx_count_distinct({c}) AS approx_unique,
              avg(CASE WHEN regexp_matches(CAST({c} AS VARCHAR), '^[0-9]{{6}}$') THEN 1 ELSE 0 END) AS regex_yyyymm,
              avg(CASE WHEN regexp_matches(CAST({c} AS VARCHAR), '^(0[1-9]|1[5-6]|0[2-3]|09|96)$') THEN 1 ELSE 0 END) AS regex_zbc
            FROM {t};
            """
            r = con.execute(q).fetchone()
            rows.append({
                "column": r[0],
                "type": r[1],
                "min": None,
                "max": None,
                "approx_unique": safe_int(r[4]),
                "regex_yyyymm": float(r[5]) if r[5] is not None else 0.0,
                "regex_zbc": float(r[6]) if r[6] is not None else 0.0
            })

    df = pd.DataFrame(rows)

    # scoring
    scored = []
    for _, r in df.iterrows():
        base = r.to_dict()
        base.update(score_row(base))
        scored.append(base)

    out = pd.DataFrame(scored)
    out_csv = OUT_DIR / "temporal_field_candidates.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print("OK ->", out_csv)

    # elegir "mejor" por score (columna con score máximo)
    def best(score_col: str):
        tmp = out.sort_values(score_col, ascending=False)
        return tmp.iloc[0]["column"]

    mapping = {
        "temporal_glob": glob,
        "loan_id": "loan_id" if "loan_id" in cols else None,
        "period": best("score_period"),
        "dq": best("score_dq"),
        "zero_balance_code": best("score_zbc"),
        "upb": best("score_upb"),
        "rate": best("score_rate"),
    }

    map_path = OUT_DIR / "risk_mapping.json"
    map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print("OK ->", map_path)
    print("MAPPING SUGERIDO:\n", json.dumps(mapping, indent=2))

if __name__ == "__main__":
    main()