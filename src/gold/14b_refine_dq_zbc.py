from __future__ import annotations
from pathlib import Path
import json
import duckdb
import pandas as pd

CANDIDATES_CSV = Path("data/gold/risk/temporal_field_candidates.csv")
MAPPING_JSON = Path("data/gold/risk/risk_mapping.json")
OUT_DIR = Path("data/gold/risk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# patrones típicos (flexibles) para detectar
DQ_REGEX = r"^(0|[1-9][0-9]{0,2}|R|XX)$"  # 0..999 o R/XX
ZBC_REGEX = r"^(0?1|0?2|0?3|0?9|15|16|96)$"  # 01/1, 02/2, 03/3, 09/9, 15,16,96

# cuántas filas muestreamos para ratios (rápido)
SAMPLE_ROWS = 200000
# candidato: columnas con cardinalidad baja (códigos)
MAX_UNIQUE_FOR_CODES = 200

def main():
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(f"No existe {CANDIDATES_CSV}. Corre primero el 14.")

    cand = pd.read_csv(CANDIDATES_CSV)

    # cargar mapping actual (tiene period_date, upb, rate ya sugeridos)
    mp = json.loads(MAPPING_JSON.read_text(encoding="utf-8"))
    glob = mp["temporal_glob"]
    t = f"read_parquet('{glob}', hive_partitioning=true)"

    # columnas candidatas a códigos: baja cardinalidad y no loan_id
    code_cands = cand[(cand["approx_unique"] <= MAX_UNIQUE_FOR_CODES) & (cand["column"] != "loan_id")].copy()
    if code_cands.empty:
        raise ValueError("No encontré columnas con baja cardinalidad para buscar DQ/ZBC.")

    con = duckdb.connect()

    results = []
    for col in code_cands["column"].tolist():
        # ratio de matches sobre muestra
        q = f"""
        WITH s AS (
          SELECT {col} AS v
          FROM {t}
          WHERE {col} IS NOT NULL
          LIMIT {SAMPLE_ROWS}
        )
        SELECT
          '{col}' AS column,
          avg(CASE WHEN regexp_matches(trim(CAST(v AS VARCHAR)), '{DQ_REGEX}') THEN 1 ELSE 0 END) AS dq_ratio,
          avg(CASE WHEN regexp_matches(trim(CAST(v AS VARCHAR)), '{ZBC_REGEX}') THEN 1 ELSE 0 END) AS zbc_ratio,
          approx_count_distinct(v) AS approx_unique_sample
        FROM s;
        """
        r = con.execute(q).fetchone()
        results.append({
            "column": r[0],
            "dq_ratio": float(r[1]) if r[1] is not None else 0.0,
            "zbc_ratio": float(r[2]) if r[2] is not None else 0.0,
            "approx_unique_sample": int(r[3]) if r[3] is not None else 0
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "dq_zbc_refine_scores.csv", index=False)

    # elegir mejores
    best_dq = df.sort_values(["dq_ratio", "approx_unique_sample"], ascending=[False, True]).iloc[0]
    best_zbc = df.sort_values(["zbc_ratio", "approx_unique_sample"], ascending=[False, True]).iloc[0]

    # umbrales mínimos para aceptar
    if best_dq["dq_ratio"] < 0.30:
        print("WARN: No encontré una columna DQ clara (dq_ratio<0.30). Probablemente DQ no está en tu temporal GOLD.")
    else:
        mp["dq"] = best_dq["column"]

    if best_zbc["zbc_ratio"] < 0.10:
        print("WARN: No encontré una columna ZBC clara (zbc_ratio<0.10). Probablemente ZBC no está en tu temporal GOLD.")
    else:
        mp["zero_balance_code"] = best_zbc["column"]

    # guardar mapping corregido
    MAPPING_JSON.write_text(json.dumps(mp, indent=2), encoding="utf-8")
    print("OK ->", OUT_DIR / "dq_zbc_refine_scores.csv")
    print("OK ->", MAPPING_JSON)
    print("MAPPING AJUSTADO:\n", json.dumps(mp, indent=2))

    # mostrar top 5 para que lo veas
    print("\nTop DQ candidates:")
    print(df.sort_values("dq_ratio", ascending=False).head(5).to_string(index=False))
    print("\nTop ZBC candidates:")
    print(df.sort_values("zbc_ratio", ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    main()