from pathlib import Path
import json
import duckdb
import pandas as pd

# Usa un archivo grande para perfilar; si quieres, cambia a uno pequeño.
SAMPLE_FILE = "data/parquet/performance/year=2020/quarter=Q4/2020Q4.parquet"
OUT_PROFILE = Path("data/logs/column_profile.csv")
OUT_SELECT = Path("data/logs/selected_columns.json")

def main():
    OUT_PROFILE.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{SAMPLE_FILE.replace('\\', '/')}');")

    # tomar una muestra manejable
    df = con.execute("SELECT * FROM t LIMIT 200000").df()

    cols = df.columns.tolist()

    rows = []
    selected = {
        "loan_id": "c001",
        "period": "c002",
        "drop": ["c000"],
        "numeric": [],
        "codes": [],
        "text": []
    }

    # regex simples
    import re
    num_re = re.compile(r"^-?\d+(\.\d+)?$")         # 123, -10, 3.14
    mmYYYY_re = re.compile(r"^\d{6}$")              # 102020 etc
    short_code_re = re.compile(r"^[A-Z0-9]{1,3}$")  # R, RA, 0, 1, 2, etc

    for c in cols:
        s = df[c].astype("string")
        non_null = s.notna() & (s != "nan") & (s != "<NA>") & (s != "")
        nn = int(non_null.sum())
        total = len(s)
        nn_ratio = nn / total if total else 0

        if nn == 0:
            rows.append([c, nn_ratio, 0, 0, 0, 0])
            continue

        s_nn = s[non_null]
        # ratios
        numeric_ratio = s_nn.map(lambda x: bool(num_re.match(str(x)))).mean()
        mmYYYY_ratio = s_nn.map(lambda x: bool(mmYYYY_re.match(str(x)))).mean()
        short_code_ratio = s_nn.map(lambda x: bool(short_code_re.match(str(x)))).mean()
        approx_unique = s_nn.nunique(dropna=True)

        rows.append([c, nn_ratio, numeric_ratio, mmYYYY_ratio, short_code_ratio, approx_unique])

        # selección automática (heurística robusta)
        if c in selected["drop"]:
            continue
        if c in (selected["loan_id"], selected["period"]):
            continue

        # numéricas “buenas”
        if nn_ratio > 0.30 and numeric_ratio > 0.98:
            selected["numeric"].append(c)
        # códigos pequeños (morosidad/flags/clases)
        elif nn_ratio > 0.30 and short_code_ratio > 0.80 and approx_unique <= 200:
            selected["codes"].append(c)
        # texto (nombres, entidades) — lo dejamos opcional, pero útil para interpretar clusters
        elif nn_ratio > 0.30 and approx_unique <= 5000:
            selected["text"].append(c)

    profile = pd.DataFrame(rows, columns=[
        "col", "non_null_ratio", "numeric_ratio", "mmYYYY_ratio", "short_code_ratio", "approx_unique"
    ]).sort_values(["numeric_ratio", "short_code_ratio", "non_null_ratio"], ascending=False)

    profile.to_csv(OUT_PROFILE, index=False, encoding="utf-8")
    OUT_SELECT.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    print("OK profile:", OUT_PROFILE)
    print("OK selected:", OUT_SELECT)
    print("\nSeleccionadas:")
    print("loan_id:", selected["loan_id"], " period:", selected["period"], " drop:", selected["drop"])
    print("numeric:", len(selected["numeric"]), " codes:", len(selected["codes"]), " text:", len(selected["text"]))

if __name__ == "__main__":
    main()