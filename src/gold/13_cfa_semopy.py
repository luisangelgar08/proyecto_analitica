from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from semopy import Model, calc_stats

TAB = "data/gold/tabular/loan_features.parquet"
OUT_DIR = Path("data/gold/afe/cfa_4f")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_N = 30000

def main():
    # Modelo 4 factores (ítems fuertes, sin year y sin F3)
    model_syntax = """
    F1 =~ mean_num_c012 + mean_num_c016 + mean_num_c017
    F2 =~ mean_num_c007 + mean_num_c008 + mean_num_c015 + mean_num_c048
    F4 =~ mean_num_c018 + mean_num_c014 + mean_num_c013
    F5 =~ mean_num_c019 + mean_num_c020 + mean_num_c032

    F1 ~~ F2
    F1 ~~ F4
    F1 ~~ F5
    F2 ~~ F4
    F2 ~~ F5
    F4 ~~ F5
    """.strip()

    (OUT_DIR / "cfa_model.txt").write_text(model_syntax, encoding="utf-8")

    vars_used = [
        "mean_num_c012","mean_num_c016","mean_num_c017",
        "mean_num_c007","mean_num_c008","mean_num_c015","mean_num_c048",
        "mean_num_c018","mean_num_c014","mean_num_c013",
        "mean_num_c019","mean_num_c020","mean_num_c032"
    ]

    schema_cols = set(pq.ParquetFile(TAB).schema.names)
    missing = [c for c in vars_used if c not in schema_cols]
    if missing:
        raise ValueError(f"Faltan columnas en loan_features.parquet: {missing}")

    df = pd.read_parquet(TAB, columns=vars_used)
    df = df.apply(pd.to_numeric, errors="coerce")
    if len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
    df = df.fillna(df.mean(numeric_only=True))

    m = Model(model_syntax)
    m.fit(df)
    stats = calc_stats(m)

    out_stats = pd.DataFrame([{
        "chi2": stats.get("chi2", np.nan),
        "dof": stats.get("DoF", np.nan),
        "p_value": stats.get("p-value", np.nan),
        "CFI": stats.get("CFI", np.nan),
        "RMSEA": stats.get("RMSEA", np.nan),
        "SRMR": stats.get("SRMR", np.nan),
    }])
    out_stats.to_csv(OUT_DIR / "cfa_fit_measures.csv", index=False)

    params = m.inspect()
    params.to_csv(OUT_DIR / "cfa_params.csv", index=False)

    print("OK ->", OUT_DIR / "cfa_model.txt")
    print("OK ->", OUT_DIR / "cfa_fit_measures.csv")
    print("OK ->", OUT_DIR / "cfa_params.csv")
    print("\nFIT:\n", out_stats.to_string(index=False))

if __name__ == "__main__":
    main()