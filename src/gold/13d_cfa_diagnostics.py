from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from semopy import Model
from semopy.stats import calc_chi2, calc_dof, calc_rmsea, calc_cfi

TAB = "data/gold/tabular/loan_features.parquet"
MODEL_TXT = "data/gold/afe/cfa_4f_robust/cfa_model.txt"   # usa tu modelo robusto
OUT_DIR = Path("data/gold/afe/cfa_4f_robust")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_N = 30000

def zscore(df):
    mu = df.mean(numeric_only=True)
    sd = df.std(numeric_only=True).replace(0, np.nan)
    return ((df - mu) / sd).fillna(0)

def to_corr(S):
    d = np.sqrt(np.diag(S))
    d[d == 0] = np.nan
    return S / np.outer(d, d)

def srmr_from_corr(R, Rhat):
    # SRMR = sqrt(2/(p(p+1)) * sum_{i<=j} (R-Rhat)^2 )
    p = R.shape[0]
    tri = np.triu_indices(p)
    diff2 = (R[tri] - Rhat[tri]) ** 2
    return float(np.sqrt((2.0 / (p * (p + 1))) * diff2.sum()))

def main():
    model_syntax = Path(MODEL_TXT).read_text(encoding="utf-8")

    # columnas usadas en el modelo (observadas)
    # extra simple: busca tokens mean_num_...
    import re
    vars_used = sorted(set(re.findall(r"mean_num_c\d+|mean_num_year", model_syntax)))

    # cargar datos
    schema_cols = set(pq.ParquetFile(TAB).schema.names)
    missing = [c for c in vars_used if c not in schema_cols]
    if missing:
        raise ValueError(f"Faltan columnas en loan_features.parquet: {missing}")

    df = pd.read_parquet(TAB, columns=vars_used).apply(pd.to_numeric, errors="coerce")
    if len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
    df = df.fillna(df.mean(numeric_only=True))
    df = zscore(df)

    # fit
    m = Model(model_syntax)
    m.fit(df)

    dof = calc_dof(m)
    chi2, pval = calc_chi2(m, dof=dof)
    rmsea = calc_rmsea(m, chi2=chi2, dof=dof)
    cfi = calc_cfi(m, chi2=chi2, dof=dof)

    # SRMR manual: correlación observada vs implicada
    X = df.to_numpy(dtype=np.float64)
    S = np.cov(X, rowvar=False, bias=True)  # cov observada
    sigma_hat, _ = m.calc_sigma()           # cov implicada por el modelo
    R = to_corr(S)
    Rhat = to_corr(sigma_hat)
    srmr = srmr_from_corr(R, Rhat)

    out = pd.DataFrame([{
        "N": len(df),
        "chi2": chi2,
        "dof": dof,
        "p_value": pval,
        "CFI": cfi,
        "RMSEA": rmsea,
        "SRMR": srmr
    }])
    out.to_csv(OUT_DIR / "cfa_fit_measures_fixed.csv", index=False)
    print("OK ->", OUT_DIR / "cfa_fit_measures_fixed.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()