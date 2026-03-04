from __future__ import annotations
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IN_TAB = "data/gold/tabular/loan_features.parquet"
OUT_DIR = Path("data/gold/afe/evidence")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tamaño de muestra para cálculos estadísticos (para que corra rápido)
N_SAMPLE = 50000   # puedes subir a 100000 si aguanta

def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")
    con.execute(f"CREATE VIEW gf AS SELECT * FROM read_parquet('{IN_TAB}', hive_partitioning=false);")

    cols = con.execute("DESCRIBE gf").df()["column_name"].tolist()
    all_feats = [c for c in cols if c.startswith(("mean_num_", "std_num_", "min_num_", "max_num_"))]

    # ========== E1) Missingness por feature y por familia ==========
    total = con.execute("SELECT COUNT(*) FROM gf;").fetchone()[0]

    rows = []
    for c in all_feats:
        nulls = con.execute(f"SELECT SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) FROM gf;").fetchone()[0]
        rows.append((c, nulls / total))

    df_miss = pd.DataFrame(rows, columns=["feature", "null_ratio"])
    df_miss["family"] = df_miss["feature"].str.split("_").str[0]  # mean/std/min/max
    df_miss["base"] = df_miss["feature"].str.extract(r"(c\d{3})", expand=False)

    df_miss.to_csv(OUT_DIR / "E1_missingness_by_feature.csv", index=False)

    df_miss_fam = (df_miss.groupby("family", as_index=False)
                         .agg(n_features=("feature","count"),
                              avg_null_ratio=("null_ratio","mean"),
                              max_null_ratio=("null_ratio","max"))
                         .sort_values("avg_null_ratio", ascending=True))
    df_miss_fam.to_csv(OUT_DIR / "E1_missingness_summary_by_family.csv", index=False)

    # ========== Tomar muestra para correlaciones / outliers / PCA ==========
    # DuckDB sample: rápido y sin cargar todo
    sample_sql = f"""
    SELECT {", ".join(all_feats)}
    FROM gf
    USING SAMPLE {N_SAMPLE} ROWS;
    """
    df = con.execute(sample_sql).df()

    # Convertir todo a float (pueden haber NULL)
    for c in all_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ========== E2) Redundancia: correlación dentro de cada base cXXX ==========
    # Para cada base cXXX, calculamos corr(mean, std), corr(mean, min), corr(mean, max) en la muestra.
    bases = sorted(df_miss["base"].dropna().unique().tolist())

    corr_rows = []
    for b in bases:
        m = f"mean_num_{b}"
        s = f"std_num_{b}"
        mi = f"min_num_{b}"
        ma = f"max_num_{b}"

        # solo si existen (por seguridad)
        if m not in df.columns:
            continue

        def corr(a, c):
            if a in df.columns and c in df.columns:
                x = df[[a, c]].dropna()
                if len(x) >= 500:  # mínimo para que tenga sentido
                    return float(x[a].corr(x[c]))
            return np.nan

        corr_rows.append({
            "base": b,
            "corr_mean_std": corr(m, s),
            "corr_mean_min": corr(m, mi),
            "corr_mean_max": corr(m, ma),
        })

    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(OUT_DIR / "E2_within_base_correlations.csv", index=False)

    # Resumen global de redundancia (promedios)
    df_corr_sum = pd.DataFrame([{
        "avg_corr_mean_std": float(np.nanmean(df_corr["corr_mean_std"])),
        "avg_corr_mean_min": float(np.nanmean(df_corr["corr_mean_min"])),
        "avg_corr_mean_max": float(np.nanmean(df_corr["corr_mean_max"])),
        "median_corr_mean_std": float(np.nanmedian(df_corr["corr_mean_std"])),
        "median_corr_mean_min": float(np.nanmedian(df_corr["corr_mean_min"])),
        "median_corr_mean_max": float(np.nanmedian(df_corr["corr_mean_max"])),
    }])
    df_corr_sum.to_csv(OUT_DIR / "E2_correlation_summary.csv", index=False)

    # ========== E3) Outliers (robusto) por familia ==========
    # Usamos p01/p99 para medir “extremos” por columna y luego promediamos por familia.
    out_rows = []
    for c in all_feats:
        x = df[c].dropna()
        if len(x) < 1000:
            continue
        p01 = np.nanpercentile(x, 1)
        p99 = np.nanpercentile(x, 99)
        # tasa de extremos (fuera de [p01, p99])
        rate = float(((x < p01) | (x > p99)).mean())
        out_rows.append((c, rate))

    df_out = pd.DataFrame(out_rows, columns=["feature", "extreme_rate_p01_p99"])
    df_out["family"] = df_out["feature"].str.split("_").str[0]
    df_out.to_csv(OUT_DIR / "E3_outlier_rate_by_feature.csv", index=False)

    df_out_fam = (df_out.groupby("family", as_index=False)
                        .agg(avg_extreme_rate=("extreme_rate_p01_p99","mean"),
                             max_extreme_rate=("extreme_rate_p01_p99","max"),
                             n_features=("feature","count"))
                        .sort_values("avg_extreme_rate", ascending=True))
    df_out_fam.to_csv(OUT_DIR / "E3_outlier_rate_summary_by_family.csv", index=False)

    # ========== E4) PCA comparativo (muestra) ==========
    # Comparar PCA con 28 mean features vs 112 (mean+std+min+max) en la muestra.
    # (Esto NO necesita construir matriz gigante; solo la muestra.)
    mean_cols = [f"mean_num_{b}" for b in bases if f"mean_num_{b}" in df.columns]
    full_cols = []
    for b in bases:
        for pref in ["mean_num_", "std_num_", "min_num_", "max_num_"]:
            c = f"{pref}{b}"
            if c in df.columns:
                full_cols.append(c)

    def pca_report(X: pd.DataFrame, name: str):
        # imputación simple por media + estandarización
        X2 = X.copy()
        X2 = X2.fillna(X2.mean(numeric_only=True))
        scaler = StandardScaler()
        Z = scaler.fit_transform(X2.to_numpy(dtype=np.float64))

        pca = PCA(n_components=min(20, Z.shape[1]))
        pca.fit(Z)
        explained = pca.explained_variance_ratio_
        cum = np.cumsum(explained)

        out = pd.DataFrame({
            "model": name,
            "component": np.arange(1, len(explained)+1),
            "pve": explained,
            "pve_cum": cum
        })
        return out

    rep_mean = pca_report(df[mean_cols], "PCA_28_means")
    rep_full = pca_report(df[full_cols], "PCA_112_full")

    rep = pd.concat([rep_mean, rep_full], ignore_index=True)
    rep.to_csv(OUT_DIR / "E4_pca_comparison_sample.csv", index=False)

    # Gráfico comparativo (opcional)
    plt.figure()
    for model_name, grp in rep.groupby("model"):
        plt.plot(grp["component"], grp["pve_cum"], marker="o", label=model_name)
    plt.xlabel("Componentes")
    plt.ylabel("PVE acumulada")
    plt.title("Comparación PCA (muestra): 28 means vs 112 full")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_E4_pca_cum_comparison.png", dpi=200)
    plt.close()

    print("OK evidence ->", OUT_DIR)
    print("E1 missingness summary:\n", df_miss_fam)
    print("E2 correlation summary:\n", df_corr_sum)
    print("E3 outlier summary:\n", df_out_fam)
    print("E4 PCA comparison saved:", OUT_DIR / "E4_pca_comparison_sample.csv")

if __name__ == "__main__":
    main()