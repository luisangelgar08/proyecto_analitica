from __future__ import annotations

from pathlib import Path
import sys
import json

import pandas as pd
import pyarrow.parquet as pq

# ---------- helpers ----------
def human_mb(nbytes: int) -> str:
    return f"{nbytes / (1024**2):.2f} MB"

def assert_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"{name}: faltan columnas {missing}. Tiene: {list(df.columns)}")

def assert_between(series: pd.Series, lo: float, hi: float, name: str):
    if series.dropna().empty:
        raise AssertionError(f"{name}: serie vacía")
    mn = float(series.min())
    mx = float(series.max())
    if mn < lo - 1e-12 or mx > hi + 1e-12:
        raise AssertionError(f"{name}: fuera de rango [{lo},{hi}] (min={mn}, max={mx})")

def assert_nonempty_file(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.stat().st_size <= 0:
        raise AssertionError(f"{path}: tamaño 0")
    return p

def read_parquet_head(path: str, n: int = 5) -> pd.DataFrame:
    # lectura ligera (pandas usa pyarrow debajo)
    return pd.read_parquet(path, engine="pyarrow").head(n)

def parquet_rows(path: str) -> int:
    return pq.ParquetFile(path).metadata.num_rows

# ---------- required files (same idea as previous) ----------
REQUIRED = [
    # VAE
    "data/gold/vae_ld5_gpu/model.pt",
    "data/gold/vae_ld5_gpu/loss_history.csv",
    "data/gold/vae_ld5_gpu/loss_curve.png",
    "data/gold/vae_ld5_gpu/vae_embeddings.parquet",
    "data/gold/vae_ld5_gpu/recon_error.parquet",
    "data/gold/vae_ld5_gpu/clustering/vae_silhouette.csv",
    "data/gold/vae_ld5_gpu/clustering/fig_vae_silhouette.png",
    "data/gold/vae_ld5_gpu/clustering/fig_vae_ward_dendrogram.png",
    "data/gold/vae_ld5_gpu/clustering/vae_kmeans_labels.parquet",
    "data/gold/vae_ld5_gpu/clustering/vae_cluster_sizes.csv",
    "data/gold/vae_ld5_gpu/clustering/vae_cluster_profile_embeddings.csv",
    "data/gold/vae_ld5_gpu/clustering/vae_cluster_profile_means22.csv",

    # PCA/AFE
    "data/gold/afe/pca/pca_explained_variance.csv",
    "data/gold/afe/pca/scree_plot.png",
    "data/gold/afe/pca/pca_scores.parquet",
    "data/gold/afe/pca/pca_loadings.csv",
    "data/gold/afe/efa/efa_pc_varimax_loadings.csv",
    "data/gold/afe/efa/efa_factor_scores.parquet",
    "data/gold/afe/efa/efa_variance_by_factor.csv",

    # Clustering factors winner
    "data/gold/clustering/compare_spaces_best.csv",
    "data/gold/clustering/compare_spaces_silhouette.csv",
    "data/gold/clustering/kmeans_labels_winner.parquet",
    "data/gold/clustering/kmeans_centroids_winner.csv",
    "data/gold/clustering/winner_cluster_sizes.csv",
    "data/gold/clustering/winner_cluster_profile_factors.csv",
    "data/gold/clustering/winner_cluster_profile_means22.csv",

    # CFA
    "data/gold/afe/cfa_4f_robust/cfa_model.txt",
    "data/gold/afe/cfa_4f_robust/cfa_fit_measures_fixed.csv",

    # Risk validation
    "data/gold/risk/loan_risk_metrics.parquet",
    "data/gold/risk/vae_cluster_risk_rates.csv",
    "data/gold/risk/factor_cluster_risk_rates.csv",

    # Stability (period_date)
    "data/gold/risk/vae_cluster_share_by_period_year.csv",
    "data/gold/risk/factor_cluster_share_by_period_year.csv",
    "data/gold/risk/plots/vae_share_by_period_year.png",
    "data/gold/risk/plots/factor_share_by_period_year.png",
]

# ---------- sanity checks ----------
def sanity_checks() -> list[str]:
    """Return list of warning strings; raise AssertionError on hard fails."""
    warns: list[str] = []

    # --- VAE embeddings ---
    emb_path = "data/gold/vae_ld5_gpu/vae_embeddings.parquet"
    n_emb = parquet_rows(emb_path)
    if n_emb < 10000:
        raise AssertionError(f"VAE embeddings: muy pocas filas ({n_emb})")
    df_emb = read_parquet_head(emb_path, 5)
    assert_cols(df_emb, ["loan_id", "z1", "z2", "z3", "z4", "z5"], "vae_embeddings")

    # --- VAE silhouette ---
    sil = pd.read_csv("data/gold/vae_ld5_gpu/clustering/vae_silhouette.csv")
    assert_cols(sil, ["k", "silhouette"], "vae_silhouette")
    assert_between(sil["silhouette"], -1.0, 1.0, "vae_silhouette.silhouette")
    best_k = int(sil.sort_values("silhouette", ascending=False).iloc[0]["k"])
    if best_k < 2:
        raise AssertionError("vae_silhouette: best_k < 2 (imposible)")
    # Si quieres forzar lo esperado:
    if best_k != 2:
        warns.append(f"VAE: best_k detectado = {best_k} (esperado 2 según tu corrida previa)")

    # --- VAE cluster sizes ---
    sz = pd.read_csv("data/gold/vae_ld5_gpu/clustering/vae_cluster_sizes.csv")
    assert_cols(sz, ["cluster", "n_loans"], "vae_cluster_sizes")
    if sz["n_loans"].sum() != n_emb:
        warns.append(f"VAE: suma n_loans ({sz['n_loans'].sum()}) != embeddings rows ({n_emb})")

    # --- Factor compare spaces ---
    csb = pd.read_csv("data/gold/clustering/compare_spaces_best.csv")
    assert_cols(csb, ["space", "k", "silhouette"], "compare_spaces_best")
    assert_between(csb["silhouette"], -1.0, 1.0, "compare_spaces_best.silhouette")

    # --- CFA measures fixed ---
    cfa = pd.read_csv("data/gold/afe/cfa_4f_robust/cfa_fit_measures_fixed.csv")
    assert_cols(cfa, ["N", "chi2", "dof", "p_value", "CFI", "RMSEA", "SRMR"], "cfa_fit_measures_fixed")
    # rangos razonables
    assert_between(cfa["CFI"], 0.0, 1.0, "CFA.CFI")
    assert_between(cfa["SRMR"], 0.0, 1.0, "CFA.SRMR")
    # RMSEA puede ser > 0.1 en tu caso, no lo marcamos como fail, solo warning
    rmsea = float(cfa["RMSEA"].iloc[0])
    if rmsea > 0.12:
        warns.append(f"CFA: RMSEA alto ({rmsea:.4f}) — OK si lo reportas como misfit absoluto y lo justificas.")

    # --- Risk rates (VAE & Factor) ---
    for name in ["vae", "factor"]:
        path = f"data/gold/risk/{name}_cluster_risk_rates.csv"
        rr = pd.read_csv(path)
        assert_cols(rr, ["cluster", "n_loans", "ever_30_rate", "ever_60_rate", "ever_90_rate", "ever_180_rate",
                         "terminated_rate", "avg_rate", "avg_max_upb", "avg_amort_ratio"], f"{name}_risk_rates")
        for c in ["ever_30_rate", "ever_60_rate", "ever_90_rate", "ever_180_rate", "terminated_rate"]:
            assert_between(rr[c], 0.0, 1.0, f"{name}.{c}")
        if rr["n_loans"].sum() <= 0:
            raise AssertionError(f"{name}_risk_rates: n_loans suma <= 0")

    # --- Stability shares by period year ---
    for name in ["vae", "factor"]:
        sp = pd.read_csv(f"data/gold/risk/{name}_cluster_share_by_period_year.csv")
        assert_cols(sp, ["year", "cluster", "n_loans", "total_year", "share"], f"{name}_share_by_period_year")
        assert_between(sp["share"], 0.0, 1.0, f"{name}.share")
        # validar que share por año sume ~1
        sums = sp.groupby("year")["share"].sum()
        bad = sums[(sums < 0.99) | (sums > 1.01)]
        if not bad.empty:
            warns.append(f"{name}: share por año no suma ~1 en años {list(bad.index)} (min={bad.min():.3f}, max={bad.max():.3f})")

    return warns

def main():
    print("VERIFY OUTPUTS + SANITY CHECKS")
    print("-" * 70)

    ok = 0
    missing = []

    for rel in REQUIRED:
        p = Path(rel)
        if p.exists() and p.stat().st_size > 0:
            ok += 1
            print(f"[OK]   {rel} ({human_mb(p.stat().st_size)})")
        else:
            missing.append(rel)
            print(f"[MISS] {rel}")

    print("-" * 70)
    print(f"Files OK: {ok}/{len(REQUIRED)}")

    if missing:
        print("\nRESULT: FAIL ❌ (faltan archivos)")
        for m in missing:
            print(" -", m)
        sys.exit(1)

    # Sanity checks
    try:
        warns = sanity_checks()
        print("\nSanity checks: PASS ✅")
        if warns:
            print("\nWarnings (no bloquean):")
            for w in warns:
                print(" -", w)
        print("\nRESULT: PASS ✅ (listo para documentación)")
        sys.exit(0)
    except Exception as e:
        print("\nSanity checks: FAIL ❌")
        print("Reason:", str(e))
        sys.exit(2)

if __name__ == "__main__":
    main()