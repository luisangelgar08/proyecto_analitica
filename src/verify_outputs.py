from __future__ import annotations

from pathlib import Path
import sys

# ---- Lista de artefactos esperados (ajusta si cambiaste rutas) ----
REQUIRED = [
    # ===== VAE =====
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

    # ===== PCA / AFE =====
    "data/gold/afe/pca/pca_explained_variance.csv",
    "data/gold/afe/pca/scree_plot.png",
    "data/gold/afe/pca/pca_scores.parquet",
    "data/gold/afe/pca/pca_loadings.csv",
    "data/gold/afe/efa/efa_pc_varimax_loadings.csv",
    "data/gold/afe/efa/efa_factor_scores.parquet",
    "data/gold/afe/efa/efa_variance_by_factor.csv",

    # ===== Clustering Factor winner =====
    "data/gold/clustering/compare_spaces_best.csv",
    "data/gold/clustering/compare_spaces_silhouette.csv",
    "data/gold/clustering/kmeans_labels_winner.parquet",
    "data/gold/clustering/kmeans_centroids_winner.csv",
    "data/gold/clustering/winner_cluster_sizes.csv",
    "data/gold/clustering/winner_cluster_profile_factors.csv",
    "data/gold/clustering/winner_cluster_profile_means22.csv",

    # ===== CFA =====
    "data/gold/afe/cfa_4f_robust/cfa_model.txt",
    "data/gold/afe/cfa_4f_robust/cfa_fit_measures_fixed.csv",

    # ===== Risk validation =====
    "data/gold/risk/loan_risk_metrics.parquet",
    "data/gold/risk/vae_cluster_risk_rates.csv",
    "data/gold/risk/factor_cluster_risk_rates.csv",

    # ===== Stability =====
    "data/gold/risk/vae_cluster_share_by_first_year.csv",
    "data/gold/risk/factor_cluster_share_by_first_year.csv",
    "data/gold/risk/vae_cluster_share_by_period_year.csv",
    "data/gold/risk/factor_cluster_share_by_period_year.csv",
    "data/gold/risk/plots/vae_share_by_period_year.png",
    "data/gold/risk/plots/factor_share_by_period_year.png",
]

# Algunas rutas pueden variar según tu repo. Si quieres, agrega aquí alternativas.
OPTIONAL_ANY_OF = [
    # Ejemplo: si tu PCA/AFE vive en otra carpeta, pon alternativas.
    # ("data/gold/afe/pca/scree_plot.png", "data/gold/afe/pca22/scree_plot.png"),
]

def human_mb(nbytes: int) -> str:
    return f"{nbytes / (1024**2):.2f} MB"

def exists_any(paths: tuple[str, ...]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None

def main():
    base = Path(".").resolve()
    print("VERIFY OUTPUTS")
    print("Base:", base)
    print("-" * 60)

    ok = []
    missing = []

    # check required
    for rel in REQUIRED:
        p = Path(rel)
        if p.exists():
            ok.append(rel)
            size = p.stat().st_size
            print(f"[OK]   {rel}  ({human_mb(size)})")
        else:
            missing.append(rel)
            print(f"[MISS] {rel}")

    # check optional alternatives
    for group in OPTIONAL_ANY_OF:
        found = exists_any(group)
        if found is None:
            print(f"[ALT-MISS] none of: {group}")
        else:
            print(f"[ALT-OK]   {found}")

    print("-" * 60)
    print(f"OK: {len(ok)}")
    print(f"MISSING: {len(missing)}")

    if missing:
        print("\nMissing files (copy/paste):")
        for m in missing:
            print(" -", m)
        print("\nRESULT: FAIL ❌ (faltan artefactos)")
        sys.exit(1)

    print("\nRESULT: PASS ✅ (todo listo)")
    sys.exit(0)

if __name__ == "__main__":
    main()