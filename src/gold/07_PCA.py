from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

IN_MAT = "data/gold/afe/afe_matrix_22.parquet"
FEAT_LIST = Path("data/gold/afe/afe_feature_list_22.txt")

OUT_DIR = Path("data/gold/afe/pca")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    feats = [l.strip() for l in FEAT_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]
    z_cols = [f"z_{c}" for c in feats]

    # Cargar matriz (solo loan_id + z-features)
    con = duckdb.connect()
    df = con.execute(
        "SELECT " + ", ".join(["loan_id"] + z_cols) + f" FROM read_parquet('{IN_MAT}');"
    ).df()

    X = df[z_cols].to_numpy(dtype=np.float64)

    n_comp = min(len(z_cols), 22)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    cum = np.cumsum(explained)

    # 1) Varianza explicada
    out_ev = OUT_DIR / "pca_explained_variance.csv"
    pd.DataFrame({
        "component": np.arange(1, n_comp + 1),
        "pve": explained,
        "pve_cum": cum
    }).to_csv(out_ev, index=False)

    # 2) Scree plot
    plt.figure()
    plt.plot(np.arange(1, n_comp + 1), explained, marker="o")
    plt.xlabel("Componente")
    plt.ylabel("PVE (Varianza explicada)")
    plt.title("Scree Plot - PCA (AFE matrix 22 features)")
    plt.tight_layout()
    out_fig = OUT_DIR / "scree_plot.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()

    # 3) Scores (representación latente)
    df_scores = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, n_comp + 1)])
    df_scores.insert(0, "loan_id", df["loan_id"])
    out_scores = OUT_DIR / "pca_scores.parquet"
    df_scores.to_parquet(out_scores, index=False)

    # 4) Loadings (relación feature ↔ componente)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feats,
        columns=[f"PC{i}" for i in range(1, n_comp + 1)]
    )
    out_load = OUT_DIR / "pca_loadings.csv"
    loadings.to_csv(out_load)

    print("OK ->", out_ev)
    print("OK ->", out_fig)
    print("OK ->", out_scores)
    print("OK ->", out_load)
    print("PVE cum @5:", float(cum[4]))
    print("PVE cum @10:", float(cum[9]) if n_comp >= 10 else float(cum[-1]))

if __name__ == "__main__":
    main()