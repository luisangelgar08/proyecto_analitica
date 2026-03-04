from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

IN_SCORES = "data/gold/afe/efa/efa_factor_scores.parquet"
OUT_DIR = Path("data/gold/clustering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_FACTORS = ["F1","F2","F3","F4","F5"]
SAMPLE_N = 5000

def main():
    df = pd.read_parquet(IN_SCORES, columns=["loan_id"] + USE_FACTORS)
    df_s = df.sample(min(SAMPLE_N, len(df)), random_state=42).reset_index(drop=True)

    X = df_s[USE_FACTORS].to_numpy(dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)

    Z = linkage(Xs, method="ward")

    plt.figure(figsize=(10, 6))
    dendrogram(Z, no_labels=True, count_sort="descending")
    plt.title(f"Dendrograma Ward (muestra n={len(df_s)}) sobre F1–F5")
    plt.tight_layout()
    out_fig = OUT_DIR / "fig_ward_dendrogram_winner_F1toF5.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()

    print("OK ->", out_fig)

if __name__ == "__main__":
    main()