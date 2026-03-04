import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("data/gold/risk/plots")
OUT.mkdir(parents=True, exist_ok=True)

vae = pd.read_csv("data/gold/risk/vae_cluster_risk_rates.csv")
fac = pd.read_csv("data/gold/risk/factor_cluster_risk_rates.csv")

def bar(df, title, outname):
    x = df["cluster"].astype(str).tolist()
    plt.figure()
    plt.bar([f"{c}-E90" for c in x], df["ever_90_rate"]*100)
    plt.bar([f"{c}-E180" for c in x], df["ever_180_rate"]*100)
    plt.ylabel("Rate (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / outname, dpi=200)
    plt.close()

bar(vae, "VAE clusters: Ever90 vs Ever180", "vae_ever90_ever180.png")
bar(fac, "Factor clusters: Ever90 vs Ever180", "factor_ever90_ever180.png")

print("OK ->", OUT)