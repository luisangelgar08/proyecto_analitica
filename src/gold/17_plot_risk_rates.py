# src/gold/17_plot_risk_rates.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("data/gold/risk/plots")
OUT.mkdir(parents=True, exist_ok=True)

def plot_one(path, title, outname):
    df = pd.read_csv(path)
    x = df["cluster"].astype(str).tolist()
    plt.figure()
    plt.plot(x, df["ever_30_rate"], marker="o", label="Ever30+")
    plt.plot(x, df["ever_90_rate"], marker="o", label="Ever90+")
    plt.xlabel("cluster")
    plt.ylabel("rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / outname, dpi=200)
    plt.close()

plot_one("data/gold/risk/vae_cluster_risk_rates.csv",
         "VAE clusters: tasas de mora (ever)", "vae_risk_rates.png")

# si existe el de factores:
try:
    plot_one("data/gold/risk/factor_cluster_risk_rates.csv",
             "Factor clusters: tasas de mora (ever)", "factor_risk_rates.png")
except FileNotFoundError:
    pass

print("OK ->", OUT)