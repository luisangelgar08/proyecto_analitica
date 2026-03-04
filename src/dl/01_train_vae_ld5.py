from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

IN_PARQUET = "data/gold/afe/afe_matrix_22.parquet"
OUT_DIR = Path("data/gold/vae_ld5_gpu")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== config =====
SEED = 42
LATENT_DIM = 5              # recomendado
HIDDEN = 128                # un poco más potente pero sigue rápido en GPU
BATCH_SIZE = 4096           # más grande para GPU (ajusta si te da OOM)
LR = 1e-3
EPOCHS = 200
BETA = 1.0
PATIENCE = 12               # early stopping
MIN_DELTA = 1e-5            # mejora mínima

# Para que sea rápido:
NUM_WORKERS = 2
PIN_MEMORY = True

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class VAE(nn.Module):
    def __init__(self, d_in: int, hidden: int, z_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_in),
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar

def loss_fn(x, xhat, mu, logvar, beta=1.0):
    recon = nn.functional.mse_loss(xhat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

def main():
    set_seed(SEED)

    # ---- GPU check ----
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    print("CUDA available:", cuda_ok)
    print("Device:", device)

    if cuda_ok:
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory/1024**3:.2f} GB")

    # ---- load data ----
    df = pd.read_parquet(IN_PARQUET)
    z_cols = [c for c in df.columns if c.startswith("z_")]
    X = df[z_cols].to_numpy(dtype=np.float32)

    # split
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.8 * n)
    tr, va = idx[:n_train], idx[n_train:]

    X_train = torch.tensor(X[tr])
    X_val = torch.tensor(X[va])

    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and cuda_ok,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and cuda_ok,
        drop_last=False,
    )

    model = VAE(d_in=X.shape[1], hidden=HIDDEN, z_dim=LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    bad = 0
    hist = []

    # (opcional) para acelerar en GPU
    torch.set_float32_matmul_precision("high")

    for ep in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tr_loss = tr_rec = tr_kl = 0.0
        tr_n = 0

        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            xhat, mu, logvar = model(xb)
            loss, recon, kl = loss_fn(xb, xhat, mu, logvar, beta=BETA)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_rec += recon.item() * bs
            tr_kl += kl.item() * bs
            tr_n += bs

        tr_loss /= tr_n
        tr_rec /= tr_n
        tr_kl /= tr_n

        # ---- val ----
        model.eval()
        va_loss = va_rec = va_kl = 0.0
        va_n = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device, non_blocking=True)
                xhat, mu, logvar = model(xb)
                loss, recon, kl = loss_fn(xb, xhat, mu, logvar, beta=BETA)
                bs = xb.size(0)
                va_loss += loss.item() * bs
                va_rec += recon.item() * bs
                va_kl += kl.item() * bs
                va_n += bs

        va_loss /= va_n
        va_rec /= va_n
        va_kl /= va_n

        # GPU memory info (opcional)
        if cuda_ok:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"epoch={ep:03d} train={tr_loss:.5f} val={va_loss:.5f} (rec={va_rec:.5f}, kl={va_kl:.5f}) | maxVRAM={mem:.2f}GB")
        else:
            print(f"epoch={ep:03d} train={tr_loss:.5f} val={va_loss:.5f} (rec={va_rec:.5f}, kl={va_kl:.5f})")

        hist.append([ep, tr_loss, tr_rec, tr_kl, va_loss, va_rec, va_kl])

        # ---- early stopping ----
        if va_loss < best_val - MIN_DELTA:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), OUT_DIR / "model.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    # ---- save history + plot ----
    hist_df = pd.DataFrame(hist, columns=[
        "epoch","train_loss","train_recon","train_kl","val_loss","val_recon","val_kl"
    ])
    hist_df.to_csv(OUT_DIR / "loss_history.csv", index=False)

    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("VAE (latent_dim=5) - loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png", dpi=200)
    plt.close()

    # ---- embeddings + recon error ----
    model.load_state_dict(torch.load(OUT_DIR / "model.pt", map_location=device))
    model.eval()

    X_all = torch.tensor(X).to(device)
    with torch.no_grad():
        xhat, mu, logvar = model(X_all)
        recon_mse = torch.mean((xhat - X_all) ** 2, dim=1).cpu().numpy()
        mu_np = mu.cpu().numpy()

    emb = pd.DataFrame(mu_np, columns=[f"z{i+1}" for i in range(LATENT_DIM)])
    emb.insert(0, "loan_id", df["loan_id"].values)
    emb.to_parquet(OUT_DIR / "vae_embeddings.parquet", index=False)

    err = pd.DataFrame({"loan_id": df["loan_id"].values, "recon_mse": recon_mse})
    err.to_parquet(OUT_DIR / "recon_error.parquet", index=False)

    print("OK ->", OUT_DIR / "model.pt")
    print("OK ->", OUT_DIR / "loss_history.csv")
    print("OK ->", OUT_DIR / "loss_curve.png")
    print("OK ->", OUT_DIR / "vae_embeddings.parquet")
    print("OK ->", OUT_DIR / "recon_error.parquet")

if __name__ == "__main__":
    main()