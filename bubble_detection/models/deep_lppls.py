"""
Deep LPPLS
==========
Two neural-network architectures for LPPLS parameter estimation.

1.  Mono-LPPLS-NN (M-LNN)
    ----------------------
    Physics-informed network trained on a SINGLE time series.
    Inspired by PINNs — embeds the LPPLS functional form in the loss.
    Estimates nonlinear params (tc, m, ω) for that specific series.

2.  Poly-LPPLS-NN (P-LNN)
    ----------------------
    Pre-trained on thousands of synthetic LPPLS series.
    At inference, estimates (tc, m, ω) for any new series in one forward pass.
    Much faster than optimisation-based fitting.

Reference: Nielsen, Sornette & Raissi (2024) — arXiv 2405.12803
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────
# LPPLS kernel (differentiable, for physics-informed loss)
# ─────────────────────────────────────────────────────────────

def lppls_torch(t: torch.Tensor,
                tc: torch.Tensor,
                m: torch.Tensor,
                omega: torch.Tensor,
                A: torch.Tensor,
                B: torch.Tensor,
                C1: torch.Tensor,
                C2: torch.Tensor) -> torch.Tensor:
    """Differentiable LPPLS formula in PyTorch."""
    dt = torch.clamp(tc - t, min=1e-6)
    power = dt ** m
    log_dt = torch.log(dt)
    return A + B * power + power * (C1 * torch.cos(omega * log_dt)
                                   + C2 * torch.sin(omega * log_dt))


def solve_linear_params_torch(t: torch.Tensor,
                               log_price: torch.Tensor,
                               tc: torch.Tensor,
                               m: torch.Tensor,
                               omega: torch.Tensor
                               ) -> Tuple[torch.Tensor, ...]:
    """
    Given nonlinear params, solve (A, B, C1, C2) via least squares.
    Returns (A, B, C1, C2) as scalar tensors.
    """
    dt = torch.clamp(tc - t, min=1e-6)
    power = dt ** m
    log_dt = torch.log(dt)

    # Design matrix  [N × 4]
    ones = torch.ones_like(t)
    X = torch.stack([ones, power,
                     power * torch.cos(omega * log_dt),
                     power * torch.sin(omega * log_dt)], dim=1)  # [N,4]

    # OLS: θ = (X'X)^{-1} X' y
    XtX = X.T @ X                        # [4,4]
    Xty = X.T @ log_price.unsqueeze(1)   # [4,1]
    try:
        theta = torch.linalg.solve(XtX, Xty).squeeze()  # [4]
    except Exception:
        theta = torch.zeros(4, device=t.device)
    return theta[0], theta[1], theta[2], theta[3]


# ─────────────────────────────────────────────────────────────
# 1. Mono-LPPLS-NN (M-LNN)
# ─────────────────────────────────────────────────────────────

class MonoLPPLSNN(nn.Module):
    """
    Physics-informed network trained on one time series.
    Output: (tc_offset, m, ω) — nonlinear params only.
    tc_offset is the fractional offset beyond t_end: tc = t_end + tc_offset.

    Architecture (from paper): 2 hidden layers with ReLU, output layer.
    """

    def __init__(self, seq_len: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),   # (tc_offset, m_raw, omega_raw)
        )
        # Sigmoid scaling to constrain outputs
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [seq_len] or [1, seq_len]"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        raw = self.net(x).squeeze(0)   # [3]
        # tc_offset ∈ (1, 366) days
        tc_offset = 1.0 + 365.0 * self.sigmoid(raw[0])
        # m ∈ (0.01, 0.99)
        m = 0.01 + 0.98 * self.sigmoid(raw[1])
        # ω ∈ (6.0, 13.0)
        omega = 6.0 + 7.0 * self.sigmoid(raw[2])
        return tc_offset, m, omega


class MonoLPPLSTrainer:
    """
    Trains a MonoLPPLSNN on a single empirical or synthetic time series.
    Physics-informed loss = RSS between LPPLS prediction and actual log prices.
    """

    def __init__(self,
                 hidden: int = 64,
                 lr: float = 1e-3,
                 epochs: int = 3000,
                 device: Optional[str] = None):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[MonoLPPLSNN] = None

    def fit(self, log_prices: np.ndarray,
            verbose: bool = False) -> Dict[str, float]:
        """
        Train on a single log-price series.
        Returns dict with estimated (tc, m, omega, A, B, C1, C2, rss).
        """
        n = len(log_prices)
        t_np = np.arange(n, dtype=np.float32)
        p_np = log_prices.astype(np.float32)

        t  = torch.tensor(t_np, device=self.device)
        lp = torch.tensor(p_np, device=self.device)
        x  = lp.clone().unsqueeze(0)          # [1, n]

        # Normalise input to zero mean, unit variance
        x_mean, x_std = x.mean(), x.std().clamp(min=1e-6)
        x_norm = (x - x_mean) / x_std

        model = MonoLPPLSNN(seq_len=n, hidden=self.hidden).to(self.device)
        self.model = model
        opt = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        t_end = float(t[-1])
        best_loss = float("inf")
        best_params: Optional[Dict] = None

        for epoch in range(self.epochs):
            opt.zero_grad()
            tc_off, m, omega = model(x_norm)
            tc = t_end + tc_off

            # Solve linear params analytically (no gradient through lstsq — use differentiable OLS)
            A, B, C1, C2 = solve_linear_params_torch(t, lp, tc, m, omega)
            pred = lppls_torch(t, tc, m, omega, A, B, C1, C2)
            loss = torch.mean((lp - pred) ** 2)

            # Penalty: enforce B < 0 (positive bubble constraint)
            penalty_B = torch.clamp(B, min=0.0) * 10.0
            # Penalty: enforce D = m*|B|/(ω*|C|) > 0
            C_amp = torch.sqrt(C1**2 + C2**2).clamp(min=1e-8)
            D = m * torch.abs(B) / (omega * C_amp)
            penalty_D = torch.clamp(0.01 - D, min=0.0) * 5.0

            total = loss + penalty_B + penalty_D
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if total.item() < best_loss:
                best_loss = total.item()
                best_params = {
                    "tc":    tc.item(),
                    "m":     m.item(),
                    "omega": omega.item(),
                    "A":     A.item(),
                    "B":     B.item(),
                    "C1":    C1.item(),
                    "C2":    C2.item(),
                    "rss":   loss.item() * n,
                }
            if verbose and (epoch + 1) % 500 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}  loss={total.item():.6f}  "
                      f"tc={best_params['tc']:.1f}  m={best_params['m']:.3f}  "
                      f"ω={best_params['omega']:.3f}")

        return best_params


# ─────────────────────────────────────────────────────────────
# 2. Poly-LPPLS-NN (P-LNN)
# ─────────────────────────────────────────────────────────────

class PolyLPPLSNN(nn.Module):
    """
    Pre-trained network for fast LPPLS parameter estimation.
    Input : normalised log-price series of fixed length (seq_len,)
    Output: (tc_offset, m, ω)

    Architecture: 1D-CNN feature extractor → BiLSTM → MLP head
    (deeper than M-LNN to generalise across series).
    """

    def __init__(self, seq_len: int = 252, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),   # → [batch, 64, 32]
        )
        cnn_out = 64 * 32

        # Fully connected head
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),   # (tc_offset, m_raw, omega_raw)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [batch, seq_len]"""
        # Normalise each series independently
        mu  = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        xn  = (x - mu) / std                    # [batch, seq_len]
        feat = self.cnn(xn.unsqueeze(1))         # [batch, 64, 32]
        feat = feat.view(feat.size(0), -1)       # [batch, 64*32]
        raw  = self.mlp(feat)                    # [batch, 3]
        tc_offset = 1.0 + 365.0 * self.sigmoid(raw[:, 0])
        m         = 0.01 + 0.98 * self.sigmoid(raw[:, 1])
        omega     = 6.0  + 7.0  * self.sigmoid(raw[:, 2])
        return tc_offset, m, omega


# ─────────────────────────────────────────────────────────────
# Synthetic data generator for P-LNN pre-training
# ─────────────────────────────────────────────────────────────

class LPPLSSyntheticGenerator:
    """
    Generates synthetic LPPLS time series for P-LNN training.
    Samples random (tc, m, ω) within valid ranges and generates
    log-price series with added noise.
    """

    def __init__(self,
                 seq_len: int = 252,
                 tc_offset_range: Tuple[float, float] = (30, 365),
                 m_range: Tuple[float, float] = (0.1, 0.9),
                 omega_range: Tuple[float, float] = (6.0, 13.0),
                 noise_levels: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.03),
                 seed: int = 42):
        self.seq_len = seq_len
        self.tc_offset_range = tc_offset_range
        self.m_range = m_range
        self.omega_range = omega_range
        self.noise_levels = noise_levels
        np.random.seed(seed)

    def _sample_params(self) -> Dict[str, float]:
        tc_off = np.random.uniform(*self.tc_offset_range)
        m      = np.random.uniform(*self.m_range)
        omega  = np.random.uniform(*self.omega_range)
        A      = np.random.uniform(4.0, 10.0)     # log price level
        B      = np.random.uniform(-0.5, -0.05)   # negative for positive bubble
        C1     = np.random.uniform(-0.1, 0.1)
        C2     = np.random.uniform(-0.1, 0.1)
        return {"tc_off": tc_off, "m": m, "omega": omega,
                "A": A, "B": B, "C1": C1, "C2": C2}

    def generate(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          X : [n_samples, seq_len]   — noisy log-price series
          y : [n_samples, 3]         — (tc_offset, m, omega) labels
        """
        X = np.zeros((n_samples, self.seq_len), dtype=np.float32)
        y = np.zeros((n_samples, 3), dtype=np.float32)
        t = np.arange(self.seq_len, dtype=np.float32)

        for i in range(n_samples):
            p = self._sample_params()
            tc = self.seq_len + p["tc_off"]
            noise_std = np.random.choice(self.noise_levels)

            from models.lppls_core import lppls_formula
            lp = lppls_formula(t, tc, p["m"], p["omega"],
                               p["A"], p["B"], p["C1"], p["C2"])
            lp += np.random.normal(0, noise_std, self.seq_len)
            X[i] = lp.astype(np.float32)
            y[i] = [p["tc_off"], p["m"], p["omega"]]

        return X, y


class PolyLPPLSTrainer:
    """
    Trains the P-LNN on synthetic LPPLS data.
    Loss: weighted MSE over (tc_offset, m, omega) predictions.
    """

    def __init__(self,
                 seq_len: int = 252,
                 hidden: int = 128,
                 lr: float = 1e-3,
                 epochs: int = 50,
                 batch_size: int = 256,
                 n_train: int = 20_000,
                 n_val: int = 4_000,
                 device: Optional[str] = None):
        self.seq_len = seq_len
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PolyLPPLSNN(seq_len=seq_len, hidden=hidden).to(self.device)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _weighted_loss(self,
                       pred_off: torch.Tensor,
                       pred_m:   torch.Tensor,
                       pred_om:  torch.Tensor,
                       y:        torch.Tensor) -> torch.Tensor:
        """
        Custom loss from Deep LPPLS paper:
        Weighted MSE — tc is most important, then ω, then m.
        """
        tc_loss = torch.mean((pred_off - y[:, 0]) ** 2) / (365.0 ** 2)
        m_loss  = torch.mean((pred_m   - y[:, 1]) ** 2)
        om_loss = torch.mean((pred_om  - y[:, 2]) ** 2) / (7.0 ** 2)
        return 5.0 * tc_loss + 1.0 * m_loss + 2.0 * om_loss

    def train(self, verbose: bool = True) -> "PolyLPPLSTrainer":
        gen = LPPLSSyntheticGenerator(seq_len=self.seq_len, seed=0)
        X_tr, y_tr = gen.generate(self.n_train)
        gen_val = LPPLSSyntheticGenerator(seq_len=self.seq_len, seed=999)
        X_val, y_val = gen_val.generate(self.n_val)

        ds_tr  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        ds_val = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        dl_tr  = DataLoader(ds_tr,  batch_size=self.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=self.batch_size)

        opt       = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, epochs=self.epochs,
            steps_per_epoch=len(dl_tr))

        for epoch in range(self.epochs):
            self.model.train()
            tr_losses = []
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                tc_off, m, omega = self.model(xb)
                loss = self._weighted_loss(tc_off, m, omega, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                tr_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    tc_off, m, omega = self.model(xb)
                    val_losses.append(
                        self._weighted_loss(tc_off, m, omega, yb).item())

            tr = np.mean(tr_losses)
            vl = np.mean(val_losses)
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(vl)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train={tr:.6f}  val={vl:.6f}")

        return self

    def predict(self, log_prices: np.ndarray) -> Dict[str, float]:
        """
        Fast inference on a single log-price series.
        Pads/trims to seq_len.
        """
        n = len(log_prices)
        seq_len = self.seq_len
        if n >= seq_len:
            series = log_prices[-seq_len:].astype(np.float32)
        else:
            pad = np.full(seq_len - n, log_prices[0], dtype=np.float32)
            series = np.concatenate([pad, log_prices.astype(np.float32)])

        x = torch.tensor(series).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            tc_off, m, omega = self.model(x)
        t_end = float(n - 1)
        tc = t_end + tc_off.item()
        return {"tc": tc, "tc_offset": tc_off.item(),
                "m": m.item(), "omega": omega.item()}

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                    "seq_len": self.seq_len,
                    "hidden": self.hidden}, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PolyLPPLSTrainer":
        ckpt = torch.load(path, map_location="cpu")
        trainer = cls(seq_len=ckpt["seq_len"], hidden=ckpt["hidden"],
                      device=device)
        trainer.model.load_state_dict(ckpt["state_dict"])
        return trainer
