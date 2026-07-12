"""
HLPPL — Hyped Log-Periodic Power Law Model
==========================================
Reference: Cao, Shao, Yan & Geman (2025) — arXiv 2510.10878

Architecture
------------
HLPPL has two distinct phases:

PHASE 1 — BubbleScore construction (deterministic)
  • Fit LPPLS → extract normalised residual ε_norm(t)
  • Compute Hype Index H(t)  — media-attention proxy (e.g. volume z-score)
  • Compute Sentiment Score S(t) — text/NLP proxy (e.g. RSI-based or real NLP)
  • BubbleScore(t) = ε_norm(t) ± α₁·H(t) + α₂·S(t)
        + if ε_norm > 0  (positive bubble)
        - if ε_norm < 0  (negative bubble)

PHASE 2 — Dual-Stream Transformer (predictive)
  Two parallel input streams → two Transformer encoders → fusion → output
    Stream 1 (price stream)    : closing price, volume, log-return, volatility
    Stream 2 (signal stream)   : BubbleScore, LPPL residual, Hype, Sentiment
  The encoders share the same architecture but have separate weights.
  Fusion: concatenate CLS tokens → MLP → scalar BubbleScore prediction

Output: continuous BubbleScore ∈ ℝ
  Sign   → bubble direction (positive = overvalued, negative = undervalued)
  Magnitude → bubble intensity

Label generation (training)
  BubbleScore labels come from Phase 1.
  The transformer learns to predict FUTURE BubbleScore from PAST features.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, List, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.lppls_core import LPPLSFitter, lppls_formula


# ─────────────────────────────────────────────────────────────
# Phase 1 helpers
# ─────────────────────────────────────────────────────────────

def compute_lppl_residual(log_prices: np.ndarray,
                          window: int = 252,
                          step: int = 1) -> np.ndarray:
    """
    Rolling LPPLS residual ε(t):
      For each day t, fit LPPLS on the preceding `window` days.
      ε(t) = log_price(t) - LPPLS_fit(t)   [in-sample last-point residual]
    Returns ε_norm(t) normalised to [-1, 1] via rolling z-score.
    """
    n = len(log_prices)
    residual = np.full(n, np.nan)
    fitter = LPPLSFitter(method="nm", n_restarts=3, maxiter=1000)

    for t in range(window, n, step):
        t_win = np.arange(window, dtype=float)
        p_win = log_prices[t - window: t]
        fit = fitter.fit(t_win, p_win)
        if fit is not None:
            from models.lppls_core import lppls_formula
            pred = lppls_formula(
                t_win, fit.tc, fit.m, fit.omega,
                fit.A, fit.B, fit.C1, fit.C2)
            # last point residual
            residual[t - 1] = p_win[-1] - pred[-1]

    # Forward-fill NaNs; if nothing computed, use zero baseline
    for i in range(1, n):
        if np.isnan(residual[i]):
            residual[i] = residual[i - 1] if not np.isnan(residual[i - 1]) else 0.0
    residual = np.nan_to_num(residual, nan=0.0)

    # Normalise with rolling z-score (window=63)
    eps_norm = np.zeros(n)
    roll_w = 63
    for i in range(n):
        start = max(0, i - roll_w + 1)
        chunk = residual[start: i + 1]
        mu  = np.nanmean(chunk) if not np.all(np.isnan(chunk)) else 0.0
        std = np.nanstd(chunk) if not np.all(np.isnan(chunk)) else 0.0
        if std < 1e-8:
            eps_norm[i] = 0.0
        else:
            eps_norm[i] = np.clip((residual[i] - mu) / std, -3.0, 3.0) / 3.0

    return eps_norm          # ∈ [-1, 1]


def compute_hype_index(volume: np.ndarray,
                       window: int = 21) -> np.ndarray:
    """
    Hype Index H(t): volume-based proxy for media/retail attention.
    H(t) = z-score of log(volume) over trailing `window` days → clipped [0, 1].
    In production, replace with actual Google Trends or news-count data.
    """
    log_vol = np.log(np.maximum(volume, 1.0))
    H = np.zeros(len(log_vol))
    for i in range(len(log_vol)):
        start = max(0, i - window + 1)
        chunk = log_vol[start: i + 1]
        mu  = chunk.mean()
        std = chunk.std()
        if std < 1e-8:
            H[i] = 0.0
        else:
            z = (log_vol[i] - mu) / std
            H[i] = float(np.clip(z / 3.0, 0.0, 1.0))   # only positive spikes = hype
    return H


def compute_sentiment_score(close: np.ndarray,
                            window_fast: int = 14,
                            window_slow: int = 50) -> np.ndarray:
    """
    Sentiment proxy S(t) from price action (RSI-based).
    Range [-1, 1] where +1 = extreme bullish sentiment, -1 = extreme bearish.
    In production: replace with FinBERT / VADER on news headlines.
    """
    n = len(close)
    rsi = np.zeros(n)
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.abs(np.minimum(delta, 0.0))

    for i in range(window_fast, n):
        avg_gain = gain[i - window_fast: i].mean()
        avg_loss = loss[i - window_fast: i].mean()
        if avg_loss < 1e-10:
            rsi[i] = 1.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = rs / (1.0 + rs)    # ∈ [0, 1]

    # Map RSI → sentiment: RSI=1 → S=+1, RSI=0 → S=-1
    sentiment = 2.0 * rsi - 1.0

    # Dampen with slow trend: if price < slow MA, reduce positive sentiment
    slow_ma = np.convolve(close, np.ones(window_slow) / window_slow, mode="same")
    trend = np.where(close > slow_ma, 1.0, -1.0)
    sentiment = 0.7 * sentiment + 0.3 * np.sign(trend)
    return np.clip(sentiment, -1.0, 1.0)


def compute_bubble_score(eps_norm: np.ndarray,
                         hype:     np.ndarray,
                         sentiment: np.ndarray,
                         alpha1: float = 0.3,
                         alpha2: float = 0.2) -> np.ndarray:
    """
    HLPPL BubbleScore formula (Cao et al. 2025):

        BubbleScore(t) =
            ε_norm(t) + α₁·H(t) + α₂·S(t)    if ε_norm(t) > 0  (positive bubble)
            ε_norm(t) - α₁·H(t) + α₂·S(t)    if ε_norm(t) < 0  (negative bubble)

    Sign of BubbleScore:  positive → overpriced bubble
                          negative → underpriced / negative bubble
    """
    bubble_score = np.where(
        eps_norm > 0,
        eps_norm + alpha1 * hype + alpha2 * sentiment,
        eps_norm - alpha1 * hype + alpha2 * sentiment,
    )
    return bubble_score


# ─────────────────────────────────────────────────────────────
# Phase 2 — Dual-Stream Transformer
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StreamEncoder(nn.Module):
    """
    Single-stream Transformer encoder.
    Input : [batch, seq_len, n_features]
    Output: [batch, d_model]  (CLS token representation)
    """

    def __init__(self,
                 n_features: int,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, n_features]"""
        x = self.input_proj(x)                           # [B, T, d_model]
        x = self.pos_enc(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, d_model]
        x   = torch.cat([cls, x], dim=1)                 # [B, T+1, d_model]
        x   = self.transformer(x)                        # [B, T+1, d_model]
        return x[:, 0, :]                                 # CLS → [B, d_model]


class DualStreamTransformer(nn.Module):
    """
    Dual-stream Transformer for BubbleScore prediction.

    Stream 1 (price stream):  [close, volume, log_return, realized_vol, ...] × seq_len
    Stream 2 (signal stream): [eps_norm, bubble_score, hype, sentiment]       × seq_len

    Fusion: concat CLS → LayerNorm → MLP → scalar output
    """

    PRICE_FEATURES  = 4    # close_norm, vol_norm, log_ret, realised_vol
    SIGNAL_FEATURES = 4    # eps_norm, bubble_score, hype, sentiment

    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.price_encoder  = StreamEncoder(self.PRICE_FEATURES,
                                            d_model, n_heads, n_layers, dropout)
        self.signal_encoder = StreamEncoder(self.SIGNAL_FEATURES,
                                            d_model, n_heads, n_layers, dropout)
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),    # scalar BubbleScore
        )

    def forward(self,
                x_price:  torch.Tensor,
                x_signal: torch.Tensor) -> torch.Tensor:
        """
        x_price:  [batch, seq_len, PRICE_FEATURES]
        x_signal: [batch, seq_len, SIGNAL_FEATURES]
        Returns:  [batch] — predicted BubbleScore
        """
        h_price  = self.price_encoder(x_price)    # [B, d_model]
        h_signal = self.signal_encoder(x_signal)  # [B, d_model]
        h_fused  = torch.cat([h_price, h_signal], dim=-1)  # [B, 2*d_model]
        out = self.fusion(h_fused).squeeze(-1)    # [B]
        return out


# ─────────────────────────────────────────────────────────────
# Full HLPPL Pipeline
# ─────────────────────────────────────────────────────────────

class HLPPLFeatureBuilder:
    """
    Builds price-stream and signal-stream features from raw OHLCV data.
    Must be fit on training data only to avoid leakage.
    """

    def __init__(self,
                 lppl_window: int = 252,
                 hype_window: int = 21,
                 alpha1: float = 0.3,
                 alpha2: float = 0.2):
        self.lppl_window = lppl_window
        self.hype_window = hype_window
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        # Fitted normalisation stats
        self._price_mean: Optional[np.ndarray] = None
        self._price_std:  Optional[np.ndarray] = None

    def build_features(self,
                       close:  np.ndarray,
                       volume: np.ndarray,
                       fit_normaliser: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          price_features  : [n, 4]  (close_norm, vol_norm, log_ret, realised_vol)
          signal_features : [n, 4]  (eps_norm, bubble_score, hype, sentiment)
          bubble_score    : [n]     (labels for supervised training)
        """
        n = len(close)
        log_price = np.log(np.maximum(close, 1e-10))
        log_volume = np.log(np.maximum(volume, 1.0))

        # — Log return and realised vol —
        log_ret = np.diff(log_price, prepend=log_price[0])
        realised_vol = np.array([
            log_ret[max(0, i-21): i+1].std()
            for i in range(n)
        ])

        # — Signal features —
        eps_norm  = compute_lppl_residual(log_price, window=self.lppl_window)
        hype      = compute_hype_index(volume, window=self.hype_window)
        sentiment = compute_sentiment_score(close)
        bubble_score = compute_bubble_score(eps_norm, hype, sentiment,
                                            self.alpha1, self.alpha2)

        # — Price features —
        price_raw = np.column_stack([
            log_price, log_volume, log_ret, realised_vol
        ]).astype(np.float32)   # [n, 4]

        if fit_normaliser:
            self._price_mean = price_raw.mean(axis=0)
            self._price_std  = price_raw.std(axis=0).clip(1e-8)

        price_norm = (price_raw - self._price_mean) / self._price_std

        signal_features = np.column_stack([
            eps_norm, bubble_score, hype, sentiment
        ]).astype(np.float32)

        return price_norm, signal_features, bubble_score


class HLPPLModel:
    """
    End-to-end HLPPL model:
      1. Build BubbleScore labels from LPPL residual + Hype + Sentiment.
      2. Train Dual-Stream Transformer to forecast next-day BubbleScore.
    """

    def __init__(self,
                 seq_len: int = 60,          # lookback window for Transformer
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 lr: float = 3e-4,
                 epochs: int = 30,
                 batch_size: int = 64,
                 alpha1: float = 0.3,
                 alpha2: float = 0.2,
                 lppl_window: int = 252,
                 device: Optional[str] = None):
        self.seq_len    = seq_len
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.device     = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.feature_builder = HLPPLFeatureBuilder(
            lppl_window=lppl_window, alpha1=alpha1, alpha2=alpha2)
        self.transformer = DualStreamTransformer(
            d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout).to(self.device)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    # ── Dataset preparation (NO LEAKAGE) ─────────────────────

    def _make_sequences(self,
                        price_feat:  np.ndarray,
                        signal_feat: np.ndarray,
                        labels:      np.ndarray,
                        forecast_horizon: int = 1
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sliding-window sequences. Label at t = bubble_score at t+horizon.
        Ensures NO LOOKAHEAD: features up to t, label from t+horizon.
        """
        n = len(labels)
        X_price, X_signal, Y = [], [], []
        for i in range(self.seq_len, n - forecast_horizon):
            X_price.append(price_feat[i - self.seq_len: i])
            X_signal.append(signal_feat[i - self.seq_len: i])
            Y.append(labels[i + forecast_horizon - 1])
        return (torch.tensor(np.array(X_price), dtype=torch.float32),
                torch.tensor(np.array(X_signal), dtype=torch.float32),
                torch.tensor(np.array(Y), dtype=torch.float32))

    # ── Training ──────────────────────────────────────────────

    def fit(self,
            close_train:  np.ndarray,
            volume_train: np.ndarray,
            close_val:    Optional[np.ndarray] = None,
            volume_val:   Optional[np.ndarray] = None,
            forecast_horizon: int = 1,
            verbose: bool = True) -> "HLPPLModel":
        """
        Train on (close_train, volume_train).
        Validation data must be temporally AFTER training data.
        """
        print("Building training features (LPPL fitting takes time)...")
        pf_tr, sf_tr, bs_tr = self.feature_builder.build_features(
            close_train, volume_train, fit_normaliser=True)

        Xp_tr, Xs_tr, Y_tr = self._make_sequences(pf_tr, sf_tr, bs_tr, forecast_horizon)
        ds_tr = TensorDataset(Xp_tr, Xs_tr, Y_tr)
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)

        has_val = close_val is not None and len(close_val) > self.seq_len + 5
        if has_val:
            print("Building validation features...")
            pf_val, sf_val, bs_val = self.feature_builder.build_features(
                close_val, volume_val, fit_normaliser=False)   # use train stats!
            Xp_val, Xs_val, Y_val = self._make_sequences(
                pf_val, sf_val, bs_val, forecast_horizon)
            ds_val = TensorDataset(Xp_val, Xs_val, Y_val)
            dl_val = DataLoader(ds_val, batch_size=self.batch_size)

        opt = optim.AdamW(self.transformer.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss(delta=0.5)

        for epoch in range(self.epochs):
            self.transformer.train()
            tr_losses = []
            for xp, xs, yb in dl_tr:
                xp, xs, yb = (xp.to(self.device),
                               xs.to(self.device),
                               yb.to(self.device))
                opt.zero_grad()
                pred = self.transformer(xp, xs)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                opt.step()
                tr_losses.append(loss.item())
            scheduler.step()

            tr_mean = float(np.mean(tr_losses))
            self.history["train_loss"].append(tr_mean)

            if has_val:
                self.transformer.eval()
                vl_losses = []
                with torch.no_grad():
                    for xp, xs, yb in dl_val:
                        xp, xs, yb = (xp.to(self.device),
                                       xs.to(self.device),
                                       yb.to(self.device))
                        pred = self.transformer(xp, xs)
                        vl_losses.append(loss_fn(pred, yb).item())
                vl_mean = float(np.mean(vl_losses))
                self.history["val_loss"].append(vl_mean)
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                          f"train={tr_mean:.6f}  val={vl_mean:.6f}")
            elif verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}  train={tr_mean:.6f}")

        return self

    # ── Inference ─────────────────────────────────────────────

    def predict(self,
                close:  np.ndarray,
                volume: np.ndarray) -> np.ndarray:
        """
        Predict BubbleScore for each available time step.
        Returns array of length (n - seq_len).
        """
        pf, sf, _ = self.feature_builder.build_features(
            close, volume, fit_normaliser=False)
        n = len(close)
        preds = []
        self.transformer.eval()
        with torch.no_grad():
            for i in range(self.seq_len, n):
                xp = torch.tensor(pf[i - self.seq_len: i],
                                  dtype=torch.float32).unsqueeze(0).to(self.device)
                xs = torch.tensor(sf[i - self.seq_len: i],
                                  dtype=torch.float32).unsqueeze(0).to(self.device)
                preds.append(self.transformer(xp, xs).item())
        return np.array(preds)

    def get_bubble_labels_only(self,
                               close:  np.ndarray,
                               volume: np.ndarray) -> np.ndarray:
        """Phase 1 only — returns BubbleScore without Transformer."""
        pf, sf, bs = self.feature_builder.build_features(
            close, volume, fit_normaliser=False)
        return bs

    def save(self, path: str) -> None:
        torch.save({
            "transformer": self.transformer.state_dict(),
            "feature_builder_mean": self.feature_builder._price_mean,
            "feature_builder_std":  self.feature_builder._price_std,
            "seq_len": self.seq_len,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.transformer.load_state_dict(ckpt["transformer"])
        self.feature_builder._price_mean = ckpt["feature_builder_mean"]
        self.feature_builder._price_std  = ckpt["feature_builder_std"]
