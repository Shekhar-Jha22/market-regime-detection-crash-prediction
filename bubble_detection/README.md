# bubble_detection — Module Reference

## Entry Point

```bash
python run.py [options]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--ticker` | str | `SPY` | Yahoo Finance ticker symbol |
| `--start` | str | `2013-01-01` | History start date |
| `--end` | str | `2024-12-31` | History end date |
| `--horizon` | int | `5` | Crash forecast window (trading days) |
| `--crash-threshold` | float | `0.05` | Log-return drop considered a crash |
| `--demo` | flag | off | Use synthetic data instead of downloading |
| `--output` | str | `output` | Directory for all output files |

---

## Models

### `models/lppls_core.py` — LPPLS Core

Implements the Log-Periodic Power Law Singularity model.

**Formula**  
`ln P(t) = A + B(tc−t)^m + (tc−t)^m [C1·cos(ω·ln(tc−t)) + C2·sin(ω·ln(tc−t))]`

**Key classes**

| Class | Purpose |
|---|---|
| `LPPLSFitter` | Fit LPPLS to a single window via DE or multi-start Nelder-Mead |
| `LPPLSConfidenceModel` | Multi-window rolling confidence (fraction of qualifying fits) |

**Filter criteria** (Filimonov & Sornette 2013)

| Parameter | Range |
|---|---|
| m (power exponent) | (0.01, 0.99) |
| ω (log-period frequency) | (6.0, 13.0) |
| tc (days to critical time) | (t_end, t_end + 730) |
| B | < 0 for positive bubble |
| D = m\|B\|/(ω·C) | ≥ 0 (oscillation damping) |

---

### `models/hlppl.py` — HLPPL BubbleScore

Phase 1 (deterministic, used by default in `run.py`):

```
BubbleScore(t) = ε_norm(t) ± α₁·H(t) + α₂·S(t)
```

- `ε_norm`: rolling LPPLS residual, z-scored → [-1, 1]
- `H(t)`: volume z-score (hype proxy), clamped to [0, 1]  
- `S(t)`: RSI-based sentiment → [-1, 1]; **causal** trailing MA for slow-trend dampening

Phase 2 (optional, full model): Dual-Stream Transformer on price and signal features.

**Key functions**

| Function | Description |
|---|---|
| `compute_lppl_residual(log_prices, window, step)` | Rolling LPPLS fit → normalised residual |
| `compute_hype_index(volume, window)` | Volume z-score proxy for media attention |
| `compute_sentiment_score(close)` | RSI + slow-MA trend dampening |
| `compute_bubble_score(eps_norm, hype, sentiment)` | HLPPL Phase 1 formula |

---

### `models/deep_lppls.py` — Neural LPPLS Estimators

**MonoLPPLSNN** — physics-informed network for a single series  
- 2-layer MLP with physics loss (LPPLS RSS + penalty for B < 0, D ≥ 0.01)  
- Output: `(tc_offset, m, ω)`

**PolyLPPLSNN** — pre-trained 1D-CNN for fast inference  
- Architecture: `Conv1d(1→32) → Conv1d(32→64) → AdaptiveAvgPool(32) → MLP`  
- Pre-trained on 10 000 synthetic LPPLS series  
- Inference: one forward pass per window; no iterative optimisation

**PolyLPPLSTrainer** — manages synthetic data generation and training  
- Loss: weighted MSE — `5·L_tc + 1·L_m + 2·L_ω` (normalised by parameter range)  
- Optimizer: AdamW + OneCycleLR

---

### `models/hawkes.py` — Hawkes Process

Exponential-kernel Hawkes process:  
`λ(t) = μ + Σ_{tᵢ<t} α·exp(-β·(t−tᵢ))`

- **Fitting**: MLE via L-BFGS-B (10 random restarts), stationarity enforced (α < β)
- **Crash probability**: exact closed-form integrated intensity  
  `P(≥1 event in [t, t+H]) = 1 − exp(−Λ(t, t+H))`
- **Event detection**: 2σ threshold on 63-day rolling distribution (Bacry et al. 2015)

---

### `models/market_hmm.py` — Hidden Markov Model

Gaussian HMM over 6 features:  
`[log_ret, realised_vol_21d, skew_21d, log_vol_z_63d, lppls_conf, bubble_score]`

- 4 states learned via EM; ranked post-hoc by mean realised volatility
- Crash probability: filtered state distribution × A^k transition matrix  
  `P(crash in [t+1,t+H]) = 1 − Π_{k=1}^{H}(1 − (π_t@A^k)·crash_mask)`

---

### `models/ensemble.py` — Logistic Meta-Learner

7-feature signal matrix fed into penalised logistic regression:

| Signal | Source |
|---|---|
| `lppls_conf` | LPPLS positive confidence |
| `bubble_score_pos` / `bubble_score_neg` | HLPPL score split by sign |
| `tc_proximity` | 1/(tc_offset+1) |
| `m_raw` | LPPLS exponent |
| `hawkes_prob` | Hawkes 5-day probability |
| `hmm_prob` | HMM 5-day probability |

Calibration: **Platt scaling** on held-out validation set  
Output: `CrashProbabilityResult` with `.crash_prob`, `.weights`, `.signal_matrix`

---

## Training Framework

### `training/walk_forward.py`

Expanding-window walk-forward CV with configurable gap (embargo).

```
Fold k:  Train [0 .. Tk]   GAP   Val [Tk+gap .. Tk+gap+val_size]
                                              ↑ normalisation fitted here only
Test:    [n - val_size .. n]  ← never seen during training or calibration
```

`check_no_leakage()` asserts zero overlap and minimum gap at runtime.

---

## Evaluation

### `evaluation/metrics.py`

| Group | Metrics |
|---|---|
| Regression | MSE, MAE, RMSE, R², Spearman ρ, Direction accuracy |
| Classification | Precision, Recall, F1, AUC-ROC, AUC-PR, MCC |
| Backtest | Annualised return, Sharpe, Sortino, Max drawdown, Calmar, Win rate |
| Bubble-specific | Lead-time to crash (mean/median/std), N confirmed events |
