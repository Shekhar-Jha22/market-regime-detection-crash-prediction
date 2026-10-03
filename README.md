# Market Crash Prediction — Ensemble Bubble-Detection Suite

A five-model ensemble that produces a calibrated daily **P(crash in next H days)** for any equity ticker.  
Zero-leakage walk-forward design: normalisation stats, meta-learner, and calibrator are each fitted on strictly disjoint data splits.

---

## Architecture

```
Raw OHLCV
   │
   ├─ [1] LPPLS Confidence Model      → pos_confidence ∈ [0,1]
   │       Multi-window log-spaced fits; fraction passing all filter criteria.
   │       Ref: Filimonov & Sornette (2013), Sornette et al. (2017)
   │
   ├─ [2] HLPPL BubbleScore           → signed score ∈ ℝ
   │       LPPL residual + Hype Index (volume z-score) + Sentiment (RSI-based).
   │       Ref: Cao, Shao, Yan & Geman (2025) — arXiv 2510.10878
   │
   ├─ [3] Poly-LPPLS-NN               → tc_offset, m  (fast LPPLS params)
   │       1D-CNN pre-trained on synthetic LPPLS series.
   │       Ref: Nielsen, Sornette & Raissi (2024) — arXiv 2405.12803
   │
   ├─ [4] Hawkes Process              → P(≥1 crash event in next H days)
   │       Exponential kernel, MLE via L-BFGS-B; closed-form integrated intensity.
   │       Ref: Hawkes (1971); Bacry, Mastromatteo & Muzy (2015)
   │
   └─ [5] Hidden Markov Model         → P(crash regime in next H days)
           Gaussian HMM over [log_ret, realised_vol, skew, vol_z, LPPLS, BubbleScore].
           States ranked by volatility; crash probability via A^k.
           Ref: Baum & Petrie (1966); Engel & Rodrigues (2012)
               │
               ▼
   [Ensemble] Logistic meta-learner (L2) trained on train signals
              + Platt calibrator fitted on held-out val signals
              → calibrated P(crash | next H days)  per day
```

---

## Leakage Controls

| Layer | Control |
|---|---|
| Features | All rolling stats use only past data |
| LPPLS residual | Fitted on `[:train_end]` only |
| Signal matrix normalisation | `fit_normaliser=True` on train split only |
| Meta-learner | Trained on train signals + labels |
| Calibrator | Fitted on val signals + labels (disjoint from train) |
| Test set | Never touched until final `evaluate_all()` |
| Walk-forward gap | 21-day embargo between train end and val start |

---

## Quickstart

```bash
pip install -r bubble_detection/requirements.txt

# Demo mode (synthetic data, ~2 min)
python bubble_detection/run.py --demo

# Real ticker
python bubble_detection/run.py --ticker SPY --start 2010-01-01 --end 2024-12-31

# Custom crash threshold and horizon
python bubble_detection/run.py --ticker NVDA --horizon 10 --crash-threshold 0.07
```

### Outputs (written to `output/`)

| File | Description |
|---|---|
| `daily_crash_probability.csv` | One calibrated P(crash) per day |
| `crash_probability.png` | Full 5-panel dashboard |
| `test_metrics.csv` | AUC-ROC, F1, Sharpe on held-out test set |
| `signal_summary.csv` | Per-signal weights and stats |

---

## Repository Layout

```
bubble_detection/
├── run.py                  # End-to-end pipeline entry point
├── requirements.txt
├── models/
│   ├── lppls_core.py       # LPPLS fitter + multi-window confidence model
│   ├── deep_lppls.py       # Mono-LNN (physics-informed) + Poly-LNN (pre-trained CNN)
│   ├── hlppl.py            # HLPPL Phase 1 (BubbleScore) + Phase 2 (Dual-Stream Transformer)
│   ├── hawkes.py           # Hawkes process — exponential kernel MLE
│   ├── market_hmm.py       # Gaussian HMM for regime detection
│   └── ensemble.py         # Logistic meta-learner + Platt calibration
├── training/
│   └── walk_forward.py     # Zero-leakage walk-forward splitter + trainer
├── evaluation/
│   └── metrics.py          # Regression, classification, backtest, and lead-time metrics
└── output/                 # Auto-created; all results land here
```

---

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--horizon` | 5 | Forecast window in trading days |
| `--crash-threshold` | 0.05 | Log-return drop defining a crash event |
| LPPLS `max_window` | 1260 | ~5 trading years (Sornette 2003) |
| LPPLS `tc_dt_max` | 730 | Max days to tc from window end |
| Hawkes `threshold_sigma` | 2.0 | σ-multiple for extreme return detection (Bacry 2015) |
| HMM `n_states` | 4 | Normal / Elevated / Stressed / Crash |
| HMM `n_iter` | 200 | EM iterations |
| Poly-LNN `n_train` | 10 000 | Synthetic series for pre-training |

---

## References

- Johansen, Ledoit & Sornette (2000). *Crashes as Critical Points.* IJTAF.  
- Filimonov & Sornette (2013). *A Stable and Robust Calibration Scheme of the Log-Periodic Power Law Model.* Physica A.  
- Sornette et al. (2017). *Real-time prediction and post-mortem analysis of the Shanghai 2015 stock market bubble crash.* J Invest Strategies.  
- Hawkes (1971). *Spectra of some self-exciting and mutually exciting point processes.* Biometrika.  
- Bacry, Mastromatteo & Muzy (2015). *Hawkes processes in finance.* Market Microstructure and Liquidity.  
- Cao, Shao, Yan & Geman (2025). *HLPPL: Hyped Log-Periodic Power Law.* arXiv 2510.10878.  
- Nielsen, Sornette & Raissi (2024). *Deep LPPLS.* arXiv 2405.12803.  
