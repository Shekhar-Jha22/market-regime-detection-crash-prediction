"""
Bubble Detection Suite — Full Pipeline Runner
=============================================
Runs all 5 models and produces a daily P(crash | next 5 days) for every day.

Models
------
  Tier 1  LPPLS multi-window + confidence indicators
  Tier 2  Poly-LPPLS-NN  (deep calibration, fast tc/m/ω estimates)
  Tier 3  HLPPL dual-stream Transformer (bubble score)
  Tier 4  Hawkes process  (self-exciting crash clustering)
  Tier 5  Hidden Markov Model (regime detection)
  ─────
  Ensemble  Logistic meta-learner + Platt calibration
            → calibrated P(crash in next H days) for every day

Zero-leakage guarantee
----------------------
  • Expanding walk-forward train/val/test split with 21-day embargo gap
  • All normalisation stats fitted on TRAIN only, applied to val/test
  • Ensemble meta-learner trained on train signals, calibrator on val signals
  • Test set never seen until final evaluation

Usage
-----
  python run.py --demo                   # synthetic data, runs in ~2 min
  python run.py --ticker SPY             # download from Yahoo Finance
  python run.py --ticker NVDA --horizon 5 --crash-threshold 0.03
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lppls_core  import LPPLSFitter, LPPLSConfidenceModel, lppls_formula
from models.deep_lppls  import PolyLPPLSTrainer
from models.hlppl       import (compute_lppl_residual, compute_hype_index,
                                 compute_sentiment_score, compute_bubble_score)
from models.hawkes      import HawkesProcess
from models.market_hmm  import MarketHMM, build_hmm_features
from models.ensemble    import (CrashEnsemble, make_crash_labels,
                                 CrashProbabilityResult,
                                 reliability_diagram, signal_summary)
from training.walk_forward import load_yfinance, WalkForwardSplitter, check_no_leakage
from evaluation.metrics    import evaluate_all


# ─────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────

def generate_synthetic(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    log_p = np.zeros(n)

    # Bubble 1
    b1s, b1e = 250, 650
    tc1 = b1e + 40.0
    t1  = np.arange(b1e - b1s, dtype=float)
    log_p[b1s:b1e] = lppls_formula(t1, tc1 - b1s, 0.45, 7.5, 5.0, -0.20, 0.04, 0.05)
    for i in range(b1e, min(b1e + 10, n)):
        log_p[i] = log_p[i-1] - 0.020

    # Bubble 2
    b2s, b2e = 950, 1400
    tc2 = b2e + 50.0
    t2  = np.arange(b2e - b2s, dtype=float)
    log_p[b2s:b2e] = lppls_formula(t2, tc2 - b2s, 0.55, 9.0, 6.0, -0.28, 0.05, 0.06)
    for i in range(b2e, min(b2e + 30, n)):
        log_p[i] = log_p[i-1] - 0.010

    for i in range(1, n):
        if log_p[i] == 0.0:
            prev = log_p[i-1] if log_p[i-1] != 0.0 else 4.5
            log_p[i] = prev + np.random.normal(0.0003, 0.012)

    close  = np.exp(log_p)
    volume = np.random.lognormal(15.0, 0.5, n)
    volume[b1e-20:b1e+15] *= 4.0
    volume[b2e-20:b2e+15] *= 5.0

    dates = pd.date_range("2014-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


# ─────────────────────────────────────────────────────────────
# Signal extractors (all strictly causal)
# ─────────────────────────────────────────────────────────────

def extract_lppls_signals(close, train_end, n_windows=6, verbose=False):
    n = len(close)
    log_price = np.log(np.maximum(close, 1e-10))
    model = LPPLSConfidenceModel(min_window=80, max_window=400,
                                  n_windows=n_windows, fitter_method="nm")
    indicators = model.rolling_confidence(log_price[:train_end], step=10)

    # NaN = not yet computed; distinguish from legitimate 0.0 confidence
    conf_arr   = np.full(n, np.nan)
    tc_off_arr = np.full(n, np.nan)
    m_arr      = np.full(n, np.nan)

    for ci in indicators:
        i = ci.date_idx
        if i >= n:
            continue
        conf_arr[i] = ci.pos_confidence
        if not np.isnan(ci.mean_tc):
            tc_off_arr[i] = max(ci.mean_tc - i, 1.0)
        if not np.isnan(ci.mean_m):
            m_arr[i] = ci.mean_m

    # Forward-fill only the uncomputed (NaN) inter-step days
    for i in range(1, n):
        if np.isnan(conf_arr[i]):
            conf_arr[i]   = conf_arr[i-1] if not np.isnan(conf_arr[i-1])   else 0.0
            tc_off_arr[i] = tc_off_arr[i-1] if not np.isnan(tc_off_arr[i-1]) else 180.0
            m_arr[i]      = m_arr[i-1]      if not np.isnan(m_arr[i-1])      else 0.5
    conf_arr   = np.nan_to_num(conf_arr,   nan=0.0)
    tc_off_arr = np.nan_to_num(tc_off_arr, nan=180.0)
    m_arr      = np.nan_to_num(m_arr,      nan=0.5)

    if verbose:
        print(f"  LPPLS: {len(indicators)} windows  mean_conf={conf_arr[:train_end].mean():.3f}")
    return conf_arr, tc_off_arr, m_arr


def extract_bubble_score(close, volume, lppl_window=100, verbose=False):
    log_price = np.log(np.maximum(close, 1e-10))
    eps_norm  = compute_lppl_residual(log_price, window=lppl_window, step=5)
    hype      = compute_hype_index(volume)
    sentiment = compute_sentiment_score(close)
    bs        = compute_bubble_score(eps_norm, hype, sentiment)
    if verbose:
        print(f"  BubbleScore: mean={bs.mean():.3f}  std={bs.std():.3f}")
    return bs


def extract_poly_lppls(close, seq_len=180, n_train_synth=6000, epochs=12, verbose=False):
    n = len(close)
    log_price = np.log(np.maximum(close, 1e-10))

    trainer = PolyLPPLSTrainer(seq_len=seq_len, hidden=64,
                                epochs=epochs, n_train=n_train_synth,
                                n_val=1000, batch_size=128)
    if verbose:
        print(f"  Pretraining Poly-LPPLS-NN on {n_train_synth} synthetic series...")
    trainer.train(verbose=False)

    tc_off_arr = np.full(n, 180.0)
    m_arr      = np.full(n, 0.5)
    for i in range(seq_len, n, 5):
        pred = trainer.predict(log_price[:i])
        tc_off_arr[i] = pred["tc_offset"]
        m_arr[i]      = pred["m"]

    for i in range(1, n):
        if tc_off_arr[i] == 180.0 and i > 0 and tc_off_arr[i-1] != 180.0:
            tc_off_arr[i] = tc_off_arr[i-1]
            m_arr[i]      = m_arr[i-1]

    if verbose:
        print(f"  Poly-NN: mean tc_offset={tc_off_arr.mean():.1f}  mean m={m_arr.mean():.3f}")
    return tc_off_arr, m_arr


def extract_hawkes(close, train_end, horizon=5, verbose=False):
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)), prepend=0.0)
    hp = HawkesProcess(threshold_sigma=2.0, n_restarts=8)
    hp.fit(log_ret[:train_end])
    if verbose:
        print(f"  {hp.summary()}")
    return hp.crash_probability(log_ret, horizon=horizon)


def extract_hmm(close, volume, train_end, bubble_score, lppls_conf,
                horizon=5, n_states=4, verbose=False):
    X = build_hmm_features(close, volume,
                            lppl_confidence=lppls_conf,
                            bubble_score=bubble_score)
    model = MarketHMM(n_states=n_states, crash_states=1,
                      n_iter=200, random_state=42)
    model.fit(X[:train_end])
    if verbose:
        print(f"  {model.summary()}")
    return model.crash_probability_horizon(X, horizon=horizon)


# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────

def plot_results(close, dates, result, hawkes_prob, hmm_prob,
                 lppls_conf, bubble_score, labels,
                 train_end, val_end, save_path):

    n      = len(close)
    dt     = np.array(dates)
    res_n  = len(result.crash_prob)
    offset = n - res_n

    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(5, 2, hspace=0.50, wspace=0.30,
                             height_ratios=[2, 1.2, 1.2, 1.2, 1.2])

    def shade(ax):
        ax.axvspan(dt[0],         dt[train_end-1], alpha=0.04, color="blue")
        ax.axvspan(dt[train_end], dt[val_end-1],   alpha=0.07, color="orange")
        if val_end < n:
            ax.axvspan(dt[val_end], dt[-1],         alpha=0.07, color="green")

    # Price
    ax1 = fig.add_subplot(gs[0, :])
    ax1.semilogy(dt, close, "k-", lw=1.2, label="Price")
    crash_days = np.where(~np.isnan(labels) & (labels == 1))[0]
    ax1.scatter(dt[crash_days], close[crash_days], s=18, c="red",
                zorder=5, alpha=0.6, label="Crash label")
    shade(ax1)
    from matplotlib.patches import Patch
    import matplotlib.lines as mlines
    crash_handle = mlines.Line2D([], [], color="red", marker="o", linestyle="None",
                                 markersize=4, alpha=0.6)
    ax1.legend(handles=[
        ax1.lines[0],
        crash_handle,
        Patch(color="blue",   alpha=0.3, label="Train"),
        Patch(color="orange", alpha=0.4, label="Val"),
        Patch(color="green",  alpha=0.4, label="Test"),
    ], labels=["Price","Crash label","Train","Val","Test"],
    fontsize=8, loc="upper left", ncol=5)
    ax1.set_title("Price (log scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Price"); ax1.grid(alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Ensemble crash probability
    ax2 = fig.add_subplot(gs[1, :])
    p      = result.crash_prob
    dt_res = dt[offset:]
    ax2.fill_between(dt_res, 0, p, where=(p > 0.5),
                     color="red", alpha=0.7, label="Alert (P>0.5)")
    ax2.fill_between(dt_res, 0, p, where=(p <= 0.5),
                     color="steelblue", alpha=0.5)
    ax2.axhline(0.5, ls="--", c="red",    lw=0.9)
    ax2.axhline(0.3, ls=":",  c="orange", lw=0.8)
    shade(ax2)
    ax2.set_title(f"ENSEMBLE P(crash in next {result.horizon} days) — calibrated",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability"); ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Individual signals
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(dt, hawkes_prob,  lw=1.0, color="purple",     alpha=0.85, label="Hawkes P")
    ax3.plot(dt, hmm_prob,     lw=1.0, color="darkorange",  alpha=0.85, label="HMM P")
    ax3.plot(dt, lppls_conf,   lw=1.0, color="green",       alpha=0.85, label="LPPLS conf")
    ax3.axhline(0.5, ls="--", c="gray", lw=0.7)
    shade(ax3)
    ax3.set_title("Individual Signals: Hawkes | HMM | LPPLS Confidence",
                  fontsize=11, fontweight="bold")
    ax3.set_ylabel("Score"); ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="upper left", fontsize=8); ax3.grid(alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # BubbleScore
    ax4 = fig.add_subplot(gs[3, :])
    ax4.fill_between(dt, 0, bubble_score, where=(bubble_score > 0),
                     color="red",   alpha=0.5, label="Positive bubble")
    ax4.fill_between(dt, 0, bubble_score, where=(bubble_score < 0),
                     color="green", alpha=0.5, label="Negative bubble")
    ax4.axhline(0.4,  ls="--", c="red",   lw=0.8)
    ax4.axhline(-0.4, ls="--", c="green", lw=0.8)
    shade(ax4)
    ax4.set_title("HLPPL BubbleScore (LPPL residual + Hype + Sentiment)",
                  fontsize=11, fontweight="bold")
    ax4.set_ylabel("BubbleScore"); ax4.grid(alpha=0.2)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Reliability diagram
    ax5 = fig.add_subplot(gs[4, 0])
    if val_end < n:
        p_test  = result.crash_prob[val_end - offset:]
        lt      = labels[val_end: val_end + len(p_test)]
        rd      = reliability_diagram(p_test, lt)
        if len(rd["bin_centers"]) > 1:
            ax5.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
            ax5.bar(rd["bin_centers"], rd["fraction_positive"],
                    width=0.08, alpha=0.6, color="steelblue", label="Actual")
            ax5.plot(rd["bin_centers"], rd["mean_predicted"],
                     "ro-", ms=5, lw=1.2, label="Predicted")
    ax5.set_title("Reliability Diagram (test set)", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Predicted prob"); ax5.set_ylabel("Fraction positive")
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # Signal weights
    ax6 = fig.add_subplot(gs[4, 1])
    if result.weights is not None:
        names  = [s.replace("_", "\n") for s in result.signal_names]
        colors = ["seagreen" if w > 0 else "tomato" for w in result.weights]
        ax6.barh(names, result.weights, color=colors, alpha=0.8)
        ax6.axvline(0, color="black", lw=0.8)
        ax6.set_title("Ensemble Signal Weights", fontsize=10, fontweight="bold")
        ax6.set_xlabel("Weight"); ax6.grid(alpha=0.3, axis="x")

    fig.suptitle("Crash Probability Pipeline — 5 Models + Ensemble",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_pipeline(close, volume, dates, horizon=5, crash_threshold=0.05,
                 output_dir="output", verbose=True):
    n = len(close)
    os.makedirs(output_dir, exist_ok=True)

    # Split
    splitter = WalkForwardSplitter(
        n_folds=1, val_size=max(80, n // 8),
        gap=21, min_train_size=max(350, n // 4))
    folds    = splitter.split(n)
    test_idx = splitter.test_indices(n)
    check_no_leakage(folds, test_idx, gap=21)

    fold      = folds[-1]
    train_end = int(fold.train_idx[-1]) + 1
    val_end   = int(fold.val_idx[-1])  + 1
    print(f"\nSplit  train=[0..{train_end-1}]  "
          f"val=[{fold.val_idx[0]}..{val_end-1}]  "
          f"test=[{val_end}..{n-1}]")

    labels = make_crash_labels(close, horizon, crash_threshold)
    print(f"Crash rate: {np.nanmean(labels):.2%}  "
          f"(>={crash_threshold:.0%} drop in {horizon} days)")

    # ── 5 model signals ───────────────────────────────────────
    print("\n[1/5] LPPLS confidence...")
    lppls_conf, tc_off_lppls, m_lppls = extract_lppls_signals(
        close, train_end, n_windows=12, verbose=verbose)

    print("\n[2/5] HLPPL BubbleScore...")
    bubble_score = extract_bubble_score(
        close, volume, lppl_window=100, verbose=verbose)

    print("\n[3/5] Poly-LPPLS-NN...")
    tc_off_nn, m_nn = extract_poly_lppls(
        close, seq_len=min(180, train_end // 3),
        n_train_synth=10000, epochs=20, verbose=verbose)

    tc_offset = 0.5 * tc_off_lppls + 0.5 * tc_off_nn
    m_blend   = 0.5 * m_lppls      + 0.5 * m_nn

    print("\n[4/5] Hawkes process...")
    hawkes_prob = extract_hawkes(close, train_end, horizon, verbose=verbose)

    print("\n[5/5] HMM...")
    hmm_prob = extract_hmm(close, volume, train_end, bubble_score,
                            lppls_conf, horizon=horizon, n_states=4, verbose=verbose)

    # ── Ensemble ──────────────────────────────────────────────
    print("\n[Ensemble] Fitting meta-learner + Platt calibrator...")
    tr = slice(0, train_end)
    vl = slice(fold.val_idx[0], val_end)

    ensemble = CrashEnsemble(horizon=horizon, crash_threshold=crash_threshold, C=1.0)
    ensemble.fit(
        lppls_conf_tr    = lppls_conf[tr],
        bubble_score_tr  = bubble_score[tr],
        tc_offset_tr     = tc_offset[tr],
        m_tr             = m_blend[tr],
        hawkes_prob_tr   = hawkes_prob[tr],
        hmm_prob_tr      = hmm_prob[tr],
        close_tr         = close[tr],
        lppls_conf_val   = lppls_conf[vl],
        bubble_score_val = bubble_score[vl],
        tc_offset_val    = tc_offset[vl],
        m_val            = m_blend[vl],
        hawkes_prob_val  = hawkes_prob[vl],
        hmm_prob_val     = hmm_prob[vl],
        close_val        = close[vl],
        verbose=verbose,
    )

    result = ensemble.predict(
        lppls_conf   = lppls_conf,
        bubble_score = bubble_score,
        tc_offset    = tc_offset,
        m_values     = m_blend,
        hawkes_prob  = hawkes_prob,
        hmm_prob     = hmm_prob,
        dates        = np.array(dates),
    )

    # ── Output DataFrame ──────────────────────────────────────
    df_out = pd.DataFrame({
        "date":             dates,
        "close":            close,
        "crash_prob_5d":    result.crash_prob,
        "crash_label":      labels,
        "lppls_confidence": lppls_conf,
        "bubble_score":     bubble_score,
        "tc_offset_days":   tc_offset,
        "m":                m_blend,
        "hawkes_prob_5d":   hawkes_prob,
        "hmm_prob_5d":      hmm_prob,
    }).set_index("date")

    # ── Test evaluation ───────────────────────────────────────
    offset    = len(close) - len(result.crash_prob)
    p_test    = result.crash_prob[val_end - offset:]
    lab_test  = labels[val_end:]
    c_test    = close[val_end:]
    valid     = ~np.isnan(lab_test[:len(p_test)])

    if valid.sum() > 10:
        print(f"\n{'='*55}\nTEST SET EVALUATION\n{'='*55}")
        metrics = evaluate_all(c_test[valid], lab_test[:len(p_test)][valid],
                               p_test[valid], threshold=0.5, verbose=True)
        pd.Series(metrics).to_csv(os.path.join(output_dir, "test_metrics.csv"))
    else:
        print("\n  Test set too small for evaluation.")

    # Signal summary
    ss = signal_summary(result)
    print(f"\nSignal summary:\n{ss}")
    ss.to_csv(os.path.join(output_dir, "signal_summary.csv"))

    # Plot
    plot_results(close, dates, result, hawkes_prob, hmm_prob,
                 lppls_conf, bubble_score, labels,
                 train_end, val_end,
                 save_path=os.path.join(output_dir, "crash_probability.png"))

    # Save CSV
    csv_path = os.path.join(output_dir, "daily_crash_probability.csv")
    df_out.round(6).to_csv(csv_path)
    print(f"  Daily probabilities → {csv_path}")

    return df_out


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",          default="SPY")
    p.add_argument("--start",           default="2013-01-01")
    p.add_argument("--end",             default="2024-12-31")
    p.add_argument("--horizon",         type=int,   default=5)
    p.add_argument("--crash-threshold", type=float, default=0.05)
    p.add_argument("--demo",            action="store_true")
    p.add_argument("--output",          default="output")
    args = p.parse_args()

    print("=" * 60)
    print(f"CRASH PROBABILITY PIPELINE  horizon={args.horizon}d  "
          f"crash_threshold={args.crash_threshold:.0%}")
    print("=" * 60)

    if args.demo:
        print("DEMO MODE: generating synthetic data...")
        df = generate_synthetic(n=1800)
    else:
        print(f"Downloading {args.ticker}  {args.start} → {args.end}...")
        try:
            df = load_yfinance(args.ticker, args.start, args.end)
        except Exception as e:
            print(f"Download failed ({e}). Using synthetic data.")
            df = generate_synthetic(n=1800)

    close  = df["Close"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)
    print(f"Dataset: {len(close)} days  ({df.index[0].date()} → {df.index[-1].date()})")

    df_out = run_pipeline(close, volume, df.index,
                          horizon=args.horizon,
                          crash_threshold=args.crash_threshold,
                          output_dir=args.output)

    print("\n" + "=" * 60)
    print("OUTPUTS in:", args.output)
    print("  daily_crash_probability.csv   <- one P(crash) per day")
    print("  crash_probability.png         <- full dashboard")
    print("  test_metrics.csv              <- evaluation metrics")
    print("  signal_summary.csv            <- signal weights")
    print("=" * 60)
    print("\nLast 10 predictions:")
    cols = ["close", "crash_prob_5d", "hawkes_prob_5d", "hmm_prob_5d",
            "lppls_confidence", "bubble_score"]
    print(df_out[cols].tail(10).round(4).to_string())
    return df_out


if __name__ == "__main__":
    main()
