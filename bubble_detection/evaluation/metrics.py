"""
Evaluation Framework
====================
Performance metrics for bubble detection models.

Metrics are grouped into four categories:

  A. Regression / Score quality
     MSE, MAE, RMSE, R², Spearman correlation, Direction accuracy

  B. Classification (bubble / no-bubble at threshold)
     Precision, Recall, F1, AUC-ROC, AUC-PR, Matthews Correlation Coefficient

  C. Financial performance (backtest)
     Annualised return, Sharpe ratio, Sortino ratio, Max drawdown,
     Calmar ratio, Win rate, Profit factor

  D. Bubble-specific diagnostics
     Lead-time analysis (how many days before peak does signal fire?)
     False positive / false negative rates
     Confidence calibration
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────
# A. Regression metrics
# ─────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray) -> Dict[str, float]:
    """MSE, MAE, RMSE, R², Spearman ρ, direction accuracy."""
    residuals = y_true - y_pred
    mse  = float(np.mean(residuals ** 2))
    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))

    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - np.sum(residuals ** 2) / ss_tot if ss_tot > 1e-10 else float("nan")

    rho, _ = stats.spearmanr(y_true, y_pred)

    # Direction accuracy: sign agreement
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {"MSE": mse, "MAE": mae, "RMSE": rmse,
            "R2": float(r2), "Spearman": float(rho),
            "DirectionAcc": dir_acc}


# ─────────────────────────────────────────────────────────────
# B. Classification metrics
# ─────────────────────────────────────────────────────────────

def classification_metrics(y_true_score: np.ndarray,
                            y_pred_score: np.ndarray,
                            threshold: float = 0.4,
                            pos_label: int = 1) -> Dict[str, float]:
    """
    Convert continuous scores to binary labels at `threshold`,
    then compute classification metrics.
    """
    y_true = (y_true_score > threshold).astype(int)
    y_pred = (y_pred_score > threshold).astype(int)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = TP / (TP + FP + 1e-10)
    recall    = TP / (TP + FN + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    fpr       = FP / (FP + TN + 1e-10)
    fnr       = FN / (FN + TP + 1e-10)

    # Matthews Correlation Coefficient
    denom = np.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
    mcc   = (TP*TN - FP*FN) / (denom + 1e-10)

    # ROC-AUC and PR-AUC (via trapezoidal approximation)
    auc_roc, auc_pr = _compute_aucs(y_true, y_pred_score)

    return {
        "Precision": precision, "Recall": recall, "F1": f1,
        "Specificity": specificity, "FPR": fpr, "FNR": fnr,
        "MCC": float(mcc), "AUC_ROC": auc_roc, "AUC_PR": auc_pr,
    }


def _compute_aucs(y_true: np.ndarray,
                  y_score: np.ndarray) -> Tuple[float, float]:
    """Compute ROC-AUC and PR-AUC via manual trapz (no sklearn required)."""
    thresholds = np.sort(np.unique(y_score))[::-1]
    tprs, fprs, precs, recs = [1.0], [1.0], [], []

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        tprs.append(TP / (TP + FN + 1e-10))
        fprs.append(FP / (FP + TN + 1e-10))
        precs.append(TP / (TP + FP + 1e-10))
        recs.append(TP / (TP + FN + 1e-10))

    tprs.append(0.0); fprs.append(0.0)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc_roc = float(np.abs(_trapz(tprs, fprs)))
    if precs:
        auc_pr = float(np.abs(_trapz(precs, recs)))
    else:
        auc_pr = float("nan")
    return auc_roc, auc_pr


# ─────────────────────────────────────────────────────────────
# C. Financial / backtest metrics
# ─────────────────────────────────────────────────────────────

def backtest_strategy(close: np.ndarray,
                      bubble_score: np.ndarray,
                      entry_threshold: float = 0.5,
                      exit_threshold:  float = 0.1,
                      short_on_negative: bool = True,
                      transaction_cost: float = 0.001,
                      annualisation: int = 252) -> Dict[str, float]:
    """
    Simple threshold-based backtest:
      - LONG  when bubble_score <  -entry_threshold (negative bubble = undervalued)
      - SHORT when bubble_score >  +entry_threshold (positive bubble = overvalued)
      - EXIT  when |bubble_score| < exit_threshold

    Parameters
    ----------
    close              : price array (same length as bubble_score)
    bubble_score       : model output
    entry_threshold    : |score| above which to enter a position
    exit_threshold     : |score| below which to exit
    short_on_negative  : if False, only trade negative bubbles (long only)
    transaction_cost   : per-trade cost (fraction of price)
    """
    n = len(bubble_score)
    close = close[-n:]   # align lengths

    log_ret  = np.diff(np.log(np.maximum(close, 1e-10)), prepend=0.0)
    position = np.zeros(n)   # +1 = long, -1 = short, 0 = flat
    pos_prev = 0

    for i in range(1, n):
        score = bubble_score[i - 1]   # use lagged score (no lookahead)
        if abs(score) > entry_threshold:
            if score > 0:
                position[i] = -1.0 if short_on_negative else 0.0   # short overvalued
            else:
                position[i] = +1.0   # long undervalued
        elif abs(score) < exit_threshold:
            position[i] = 0.0
        else:
            position[i] = pos_prev   # hold
        pos_prev = position[i]

    # Strategy returns (position[i] applied to ret[i])
    # Transaction cost on position changes
    pos_change = np.abs(np.diff(position, prepend=position[0]))
    strat_ret  = position * log_ret - transaction_cost * pos_change

    # Financial metrics
    total_return = float(np.expm1(np.sum(strat_ret)))
    n_years      = n / annualisation

    ann_return = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1)
    ann_vol    = float(np.std(strat_ret) * np.sqrt(annualisation))
    sharpe     = ann_return / (ann_vol + 1e-10)

    # Sortino (downside vol)
    neg_ret     = strat_ret[strat_ret < 0]
    downside_std = float(np.std(neg_ret) * np.sqrt(annualisation)) if len(neg_ret) > 1 else 1e-10
    sortino      = ann_return / downside_std

    # Max drawdown
    cum_ret   = np.cumsum(strat_ret)
    roll_max  = np.maximum.accumulate(cum_ret)
    drawdown  = cum_ret - roll_max
    max_dd    = float(np.min(drawdown))

    calmar = ann_return / (abs(max_dd) + 1e-10)

    # Win rate and profit factor
    daily_pnl = strat_ret[position != 0] if np.any(position != 0) else strat_ret
    wins  = daily_pnl[daily_pnl > 0]
    losses = daily_pnl[daily_pnl < 0]
    win_rate     = len(wins) / (len(wins) + len(losses) + 1e-10)
    profit_factor = wins.sum() / (abs(losses.sum()) + 1e-10)

    n_trades = int(np.sum(np.diff(position, prepend=position[0]) != 0))

    return {
        "AnnReturn":    ann_return,
        "AnnVol":       ann_vol,
        "Sharpe":       sharpe,
        "Sortino":      sortino,
        "MaxDrawdown":  max_dd,
        "Calmar":       calmar,
        "WinRate":      win_rate,
        "ProfitFactor": profit_factor,
        "NTrades":      float(n_trades),
        "TotalReturn":  total_return,
    }


# ─────────────────────────────────────────────────────────────
# D. Bubble-specific diagnostics
# ─────────────────────────────────────────────────────────────

def bubble_lead_time(bubble_score: np.ndarray,
                     close: np.ndarray,
                     signal_threshold: float = 0.5,
                     crash_pct: float = -0.20,
                     max_horizon: int = 252) -> Dict[str, float]:
    """
    For each period where signal fires (score > threshold),
    measure how many days until a ≥ crash_pct drawdown occurs.

    Returns mean / median / std of lead times.
    Only counts events where a crash actually follows within max_horizon days.
    """
    n = len(bubble_score)
    close = close[-n:]
    lead_times = []
    i = 0
    while i < n:
        if bubble_score[i] > signal_threshold:
            # Find peak within horizon
            horizon_end = min(i + max_horizon, n)
            future_log_ret = np.log(close[i: horizon_end] / close[i])
            crash_mask = future_log_ret <= crash_pct
            if crash_mask.any():
                lead_time = int(np.argmax(crash_mask))
                lead_times.append(lead_time)
            i += 5   # skip forward
        else:
            i += 1

    if not lead_times:
        return {"LeadTimeMean": float("nan"), "LeadTimeMedian": float("nan"),
                "LeadTimeStd": float("nan"), "N_Events": 0.0}
    return {
        "LeadTimeMean":   float(np.mean(lead_times)),
        "LeadTimeMedian": float(np.median(lead_times)),
        "LeadTimeStd":    float(np.std(lead_times)),
        "N_Events":       float(len(lead_times)),
    }


# ─────────────────────────────────────────────────────────────
# Master evaluation function
# ─────────────────────────────────────────────────────────────

def evaluate_all(close:         np.ndarray,
                 bubble_score_true: np.ndarray,
                 bubble_score_pred: np.ndarray,
                 threshold:     float = 0.4,
                 verbose:       bool = True) -> Dict[str, float]:
    """
    Run all metric groups and return a flat dict.
    """
    results: Dict[str, float] = {}

    # A — Regression
    reg = regression_metrics(bubble_score_true, bubble_score_pred)
    results.update({f"reg_{k}": v for k, v in reg.items()})

    # B — Classification
    clf = classification_metrics(bubble_score_true, bubble_score_pred, threshold)
    results.update({f"clf_{k}": v for k, v in clf.items()})

    # C — Backtest
    bt = backtest_strategy(close, bubble_score_pred)
    results.update({f"bt_{k}": v for k, v in bt.items()})

    # D — Lead time (uses raw pred score, not labels)
    lt = bubble_lead_time(bubble_score_pred, close, threshold)
    results.update({f"lt_{k}": v for k, v in lt.items()})

    if verbose:
        print("\n" + "─" * 55)
        print("EVALUATION SUMMARY")
        print("─" * 55)
        groups = {
            "Regression":    {k: v for k, v in results.items() if k.startswith("reg_")},
            "Classification":{k: v for k, v in results.items() if k.startswith("clf_")},
            "Backtest":      {k: v for k, v in results.items() if k.startswith("bt_")},
            "Lead Time":     {k: v for k, v in results.items() if k.startswith("lt_")},
        }
        for group, metrics in groups.items():
            print(f"\n{group}:")
            for k, v in metrics.items():
                short_k = k.split("_", 1)[1]
                if isinstance(v, float) and not np.isnan(v):
                    print(f"  {short_k:<18} {v:+.4f}")
                else:
                    print(f"  {short_k:<18} {v}")
        print("─" * 55)

    return results


# ─────────────────────────────────────────────────────────────
# Cross-fold aggregator
# ─────────────────────────────────────────────────────────────

def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Given a list of metric dicts (one per fold), return a DataFrame
    with mean ± std across folds.
    """
    df = pd.DataFrame(fold_metrics)
    summary = pd.DataFrame({
        "mean": df.mean(),
        "std":  df.std(),
        "min":  df.min(),
        "max":  df.max(),
    })
    return summary.round(4)
