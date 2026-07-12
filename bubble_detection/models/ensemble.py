"""
Crash Probability Ensemble
===========================
Fuses signals from all five models into a single calibrated daily output:

    P(crash in next H days | all signals at time t)

Models / signals
----------------
    1. LPPLS Confidence    → pos_confidence score  ∈ [0, 1]
    2. BubbleScore (HLPPL) → raw bubble score       ∈ ℝ  (sign + magnitude)
    3. Poly-LPPLS-NN       → tc_offset (days to critical time) + m + ω
    4. Hawkes Process      → P(event in next H days) via survival analysis
    5. HMM                 → P(crash regime in next H days) via A^k

Ensemble method
---------------
    A logistic regression meta-learner is trained on the concatenated signals
    to predict binary crash labels (≥X% drawdown in next H days).

    Walk-forward safe:
      • Meta-learner trained on TRAIN signals + crash labels only.
      • Calibrated with Platt scaling on a held-out validation set.
      • Test set never touches training or calibration.

Output calibration
------------------
    Raw logistic output is calibrated using isotonic regression or
    Platt scaling to produce reliable probability estimates.
    Reliability diagram is available for visual inspection.

Label definition
----------------
    y(t) = 1  if  max_drawdown(close[t+1 : t+H]) ≤ −crash_threshold
    y(t) = 0  otherwise

    Default: crash_threshold = 0.05 (5% peak-to-trough in next 5 days).
    This is conservative — a 5-day 5% move in equities is a genuine tail event.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit       # sigmoid
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────

def make_crash_labels(close: np.ndarray,
                      horizon: int = 5,
                      crash_threshold: float = 0.05) -> np.ndarray:
    """
    Binary crash label for each day t:
        y(t) = 1  if  close drops ≥ crash_threshold within [t+1, t+horizon]
        y(t) = 0  otherwise
        y(t) = NaN for last `horizon` days (no future data)

    Uses log-return based drawdown:
        drawdown(t) = min(log(close[t+k]/close[t]))  for k=1..horizon
    """
    n = len(close)
    labels = np.full(n, np.nan)
    log_close = np.log(np.maximum(close, 1e-10))
    for t in range(n - horizon):
        future = log_close[t + 1: t + horizon + 1]
        drawdown = np.min(future) - log_close[t]    # ≤ 0 for drops
        labels[t] = 1.0 if drawdown <= -crash_threshold else 0.0
    return labels


# ─────────────────────────────────────────────────────────────
# Signal normalisers
# ─────────────────────────────────────────────────────────────

def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-10:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x)


# ─────────────────────────────────────────────────────────────
# Signal matrix builder
# ─────────────────────────────────────────────────────────────

class SignalMatrix:
    """
    Assembles all model outputs into a normalised signal matrix [n, n_signals].
    Normalisation stats are fitted on training data only.
    """

    SIGNAL_NAMES = [
        "lppls_conf",          # LPPLS positive confidence ∈ [0,1]
        "bubble_score_pos",    # max(BubbleScore, 0) — positive bubble strength
        "bubble_score_neg",    # max(-BubbleScore, 0) — negative bubble strength
        "tc_proximity",        # 1 / (tc_offset + 1) — how close to critical time
        "m_raw",               # LPPLS power-law exponent (higher → faster acceleration)
        "hawkes_prob",         # Hawkes P(event in horizon)
        "hmm_prob",            # HMM P(crash state in horizon)
    ]

    def __init__(self):
        self._signal_mean: Optional[np.ndarray] = None
        self._signal_std:  Optional[np.ndarray] = None

    def build(self,
              lppls_confidence:   np.ndarray,
              bubble_score:       np.ndarray,
              tc_offset:          np.ndarray,
              m_values:           np.ndarray,
              hawkes_prob:        np.ndarray,
              hmm_prob:           np.ndarray,
              fit_normaliser: bool = True) -> np.ndarray:
        """
        Parameters (all arrays length n, aligned in time)
        ----------
        lppls_confidence : LPPLS positive bubble confidence ∈ [0, 1]
        bubble_score     : HLPPL BubbleScore (signed)
        tc_offset        : days until estimated tc (from Deep LPPLS or classic fit)
        m_values         : LPPLS power-law exponent m
        hawkes_prob      : Hawkes 5-day crash probability
        hmm_prob         : HMM 5-day crash-state probability
        fit_normaliser   : True = compute and store stats; False = use stored stats

        Returns
        -------
        S : [n, 7]  normalised signal matrix
        """
        n = len(lppls_confidence)
        tc_prox = 1.0 / (np.abs(tc_offset) + 1.0)   # closer tc → higher signal

        raw = np.column_stack([
            np.clip(lppls_confidence, 0, 1),
            np.maximum(bubble_score, 0),
            np.maximum(-bubble_score, 0),
            tc_prox,
            m_values,
            np.clip(hawkes_prob, 0, 1),
            np.clip(hmm_prob, 0, 1),
        ]).astype(np.float64)
        raw = np.nan_to_num(raw, nan=0.0)

        if fit_normaliser:
            self._signal_mean = raw.mean(axis=0)
            self._signal_std  = raw.std(axis=0).clip(1e-8)

        normed = (raw - self._signal_mean) / self._signal_std
        return normed.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Platt scaling calibrator
# ─────────────────────────────────────────────────────────────

class PlattCalibrator:
    """
    Fits a logistic function p̂ = σ(a·f + b) to calibrate raw scores.
    Minimises binary cross-entropy on the validation set.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        valid = ~np.isnan(labels)
        s, y = scores[valid], labels[valid]
        if y.sum() < 2 or (1 - y).sum() < 2:
            return self   # not enough of each class

        def nll(params):
            a, b = params
            p = expit(a * s + b)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(nll, [1.0, 0.0], method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8})
        self.a, self.b = res.x
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return expit(self.a * scores + self.b)


# ─────────────────────────────────────────────────────────────
# Meta-learner (logistic regression with L2)
# ─────────────────────────────────────────────────────────────

class LogisticMetaLearner:
    """
    Penalised logistic regression over the signal matrix.
    Weights trained to minimise binary cross-entropy + L2 penalty.
    """

    def __init__(self, C: float = 1.0):
        self.C = C          # inverse regularisation strength (higher = less regularised)
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticMetaLearner":
        """
        X : [n, n_signals]
        y : [n,]  binary labels (NaN rows excluded)
        """
        valid = ~np.isnan(y)
        Xv, yv = X[valid], y[valid]
        n_feat = Xv.shape[1]

        def obj(params):
            w, b = params[:-1], params[-1]
            logits = Xv @ w + b
            p = expit(np.clip(logits, -30, 30))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            nll  = -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
            reg  = 0.5 / self.C * np.sum(w ** 2)
            return nll + reg

        x0 = np.zeros(n_feat + 1)
        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-10})
        self.weights_ = res.x[:-1]
        self.bias_    = res.x[-1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights_ + self.bias_
        return expit(logits)


# ─────────────────────────────────────────────────────────────
# Full Ensemble
# ─────────────────────────────────────────────────────────────

@dataclass
class CrashProbabilityResult:
    dates:          np.ndarray           # datetime index
    crash_prob:     np.ndarray           # calibrated P(crash | next H days)
    raw_score:      np.ndarray           # uncalibrated logit score
    signal_matrix:  np.ndarray           # [n, 7] signal matrix
    signal_names:   List[str]
    horizon:        int
    crash_threshold: float
    weights:        Optional[np.ndarray] = None  # meta-learner weights


class CrashEnsemble:
    """
    Full ensemble pipeline.

    fit(signals_train, labels_train, signals_val, labels_val)
    predict(signals_test) → CrashProbabilityResult

    Leakage controls
    ----------------
    • SignalMatrix normalisation fitted on train only
    • Calibrator fitted on val only (held-out from meta-learner training)
    • Test never used during fit or calibration
    """

    def __init__(self,
                 horizon: int = 5,
                 crash_threshold: float = 0.05,
                 C: float = 1.0):
        self.horizon         = horizon
        self.crash_threshold = crash_threshold
        self.signal_matrix   = SignalMatrix()
        self.meta_learner    = LogisticMetaLearner(C=C)
        self.calibrator      = PlattCalibrator()
        self._is_fitted      = False

    def fit(self,
            # Training signals
            lppls_conf_tr:  np.ndarray,
            bubble_score_tr: np.ndarray,
            tc_offset_tr:   np.ndarray,
            m_tr:           np.ndarray,
            hawkes_prob_tr: np.ndarray,
            hmm_prob_tr:    np.ndarray,
            close_tr:       np.ndarray,
            # Validation signals (for calibration only)
            lppls_conf_val:  np.ndarray,
            bubble_score_val: np.ndarray,
            tc_offset_val:   np.ndarray,
            m_val:           np.ndarray,
            hawkes_prob_val: np.ndarray,
            hmm_prob_val:    np.ndarray,
            close_val:       np.ndarray,
            verbose: bool = True) -> "CrashEnsemble":
        """
        Two-stage fitting:
          1. Train meta-learner on training signals
          2. Calibrate on validation signals (Platt scaling)
        """
        # — Crash labels —
        y_tr  = make_crash_labels(close_tr,  self.horizon, self.crash_threshold)
        y_val = make_crash_labels(close_val, self.horizon, self.crash_threshold)

        # — Build signal matrices —
        S_tr = self.signal_matrix.build(
            lppls_conf_tr, bubble_score_tr, tc_offset_tr, m_tr,
            hawkes_prob_tr, hmm_prob_tr, fit_normaliser=True)

        S_val = self.signal_matrix.build(
            lppls_conf_val, bubble_score_val, tc_offset_val, m_val,
            hawkes_prob_val, hmm_prob_val, fit_normaliser=False)

        # — Meta-learner (trained on train set) —
        self.meta_learner.fit(S_tr, y_tr)

        # — Calibration (on val set) —
        raw_val = self.meta_learner.predict_proba(S_val)
        self.calibrator.fit(raw_val, y_val)

        if verbose:
            # Quick performance check on train
            raw_tr = self.meta_learner.predict_proba(S_tr)
            cal_tr = self.calibrator.predict(raw_tr)
            valid  = ~np.isnan(y_tr)
            if valid.sum() > 0:
                preds_bin = (cal_tr[valid] > 0.5).astype(float)
                acc = (preds_bin == y_tr[valid]).mean()
                crash_rate = y_tr[valid].mean()
                print(f"  Meta-learner train acc={acc:.3f}  "
                      f"base rate={crash_rate:.3f}  "
                      f"n_train={valid.sum()}")
            # Weights
            w = self.meta_learner.weights_
            if w is not None:
                print("  Signal weights:")
                for name, wi in zip(SignalMatrix.SIGNAL_NAMES, w):
                    print(f"    {name:<22} {wi:+.4f}")

        self._is_fitted = True
        return self

    def predict(self,
                lppls_conf:   np.ndarray,
                bubble_score: np.ndarray,
                tc_offset:    np.ndarray,
                m_values:     np.ndarray,
                hawkes_prob:  np.ndarray,
                hmm_prob:     np.ndarray,
                dates:        Optional[np.ndarray] = None) -> CrashProbabilityResult:
        """
        Predict calibrated P(crash in next H days) for each day.
        """
        assert self._is_fitted, "Call .fit() first."
        S = self.signal_matrix.build(
            lppls_conf, bubble_score, tc_offset, m_values,
            hawkes_prob, hmm_prob, fit_normaliser=False)
        raw   = self.meta_learner.predict_proba(S)
        calib = self.calibrator.predict(raw)

        if dates is None:
            dates = np.arange(len(lppls_conf))

        return CrashProbabilityResult(
            dates=dates,
            crash_prob=calib,
            raw_score=raw,
            signal_matrix=S,
            signal_names=SignalMatrix.SIGNAL_NAMES,
            horizon=self.horizon,
            crash_threshold=self.crash_threshold,
            weights=self.meta_learner.weights_,
        )


# ─────────────────────────────────────────────────────────────
# Reliability diagram (calibration check)
# ─────────────────────────────────────────────────────────────

def reliability_diagram(prob: np.ndarray,
                        labels: np.ndarray,
                        n_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Check calibration: bin predicted probabilities, compute mean predicted
    vs fraction of actual positives in each bin.

    Returns dict with keys: bin_centers, mean_predicted, fraction_positive, counts.
    """
    valid    = ~np.isnan(labels)
    p, y     = prob[valid], labels[valid]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, pred_mean, frac_pos, counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        pred_mean.append(p[mask].mean())
        frac_pos.append(y[mask].mean())
        counts.append(mask.sum())

    return {
        "bin_centers":      np.array(centers),
        "mean_predicted":   np.array(pred_mean),
        "fraction_positive": np.array(frac_pos),
        "counts":           np.array(counts),
    }


# ─────────────────────────────────────────────────────────────
# Convenience: signal description table
# ─────────────────────────────────────────────────────────────

def signal_summary(result: CrashProbabilityResult) -> pd.DataFrame:
    """Return a DataFrame showing each signal's stats and weight."""
    rows = []
    for i, name in enumerate(result.signal_names):
        col = result.signal_matrix[:, i]
        rows.append({
            "signal":  name,
            "mean":    float(col.mean()),
            "std":     float(col.std()),
            "weight":  float(result.weights[i]) if result.weights is not None else float("nan"),
        })
    return pd.DataFrame(rows).set_index("signal")
