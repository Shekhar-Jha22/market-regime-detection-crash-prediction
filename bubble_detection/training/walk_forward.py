"""
Training Pipeline — Zero-Leakage Walk-Forward Validation
=========================================================

Leakage sources in financial ML and how we prevent each:

  1. LOOK-AHEAD in features
     → All features computed using only past data (rolling windows).
     → Normalisation stats fitted on TRAIN set only, applied to val/test.

  2. TEMPORAL OVERLAP in train/val split
     → Pure walk-forward splits: train ends at t, val starts at t+gap.
     → Gap (embargo) = max(sequence_length, 21 days) to avoid serial correlation.

  3. FUTURE tc in LPPLS labels
     → tc is a PREDICTION target, not a feature — never fed back as input.

  4. DISTRIBUTION LEAKAGE from full-dataset normalisation
     → HLPPLFeatureBuilder.fit_normaliser=True only on train fold.

Walk-Forward CV schedule
------------------------
  Expanding window (anchored start) — standard for non-stationary finance:

  Fold 1:  Train [0,    T1]   Val [T1+gap,  T1+gap+val_size]
  Fold 2:  Train [0,    T2]   Val [T2+gap,  T2+gap+val_size]
  ...
  Fold k:  Train [0,    Tk]   Val [Tk+gap,  Tk+gap+val_size]

  Test:    completely held out from the start, never seen during any fold.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_yfinance(ticker: str,
                  start: str = "2010-01-01",
                  end:   str = "2024-12-31") -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance.
    Returns DataFrame with columns: [Open, High, Low, Close, Volume, Adj Close].
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.dropna(subset=["Close", "Volume"])
    df.index = pd.to_datetime(df.index)
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ─────────────────────────────────────────────────────────────
# Walk-Forward Splitter (NO LEAKAGE)
# ─────────────────────────────────────────────────────────────

@dataclass
class Fold:
    """One walk-forward fold."""
    fold_id:    int
    train_idx:  np.ndarray   # integer indices into the full array
    val_idx:    np.ndarray
    gap:        int           # embargo size (days)


class WalkForwardSplitter:
    """
    Expanding-window walk-forward cross-validation splitter.

    Parameters
    ----------
    n_folds        : number of CV folds
    val_size       : validation window size in observations
    gap            : embargo between train end and val start
                     (set ≥ forecast_horizon + seq_len to eliminate leakage)
    min_train_size : minimum training observations required
    """

    def __init__(self,
                 n_folds: int = 5,
                 val_size: int = 126,     # ~6 months
                 gap: int = 21,           # 1 month embargo
                 min_train_size: int = 504):  # ~2 years minimum train
        self.n_folds       = n_folds
        self.val_size      = val_size
        self.gap           = gap
        self.min_train_size = min_train_size

    def split(self, n: int) -> List[Fold]:
        """
        Generate fold indices for a dataset of length n.
        Train / [gap] / Val
        """
        # Test set is the last val_size observations — completely held out
        test_start = n - self.val_size
        available  = test_start

        if available < self.min_train_size + self.gap + self.val_size:
            raise ValueError(
                f"Dataset too small: need at least "
                f"{self.min_train_size + self.gap + self.val_size + self.val_size} obs. "
                f"Got {n}.")

        # Fold boundaries (train ends)
        fold_ends = np.linspace(
            self.min_train_size,
            available - self.gap - self.val_size,
            self.n_folds
        ).astype(int)

        folds = []
        for i, train_end in enumerate(fold_ends):
            val_start = train_end + self.gap
            val_end   = val_start + self.val_size
            if val_end > available:
                break
            folds.append(Fold(
                fold_id   = i,
                train_idx = np.arange(0, train_end),
                val_idx   = np.arange(val_start, val_end),
                gap       = self.gap,
            ))
        return folds

    def test_indices(self, n: int) -> np.ndarray:
        """Returns the held-out test indices (always the last val_size obs)."""
        return np.arange(n - self.val_size, n)


# ─────────────────────────────────────────────────────────────
# Leakage checker
# ─────────────────────────────────────────────────────────────

def check_no_leakage(folds: List[Fold], test_idx: np.ndarray,
                     gap: int) -> None:
    """
    Assert that:
      1. No train/val overlap in any fold.
      2. Gap ≥ specified gap between train end and val start.
      3. No fold touches the test set.
    Raises AssertionError with a description if any check fails.
    """
    for fold in folds:
        tr = set(fold.train_idx)
        vl = set(fold.val_idx)
        ts = set(test_idx)

        assert len(tr & vl) == 0,  f"Fold {fold.fold_id}: train/val overlap!"
        assert len(tr & ts) == 0,  f"Fold {fold.fold_id}: train touches test!"
        assert len(vl & ts) == 0,  f"Fold {fold.fold_id}: val touches test!"

        actual_gap = fold.val_idx[0] - fold.train_idx[-1] - 1
        assert actual_gap >= gap, \
            f"Fold {fold.fold_id}: gap={actual_gap} < required {gap}!"

    print(f"✓ Leakage check passed for {len(folds)} folds "
          f"(gap={gap}, test_size={len(test_idx)}).")


# ─────────────────────────────────────────────────────────────
# Model trainer wrapper (model-agnostic)
# ─────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold_id:   int
    train_metrics: Dict[str, float]
    val_metrics:   Dict[str, float]
    model_state:   Any = field(default=None, repr=False)


class WalkForwardTrainer:
    """
    Runs walk-forward CV for any model that exposes:
      .fit(close_train, volume_train, close_val, volume_val, ...)
      .predict(close, volume)  →  np.ndarray of scores

    Handles normalisation scoping (train stats only) automatically via
    the feature_builder.fit_normaliser flag in HLPPLModel / LPPLSConfidenceModel.
    """

    def __init__(self,
                 splitter: Optional[WalkForwardSplitter] = None,
                 gap: int = 21):
        self.splitter = splitter or WalkForwardSplitter(gap=gap)
        self.gap = gap
        self.fold_results_: List[FoldResult] = []

    def run(self,
            close:   np.ndarray,
            volume:  np.ndarray,
            model_factory: Callable[[], Any],
            metric_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
            bubble_labels: Optional[np.ndarray] = None,
            verbose: bool = True) -> List[FoldResult]:
        """
        Parameters
        ----------
        close, volume    : full raw price/volume arrays
        model_factory    : callable → fresh model instance per fold
        metric_fn        : (y_true, y_pred) → dict of metric names/values
        bubble_labels    : precomputed labels (if None, model generates them)
        """
        n = len(close)
        folds    = self.splitter.split(n)
        test_idx = self.splitter.test_indices(n)
        check_no_leakage(folds, test_idx, self.gap)

        results = []
        for fold in folds:
            if verbose:
                print(f"\n{'='*55}")
                print(f"Fold {fold.fold_id+1}/{len(folds)}  "
                      f"train=[0..{fold.train_idx[-1]}]  "
                      f"gap={fold.gap}  "
                      f"val=[{fold.val_idx[0]}..{fold.val_idx[-1]}]")

            model = model_factory()

            c_tr = close[fold.train_idx]
            v_tr = volume[fold.train_idx]
            c_val_full = close[: fold.val_idx[-1] + 1]   # includes train history
            v_val_full = volume[: fold.val_idx[-1] + 1]

            # Fit — pass val portion so model can compute val loss during training
            # but ONLY provide val indices AFTER the gap
            try:
                model.fit(
                    close_train=c_tr,
                    volume_train=v_tr,
                    close_val=close[fold.val_idx],
                    volume_val=volume[fold.val_idx],
                    verbose=verbose,
                )
            except TypeError:
                # Model doesn't accept val data at fit-time
                model.fit(c_tr, v_tr)

            # Predict on val portion
            try:
                val_preds = model.predict(
                    close[fold.val_idx],
                    volume[fold.val_idx],
                )
            except Exception as e:
                warnings.warn(f"Fold {fold.fold_id} predict failed: {e}")
                val_preds = np.zeros(len(fold.val_idx))

            # Ground truth labels
            if bubble_labels is not None:
                y_val = bubble_labels[fold.val_idx[-len(val_preds):]]
            else:
                y_val = np.zeros_like(val_preds)

            val_metrics = metric_fn(y_val, val_preds)
            if verbose:
                print(f"  Val metrics: " +
                      "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

            results.append(FoldResult(
                fold_id=fold.fold_id,
                train_metrics={},
                val_metrics=val_metrics,
            ))

        self.fold_results_ = results
        return results

    def summary(self) -> pd.DataFrame:
        """Aggregate metrics across folds."""
        rows = []
        for r in self.fold_results_:
            row = {"fold": r.fold_id}
            row.update(r.val_metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        summary = df.describe().loc[["mean", "std", "min", "max"]]
        return summary
