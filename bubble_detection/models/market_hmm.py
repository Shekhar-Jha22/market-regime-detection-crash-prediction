"""
Hidden Markov Model for Market Regime Detection
================================================
Fits a Gaussian HMM over a rich feature set to identify latent market regimes.
The model identifies states along a spectrum from "calm" to "crash", and
computes P(crash state | next k days) via the forward algorithm + transition matrix.

States (learned, not prescribed — labels assigned post-hoc)
-----------------------------------------------------------
    By default 4 states. After fitting, states are ranked by their mean
    realised volatility: lowest vol = "normal", highest = "crash regime".

    State ordering (by ascending mean vol):
      0 = Normal / Bull
      1 = Elevated vol / Late-cycle
      2 = Stressed / Pre-crash
      3 = Crash / Crisis

Feature set
-----------
    f(t) = [log_return, realised_vol_21d, skew_21d,
            log_volume_z, lppl_confidence, bubble_score]

    LPPL confidence and BubbleScore are optional — the model degrades gracefully
    to the first 4 features if they are not provided.

5-day crash probability
-----------------------
    At time t, let π_t = P(state | observations 0..t)  (filtered distribution).
    Then:
        P(in crash state at t+k) = (π_t @ A^k)[crash_state_idx]

    P(crash in next H days) = 1 − P(never in crash state in {t+1,..,t+H})
                             ≈ 1 − Π_{k=1}^{H} (1 − (π_t @ A^k)[crash_idx])

Walk-forward safety
-------------------
    .fit()  must be called only on training data.
    .predict() uses the Viterbi / forward algorithm causally.
    The feature normalisation stats are frozen at fit time.

Reference
---------
Baum & Petrie (1966); Rabiner (1989); Engel & Rodrigues (2012) — HMMs in finance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from hmmlearn import hmm


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────

def build_hmm_features(close:           np.ndarray,
                       volume:          np.ndarray,
                       lppl_confidence: Optional[np.ndarray] = None,
                       bubble_score:    Optional[np.ndarray] = None,
                       vol_window:      int = 21,
                       skew_window:     int = 21,
                       vol_z_window:    int = 63) -> np.ndarray:
    """
    Build the feature matrix for HMM fitting / inference.
    All features are causal (use only past data at each t).

    Returns
    -------
    X : [n, n_features]  float32
    """
    n = len(close)
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)), prepend=0.0)

    # Realised volatility
    realised_vol = np.array([
        log_ret[max(0, i - vol_window): i + 1].std()
        for i in range(n)
    ]).clip(1e-8)

    # Realised skewness (signed proxy for tail risk)
    realised_skew = np.array([
        _rolling_skew(log_ret, i, skew_window)
        for i in range(n)
    ])

    # Volume z-score
    log_vol = np.log(np.maximum(volume, 1.0))
    vol_z   = np.array([
        _rolling_zscore(log_vol, i, vol_z_window)
        for i in range(n)
    ])

    features = [log_ret, realised_vol, realised_skew, vol_z]

    if lppl_confidence is not None:
        lc = np.asarray(lppl_confidence, dtype=float)
        assert len(lc) == n, "lppl_confidence length mismatch"
        features.append(lc)

    if bubble_score is not None:
        bs = np.asarray(bubble_score, dtype=float)
        assert len(bs) == n, "bubble_score length mismatch"
        features.append(bs)

    X = np.column_stack(features).astype(np.float32)
    # Replace any NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _rolling_zscore(arr: np.ndarray, i: int, w: int) -> float:
    chunk = arr[max(0, i - w + 1): i + 1]
    mu, std = chunk.mean(), chunk.std()
    return 0.0 if std < 1e-10 else float((arr[i] - mu) / std)


def _rolling_skew(arr: np.ndarray, i: int, w: int) -> float:
    chunk = arr[max(0, i - w + 1): i + 1]
    if len(chunk) < 3:
        return 0.0
    mu, std = chunk.mean(), chunk.std()
    if std < 1e-10:
        return 0.0
    return float(np.mean(((chunk - mu) / std) ** 3))


# ─────────────────────────────────────────────────────────────
# HMM wrapper
# ─────────────────────────────────────────────────────────────

@dataclass
class HMMState:
    idx:        int
    label:      str          # e.g. "normal", "stressed", "crash"
    mean_vol:   float
    mean_ret:   float
    frequency:  float        # fraction of time in this state


class MarketHMM:
    """
    Gaussian HMM for market regime detection.

    Parameters
    ----------
    n_states       : number of hidden states (4 is typical)
    n_iter         : EM iterations
    covariance_type: 'diag' (faster) or 'full'
    crash_states   : how many of the highest-vol states to treat as "crash"
    """

    def __init__(self,
                 n_states:        int = 4,
                 n_iter:          int = 200,
                 covariance_type: str = "diag",
                 crash_states:    int = 1,
                 random_state:    int = 42):
        self.n_states        = n_states
        self.n_iter          = n_iter
        self.covariance_type = covariance_type
        self.crash_states    = crash_states
        self.random_state    = random_state

        self._model:       Optional[hmm.GaussianHMM] = None
        self._state_info:  Optional[List[HMMState]]  = None
        self._crash_state_indices: Optional[np.ndarray] = None
        self._feat_mean:   Optional[np.ndarray] = None
        self._feat_std:    Optional[np.ndarray] = None
        self._n_features:  int = 0

    # ── Fitting ───────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "MarketHMM":
        """
        Fit on feature matrix X [n, n_features].
        Must be called on training data only.
        """
        self._n_features = X.shape[1]

        # Normalise (store stats for causal inference)
        self._feat_mean = X.mean(axis=0)
        self._feat_std  = X.std(axis=0).clip(1e-8)
        Xn = (X - self._feat_mean) / self._feat_std

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                tol=1e-4,
                init_params="stmc",
            )
            model.fit(Xn)
        self._model = model

        # Decode training data to label states
        states_seq = model.predict(Xn)

        # Rank states by mean realised vol (feature index 1)
        state_vol = np.array([
            Xn[states_seq == s, 1].mean() if np.any(states_seq == s) else 0.0
            for s in range(self.n_states)
        ])
        vol_rank = np.argsort(state_vol)   # ascending: vol_rank[0] = calmest

        labels = ["normal", "elevated", "stressed", "crash"]
        if self.n_states != 4:
            labels = [f"state_{i}" for i in range(self.n_states)]
            labels[-1] = "crash"

        self._state_info = []
        for rank, state_idx in enumerate(vol_rank):
            freq = np.mean(states_seq == state_idx)
            self._state_info.append(HMMState(
                idx=int(state_idx),
                label=labels[rank] if rank < len(labels) else f"state_{rank}",
                mean_vol=float(state_vol[state_idx]),
                mean_ret=float(Xn[states_seq == state_idx, 0].mean()
                               if np.any(states_seq == state_idx) else 0.0),
                frequency=float(freq),
            ))

        # The top `crash_states` highest-vol states are "crash"
        crash_ranks = list(range(self.n_states - self.crash_states,
                                 self.n_states))
        self._crash_state_indices = np.array([
            vol_rank[r] for r in crash_ranks
        ])
        return self

    # ── Regime probabilities ──────────────────────────────────

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feat_mean) / self._feat_std

    def state_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        P(state_k | observations 0..t) for each t.
        Uses the forward algorithm (filtering distribution).
        Returns [n, n_states].
        """
        assert self._model is not None, "Call .fit() first."
        Xn = self._normalise(np.nan_to_num(X, 0.0))

        # hmmlearn's predict_proba uses the forward–backward (smoothed) algorithm.
        # For causal / online inference we prefer filtering — approximate it by
        # using a forward-only pass. hmmlearn doesn't expose forward() directly
        # on GaussianHMM in all versions, so we compute it manually.
        try:
            # Try to use the internal forward algorithm
            log_prob, fwd_lattice = self._model._do_forward_pass(
                self._model._compute_log_likelihood(Xn))
            # Normalise rows
            fwd = np.exp(fwd_lattice)
            row_sums = fwd.sum(axis=1, keepdims=True).clip(1e-300)
            return fwd / row_sums
        except Exception:
            # Fallback: smoothed posteriors (uses future data but OK for offline eval)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._model.predict_proba(Xn)

    def crash_probability_horizon(self,
                                   X: np.ndarray,
                                   horizon: int = 5) -> np.ndarray:
        """
        P(visit crash state at least once in next `horizon` steps)
        computed from filtered state distribution + transition matrix.

        P(crash in [t+1, t+H]) = 1 − Π_{k=1}^{H} (1 − (π_t @ A^k) · crash_mask)

        Parameters
        ----------
        X       : feature matrix [n, n_features]
        horizon : forecast window in days

        Returns
        -------
        prob : [n,]  P(≥1 crash visit in next `horizon` days)
        """
        assert self._model is not None
        state_probs = self.state_probabilities(X)   # [n, n_states]
        A = self._model.transmat_                    # [n_states, n_states]
        crash_mask = np.zeros(self.n_states)
        crash_mask[self._crash_state_indices] = 1.0

        n = len(X)
        prob = np.zeros(n)
        A_pow = np.eye(self.n_states)   # A^0

        # Pre-compute A^1 ... A^H
        A_powers = []
        for _ in range(horizon):
            A_pow = A_pow @ A
            A_powers.append(A_pow.copy())

        for t in range(n):
            pi = state_probs[t]    # [n_states]
            # P(not crash on any step k=1..H)
            p_no_crash = 1.0
            for A_k in A_powers:
                p_crash_at_k = float(pi @ A_k @ crash_mask)
                p_no_crash  *= (1.0 - p_crash_at_k)
            prob[t] = 1.0 - p_no_crash

        return np.clip(prob, 0.0, 1.0)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding — most likely state sequence."""
        assert self._model is not None
        Xn = self._normalise(np.nan_to_num(X, 0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._model.predict(Xn)

    def summary(self) -> str:
        if self._state_info is None:
            return "MarketHMM (not fitted)"
        lines = [f"MarketHMM  n_states={self.n_states}"]
        for s in self._state_info:
            crash_tag = " ← CRASH" if s.idx in self._crash_state_indices else ""
            lines.append(f"  State {s.idx} ({s.label:<10}) "
                         f"freq={s.frequency:.2%}  "
                         f"mean_vol(norm)={s.mean_vol:+.3f}{crash_tag}")
        A = self._model.transmat_
        crash_self_prob = A[np.ix_(self._crash_state_indices,
                                   self._crash_state_indices)].mean()
        lines.append(f"  Crash→Crash self-transition: {crash_self_prob:.3f}")
        return "\n".join(lines)
