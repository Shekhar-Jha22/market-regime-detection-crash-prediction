"""
Hawkes Process for Crash Probability
=====================================
A Hawkes process is a self-exciting point process where each past extreme event
temporarily elevates the probability of future events — exactly the clustering
pattern seen in financial crashes ("volatility begets volatility").

Model
-----
    λ(t) = μ + Σ_{tᵢ < t} α · exp(-β · (t - tᵢ))

    μ  : baseline intensity (events/day on quiet days)
    α  : jump in intensity caused by each new event  (α < β for stationarity)
    β  : exponential decay rate (how fast excitement fades)

Events
------
    We define "crash events" as days where the log-return falls below a
    threshold τ (default τ = -1.5σ, i.e. roughly -2% on equities).
    Both single-day events and multi-day runs are supported.

5-day crash probability
-----------------------
    Exact survival analysis via the conditional intensity:

        P(at least 1 event in [t, t+H]) = 1 − exp(−Λ(t, t+H))

    where Λ(t, t+H) = ∫_t^{t+H} λ(s) ds  is the integrated intensity.
    For the exponential kernel this integral has a closed form.

Fitting
-------
    Maximum likelihood via L-BFGS-B (log-likelihood of the Hawkes process).
    Handles compensator term analytically for efficiency.

Reference
---------
Hawkes (1971); Bacry, Mastromatteo & Muzy (2015) market microstructure review.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def detect_crash_events(log_returns: np.ndarray,
                        threshold_sigma: float = 1.5,
                        min_gap: int = 1) -> np.ndarray:
    """
    Flag days whose log-return is below −threshold_sigma × rolling σ.

    Parameters
    ----------
    log_returns      : daily log-returns
    threshold_sigma  : number of σ below mean to call an event
    min_gap          : minimum gap between consecutive events (deduplication)

    Returns
    -------
    event_times : integer day indices where crash events occur
    """
    n = len(log_returns)
    roll_w = 63   # ~3-month rolling window for σ
    rolling_std  = np.array([
        log_returns[max(0, i - roll_w): i + 1].std()
        for i in range(n)
    ]).clip(1e-8)
    rolling_mean = np.array([
        log_returns[max(0, i - roll_w): i + 1].mean()
        for i in range(n)
    ])
    threshold = rolling_mean - threshold_sigma * rolling_std
    raw_events = np.where(log_returns < threshold)[0]

    # Deduplicate: keep only the first event in each cluster
    if len(raw_events) == 0:
        return raw_events
    deduped = [raw_events[0]]
    for t in raw_events[1:]:
        if t - deduped[-1] >= min_gap:
            deduped.append(t)
    return np.array(deduped)


# ─────────────────────────────────────────────────────────────
# Log-likelihood of Hawkes process (exponential kernel)
# ─────────────────────────────────────────────────────────────

def hawkes_log_likelihood(params: np.ndarray,
                          event_times: np.ndarray,
                          T: float) -> float:
    """
    Negative log-likelihood for Hawkes process with exponential kernel.
    params = [log_mu, log_alpha, log_beta]  (log-space for positivity)

    Exact closed-form compensator:
      Λ(0, T) = μ·T + (α/β) · Σ_i (1 − exp(−β·(T − tᵢ)))

    Log-likelihood:
      ℓ = Σ_i log(λ(tᵢ)) − Λ(0, T)
    """
    mu, alpha, beta = np.exp(params)
    if alpha >= beta:           # stationarity condition α < β
        return 1e20

    events = event_times.astype(float)
    n_ev   = len(events)

    # — Conditional intensity at each event time —
    # λ(tᵢ) = μ + α · Σ_{j < i} exp(-β·(tᵢ - tⱼ))
    # Computed recursively in O(n) for speed
    log_sum = 0.0
    R = 0.0   # recursive running sum of excitation
    for i in range(n_ev):
        if i == 0:
            R = 0.0
        else:
            R = np.exp(-beta * (events[i] - events[i - 1])) * (1.0 + R)
        lam_i = mu + alpha * R
        if lam_i <= 0:
            return 1e20
        log_sum += np.log(lam_i)

    # — Compensator Λ(0, T) —
    compensator = mu * T
    for t_i in events:
        compensator += (alpha / beta) * (1.0 - np.exp(-beta * (T - t_i)))

    return -(log_sum - compensator)


@dataclass
class HawkesParams:
    mu:    float   # baseline intensity
    alpha: float   # excitation jump
    beta:  float   # decay rate
    log_likelihood: float = 0.0

    @property
    def branching_ratio(self) -> float:
        """n = α/β. n < 1 required for stationarity."""
        return self.alpha / self.beta

    @property
    def mean_intensity(self) -> float:
        """Unconditional mean rate = μ / (1 − n)"""
        n = self.branching_ratio
        if n >= 1:
            return float("inf")
        return self.mu / (1.0 - n)


class HawkesProcess:
    """
    Univariate Hawkes process with exponential kernel.

    Usage
    -----
    hp = HawkesProcess()
    hp.fit(log_returns_train)
    prob5 = hp.crash_probability_5d(log_returns_full)
    """

    def __init__(self,
                 threshold_sigma: float = 1.5,
                 n_restarts: int = 10,
                 min_events: int = 5):
        self.threshold_sigma = threshold_sigma
        self.n_restarts      = n_restarts
        self.min_events      = min_events
        self.params_: Optional[HawkesParams] = None

    # ── Fitting ───────────────────────────────────────────────

    def fit(self, log_returns: np.ndarray) -> "HawkesProcess":
        """
        Fit Hawkes parameters via MLE on the training log-return series.
        """
        events = detect_crash_events(log_returns, self.threshold_sigma)
        T = float(len(log_returns))

        if len(events) < self.min_events:
            warnings.warn(
                f"Only {len(events)} crash events detected (need ≥ {self.min_events}). "
                "Falling back to baseline intensity only.")
            mu_fallback = max(len(events), 1) / T
            self.params_ = HawkesParams(mu=mu_fallback, alpha=0.01, beta=1.0)
            self.event_times_train_ = events
            return self

        best_ll  = np.inf
        best_par = None

        for seed in range(self.n_restarts):
            rng = np.random.default_rng(seed)
            # Random init in log-space
            log_mu    = rng.uniform(-5, -1)
            log_alpha = rng.uniform(-5, -1)
            log_beta  = rng.uniform(-3,  1)
            x0 = np.array([log_mu, log_alpha, log_beta])
            try:
                res = minimize(
                    hawkes_log_likelihood,
                    x0,
                    args=(events, T),
                    method="L-BFGS-B",
                    bounds=[(-10, 2), (-10, 2), (-5, 5)],
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
                if res.fun < best_ll:
                    best_ll  = res.fun
                    best_par = res.x
            except Exception:
                pass

        if best_par is None:
            raise RuntimeError("Hawkes MLE failed across all restarts.")

        mu, alpha, beta = np.exp(best_par)
        self.params_ = HawkesParams(mu=mu, alpha=alpha, beta=beta,
                                    log_likelihood=-best_ll)
        self.event_times_train_ = events
        return self

    # ── Conditional intensity ─────────────────────────────────

    def conditional_intensity(self,
                               t: float,
                               past_events: np.ndarray) -> float:
        """
        λ(t | history) = μ + Σ_{tᵢ < t} α·exp(-β·(t - tᵢ))
        """
        assert self.params_ is not None, "Call .fit() first."
        p = self.params_
        excitation = p.alpha * np.sum(
            np.exp(-p.beta * (t - past_events[past_events < t])))
        return p.mu + excitation

    # ── Integrated intensity (closed form) ───────────────────

    def _integrated_intensity(self,
                               t_start: float,
                               t_end: float,
                               past_events: np.ndarray) -> float:
        """
        Λ(t_start, t_end) = ∫_{t_start}^{t_end} λ(s) ds

        For exponential kernel:
          ∫_a^b [μ + Σ_i α·exp(-β·(s-tᵢ))] ds
          = μ·(b−a) + Σ_i (α/β)·[exp(-β·(a−tᵢ)) − exp(-β·(b−tᵢ))]  for tᵢ < a
          + Σ_i (α/β)·[1 − exp(-β·(b−tᵢ))]                            for a ≤ tᵢ < b
        """
        p = self.params_
        dt = t_end - t_start
        Lambda = p.mu * dt

        for ti in past_events:
            if ti < t_start:
                Lambda += (p.alpha / p.beta) * (
                    np.exp(-p.beta * (t_start - ti)) -
                    np.exp(-p.beta * (t_end   - ti))
                )
            elif ti < t_end:
                Lambda += (p.alpha / p.beta) * (
                    1.0 - np.exp(-p.beta * (t_end - ti))
                )
        return Lambda

    # ── 5-day crash probability (rolling) ────────────────────

    def crash_probability(self,
                          log_returns: np.ndarray,
                          horizon: int = 5,
                          use_train_history: bool = True) -> np.ndarray:
        """
        For each day t in log_returns, compute:
            P(≥1 crash in [t, t+horizon]) = 1 − exp(−Λ(t, t+horizon))

        Parameters
        ----------
        log_returns        : full return series (train + test)
        horizon            : forecast window in days
        use_train_history  : include training events as initial history

        Returns
        -------
        prob : array of length len(log_returns), P(crash | next `horizon` days)
        """
        assert self.params_ is not None, "Call .fit() first."
        n = len(log_returns)

        # All crash events in the full series
        all_events = detect_crash_events(log_returns, self.threshold_sigma)

        probs = np.zeros(n)
        for t in range(n):
            # Only use events strictly before t (causal)
            past = all_events[all_events < t]
            Lambda = self._integrated_intensity(
                float(t), float(t + horizon), past)
            probs[t] = 1.0 - np.exp(-Lambda)

        return probs

    def summary(self) -> str:
        if self.params_ is None:
            return "HawkesProcess (not fitted)"
        p = self.params_
        return (f"HawkesProcess  μ={p.mu:.4f}  α={p.alpha:.4f}  β={p.beta:.4f}  "
                f"n=α/β={p.branching_ratio:.3f}  "
                f"E[λ]={p.mean_intensity:.4f}  LL={p.log_likelihood:.2f}")
