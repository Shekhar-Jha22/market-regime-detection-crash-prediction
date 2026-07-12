"""
LPPLS Core Model
================
Implements the Log-Periodic Power Law Singularity (LPPLS) model.

Formula (linearised form):
    ln P(t) = A + B*(tc - t)^m + C*(tc - t)^m * cos(ω*ln(tc - t) - φ)

or equivalently (preferred for fitting, separates linear/nonlinear params):
    ln P(t) = A + B*(tc - t)^m + C1*(tc - t)^m*cos(ω*ln(tc - t))
                                + C2*(tc - t)^m*sin(ω*ln(tc - t))

Nonlinear params  : tc, m, ω     (fitted via optimisation / deep learning)
Linear params     : A, B, C1, C2 (solved analytically via OLS given nonlinear params)

References
----------
Johansen, Ledoit, Sornette (2000); Filimonov & Sornette (2013);
Sornette et al. (2017 confidence indicators)
"""

from __future__ import annotations

import warnings
import itertools
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import lstsq


# ─────────────────────────────────────────────────────────────
# Parameter constraints (canonical LPPLS bounds)
# ─────────────────────────────────────────────────────────────
CONSTRAINTS = {
    "m":   (0.01, 0.99),     # power-law exponent
    "ω":   (6.0,  13.0),     # log-periodic angular frequency
    "tc":  None,             # set dynamically from window
    "B":   (None, 0.0),      # B < 0 for positive bubbles (price accelerating up)
    "D":   (0.0, None),      # D = |B|*m / ω  ≥ 0 (oscillation damping condition)
}

# Filter criteria (Filimonov & Sornette 2013)
FILTER = {
    "m_range":   (0.01, 0.99),
    "ω_range":   (6.0,  13.0),
    "tc_dt_min": 0,           # tc must be after window end (days)
    "tc_dt_max": 365,         # tc must be within 1 year of window end
    "D_min":     0.0,         # oscillation-damping ratio ≥ 0
    "B_max":     0.0,         # B < 0 for positive bubble
    "rss_max":   np.inf,      # optional residual sum-of-squares cap
}


@dataclass
class LPPLSFit:
    """Container for a single LPPLS fit result."""
    tc: float           # critical time (in days from epoch or index)
    m: float            # power-law exponent
    omega: float        # log-periodic frequency
    A: float
    B: float
    C1: float
    C2: float
    rss: float          # residual sum of squares
    n_obs: int          # window length
    t_start: float      # window start (same units as tc)
    t_end: float        # window end
    qualifies: bool = True  # passes filter criteria

    @property
    def phi(self) -> float:
        return np.arctan2(self.C2, self.C1)

    @property
    def C(self) -> float:
        return np.sqrt(self.C1**2 + self.C2**2)

    @property
    def D(self) -> float:
        """Oscillation damping ratio D = m*|B| / (ω*C)"""
        if self.C == 0:
            return np.inf
        return self.m * abs(self.B) / (self.omega * self.C)


# ─────────────────────────────────────────────────────────────
# LPPLS functional form
# ─────────────────────────────────────────────────────────────

def lppls_formula(t: np.ndarray, tc: float, m: float, omega: float,
                  A: float, B: float, C1: float, C2: float) -> np.ndarray:
    """
    Full LPPLS formula (linearised).
    Returns log-price predictions for array t.
    """
    dt = np.maximum(tc - t, 1e-10)   # tc - t  (must be > 0)
    power = dt ** m
    cos_term = np.cos(omega * np.log(dt))
    sin_term = np.sin(omega * np.log(dt))
    return A + B * power + power * (C1 * cos_term + C2 * sin_term)


def _matrix_equation(t: np.ndarray, price: np.ndarray,
                      tc: float, m: float, omega: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the linear system  X @ [A, B, C1, C2]  ≈  price
    where X columns are [1, (tc-t)^m, (tc-t)^m*cos, (tc-t)^m*sin].
    Returns (X, coeffs, residuals).
    """
    dt = np.maximum(tc - t, 1e-10)
    power = dt ** m
    X = np.column_stack([
        np.ones(len(t)),
        power,
        power * np.cos(omega * np.log(dt)),
        power * np.sin(omega * np.log(dt)),
    ])
    coeffs, res, _, _ = lstsq(X, price)
    return X, coeffs, res


def lppls_residuals(nonlinear: np.ndarray,
                    t: np.ndarray,
                    price: np.ndarray) -> float:
    """Objective: RSS as function of nonlinear params [tc, m, ω]."""
    tc, m, omega = nonlinear
    if tc <= t[-1]:          # tc must be in the future
        return 1e20
    if not (0.01 < m < 0.99):
        return 1e20
    if not (6.0 < omega < 13.0):
        return 1e20
    try:
        _, coeffs, res = _matrix_equation(t, price, tc, m, omega)
        A, B, C1, C2 = coeffs
        if B > 0:           # positive bubble requires B < 0
            return 1e20
        predicted = lppls_formula(t, tc, m, omega, A, B, C1, C2)
        rss = np.sum((price - predicted) ** 2)
        return rss
    except Exception:
        return 1e20


# ─────────────────────────────────────────────────────────────
# Single-window fitter
# ─────────────────────────────────────────────────────────────

class LPPLSFitter:
    """
    Fits the LPPLS model to a single price window using
    multi-start Nelder-Mead or differential evolution.
    """

    def __init__(self,
                 method: str = "de",        # 'de' = differential evolution, 'nm' = Nelder-Mead
                 n_restarts: int = 6,       # only used for 'nm'
                 maxiter: int = 2000):
        self.method = method
        self.n_restarts = n_restarts
        self.maxiter = maxiter

    def fit(self, t: np.ndarray, log_price: np.ndarray,
            tc_bounds: Optional[Tuple[float, float]] = None) -> Optional[LPPLSFit]:
        """
        Fit LPPLS to (t, log_price).  t should be in *days from start* (integer).
        tc_bounds: (min_tc, max_tc) in same units as t.
        """
        t_end = t[-1]
        if tc_bounds is None:
            tc_bounds = (t_end + 1, t_end + 365)

        bounds_de = [tc_bounds, (0.01, 0.99), (6.0, 13.0)]

        best_rss = np.inf
        best_params = None

        if self.method == "de":
            try:
                result = differential_evolution(
                    lppls_residuals,
                    bounds=bounds_de,
                    args=(t, log_price),
                    maxiter=self.maxiter,
                    tol=1e-8,
                    seed=42,
                    workers=1,
                    popsize=10,
                )
                if result.fun < best_rss:
                    best_rss = result.fun
                    best_params = result.x
            except Exception:
                pass
        else:
            # Multi-start Nelder-Mead
            tc_range = np.linspace(tc_bounds[0], tc_bounds[1], self.n_restarts)
            for tc0 in tc_range:
                x0 = [tc0, np.random.uniform(0.1, 0.9),
                      np.random.uniform(6.0, 13.0)]
                try:
                    res = minimize(lppls_residuals, x0,
                                   args=(t, log_price),
                                   method="Nelder-Mead",
                                   options={"maxiter": self.maxiter, "xatol": 1e-6})
                    if res.fun < best_rss:
                        best_rss = res.fun
                        best_params = res.x
                except Exception:
                    pass

        if best_params is None or best_rss > 1e15:
            return None

        tc, m, omega = best_params
        _, coeffs, _ = _matrix_equation(t, log_price, tc, m, omega)
        A, B, C1, C2 = coeffs

        fit = LPPLSFit(
            tc=tc, m=m, omega=omega, A=A, B=B, C1=C1, C2=C2,
            rss=best_rss, n_obs=len(t),
            t_start=float(t[0]), t_end=float(t[-1]),
        )
        fit.qualifies = self._passes_filters(fit, t_end)
        return fit

    @staticmethod
    def _passes_filters(fit: LPPLSFit, t_end: float) -> bool:
        """Apply canonical LPPLS filter criteria."""
        if not (0.01 < fit.m < 0.99):
            return False
        if not (6.0 < fit.omega < 13.0):
            return False
        if fit.tc <= t_end:
            return False
        if fit.tc > t_end + 365:
            return False
        if fit.B > 0:
            return False
        if fit.D < 0:
            return False
        return True


# ─────────────────────────────────────────────────────────────
# Multi-window LPPLS + Confidence Indicators
# (Sornette et al. 2017 / Filimonov & Sornette 2013)
# ─────────────────────────────────────────────────────────────

@dataclass
class ConfidenceIndicator:
    """
    LPPLS confidence score at a given time point.
    Fraction of qualifying fits across all window lengths.
    """
    date_idx: int
    pos_confidence: float   # fraction qualifying for positive bubble
    neg_confidence: float   # fraction qualifying for negative bubble
    n_fits: int             # total fits attempted
    mean_tc: float          # mean critical time across qualifying fits
    std_tc: float
    mean_m: float
    mean_omega: float
    mean_B: float


class LPPLSConfidenceModel:
    """
    Multi-window LPPLS with confidence indicators.

    For each observation t, fits LPPLS over many window lengths (dt1..dt2).
    Confidence = fraction of fits that pass ALL filter criteria.

    Uses sliding windows as in the LPPLS implementation repo (sabato96).
    """

    def __init__(self,
                 window_sizes: Optional[List[int]] = None,
                 fitter_method: str = "de",
                 min_window: int = 60,
                 max_window: int = 750,
                 n_windows: int = 20,
                 verbose: bool = False):
        if window_sizes is None:
            # log-spaced window sizes (typical in literature)
            window_sizes = np.unique(
                np.round(np.logspace(
                    np.log10(min_window),
                    np.log10(max_window),
                    n_windows
                )).astype(int)
            ).tolist()
        self.window_sizes = window_sizes
        self.fitter = LPPLSFitter(method=fitter_method)
        self.verbose = verbose
        self.fits_: List[LPPLSFit] = []
        self.confidence_: List[ConfidenceIndicator] = []

    def fit_window(self, log_prices: np.ndarray,
                   t: Optional[np.ndarray] = None) -> List[LPPLSFit]:
        """
        Fit LPPLS over ALL window sizes ending at the last observation.
        Returns list of fits (both qualifying and not).
        """
        n = len(log_prices)
        if t is None:
            t = np.arange(n, dtype=float)
        fits = []
        for ws in self.window_sizes:
            if ws > n:
                continue
            t_win = t[n - ws:]
            p_win = log_prices[n - ws:]
            fit = self.fitter.fit(t_win, p_win,
                                  tc_bounds=(t[-1] + 1, t[-1] + 365))
            if fit is not None:
                fits.append(fit)
        return fits

    def compute_confidence(self, fits: List[LPPLSFit]) -> Tuple[float, float]:
        """
        Compute positive/negative bubble confidence.
        Positive bubble: B < 0 and all filters pass.
        Negative bubble: mirrored (price accelerating down, B > 0).
        """
        if not fits:
            return 0.0, 0.0
        pos = sum(1 for f in fits if f.qualifies and f.B < 0)
        neg = sum(1 for f in fits if f.qualifies and f.B >= 0)
        n = len(fits)
        return pos / n, neg / n

    def rolling_confidence(self,
                           log_prices: np.ndarray,
                           t: Optional[np.ndarray] = None,
                           step: int = 5) -> List[ConfidenceIndicator]:
        """
        Roll through the time series, computing confidence at each step.
        step = how many days to skip between computations (5 = weekly).
        """
        n = len(log_prices)
        if t is None:
            t = np.arange(n, dtype=float)
        min_len = min(self.window_sizes) if self.window_sizes else 60
        indicators = []
        indices = range(min_len, n + 1, step)

        for idx in indices:
            fits = self.fit_window(log_prices[:idx], t[:idx])
            pos_c, neg_c = self.compute_confidence(fits)
            qual = [f for f in fits if f.qualifies]
            if qual:
                tcs = [f.tc for f in qual]
                ms  = [f.m  for f in qual]
                oms = [f.omega for f in qual]
                bs  = [f.B  for f in qual]
            else:
                tcs = ms = oms = bs = [np.nan]
            ci = ConfidenceIndicator(
                date_idx=idx - 1,
                pos_confidence=pos_c,
                neg_confidence=neg_c,
                n_fits=len(fits),
                mean_tc=float(np.nanmean(tcs)),
                std_tc=float(np.nanstd(tcs)),
                mean_m=float(np.nanmean(ms)),
                mean_omega=float(np.nanmean(oms)),
                mean_B=float(np.nanmean(bs)),
            )
            indicators.append(ci)
            if self.verbose:
                print(f"  idx={idx-1}  pos={pos_c:.2f}  neg={neg_c:.2f}  "
                      f"n_qual={len(qual)}/{len(fits)}")
        self.confidence_ = indicators
        return indicators
