"""Shared scoring helpers.

Analyzer scores are *risk-calibrated*, not raw clean-fractions: each analyzer
declares the rate at which a problem starts to matter (``ok``) and the rate at
which the dataset is unusable for that reason (``bad``). A dataset where 5% of
labels are wrong is not "95% fine" — it is a meaningfully noisy dataset, and
the score should say so.
"""

from __future__ import annotations

import numpy as np


def risk_score(rate: float, ok: float, bad: float) -> float:
    """Map a problem rate to a score in [0, 1]. 1.0 at/below ``ok``, 0.0 at/above ``bad``."""
    if rate <= ok:
        return 1.0
    if rate >= bad:
        return 0.0
    return float(1.0 - (rate - ok) / (bad - ok))


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-scores — outliers cannot inflate their own yardstick."""
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if not np.isfinite(mad) or mad == 0:
        # Degenerate spread: fall back to mean absolute deviation, then give up.
        mad = np.nanmean(np.abs(values - med)) / 1.2533 if values.size else 0.0
        if not np.isfinite(mad) or mad == 0:
            return np.zeros_like(values)
    return (values - med) / (1.4826 * mad)


def sample_indices(n: int, cap: int, seed: int) -> np.ndarray:
    """Deterministic, sorted sample of row positions."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def plural(n: int, one: str, many: str | None = None) -> str:
    """'1 known dataset', '3 known datasets' — never 'dataset(s)'."""
    return f"{n:,} {one if n == 1 else (many or one + 's')}"
