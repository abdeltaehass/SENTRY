"""Confidence calibration for generation reliability.

The model emits a generation confidence (geometric-mean token probability), but a
raw confidence of 0.6 doesn't mean "60% likely to be correct". This module
measures that gap (Expected Calibration Error) and fits a calibrator that maps
raw confidence -> P(report is reliable), so a deployed system can threshold on a
*meaningful* probability (e.g. "escalate to a human below 0.5 calibrated").

"Correct/reliable" is supplied by the caller as a 0/1 label per report (e.g.
hallucination-free, from eval.analysis).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """ECE: average |accuracy - confidence| over equal-width confidence bins."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    n = len(conf)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if lo == 0.0:
            mask |= conf == 0.0
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_bins(confidences, correct, n_bins: int = 10) -> list[dict]:
    """Per-bin (mean confidence, accuracy, count) for a reliability diagram."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if lo == 0.0:
            mask |= conf == 0.0
        if mask.sum() == 0:
            continue
        out.append({
            "lo": float(lo), "hi": float(hi), "count": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()),
            "accuracy": float(corr[mask].mean()),
        })
    return out


class Calibrator:
    """Isotonic-regression calibrator: raw confidence -> calibrated P(reliable).

    Fit with sklearn, but store/predict via the monotone thresholds + np.interp,
    so applying a saved calibrator needs only numpy.
    """

    def __init__(self):
        self.x_thresholds_: np.ndarray | None = None
        self.y_thresholds_: np.ndarray | None = None

    def fit(self, confidences, correct) -> "Calibrator":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float))
        self.x_thresholds_ = np.asarray(iso.X_thresholds_, dtype=float)
        self.y_thresholds_ = np.asarray(iso.y_thresholds_, dtype=float)
        return self

    def predict(self, confidences) -> np.ndarray:
        if self.x_thresholds_ is None:
            raise RuntimeError("Calibrator is not fitted")
        return np.interp(np.asarray(confidences, dtype=float),
                         self.x_thresholds_, self.y_thresholds_)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "x_thresholds": self.x_thresholds_.tolist(),
            "y_thresholds": self.y_thresholds_.tolist(),
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls()
        c.x_thresholds_ = np.asarray(d["x_thresholds"], dtype=float)
        c.y_thresholds_ = np.asarray(d["y_thresholds"], dtype=float)
        return c
