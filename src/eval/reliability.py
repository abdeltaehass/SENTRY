"""Reliability scoring for generated incident reports.

Turns the raw generation confidence into an operator-facing reliability score and
a risk level. If a fitted calibrator (eval.calibration) is provided, the score is
the calibrated P(reliable); otherwise it's the raw confidence. High-risk reports
are flagged so a human can review them before acting.
"""

from __future__ import annotations

from pathlib import Path

# Thresholds on the reliability score (heuristic — tune on a labelled val set).
LOW_RISK_AT = 0.66
ELEVATED_RISK_AT = 0.40


def assess_reliability(confidence: float, calibrator=None) -> dict:
    """Map a confidence (or calibrated probability) to a risk assessment."""
    score = float(calibrator.predict([confidence])[0]) if calibrator is not None else float(confidence)
    score = max(0.0, min(1.0, score))
    if score >= LOW_RISK_AT:
        level, flagged = "low", False
    elif score >= ELEVATED_RISK_AT:
        level, flagged = "elevated", False
    else:
        level, flagged = "high", True
    return {"reliability_score": score, "risk_level": level, "flagged": flagged}


def load_calibrator(path: str | Path = "outputs/calibrator.json"):
    """Load a saved calibrator if one exists, else None (use raw confidence)."""
    from .calibration import Calibrator

    p = Path(path)
    return Calibrator.load(p) if p.exists() else None
