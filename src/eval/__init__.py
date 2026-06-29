"""Evaluation: NLG metrics, incident-event overlap, hallucination analysis, calibration."""

from .analysis import event_diff, hallucination_stats
from .calibration import Calibrator, expected_calibration_error, reliability_bins
from .metrics import (
    benchmark_metrics,
    compute_text_metrics,
    event_overlap,
    score,
)
from .reliability import assess_reliability, load_calibrator

__all__ = [
    "compute_text_metrics", "event_overlap", "score", "benchmark_metrics",
    "event_diff", "hallucination_stats",
    "Calibrator", "expected_calibration_error", "reliability_bins",
    "assess_reliability", "load_calibrator",
]
