"""Hermetic tests for the eval package (metrics, analysis, calibration) + multiview."""

import numpy as np
from PIL import Image

from eval.analysis import event_diff, hallucination_stats
from eval.calibration import Calibrator, expected_calibration_error, reliability_bins
from eval.metrics import (
    _asserted_events,
    _mentioned_events,
    benchmark_metrics,
    event_overlap,
)
from model.multiview import compose_views


# --- event overlap ----------------------------------------------------------

def test_mentioned_events():
    assert _mentioned_events("an intruder with a knife") == {"intrusion", "weapon"}
    assert _mentioned_events("the area is empty and quiet") == set()


def test_negation_excludes_from_asserted():
    text = "intruder present, no weapon visible"
    assert _mentioned_events(text) == {"intrusion", "weapon"}
    assert _asserted_events(text) == {"intrusion"}  # weapon is negated


def test_event_overlap_perfect_mention():
    m = event_overlap(["intruder with a weapon"], ["an armed intruder"])
    assert m["event_f1"] == 100.0


# --- hallucination analysis -------------------------------------------------

def test_event_diff_and_stats():
    d = event_diff("intruder with a weapon", "intruder, no weapon")
    assert d["matched"] == {"intrusion"}
    assert d["hallucinated"] == {"weapon"}
    assert d["omitted"] == set()

    s = hallucination_stats(["intruder with a weapon"], ["intruder, no weapon"])
    assert s["hallucination_rate"] == 100.0
    assert s["asserted_precision"] == 50.0  # 2 asserted, 1 supported


# --- benchmark metrics ------------------------------------------------------

def test_benchmark_metrics_identical():
    m = benchmark_metrics(["an intruder climbs the fence"], ["an intruder climbs the fence"])
    assert m["bleu1"] > 0.9
    assert m["rougeL"] > 0.99


# --- calibration ------------------------------------------------------------

def test_ece_overconfident_is_high():
    assert expected_calibration_error(np.full(20, 0.95), np.zeros(20)) > 0.8


def test_calibrator_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 400)
    correct = (rng.uniform(0, 1, 400) < conf**2).astype(float)
    cal = Calibrator().fit(conf, correct)
    p = cal.predict([0.1, 0.9])
    assert p.shape == (2,) and 0.0 <= p[0] <= 1.0
    path = tmp_path / "cal.json"
    cal.save(path)
    assert np.allclose(p, Calibrator.load(path).predict([0.1, 0.9]))
    assert sum(b["count"] for b in reliability_bins(conf, correct)) == 400


# --- multi-view -------------------------------------------------------------

def test_compose_views_shapes():
    a = Image.new("RGB", (40, 50), (10, 10, 10))
    b = Image.new("RGB", (60, 30), (200, 200, 200))
    a.save("/tmp/_sv0.png"); b.save("/tmp/_sv1.png")
    assert compose_views(["/tmp/_sv0.png"]).size == (40, 50)            # single unchanged
    assert compose_views(["/tmp/_sv0.png", "/tmp/_sv1.png"], size=224).size == (448, 224)
