"""Hermetic tests for the reliability scorer."""

from eval.reliability import assess_reliability


def test_low_risk_not_flagged():
    r = assess_reliability(0.9)
    assert r["risk_level"] == "low" and not r["flagged"]
    assert r["reliability_score"] == 0.9


def test_elevated_risk():
    r = assess_reliability(0.5)
    assert r["risk_level"] == "elevated" and not r["flagged"]


def test_high_risk_flagged():
    r = assess_reliability(0.3)
    assert r["risk_level"] == "high" and r["flagged"] is True


def test_calibrator_overrides_raw_confidence():
    class _Cal:
        def predict(self, xs):
            return [0.95 for _ in xs]   # calibrated up

    r = assess_reliability(0.2, _Cal())
    assert r["reliability_score"] == 0.95 and r["risk_level"] == "low"
