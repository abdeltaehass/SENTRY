"""Hallucination & reliability analysis for generated reports.

We treat the **reference report as ground truth** and compare the *asserted*
events (negation-aware, from eval.metrics) in the generation against the reference:

    hallucinated = events asserted by the model but NOT in the reference
    omitted      = events in the reference the model failed to assert
    matched      = events both agree are present

This is a *reference-grounded proxy* for hallucination. True image-grounded
hallucination (the event is absent from the pixels) needs human review. The proxy
is still informative: it flags reports that invent events the reference never has.
"""

from __future__ import annotations

from .metrics import _asserted_events


def event_diff(prediction: str, reference: str) -> dict[str, set[str]]:
    """Per-report matched / hallucinated / omitted asserted events."""
    pred = _asserted_events(prediction)
    ref = _asserted_events(reference)
    return {"matched": pred & ref, "hallucinated": pred - ref, "omitted": ref - pred}


def hallucination_stats(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Aggregate hallucination / omission statistics over a set of reports."""
    n = len(predictions)
    if n == 0:
        return {"n": 0}
    diffs = [event_diff(p, r) for p, r in zip(predictions, references)]

    reports_with_halluc = sum(1 for d in diffs if d["hallucinated"])
    reports_with_omit = sum(1 for d in diffs if d["omitted"])
    total_halluc = sum(len(d["hallucinated"]) for d in diffs)
    total_omit = sum(len(d["omitted"]) for d in diffs)
    total_asserted = sum(len(d["matched"]) + len(d["hallucinated"]) for d in diffs)

    return {
        "n": n,
        "hallucination_rate": 100.0 * reports_with_halluc / n,   # % reports w/ >=1 invented event
        "omission_rate": 100.0 * reports_with_omit / n,
        "mean_hallucinated_per_report": total_halluc / n,
        "mean_omitted_per_report": total_omit / n,
        "asserted_events_total": total_asserted,
        "hallucinated_events_total": total_halluc,
        # of all events the model asserts, how many are supported by the reference
        "asserted_precision": (100.0 * (total_asserted - total_halluc) / total_asserted)
        if total_asserted else 0.0,
    }
