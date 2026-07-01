"""Structured incident-report schema — the productionizable output layer.

Freeform text is a demo; a validated JSON object is something a downstream system
(SOC dashboard, alerting rule, case-management API) can actually consume. This
module turns SENTRY's signals into one typed record:

    {
      "timestamp": "14:32:07",
      "location": "east entrance",
      "incident_type": "unattended object",
      "confidence": 0.81,
      "description": "Individual left a backpack near the door and exited frame",
      "hallucination_flag": false,
      "grounding_regions": ["lower-left quadrant"]
    }

The fields are assembled from what the pipeline already produces, so the schema
is grounded rather than re-hallucinated:

  - ``description``        the generated report text
  - ``incident_type``     classified from the description with the same event
                          lexicon + negation handling as event-F1
                          (``eval.metrics.detect_events``)
  - ``confidence`` /      the generation confidence and the reliability flag
    ``hallucination_flag``(``eval.reliability.assess_reliability``)
  - ``grounding_regions`` the Grad-CAM heatmap reduced to named image regions
  - ``timestamp`` /       carried from frame/camera metadata (or a demo stand-in)
    ``location``

Two entry points cover both implementation routes the task allows:

  - ``IncidentReport.from_signals`` — the **post-generation** assembly path,
    always emits a schema-valid object.
  - ``parse_incident`` — a robust **parse + validate + repair** layer for JSON a
    model emits directly (e.g. under constrained/guided decoding), tolerant of
    prose around the object and light formatting damage.

``INCIDENT_JSON_SCHEMA`` is the JSON Schema for the record — usable for external
validation and ready to feed a grammar-constrained decoder.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from eval.metrics import detect_events

# --- controlled vocabulary --------------------------------------------------

# Event categories (eval.metrics) mapped onto operator-facing incident types.
_EVENT_TO_INCIDENT: dict[str, str] = {"abandoned object": "unattended object"}

# Severity-ordered: the most critical asserted event wins when several co-occur.
_INCIDENT_PRIORITY: tuple[str, ...] = (
    "weapon", "violence", "fire/smoke", "intrusion", "theft", "abandoned object",
    "vandalism", "fall", "tailgating", "vehicle", "loitering", "crowd",
)

NORMAL_INCIDENT = "normal"
UNKNOWN_INCIDENT = "other"

INCIDENT_TYPES: tuple[str, ...] = (
    *(_EVENT_TO_INCIDENT.get(e, e) for e in _INCIDENT_PRIORITY),
    NORMAL_INCIDENT, UNKNOWN_INCIDENT,
)

_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


class SchemaError(ValueError):
    """Raised when a record cannot be made schema-valid (e.g. empty description)."""


# --- incident classification ------------------------------------------------


# An object left/placed and (implicitly) walked away from. The shared event
# lexicon's "abandoned object" terms are deliberately narrow for event-F1, so the
# structured layer adds object-abandonment cues here without perturbing that metric.
_OBJECT_NOUN = re.compile(
    r"\b(bag|backpack|rucksack|suitcase|luggage|package|parcel|box|item|object|"
    r"belongings|briefcase|duffel)\b", re.IGNORECASE)
_LEAVE_VERB = re.compile(
    r"\b(unattended|abandon\w*|left behind|left|leaves|leaving|drops?|dropped|"
    r"places?|placed|sets? down|set down|discard\w*|ditch\w*|dump\w*)\b", re.IGNORECASE)


def _is_abandonment(text: str) -> bool:
    """True when the text describes an object being left/placed unattended."""
    t = text or ""
    if re.search(r"\bunattended\b|\babandon\w*\b", t, re.IGNORECASE):
        return True
    return bool(_OBJECT_NOUN.search(t) and _LEAVE_VERB.search(t))


def classify_incident(description: str) -> str:
    """Pick the incident type from a description via the shared event lexicon.

    Returns the most severe asserted event (mapped to an incident label), or
    ``"normal"`` when nothing anomalous is asserted.
    """
    events = detect_events(description or "", asserted_only=True, word_boundary=True)
    if _is_abandonment(description or ""):
        events = events | {"abandoned object"}
    for event in _INCIDENT_PRIORITY:
        if event in events:
            return _EVENT_TO_INCIDENT.get(event, event)
    return NORMAL_INCIDENT


def _coerce_incident_type(value) -> str:
    """Normalize a free-form incident_type onto the controlled vocabulary."""
    text = str(value or "").strip().lower()
    if text in INCIDENT_TYPES:
        return text
    return _EVENT_TO_INCIDENT.get(text, UNKNOWN_INCIDENT if text else NORMAL_INCIDENT)


# --- grounding heatmap -> named regions -------------------------------------

_QUADRANT_LABELS = {
    (0, 0): "upper-left quadrant", (0, 1): "upper-right quadrant",
    (1, 0): "lower-left quadrant", (1, 1): "lower-right quadrant",
}


def regions_from_cam(cam, *, min_share: float = 0.30) -> list[str]:
    """Reduce a Grad-CAM grid to the image quadrant(s) it concentrates on.

    Splits the map into 2x2 quadrants, and returns those holding at least
    ``min_share`` of the total activation (always at least the hottest), ordered
    by activation. An empty/flat map yields ``[]``.
    """
    import numpy as np

    arr = np.asarray(cam, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return []
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return []

    h, w = arr.shape
    my, mx = max(1, h // 2), max(1, w // 2)
    quads = {
        (0, 0): arr[:my, :mx].sum(), (0, 1): arr[:my, mx:].sum(),
        (1, 0): arr[my:, :mx].sum(), (1, 1): arr[my:, mx:].sum(),
    }
    ranked = sorted(quads.items(), key=lambda kv: kv[1], reverse=True)
    picked = [cell for cell, mass in ranked if mass / total >= min_share]
    if not picked:
        picked = [ranked[0][0]]
    return [_QUADRANT_LABELS[cell] for cell in picked]


# --- timestamp --------------------------------------------------------------


def now_timestamp() -> str:
    """Wall-clock ``HH:MM:SS`` — a stand-in when no frame/camera time is known."""
    return datetime.now().strftime("%H:%M:%S")


def _coerce_timestamp(value) -> str | None:
    """Normalize to a valid ``HH:MM:SS``; salvage an ISO/short time; else None.

    Range-validated via ``strptime`` (so ``25:99:99`` is rejected, not just
    shape-matched).
    """
    if value is None:
        return None
    text = str(value).strip()
    for fmt, width in (("%H:%M:%S", 8), ("%Y-%m-%dT%H:%M:%S", 19),
                       ("%Y-%m-%d %H:%M:%S", 19), ("%H:%M", 5)):
        try:
            return datetime.strptime(text[:width], fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    return None


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "flagged")
    return bool(value)


def _coerce_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf != conf:  # NaN
        return 0.0
    return max(0.0, min(1.0, conf))


def _coerce_regions(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    try:
        items = list(value)
    except TypeError:
        return []
    return [str(v).strip() for v in items if str(v).strip()]


# --- the record -------------------------------------------------------------


@dataclass
class IncidentReport:
    """A validated structured incident report (see module docstring for the schema)."""

    incident_type: str
    confidence: float
    description: str
    hallucination_flag: bool = False
    grounding_regions: list[str] = field(default_factory=list)
    timestamp: str | None = None
    location: str | None = None

    @classmethod
    def from_signals(
        cls,
        description: str,
        *,
        confidence: float,
        reliability: dict | None = None,
        cam=None,
        incident_type: str | None = None,
        timestamp: str | None = None,
        location: str | None = None,
    ) -> IncidentReport:
        """Assemble a schema-valid record from the pipeline's signals.

        ``reliability`` is an ``assess_reliability`` dict (supplies the flag and,
        if present, the calibrated score); ``cam`` is a Grad-CAM grid. Any field
        left ``None`` is derived: ``incident_type`` from the description,
        ``grounding_regions`` from ``cam``.
        """
        rel = reliability or {}
        score = rel.get("reliability_score", confidence)
        record = cls(
            incident_type=incident_type or classify_incident(description),
            confidence=score,
            description=description,
            hallucination_flag=bool(rel.get("flagged", False)),
            grounding_regions=regions_from_cam(cam) if cam is not None else [],
            timestamp=timestamp,
            location=location,
        )
        return record.validated()

    def validated(self) -> IncidentReport:
        """Coerce every field onto the schema, in place; raise only if unusable."""
        self.description = str(self.description or "").strip()
        if not self.description:
            raise SchemaError("description is required and must be non-empty")
        self.incident_type = _coerce_incident_type(self.incident_type)
        self.confidence = _coerce_confidence(self.confidence)
        self.hallucination_flag = _coerce_bool(self.hallucination_flag)
        self.grounding_regions = _coerce_regions(self.grounding_regions)
        self.timestamp = _coerce_timestamp(self.timestamp)
        self.location = str(self.location).strip() if self.location else None
        return self

    def to_dict(self) -> dict:
        """Ordered dict matching the documented schema."""
        return {
            "timestamp": self.timestamp,
            "location": self.location,
            "incident_type": self.incident_type,
            "confidence": round(self.confidence, 4),
            "description": self.description,
            "hallucination_flag": self.hallucination_flag,
            "grounding_regions": self.grounding_regions,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# --- robust parse of model-emitted JSON -------------------------------------


def extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` in ``text`` (string-literal aware).

    Constrained/guided decoding may wrap the object in prose or code fences; this
    recovers just the object so ``json.loads`` gets clean input.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def coerce_incident(obj: dict) -> IncidentReport:
    """Build a validated record from a (possibly messy) dict, repairing fields."""
    if not isinstance(obj, dict):
        raise SchemaError(f"expected a JSON object, got {type(obj).__name__}")
    record = IncidentReport(
        incident_type=obj.get("incident_type", ""),
        confidence=obj.get("confidence", 0.0),
        description=obj.get("description", ""),
        hallucination_flag=obj.get("hallucination_flag", False),
        grounding_regions=obj.get("grounding_regions", []),
        timestamp=obj.get("timestamp"),
        location=obj.get("location"),
    ).validated()
    # A missing or unrecognized incident_type is re-derived from the description
    # (a blank type coerces to "normal", an unknown one to "other").
    raw = str(obj.get("incident_type") or "").strip()
    if not raw or record.incident_type == UNKNOWN_INCIDENT:
        record.incident_type = classify_incident(record.description)
    return record


def parse_incident(source) -> IncidentReport:
    """Parse a dict or JSON-ish string into a validated :class:`IncidentReport`.

    Tolerates prose/code-fences around the object. Raises :class:`SchemaError`
    when no JSON object can be recovered or the description is empty.
    """
    if isinstance(source, IncidentReport):
        return source.validated()
    if isinstance(source, dict):
        return coerce_incident(source)
    snippet = extract_json_object(str(source))
    if snippet is None:
        raise SchemaError("no JSON object found in model output")
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc}") from exc
    return coerce_incident(obj)


# --- guided-decoding prompt + JSON Schema -----------------------------------

STRUCTURED_PROMPT = (
    "Report this surveillance frame as a single JSON object with exactly these keys: "
    "timestamp, location, incident_type, confidence, description, hallucination_flag, "
    "grounding_regions. Output only the JSON."
)

INCIDENT_JSON_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SENTRY incident report",
    "type": "object",
    "additionalProperties": False,
    "required": ["incident_type", "confidence", "description", "hallucination_flag",
                 "grounding_regions"],
    "properties": {
        "timestamp": {"type": ["string", "null"], "pattern": _TIMESTAMP_RE.pattern},
        "location": {"type": ["string", "null"]},
        "incident_type": {"type": "string", "enum": list(INCIDENT_TYPES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "description": {"type": "string", "minLength": 1},
        "hallucination_flag": {"type": "boolean"},
        "grounding_regions": {"type": "array", "items": {"type": "string"}},
    },
}


def to_dict(report: IncidentReport) -> dict:
    """Convenience: full field dump (dataclass order) for a report."""
    return asdict(report)
