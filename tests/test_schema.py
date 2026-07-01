"""Hermetic tests for the structured incident-report schema (no model/torch).

Covers incident classification (reusing the event lexicon with whole-word
matching), Grad-CAM -> named regions, signal assembly, and the robust
parse/validate/repair layer for model-emitted JSON.
"""

import json

import numpy as np
import pytest

from eval.metrics import _asserted_events, detect_events
from schema.incident import (
    INCIDENT_JSON_SCHEMA,
    INCIDENT_TYPES,
    IncidentReport,
    SchemaError,
    classify_incident,
    extract_json_object,
    parse_incident,
    regions_from_cam,
)

# --- incident classification ------------------------------------------------

def test_classify_maps_events_to_incident_types():
    assert classify_incident("two people fighting, one appears armed") == "weapon"
    assert classify_incident("a physical fight broke out") == "violence"
    assert classify_incident("smoke and flames near the exit") == "fire/smoke"
    assert classify_incident("Routine footage, nothing unusual.") == "normal"


def test_classify_abandoned_object_is_unattended():
    assert classify_incident(
        "Individual left a backpack near the door and exited frame") == "unattended object"
    assert classify_incident(
        "An individual leaves a bag near the dock, unattended.") == "unattended object"
    # object carried in (not left) is not an abandonment
    assert classify_incident(
        "A person enters carrying a backpack. No anomalies observed.") == "normal"


def test_classify_respects_negation():
    assert classify_incident("A person enters. No weapon seen.") == "normal"


def test_classify_whole_word_matching():
    # "car" must not fire on "carrying"; "van" must not fire on "vandalism"
    assert classify_incident("a guard carrying a clipboard walks by") == "normal"
    assert classify_incident("someone spray-painted graffiti, clear vandalism") == "vandalism"
    # plurals still match
    assert classify_incident("two vehicles collided at the intersection") == "vehicle"


def test_classified_types_are_in_vocabulary():
    for desc in ["armed robbery", "a fire started", "loitering by the gate", "a car crash"]:
        assert classify_incident(desc) in INCIDENT_TYPES


def test_detect_events_word_boundary_flag():
    # substring default (event-F1 behaviour) matches "car" inside "carrying"
    assert "vehicle" in detect_events("carrying a bag", word_boundary=False)
    # whole-word (structured layer) does not
    assert "vehicle" not in detect_events("carrying a bag", word_boundary=True)
    # the private helper still defaults to the lenient substring behaviour
    assert _asserted_events("carrying a bag") == {"vehicle"}


# --- grounding regions ------------------------------------------------------

def test_regions_from_cam_picks_hot_quadrant():
    cam = np.zeros((4, 4), dtype=np.float32)
    cam[2:, :2] = 1.0                                  # bottom-left hot
    assert regions_from_cam(cam) == ["lower-left quadrant"]


def test_regions_from_cam_multiple_and_empty():
    cam = np.zeros((4, 4), dtype=np.float32)
    cam[:2, 2:] = 1.0                                  # top-right
    cam[2:, :2] = 1.0                                  # bottom-left, equal mass
    regions = regions_from_cam(cam)
    assert set(regions) == {"upper-right quadrant", "lower-left quadrant"}
    assert regions_from_cam(np.zeros((4, 4))) == []    # flat map -> no region
    assert regions_from_cam(np.array([1.0])) == []     # non-2D -> no region


# --- assembly from signals --------------------------------------------------

def test_from_signals_matches_documented_shape():
    cam = np.zeros((4, 4), dtype=np.float32)
    cam[2:, :2] = 1.0
    rel = {"reliability_score": 0.81, "flagged": False, "risk_level": "low"}
    rec = IncidentReport.from_signals(
        "Individual left a backpack near the door and exited frame",
        confidence=0.81, reliability=rel, cam=cam,
        timestamp="14:32:07", location="east entrance",
    )
    d = rec.to_dict()
    assert list(d.keys()) == ["timestamp", "location", "incident_type", "confidence",
                              "description", "hallucination_flag", "grounding_regions"]
    assert d["incident_type"] == "unattended object"
    assert d["confidence"] == 0.81 and d["hallucination_flag"] is False
    assert d["grounding_regions"] == ["lower-left quadrant"]
    assert json.loads(rec.to_json()) == d               # round-trips as JSON


def test_from_signals_uses_calibrated_score_and_flag():
    rel = {"reliability_score": 0.22, "flagged": True, "risk_level": "high"}
    rec = IncidentReport.from_signals("a person with a gun", confidence=0.5, reliability=rel)
    assert rec.confidence == 0.22 and rec.hallucination_flag is True
    assert rec.incident_type == "weapon" and rec.grounding_regions == []


# --- validation / coercion --------------------------------------------------

def test_validated_clamps_and_coerces():
    rec = IncidentReport(incident_type="BOGUS", confidence=1.7,
                         description="  a fight  ", hallucination_flag="true",
                         grounding_regions="center", timestamp="25:99:99").validated()
    assert rec.incident_type == "other"                 # unknown -> other
    assert rec.confidence == 1.0                         # clamped to [0, 1]
    assert rec.description == "a fight"                  # stripped
    assert rec.hallucination_flag is True               # "true" -> True
    assert rec.grounding_regions == ["center"]          # str -> [str]
    assert rec.timestamp is None                         # invalid time dropped


def test_empty_description_raises():
    with pytest.raises(SchemaError):
        IncidentReport(incident_type="normal", confidence=0.5, description="   ").validated()


# --- robust parsing of model output -----------------------------------------

def test_extract_json_object_ignores_prose_and_braces_in_strings():
    text = 'Here you go:\n```json\n{"a": "has } brace", "b": 1}\n``` done'
    assert extract_json_object(text) == '{"a": "has } brace", "b": 1}'
    assert extract_json_object("no object here") is None


def test_parse_incident_repairs_messy_json():
    messy = (
        'Sure! {"timestamp": "2026-07-01T14:32:07", "location": "dock", '
        '"incident_type": "BOMB", "confidence": "1.4", '
        '"description": "a bag left unattended by the wall", '
        '"hallucination_flag": "yes", "grounding_regions": "lower-left quadrant"}'
    )
    rec = parse_incident(messy)
    assert rec.incident_type == "unattended object"     # invalid type re-derived from text
    assert rec.confidence == 1.0                         # "1.4" -> clamped
    assert rec.timestamp == "14:32:07"                   # ISO datetime -> HH:MM:SS
    assert rec.hallucination_flag is True
    assert rec.grounding_regions == ["lower-left quadrant"]


def test_parse_incident_from_dict_and_failure():
    rec = parse_incident({"confidence": 0.6, "description": "an armed robbery in progress"})
    assert rec.incident_type in ("weapon", "theft")     # derived from description
    with pytest.raises(SchemaError):
        parse_incident("not json at all")


# --- JSON Schema ------------------------------------------------------------

def test_json_schema_enumerates_incident_types():
    props = INCIDENT_JSON_SCHEMA["properties"]
    assert set(props["incident_type"]["enum"]) == set(INCIDENT_TYPES)
    assert props["confidence"]["minimum"] == 0.0 and props["confidence"]["maximum"] == 1.0
    assert "description" in INCIDENT_JSON_SCHEMA["required"]
