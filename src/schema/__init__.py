"""Structured incident-report output layer (see ``schema.incident``)."""

from __future__ import annotations

from .incident import (
    INCIDENT_JSON_SCHEMA,
    INCIDENT_TYPES,
    STRUCTURED_PROMPT,
    IncidentReport,
    SchemaError,
    classify_incident,
    coerce_incident,
    extract_json_object,
    parse_incident,
    regions_from_cam,
)

__all__ = [
    "INCIDENT_JSON_SCHEMA",
    "INCIDENT_TYPES",
    "STRUCTURED_PROMPT",
    "IncidentReport",
    "SchemaError",
    "classify_incident",
    "coerce_incident",
    "extract_json_object",
    "parse_incident",
    "regions_from_cam",
]
