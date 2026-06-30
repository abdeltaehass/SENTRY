"""Dataset-specific adapters that parse a public surveillance/anomaly dataset's
raw annotations into SENTRY records (the JSONL/manifest contract).

Each adapter is responsible for one source and exposes ``build_records`` plus a
``python -m data.datasets.<name>`` CLI. The shared output plumbing (CSV/JSONL +
summary + leakage check) lives in ``data.manifest``; the dataset cards documenting
provenance and bias live in ``data/cards/``.

Available adapters:
    ucf_crime  — UCF-Crime (1,900 real surveillance videos, 13 anomaly classes).
"""
