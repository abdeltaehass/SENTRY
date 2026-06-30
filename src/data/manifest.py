"""Dataset-agnostic manifest writer.

A *manifest* is the cleaned, auditable index of a dataset after a dataset-specific
adapter (e.g. ``data.datasets.ucf_crime``) has parsed its raw annotations into
SENTRY records. It is the artifact you commit / review / diff — small, text, and
free of the multi-GB media — so the provenance of every training example is
inspectable without opening a single video.

This module writes three things from one list of records:

  1. ``manifest.csv``      — one flat row per clip/frame (open in a spreadsheet).
  2. ``{split}.jsonl``     — the per-split files the dataloader consumes
                             (``data.dataset.IncidentReportDataset``).
  3. ``manifest.json``     — a summary: counts per split / category / event,
                             plus provenance (source, builder, timestamp, fields).

Adapters stay thin: they produce records; this writer standardizes the output
layout, the summary, and the leakage check so every dataset reports the same way.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# Columns flattened into manifest.csv, in order. Records may carry extra keys
# (e.g. image_paths) that are JSON-encoded into the catch-all `extra` column.
_CSV_COLUMNS: tuple[str, ...] = (
    "id", "split", "source", "category", "is_anomalous", "events",
    "report", "camera_id", "location_id", "video_path", "image_path",
    "frame_index", "anomaly_windows_seconds",
)


def _flatten(value: Any) -> str:
    """Render a record value as a single CSV-safe cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_csv(records: Sequence[dict], path: str | Path,
              columns: Sequence[str] = _CSV_COLUMNS) -> Path:
    """Write a flat, human-auditable CSV (one row per record)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_cols = sorted({k for r in records for k in r} - set(columns))
    header = [*columns, *( ["extra"] if extra_cols else [] )]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in records:
            row = [_flatten(r.get(c)) for c in columns]
            if extra_cols:
                row.append(_flatten({k: r[k] for k in extra_cols if k in r}))
            writer.writerow(row)
    return path


def write_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    """Write records as JSON Lines (the dataloader's input format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def summarize(records: Sequence[dict], *, source: str, group_key: str = "camera_id") -> dict:
    """Build a summary dict: split / category / event counts + integrity stats."""
    by_split = Counter(r.get("split") for r in records)
    by_category = Counter(r.get("category") for r in records)
    events = Counter(e for r in records for e in (r.get("events") or []))
    n_anom = sum(1 for r in records if r.get("is_anomalous"))
    n_groups = len({r.get(group_key) for r in records if r.get(group_key) is not None})
    return {
        "source": source,
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "n_records": len(records),
        "n_anomalous": n_anom,
        "n_normal": len(records) - n_anom,
        f"n_{group_key}s": n_groups,
        "by_split": dict(sorted(by_split.items(), key=lambda kv: str(kv[0]))),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: str(kv[0]))),
        "by_event": dict(sorted(events.items(), key=lambda kv: str(kv[0]))),
        "fields": sorted({k for r in records for k in r}),
    }


def assert_no_split_leakage(records: Sequence[dict], group_key: str = "camera_id") -> None:
    """Fail loudly if one group (camera/clip) id appears in more than one split."""
    seen: dict[Any, str] = {}
    for r in records:
        gid, split = r.get(group_key), r.get("split")
        if gid is None or split is None:
            continue
        if gid in seen and seen[gid] != split:
            raise AssertionError(
                f"{group_key} leakage: {gid!r} appears in both "
                f"{seen[gid]!r} and {split!r} splits"
            )
        seen[gid] = split


def write_manifest(records: Sequence[dict], out_dir: str | Path, *, source: str,
                   group_key: str = "camera_id", split_jsonl: bool = True) -> dict:
    """Write csv + per-split jsonl + summary json for ``records``.

    Returns the summary dict (also written to ``manifest.json``). Raises if a
    group id straddles splits, so a leaky manifest never gets written silently.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_split_leakage(records, group_key=group_key)

    write_csv(records, out_dir / "manifest.csv")
    if split_jsonl:
        by_split: dict[str, list[dict]] = {}
        for r in records:
            by_split.setdefault(str(r.get("split", "all")), []).append(r)
        for split, recs in by_split.items():
            write_jsonl(recs, out_dir / f"{split}.jsonl")

    summary = summarize(records, source=source, group_key=group_key)
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def print_summary(summary: dict) -> None:
    """Pretty-print a manifest summary for CLI output."""
    print(f"=== {summary['source']} manifest ===  {summary['n_records']} records "
          f"({summary['n_anomalous']} anomalous / {summary['n_normal']} normal)")
    for key, label in (("by_split", "split"), ("by_category", "category")):
        parts = ", ".join(f"{k}={v}" for k, v in summary[key].items())
        print(f"  by {label:8s}: {parts}")
    events = ", ".join(f"{k}={v}" for k, v in summary["by_event"].items())
    print(f"  by event   : {events or '(none)'}")
