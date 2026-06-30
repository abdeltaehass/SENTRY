"""UCF-Crime adapter: real surveillance videos -> SENTRY records.

UCF-Crime (Sultani et al., CVPR 2018, "Real-world Anomaly Detection in
Surveillance Videos") is 1,900 untrimmed real CCTV videos (~128 h, 30 fps)
spanning 13 anomaly classes plus normal footage. See ``data/cards/ucf_crime.md``
for provenance, licensing, and known biases.

This module turns the dataset's three official annotation files into the SENTRY
contract (``data.dataset.IncidentReportDataset`` / ``data.manifest``):

  Anomaly_Train.txt
  Anomaly_Test.txt
        One relative video path per line, e.g. ``Abuse/Abuse028_x264.mp4`` or
        ``Training_Normal_Videos_Anomaly/Normal_Videos_001_x264.mp4``. The folder
        prefix names the class; we honor this official train/test split.

  Temporal_Anomaly_Annotation_for_Testing_Videos.txt
        One line per *test* video, whitespace-separated, six fields:
            <filename> <class> <start1> <end1> <start2> <end2>
        Frame indices (30 fps) bound up to two anomalous segments; ``-1`` means
        "no such segment". Normal test videos use class ``Normal`` and all ``-1``.

What this adapter produces and what it does NOT:
  - Reports are *templated* from the class label (+ temporal window), not human
    prose: UCF-Crime ships category labels, not natural-language descriptions.
    This is weak supervision and is documented as a limitation in the data card.
  - Each clip becomes its own ``camera_id`` so the leakage-safe split keeps every
    frame of a clip in a single split (the official split is already clip-level).
  - Class labels are mapped onto SENTRY's event taxonomy (``eval.metrics``) so
    event-overlap F1 works on real data out of the box.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..manifest import print_summary, write_manifest
from ..preprocess import clean_text

FPS = 30  # UCF-Crime is encoded at 30 fps; temporal annotations are frame indices.

# 13 anomaly classes (folder names) + the normal-video folders.
ANOMALY_CLASSES: tuple[str, ...] = (
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
)
NORMAL_FOLDERS: tuple[str, ...] = (
    "Normal_Videos_event", "Training_Normal_Videos_Anomaly",
    "Testing_Normal_Videos_Anomaly",
)
NORMAL_CLASS = "Normal"

# Map each class onto SENTRY's event taxonomy keys (see eval.metrics.EVENT_CATEGORIES)
# so generated reports score against the same event-overlap metric used elsewhere.
CATEGORY_EVENTS: dict[str, list[str]] = {
    "Abuse": ["violence"],
    "Arrest": ["violence"],
    "Arson": ["fire/smoke"],
    "Assault": ["violence"],
    "Burglary": ["theft", "intrusion"],
    "Explosion": ["fire/smoke"],
    "Fighting": ["violence"],
    "RoadAccidents": ["vehicle"],
    "Robbery": ["theft"],
    "Shooting": ["weapon", "violence"],
    "Shoplifting": ["theft"],
    "Stealing": ["theft"],
    "Vandalism": ["vandalism"],
    NORMAL_CLASS: [],
}

# Human-readable phrase per class, used to template a factual incident report.
_CATEGORY_PHRASE: dict[str, str] = {
    "Abuse": "a person being physically abused",
    "Arrest": "an apprehension or arrest of an individual",
    "Arson": "an intentional fire being set",
    "Assault": "a physical assault on an individual",
    "Burglary": "an unlawful entry into a property, consistent with a burglary",
    "Explosion": "an explosion",
    "Fighting": "a physical fight between individuals",
    "RoadAccidents": "a road traffic accident involving one or more vehicles",
    "Robbery": "a robbery in progress",
    "Shooting": "a shooting involving a firearm",
    "Shoplifting": "the concealment or theft of goods from a retail setting",
    "Stealing": "an item being stolen",
    "Vandalism": "property being deliberately damaged or vandalized",
    NORMAL_CLASS: "routine activity with no anomalous event",
}

_VIDEO_SUFFIX = re.compile(r"_x264$", re.IGNORECASE)
_TRAILING_DIGITS = re.compile(r"\d+$")


# --- parsing the official annotation files ----------------------------------


def category_from_path(rel_path: str) -> str:
    """Infer the class from a split-file line (``Class/Video_x264.mp4``).

    Falls back to the filename prefix (``Abuse028`` -> ``Abuse``) when the line
    has no folder, and normalizes any normal-video folder to ``Normal``.
    """
    parts = re.split(r"[\\/]", rel_path.strip())
    folder = parts[0] if len(parts) > 1 else ""
    if folder in NORMAL_FOLDERS or "Normal" in folder:
        return NORMAL_CLASS
    for cls in ANOMALY_CLASSES:
        if folder == cls:
            return cls
    # No/unknown folder: derive from the stem, e.g. "Fighting047_x264".
    stem = _VIDEO_SUFFIX.sub("", Path(parts[-1]).stem)
    if stem.startswith("Normal"):
        return NORMAL_CLASS
    base = _TRAILING_DIGITS.sub("", stem)
    return base if base in ANOMALY_CLASSES else (folder or base or NORMAL_CLASS)


def video_id_from_path(rel_path: str) -> str:
    """``Abuse/Abuse028_x264.mp4`` -> ``Abuse028_x264`` (the clip identity)."""
    return Path(re.split(r"[\\/]", rel_path.strip())[-1]).stem


def parse_split_file(path: str | Path) -> list[dict]:
    """Parse ``Anomaly_Train.txt`` / ``Anomaly_Test.txt`` lines into stubs.

    Returns ``[{video_id, category, rel_path}]`` (blank lines skipped).
    """
    out: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        out.append({
            "video_id": video_id_from_path(rel),
            "category": category_from_path(rel),
            "rel_path": rel.replace("\\", "/"),
        })
    return out


def parse_temporal_annotations(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    """Parse the temporal annotation file -> ``{video_id: [(start, end), ...]}``.

    Each line is ``<filename> <class> <s1> <e1> <s2> <e2>`` (frame indices, 30 fps,
    ``-1`` = absent). Returns only the valid ``(start, end)`` segments per clip;
    a normal clip (all ``-1``) maps to an empty list.
    """
    windows: dict[str, list[tuple[int, int]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cols = line.split()
        if not cols:
            continue
        vid = Path(cols[0]).stem
        nums = [int(c) for c in cols[2:] if _is_int(c)]
        segments: list[tuple[int, int]] = []
        for i in range(0, len(nums) - 1, 2):
            start, end = nums[i], nums[i + 1]
            if start >= 0 and end >= start:
                segments.append((start, end))
        windows[vid] = segments
    return windows


def _is_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


# --- report templating ------------------------------------------------------


def make_report(category: str, windows_sec: list[tuple[float, float]] | None) -> str:
    """Template a concise factual incident report from class (+ window seconds)."""
    phrase = _CATEGORY_PHRASE.get(category, "an anomalous event")
    if category == NORMAL_CLASS or not windows_sec:
        if category == NORMAL_CLASS:
            return "Routine surveillance footage with no anomalous events observed."
        return clean_text(f"Surveillance footage shows {phrase}.")
    spans = "; ".join(f"~{a:.0f}s to ~{b:.0f}s" for a, b in windows_sec)
    return clean_text(
        f"Surveillance footage shows {phrase}. "
        f"The anomalous event occurs between {spans} into the clip."
    )


# --- record building --------------------------------------------------------


def _windows_seconds(frames: list[tuple[int, int]]) -> list[tuple[float, float]]:
    return [(round(s / FPS, 2), round(e / FPS, 2)) for s, e in frames]


def build_records(
    data_root: str | Path,
    train_split: str | Path,
    test_split: str | Path,
    temporal_annotations: str | Path | None = None,
    *,
    videos_subdir: str = "Videos",
    frames_subdir: str = "frames",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """Build clip-level SENTRY records from the official UCF-Crime files.

    Args:
        data_root: dataset root (also where processed frame dirs are rooted).
        train_split / test_split: official ``Anomaly_Train/Test.txt`` paths.
        temporal_annotations: optional test-set temporal annotation file; when
            given, test clips carry per-event frame/second windows and a
            window-aware report.
        videos_subdir: folder under ``data_root`` holding the class video folders.
        frames_subdir: folder under ``data_root`` where extracted frames live.
        val_fraction: fraction of *train clips* to carve into a ``val`` split
            (deterministic, by clip id, so no clip straddles train/val).
        seed: RNG seed for the val carve-out.

    Returns one record per clip (see module docstring for the schema).
    """
    data_root = Path(data_root)
    videos_root = data_root / videos_subdir
    frames_root = data_root / frames_subdir

    windows = parse_temporal_annotations(temporal_annotations) if temporal_annotations else {}

    train_stubs = parse_split_file(train_split)
    test_stubs = parse_split_file(test_split)
    val_ids = _carve_val_ids([s["video_id"] for s in train_stubs], val_fraction, seed)

    records: list[dict] = []
    for stub, official in ((s, "train") for s in train_stubs):
        split = "val" if stub["video_id"] in val_ids else "train"
        records.append(_record(stub, split, official, windows, videos_root, frames_root))
    for stub in test_stubs:
        records.append(_record(stub, "test", "test", windows, videos_root, frames_root))
    return records


def _record(stub: dict, split: str, official_split: str,
            windows: dict[str, list[tuple[int, int]]],
            videos_root: Path, frames_root: Path) -> dict:
    vid, category = stub["video_id"], stub["category"]
    frame_windows = windows.get(vid, [])
    sec_windows = _windows_seconds(frame_windows)
    is_anom = category != NORMAL_CLASS
    return {
        "id": _VIDEO_SUFFIX.sub("", vid),
        "video_id": vid,
        "source": "UCF-Crime",
        "category": category,
        "is_anomalous": is_anom,
        "events": CATEGORY_EVENTS.get(category, []),
        "report": make_report(category, sec_windows),
        "split": split,
        "official_split": official_split,
        "fps": FPS,
        "anomaly_windows_frames": [list(w) for w in frame_windows],
        "anomaly_windows_seconds": [list(w) for w in sec_windows],
        "video_path": str(videos_root / stub["rel_path"]),
        "frame_dir": str(frames_root / vid),
        # One camera per clip -> the leakage-safe split keeps a clip's frames together.
        "camera_id": vid,
        "location_id": None,
    }


def _carve_val_ids(train_ids: list[str], fraction: float, seed: int) -> set[str]:
    """Deterministically pick a fraction of train clip ids for the val split."""
    if fraction <= 0:
        return set()
    import random

    ids = sorted(set(train_ids))
    random.Random(seed).shuffle(ids)
    n_val = int(round(fraction * len(ids)))
    return set(ids[:n_val])


# --- optional frame extraction (clip records -> frame-level records) --------


def expand_to_frames(
    records: list[dict], *, frames_per_clip: int = 8, context_frames: int = 2,
    write_images: bool = True,
) -> list[dict]:
    """Sample frames per clip into frame-level records the dataloader can train on.

    For anomaly clips with temporal windows, frames are sampled *inside* the
    anomalous segments (labelled with the clip's events/report); up to
    ``context_frames`` extra frames are sampled outside the window and labelled
    normal. Normal clips are sampled uniformly. Requires the video files and
    OpenCV when ``write_images=True``; otherwise emits records pointing at the
    expected frame paths without decoding (useful for dry-runs/manifests).
    """
    frame_records: list[dict] = []
    for rec in records:
        plan = _frame_plan(rec, frames_per_clip, context_frames)
        if write_images:
            saved = _grab_frames(rec["video_path"], rec["frame_dir"],
                                 [idx for idx, _ in plan])
        else:
            stem = Path(rec["video_id"]).name
            saved = {idx: str(Path(rec["frame_dir"]) / f"{stem}_f{idx:06d}.jpg")
                     for idx, _ in plan}
        for idx, anomalous in plan:
            if idx not in saved:
                continue
            frame_records.append({
                "id": f"{rec['video_id']}_f{idx:06d}",
                "image_path": saved[idx],
                "report": rec["report"] if anomalous else
                          "Routine surveillance footage with no anomalous events observed.",
                "events": rec["events"] if anomalous else [],
                "is_anomalous": bool(anomalous),
                "category": rec["category"] if anomalous else NORMAL_CLASS,
                "frame_index": idx,
                "source": "UCF-Crime",
                "split": rec["split"],
                "camera_id": rec["camera_id"],
                "location_id": rec["location_id"],
            })
    return frame_records


def _frame_plan(rec: dict, n_anom: int, n_context: int) -> list[tuple[int, bool]]:
    """Decide which frame indices to sample and whether each is anomalous."""
    windows = rec.get("anomaly_windows_frames") or []
    if not rec["is_anomalous"] or not windows:
        # Normal clip (or anomaly without temporal labels): uniform 0..max guess.
        span_end = max((w[1] for w in windows), default=n_anom * FPS)
        idxs = _linspace(0, max(span_end, n_anom), n_anom)
        return [(i, False) for i in idxs]

    plan: list[tuple[int, bool]] = []
    per_window = max(1, n_anom // len(windows))
    last_end = 0
    for start, end in windows:
        for i in _linspace(start, end, per_window):
            plan.append((i, True))
        last_end = max(last_end, end)
    for i in _linspace(last_end + FPS, last_end + FPS * (n_context + 1), n_context):
        plan.append((i, False))
    return plan


def _linspace(start: int, end: int, n: int) -> list[int]:
    """``n`` evenly spaced integer indices in ``[start, end]`` (inclusive-ish)."""
    if n <= 0:
        return []
    if n == 1 or end <= start:
        return [max(0, start)]
    step = (end - start) / (n - 1)
    return [max(0, int(round(start + step * k))) for k in range(n)]


def _grab_frames(video_path: str, out_dir: str, indices: list[int]) -> dict[int, str]:
    """Decode and save the requested frame indices. Returns ``{index: path}``."""
    try:
        import cv2
    except ImportError as exc:  # mirror preprocess.extract_frames behaviour
        raise ImportError(
            "Frame extraction needs OpenCV: `pip install opencv-python` "
            "(or pass --no-write-images to emit a manifest without decoding)."
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    cap = cv2.VideoCapture(str(video_path))
    saved: dict[int, str] = {}
    for idx in sorted(set(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        p = out / f"{stem}_f{idx:06d}.jpg"
        cv2.imwrite(str(p), frame)
        saved[idx] = str(p)
    cap.release()
    return saved


# --- CLI --------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a SENTRY manifest from the official UCF-Crime annotation files."
    )
    p.add_argument("--data-root", required=True,
                   help="UCF-Crime root (contains Videos/ and the split .txt files)")
    p.add_argument("--train-split", default=None,
                   help="Anomaly_Train.txt (default: <data-root>/Anomaly_Train.txt)")
    p.add_argument("--test-split", default=None,
                   help="Anomaly_Test.txt (default: <data-root>/Anomaly_Test.txt)")
    p.add_argument("--temporal", default=None,
                   help="Temporal_Anomaly_Annotation_for_Testing_Videos.txt (optional)")
    p.add_argument("--out-dir", default="data/processed/ucf_crime",
                   help="where manifest.csv / {split}.jsonl / manifest.json are written")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="fraction of train clips carved into a val split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extract-frames", action="store_true",
                   help="also decode + sample frames into frame-level JSONL (needs OpenCV)")
    p.add_argument("--frames-per-clip", type=int, default=8)
    p.add_argument("--no-write-images", action="store_true",
                   help="with --extract-frames, plan frame paths without decoding video")
    args = p.parse_args()

    root = Path(args.data_root)
    train = args.train_split or root / "Anomaly_Train.txt"
    test = args.test_split or root / "Anomaly_Test.txt"

    records = build_records(
        root, train, test, args.temporal,
        val_fraction=args.val_fraction, seed=args.seed,
    )
    summary = write_manifest(records, args.out_dir, source="UCF-Crime")
    print_summary(summary)
    print(f"  -> clip manifest: {Path(args.out_dir) / 'manifest.csv'}")

    if args.extract_frames:
        frames = expand_to_frames(
            records, frames_per_clip=args.frames_per_clip,
            write_images=not args.no_write_images,
        )
        frame_dir = Path(args.out_dir) / "frames_manifest"
        fsummary = write_manifest(frames, frame_dir, source="UCF-Crime (frames)")
        print_summary(fsummary)
        print(f"  -> frame manifest: {frame_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
