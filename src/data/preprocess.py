"""Preprocess a surveillance/anomaly dataset into the SENTRY JSONL contract.

Pipeline (scaffold — wire your specific dataset into `build_records`):
  1. (video) extract_frames   -> sampled frames per clip
  2. parse annotations        -> incident-description text per frame/scene
  3. clean_text               -> normalize whitespace
  4. split_by_group           -> split by CAMERA / LOCATION id (not by frame),
                                 so frames from one camera never straddle splits
  5. write JSONL              -> data/processed/incidents/{train,val,test}.jsonl

Output record:
  {"id": "scene_0001",
   "image_path":  ".../scene_0001_f00.jpg",
   "image_paths": [...],          # optional: multiple frames / camera views
   "report": "...",
   "camera_id": "cam_03", "location_id": "lobby_east"}

Security-camera frames are typically wide (16:9 / 4:3), unlike the model's square
input, so aspect ratio is preserved with `letterbox` at load time (data.dataset)
rather than squishing them.

PLACEHOLDER ANNOTATION SCHEMA (raw input to `build_records`, a JSON list):
  [{"id": "scene_0001",
    "camera_id": "cam_03", "location_id": "lobby_east",
    "image": "frames/scene_0001.jpg",        # OR "video": "clips/scene_0001.mp4"
    "incident_description": "A person leaves a bag unattended near the entrance."}]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

# --- image aspect handling --------------------------------------------------


def letterbox(image: Image.Image, size: int | None = None, fill=(0, 0, 0)) -> Image.Image:
    """Pad an image to a square (preserving aspect ratio); optionally resize.

    Security footage is wide; squishing it to the model's square input distorts
    geometry. Letterboxing pads the short side instead.
    """
    image = image.convert("RGB")
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    if size is not None:
        canvas = canvas.resize((size, size))
    return canvas


# --- video frame extraction (scaffold; pluggable backend) -------------------


def extract_frames(video_path: str | Path, out_dir: str | Path,
                   every_n_frames: int = 30, max_frames: int | None = None) -> list[str]:
    """Sample frames from a video to JPEGs. Returns the written frame paths.

    Uses OpenCV if available. Surveillance ingestion backends vary (RTSP, NVR
    exports, mp4); swap this for yours. `every_n_frames=30` ~= 1 fps at 30 fps.
    """
    try:
        import cv2
    except ImportError as exc:  # keep the scaffold importable without the dep
        raise ImportError(
            "extract_frames needs OpenCV: `pip install opencv-python` "
            "(or replace this function with your ingestion backend)."
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    cap = cv2.VideoCapture(str(video_path))
    paths: list[str] = []
    idx = kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n_frames == 0:
            p = out_dir / f"{stem}_f{kept:04d}.jpg"
            cv2.imwrite(str(p), frame)
            paths.append(str(p))
            kept += 1
            if max_frames and kept >= max_frames:
                break
        idx += 1
    cap.release()
    return paths


# --- text + records ---------------------------------------------------------


_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Generic normalization: collapse whitespace, strip."""
    if not text:
        return ""
    return _WS.sub(" ", text).strip()


def build_records(annotations_path: str | Path, base_dir: str | Path = ".") -> list[dict]:
    """Parse the placeholder annotation schema (see module docstring) -> records.

    Resolves `image` (or extracts frames from `video`) and attaches camera/location
    ids for the leakage-safe split. Replace the parsing to match your dataset.
    """
    base_dir = Path(base_dir)
    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    records: list[dict] = []
    for a in annotations:
        report = clean_text(a.get("incident_description"))
        if not report:
            continue
        if a.get("image"):
            image_paths = [str(base_dir / a["image"])]
        elif a.get("video"):
            image_paths = extract_frames(base_dir / a["video"], base_dir / "frames")
        else:
            continue
        records.append({
            "id": a.get("id"),
            "image_path": image_paths[0],
            "image_paths": image_paths,
            "report": report,
            "camera_id": a.get("camera_id"),
            "location_id": a.get("location_id"),
        })
    return records


# --- leakage-safe split (by camera / location, not by frame) ----------------


def split_by_group(records: list[dict], group_key: str = "camera_id",
                   ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
                   seed: int = 42) -> dict[str, list[dict]]:
    """Partition GROUPS (camera/location ids) into train/val/test.

    Same leakage-prevention pattern as a per-subject / per-source split —
    every frame from one camera/location lands in exactly one split.
    """
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[str(r.get(group_key))].append(r)

    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    assign = {
        **{g: "train" for g in groups[:n_train]},
        **{g: "val" for g in groups[n_train:n_train + n_val]},
        **{g: "test" for g in groups[n_train + n_val:]},
    }
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        splits[assign[str(r.get(group_key))]].append(r)
    return splits


def assert_no_leakage(splits: dict[str, list[dict]], group_key: str = "camera_id") -> None:
    """Fail loudly if any group id appears in more than one split."""
    sets = {name: {str(r.get(group_key)) for r in recs} for name, recs in splits.items()}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and (sets[a] & sets[b]):
                raise AssertionError(f"{group_key} leakage {a}/{b}: {sorted(sets[a] & sets[b])[:5]}")


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess a surveillance dataset into JSONL splits")
    p.add_argument("--annotations", required=True, help="raw annotations JSON (see module docstring)")
    p.add_argument("--base-dir", default=".", help="root for resolving image/video paths")
    p.add_argument("--out-dir", default="data/processed/incidents")
    p.add_argument("--group-key", default="camera_id", help="split by this id (camera_id|location_id)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    records = build_records(args.annotations, base_dir=args.base_dir)
    splits = split_by_group(records, group_key=args.group_key, seed=args.seed)
    assert_no_leakage(splits, group_key=args.group_key)

    out = Path(args.out_dir)
    for name, recs in splits.items():
        write_jsonl(recs, out / f"{name}.jsonl")

    n_groups = len({str(r.get(args.group_key)) for r in records})
    print(f"=== SENTRY preprocessing ===  kept {len(records)} records / {n_groups} {args.group_key}s")
    for name, recs in splits.items():
        g = len({str(r.get(args.group_key)) for r in recs})
        print(f"  {name:5s}: {len(recs):5d} records / {g:3d} {args.group_key}s -> {out / f'{name}.jsonl'}")
    print(f"no {args.group_key} leakage across splits ✓")


if __name__ == "__main__":
    main()
