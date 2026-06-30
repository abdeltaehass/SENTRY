"""Hermetic tests for the UCF-Crime adapter + the shared manifest writer.

No download and no model/torch: small fixture annotation files exercise parsing,
the event-taxonomy mapping, report templating, the leakage-safe split, frame
planning, and the manifest artifacts.
"""

import csv
import json

import pytest

from data.datasets import ucf_crime as ucf
from data.manifest import assert_no_split_leakage, write_manifest
from eval.metrics import EVENT_CATEGORIES

# --- fixtures: the three official UCF-Crime annotation files -----------------

TRAIN = """\
Abuse/Abuse001_x264.mp4
Burglary/Burglary033_x264.mp4
Fighting/Fighting047_x264.mp4
Shooting/Shooting008_x264.mp4
RoadAccidents/RoadAccidents022_x264.mp4
Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4
Training_Normal_Videos_Anomaly/Normal_Videos308_x264.mp4
"""

TEST = """\
Abuse/Abuse028_x264.mp4
Fighting/Fighting003_x264.mp4
Stealing/Stealing079_x264.mp4
Testing_Normal_Videos_Anomaly/Normal_Videos003_x264.mp4
"""

# filename class s1 e1 s2 e2  (frame indices @30fps; -1 = no segment)
TEMPORAL = """\
Abuse028_x264.mp4  Abuse  165  240  -1  -1
Fighting003_x264.mp4  Fighting  540  660  1230  1320
Stealing079_x264.mp4  Stealing  90  150  -1  -1
Normal_Videos003_x264.mp4  Normal  -1  -1  -1  -1
"""


@pytest.fixture
def ucf_root(tmp_path):
    (tmp_path / "Anomaly_Train.txt").write_text(TRAIN, encoding="utf-8")
    (tmp_path / "Anomaly_Test.txt").write_text(TEST, encoding="utf-8")
    (tmp_path / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").write_text(
        TEMPORAL, encoding="utf-8")
    return tmp_path


# --- parsing the split file -------------------------------------------------

def test_category_from_path_folder_and_normal():
    assert ucf.category_from_path("Abuse/Abuse028_x264.mp4") == "Abuse"
    assert ucf.category_from_path("RoadAccidents/RoadAccidents022_x264.mp4") == "RoadAccidents"
    # all three normal folders normalize to "Normal"
    cat = ucf.category_from_path
    assert cat("Normal_Videos_event/Normal_Videos_010_x264.mp4") == "Normal"
    assert cat("Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4") == "Normal"
    assert cat("Testing_Normal_Videos_Anomaly/Normal_Videos003_x264.mp4") == "Normal"


def test_category_from_path_no_folder_fallback():
    # bare filename (no folder): derive class from the stem prefix
    assert ucf.category_from_path("Fighting047_x264.mp4") == "Fighting"
    assert ucf.category_from_path("Normal_Videos308_x264.mp4") == "Normal"


def test_video_id_handles_backslashes():
    assert ucf.video_id_from_path("Abuse\\Abuse028_x264.mp4") == "Abuse028_x264"
    assert ucf.video_id_from_path("Abuse/Abuse028_x264.mp4") == "Abuse028_x264"


def test_parse_split_file(ucf_root):
    stubs = ucf.parse_split_file(ucf_root / "Anomaly_Test.txt")
    assert [s["category"] for s in stubs] == ["Abuse", "Fighting", "Stealing", "Normal"]
    assert stubs[0]["video_id"] == "Abuse028_x264"


# --- parsing the temporal annotations ---------------------------------------

def test_parse_temporal_two_segments_and_negatives(ucf_root):
    w = ucf.parse_temporal_annotations(
        ucf_root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    assert w["Abuse028_x264"] == [(165, 240)]            # single segment
    assert w["Fighting003_x264"] == [(540, 660), (1230, 1320)]  # two segments
    assert w["Normal_Videos003_x264"] == []             # all -1 -> no segments


# --- report templating ------------------------------------------------------

def test_make_report_variants():
    assert "no anomalous events" in ucf.make_report("Normal", [])
    anom = ucf.make_report("Fighting", [(18.0, 22.0)])
    assert "physical fight" in anom and "~18s to ~22s" in anom
    # anomaly without a window still produces a class-only report (train clips)
    assert ucf.make_report("Arson", None) == \
        "Surveillance footage shows an intentional fire being set."


# --- event-taxonomy alignment ----------------------------------------------

def test_category_events_use_real_taxonomy_keys():
    # every mapped event must be a real key in eval.metrics so event-F1 works
    for events in ucf.CATEGORY_EVENTS.values():
        for e in events:
            assert e in EVENT_CATEGORIES, f"{e!r} not in EVENT_CATEGORIES"
    assert set(ucf.CATEGORY_EVENTS["Shooting"]) == {"weapon", "violence"}
    assert set(ucf.CATEGORY_EVENTS["Burglary"]) == {"theft", "intrusion"}
    assert ucf.CATEGORY_EVENTS["Normal"] == []


# --- record building: official split honored, val carved, no leakage --------

def test_build_records_split_and_windows(ucf_root):
    recs = ucf.build_records(
        ucf_root, ucf_root / "Anomaly_Train.txt", ucf_root / "Anomaly_Test.txt",
        ucf_root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
        val_fraction=0.0,
    )
    by_id = {r["id"]: r for r in recs}
    # test clips keep their official split + carry temporal windows in seconds
    assert by_id["Abuse028"]["split"] == "test"
    assert by_id["Abuse028"]["anomaly_windows_seconds"] == [[5.5, 8.0]]
    assert by_id["Fighting003"]["anomaly_windows_seconds"] == [[18.0, 22.0], [41.0, 44.0]]
    # normal clip: not anomalous, no events
    assert by_id["Normal_Videos003"]["is_anomalous"] is False
    assert by_id["Normal_Videos003"]["events"] == []
    # one camera per clip; no clip straddles splits
    assert by_id["Abuse028"]["camera_id"] == "Abuse028_x264"
    assert_no_split_leakage(recs)


def test_val_carve_is_deterministic_and_leakage_safe(ucf_root):
    kw = dict(train_split=ucf_root / "Anomaly_Train.txt",
              test_split=ucf_root / "Anomaly_Test.txt", val_fraction=0.3, seed=7)
    a = ucf.build_records(ucf_root, **kw)
    b = ucf.build_records(ucf_root, **kw)
    val_a = sorted(r["id"] for r in a if r["split"] == "val")
    val_b = sorted(r["id"] for r in b if r["split"] == "val")
    assert val_a == val_b and len(val_a) >= 1        # deterministic, non-empty
    # val clips came from the official TRAIN split, never from test
    assert all(r["official_split"] == "train" for r in a if r["split"] == "val")
    assert_no_split_leakage(a)


# --- frame planning (no OpenCV / no video needed) ---------------------------

def test_expand_to_frames_labels_window_vs_context(ucf_root):
    recs = ucf.build_records(
        ucf_root, ucf_root / "Anomaly_Train.txt", ucf_root / "Anomaly_Test.txt",
        ucf_root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
        val_fraction=0.0,
    )
    abuse = next(r for r in recs if r["id"] == "Abuse028")
    frames = ucf.expand_to_frames([abuse], frames_per_clip=4, context_frames=2,
                                  write_images=False)
    anomalous = [f for f in frames if f["is_anomalous"]]
    context = [f for f in frames if not f["is_anomalous"]]
    # anomalous frames fall inside the annotated window [165, 240]
    assert anomalous and all(165 <= f["frame_index"] <= 240 for f in anomalous)
    assert all(f["events"] == ["violence"] for f in anomalous)
    # context frames are labelled normal and sit outside the window
    assert context and all(f["frame_index"] > 240 for f in context)
    assert all(f["events"] == [] and f["category"] == "Normal" for f in context)


# --- manifest writer --------------------------------------------------------

def test_write_manifest_artifacts(ucf_root, tmp_path):
    recs = ucf.build_records(
        ucf_root, ucf_root / "Anomaly_Train.txt", ucf_root / "Anomaly_Test.txt",
        ucf_root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
        val_fraction=0.0,
    )
    out = tmp_path / "manifest_out"
    summary = write_manifest(recs, out, source="UCF-Crime")

    assert (out / "manifest.csv").exists()
    assert (out / "train.jsonl").exists() and (out / "test.jsonl").exists()
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["source"] == "UCF-Crime"
    assert saved["n_records"] == len(recs)
    assert summary["by_event"]["violence"] >= 2  # Abuse + Fighting + Shooting

    # csv round-trips: one header + one row per record, ids preserved
    with (out / "manifest.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(recs)
    assert {r["id"] for r in rows} == {r["id"] for r in recs}


def test_assert_no_split_leakage_raises():
    leaky = [
        {"camera_id": "Abuse028_x264", "split": "train"},
        {"camera_id": "Abuse028_x264", "split": "test"},
    ]
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_split_leakage(leaky)
