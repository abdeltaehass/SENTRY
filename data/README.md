# Data

SENTRY trains and evaluates on a processed **JSONL per split**, one record per line:

```json
{"id": "Abuse028",
 "image_path": "data/processed/ucf_crime/frames/Abuse028_x264/Abuse028_x264_f000200.jpg",
 "report": "Surveillance footage shows a person being physically abused. The anomalous event occurs between ~6s to ~8s into the clip.",
 "events": ["violence"],
 "is_anomalous": true,
 "category": "Abuse",
 "camera_id": "Abuse028_x264",
 "location_id": null,
 "source": "UCF-Crime"}
```

- `image_paths` (plural) is optional — multiple camera views of one scene (see `model.multiview`).
- The dataloader (`data.dataset.IncidentReportDataset`) consumes these files directly.
- Split **by camera / clip**, never by frame, so a clip's frames never straddle splits.

## Real dataset: UCF-Crime (integrated)

The primary source is **UCF-Crime** — 1,900 real surveillance videos across 13
crime categories. Its provenance, collection method, and **known biases** are
documented in the **data card**: [`cards/ucf_crime.md`](cards/ucf_crime.md).
Two more datasets were evaluated and documented as complements:
[ShanghaiTech](cards/shanghaitech.md) and [VIRAT](cards/virat.md) — see the
[cards index](cards/README.md) for why UCF-Crime was chosen.

### Build the manifest

1. **Get the data** (research use): download UCF-Crime from the
   [official project page](https://www.crcv.ucf.edu/projects/real-world/) and lay
   it out as:

   ```
   data/raw/ucf_crime/
     Videos/<Class>/<Video>_x264.mp4       # 13 class folders + Normal folders
     Anomaly_Train.txt                      # official split files
     Anomaly_Test.txt
     Temporal_Anomaly_Annotation_for_Testing_Videos.txt
   ```

2. **Build a clip-level manifest** (no video decoding — fast, inspectable):

   ```bash
   PYTHONPATH=src python -m data.datasets.ucf_crime \
     --data-root data/raw/ucf_crime \
     --temporal data/raw/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
     --out-dir  data/processed/ucf_crime
   ```

   Writes `manifest.csv` (one auditable row per clip), `manifest.json` (split /
   category / event counts + provenance), and `{train,val,test}.jsonl`. The
   **official train/test split is honored**; a deterministic `val` split is
   carved from train **by clip**. A leakage check fails the build if any clip's
   frames would straddle two splits.

3. **(Optional) Extract frames** for training (needs `opencv-python`): for
   anomaly clips, frames are sampled *inside* the annotated temporal window (and a
   few outside it, labelled normal); normal clips are sampled uniformly:

   ```bash
   PYTHONPATH=src python -m data.datasets.ucf_crime \
     --data-root data/raw/ucf_crime \
     --temporal data/raw/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
     --out-dir data/processed/ucf_crime \
     --extract-frames --frames-per-clip 8
   ```

A committed, illustrative example of the output (no media, synthetic ids) lives
in [`samples/ucf_crime/`](samples/ucf_crime/) so you can see the schema without
downloading anything.

### What the reports are (and aren't)

UCF-Crime ships **category labels + temporal windows, not natural-language
descriptions**, so SENTRY **templates** a factual report per clip from its class
and window. That is honest **weak supervision** (frame → which incident type and
when), not gold human prose — see the *Reports & weak supervision* section of the
[data card](cards/ucf_crime.md). Because of this, **event-overlap F1** (does the
report flag the same incident type as the label?) is the headline metric for this
source, not BLEU/ROUGE n-gram overlap.

## Layout

```
data/
  raw/                 # the downloaded datasets (git-ignored; you provide these)
  processed/           # built manifests + extracted frames (git-ignored)
  samples/ucf_crime/   # committed example manifest (schema demo, no media)
  cards/               # dataset cards: provenance, collection, known biases
```

## Adding another dataset

Write an adapter under `src/data/datasets/<name>.py` that parses the source's raw
annotations into the record schema above (one record per clip), then reuses
`data.manifest.write_manifest(records, out_dir, source=...)` for the CSV/JSONL/
summary + leakage check. Add a matching card under `cards/`. `ucf_crime.py` is the
reference implementation.
