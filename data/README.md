# Data

SENTRY consumes a processed JSONL per split, one record per line:

```json
{"id": "scene_0001",
 "image_path":  "data/raw/incidents/frames/scene_0001.jpg",
 "image_paths": ["...cam1.jpg", "...cam2.jpg"],
 "report": "An individual climbs the perimeter fence and enters the yard. No weapon is visible."}
```

- `image_paths` is optional (multiple camera views of one scene → see `model.multiview`).
- Split **by scene / camera / time**, never by frame, to prevent train/test leakage.

The ingestion + preprocessing that produces these files is **domain-specific** and
is the main thing to build for SENTRY (sourcing a labelled surveillance-frame →
incident-report dataset is the hard part). Write splits to:

```
data/processed/incidents/{train,val,test}.jsonl
```
