# Data Card — UCF-Crime

> A data card documents what a dataset contains, how it was collected, and its
> known limitations, so anyone using it can reason about what the resulting model
> learned — and what it didn't. This card covers **UCF-Crime**, SENTRY's primary
> real-data source. Adapter: [`src/data/datasets/ucf_crime.py`](../../src/data/datasets/ucf_crime.py).

---

## At a glance

| | |
|---|---|
| **Name** | UCF-Crime (a.k.a. "Real-world Anomaly Detection in Surveillance Videos") |
| **Authors** | Waqas Sultani, Chen Chen, Mubarak Shah — Center for Research in Computer Vision, University of Central Florida |
| **Published** | CVPR 2018 |
| **Modality** | Untrimmed real-world CCTV / surveillance video (RGB, with audio) |
| **Size** | **1,900 videos**, ~**128 hours**, 30 fps |
| **Labels** | 13 anomaly classes + normal; **clip-level** class labels, plus **frame-level temporal windows** for the test set |
| **Task in original paper** | Weakly-supervised anomaly *detection* (anomaly score per frame) via multiple-instance learning |
| **Task in SENTRY** | Conditioning data for incident-**report generation** (re-purposed; see *Reports* below) |
| **Project page** | https://www.crcv.ucf.edu/projects/real-world/ |
| **Paper** | Sultani et al., *Real-world Anomaly Detection in Surveillance Videos*, CVPR 2018 — https://arxiv.org/abs/1801.04264 |

## What's in it

**13 anomaly categories** (chosen by the authors for public-safety impact):

```
Abuse   Arrest   Arson   Assault   Burglary   Explosion   Fighting
RoadAccidents   Robbery   Shooting   Shoplifting   Stealing   Vandalism
```

plus **Normal** footage (no anomalous event). Counts are roughly balanced
anomaly-vs-normal at the video level (≈950 anomaly / ≈950 normal), but the 13
anomaly classes themselves are **not** balanced (e.g. far more `RoadAccidents`
and `Normal` than `Abuse` or `Arson`).

The videos are **untrimmed and long** — a single clip may run several minutes
with the anomaly occupying only a few seconds. This is why the temporal
annotation (below) matters: most frames of an "anomaly" video are visually normal.

## How it was collected

- Videos were **sourced from the web** (YouTube and LiveLeak) using text-search
  queries for each crime type, plus extended queries in multiple languages to
  reduce sourcing bias.
- A team of **ten annotators** reviewed candidates. Videos were **discarded** if
  they were manually edited, pranks/staged, not captured by CCTV, taken by a news
  camera, a compilation, or visually ambiguous — leaving real, single-shot
  surveillance footage.
- **Temporal annotation** of the *test* set: multiple annotators independently
  marked the start/end frames of each anomalous event; the average was taken to
  reduce labeling variance.

## Annotation files this adapter reads

| File | Content | Used for |
|---|---|---|
| `Anomaly_Train.txt` | one relative video path per line, e.g. `Abuse/Abuse028_x264.mp4` | official train split + class (from folder) |
| `Anomaly_Test.txt` | same format, test videos | official test split + class |
| `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | `<file> <class> <s1> <e1> <s2> <e2>` (frame indices @30fps, `-1` = none) | per-event temporal windows on the test set |

The normal videos live under `Normal_Videos_event/`,
`Training_Normal_Videos_Anomaly/`, and `Testing_Normal_Videos_Anomaly/`; the
adapter normalizes all of these to the `Normal` class.

## How SENTRY processes it

The pipeline is in [`src/data/datasets/ucf_crime.py`](../../src/data/datasets/ucf_crime.py):

1. **Parse** the official split files → one record per clip, class inferred from
   the folder prefix. The **official train/test split is honored** (not
   re-randomized); a deterministic `val` split is carved from train **by clip**.
2. **Map** each class onto SENTRY's event taxonomy (`eval.metrics.EVENT_CATEGORIES`)
   so generated reports are scorable with event-overlap F1 on real data
   (e.g. `Shooting → {weapon, violence}`, `Burglary → {theft, intrusion}`).
3. **Template a report** per clip from its class label (and, for the test set,
   the temporal window) — see *Reports & weak supervision* below.
4. **(Optional) Extract frames**: for anomaly clips, sample frames *inside* the
   annotated window (labelled with the clip's events) plus a few *outside* it
   (labelled normal); normal clips are sampled uniformly. Produces a frame-level
   manifest the dataloader trains on.
5. **Write a manifest** (`manifest.csv` + `{train,val,test}.jsonl` +
   `manifest.json` summary) via the shared `data.manifest` writer, with a
   leakage check that no clip's frames straddle two splits.

A small, committed example of the output lives in
[`data/samples/ucf_crime/`](../samples/ucf_crime/).

## Reports & weak supervision (important)

UCF-Crime ships **class labels and temporal windows, not natural-language
descriptions.** SENTRY's reports for this dataset are therefore **programmatically
templated** from the class label + temporal window, e.g.:

> *"Surveillance footage shows a physical fight between individuals. The
> anomalous event occurs between ~18s to ~22s into the clip."*

This is a **weak-supervision** signal: it teaches the model the mapping
*frame → which incident category and when*, but **not** fine-grained free-text
description (number of people, clothing, direction of travel, etc.). Treat the
templated text as a label, not as a gold human report. Two consequences:

- Metrics like BLEU/ROUGE against templated references mostly measure
  **class/temporal agreement**, which is why **event-overlap F1** is the headline
  metric for this source, not n-gram overlap.
- For genuinely descriptive reports, pair UCF-Crime with a captioned source (see
  *Alternatives*) — UCF-Crime supplies real incident **variety and grounding**;
  a captioned set supplies **prose supervision**.

## Known biases & limitations

- **Web-sourced selection bias.** Footage that made it onto YouTube/LiveLeak is
  not a random sample of real surveillance: dramatic, shareable events are
  over-represented. The "normal" distribution is whatever the authors paired,
  not a true base rate.
- **Class imbalance.** The 13 anomaly classes are uneven; naive training will
  favor majority classes. Use class weighting / resampling and **report
  per-class metrics**, not just the average.
- **Geographic & cultural skew.** Search-query sourcing skews toward
  English-language and certain regions; scene types, vehicles, signage, and dress
  are not globally representative.
- **Visual quality variance.** Real CCTV: low resolution, compression artifacts,
  night/IR footage, timestamp overlays, watermarks, varied aspect ratios
  (handled by SENTRY's letterboxing).
- **Sparse anomalies in long clips.** Most frames of an anomaly video are normal;
  uniform frame sampling without the temporal window will mislabel them. The
  adapter uses the windows for the test set; **train clips have no temporal
  labels** in UCF-Crime, so train-frame labels are clip-level (a known weakness
  shared with the original weakly-supervised setup).
- **Sensitive content.** Real depictions of violence, crime, and people who did
  **not** consent to be in a dataset. Handle accordingly (see *Ethics & use*).
- **Label noise.** Web video + averaged human temporal labels → boundaries are
  approximate, not frame-exact.

## License & access

- UCF-Crime is released by CRCV/UCF **for academic research use**. It is **not
  redistributed in this repository** — you must obtain it from the official
  project page and accept its terms. Only the adapter, the annotation **manifest
  schema**, and a tiny synthetic sample are committed here.
- The underlying clips remain subject to their **original platform rights**;
  SENTRY treats them as research-only and ships **no weights trained to identify
  real individuals**.

## Ethics & intended use

SENTRY is a **research/educational** project. UCF-Crime is used to study
incident-**report drafting with calibrated reliability**, with a human in the
loop — **not** to build an operational surveillance or person-identification
system. Do not use this data or any model trained on it to make real security,
legal, or safety decisions about real people. See the repository's *Ethics*
section.

## Citation

```bibtex
@inproceedings{sultani2018real,
  title     = {Real-world Anomaly Detection in Surveillance Videos},
  author    = {Sultani, Waqas and Chen, Chen and Shah, Mubarak},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}
```
