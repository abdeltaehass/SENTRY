# 🛰️ SENTRY — Surveillance Incident Report Generator

> A vision-language model that turns a **surveillance camera frame into a
> structured incident report** — *and tells you when not to trust it.* Every
> report ships with a **calibrated reliability score** that flags low-confidence
> outputs for human review, plus a **Grad-CAM overlay** showing where each claim
> is grounded in the frame.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
<img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Transformers%20%C2%B7%20PEFT-FFD21E">
<img alt="Gradio" src="https://img.shields.io/badge/Gradio-demo-F97316">
<img alt="Tests" src="https://img.shields.io/badge/tests-35%20passing-2ea44f">
<img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

**▶️ Live demo:** https://huggingface.co/spaces/essencelinked/sentry-incident-reports
&nbsp;·&nbsp; **Stack:** PyTorch · Hugging Face `transformers`/`PEFT` · BLIP-2 + LoRA · Gradio

> ⚠️ Research / educational project — **not** for real security or surveillance
> decisions. Outputs are model-generated and may be wrong.

---

## What this project demonstrates

A compact, end-to-end ML-engineering portfolio piece — not just a model, but the
scaffolding that makes one trustworthy:

- **Vision-language fine-tuning** — BLIP-2 adapted with **LoRA** (parameter-efficient, ~0.3% of weights trained).
- **Real dataset integration** — a working dataloader + **cleaned annotation manifest** for **UCF-Crime** (1,900 real CCTV videos, 13 crime classes), with **data cards** documenting provenance and bias. *(See [Dataset](#dataset).)*
- **Leakage-safe evaluation** — splits by **camera/clip**, never by frame, with an automated leakage check.
- **Trustworthy outputs** — **calibration (ECE)** + a **reliability score** that flags low-confidence reports, and **Grad-CAM grounding** as a second check.
- **Task-aware metrics** — beyond BLEU/ROUGE: an **event-overlap F1** and a **hallucination rate** that measure *facts*, not fluency.
- **Engineering hygiene** — typed config, a pluggable ingestion pipeline, **35 passing tests**, `ruff`-clean data layer, and a one-click deployed demo.

## Problem

Security operations centers drown in video: a handful of staff watch hundreds of
feeds. The useful signal — *"someone left a bag by the entrance and walked away"* —
is rare and easy to miss. Vision-language models can draft a written incident
report from a frame, but they have a dangerous failure mode: they confidently
describe things that aren't there. A report generator is only useful for security
if it also tells you **when not to trust it**. SENTRY is built around that:
generation **plus** measured reliability.

## How it works

A vision-language model is adapted with parameter-efficient fine-tuning:

```
camera frame ─► ViT image encoder ─► Q-Former + projection ─► OPT-2.7B (LLM) ─► incident report
                  (frozen)              (projection: trained)    (frozen + LoRA)
```

- **Backbone:** BLIP-2 (`Salesforce/blip2-opt-2.7b`).
- **Training:** freeze the ViT, Q-Former, and OPT base; train only **LoRA adapters
  on the OPT decoder + the projection layer** — a small, fast, ~0.3% footprint.
- **Prompt-conditioned:** trained and served with the same prompt ("Describe what
  is happening in this frame, including any anomalies or notable events…"), masked
  out of the loss so the two stay consistent.
- **Wide-frame aware:** security footage (16:9 / 4:3) is **letterboxed** to square
  rather than squished, so geometry is preserved.

## Dataset

SENTRY integrates **[UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)** —
**1,900 real surveillance videos**, ~128 hours, across **13 crime categories**
(abuse, arson, assault, burglary, explosion, fighting, road accidents, robbery,
shooting, shoplifting, stealing, vandalism, arrest) plus normal footage.

| | |
|---|---|
| **Adapter** | [`src/data/datasets/ucf_crime.py`](src/data/datasets/ucf_crime.py) — parses the official `Anomaly_Train/Test.txt` + temporal-annotation files |
| **Manifest** | `manifest.csv` + `{train,val,test}.jsonl` + a `manifest.json` summary, via the reusable [`data.manifest`](src/data/manifest.py) writer |
| **Splits** | official train/test **honored**; a `val` split carved from train **by clip**; automated **leakage check** |
| **Labels → events** | each class is mapped onto SENTRY's event taxonomy (e.g. `Shooting → {weapon, violence}`) so reports score against **event-F1** on real data |
| **Data cards** | [`data/cards/`](data/cards/) — provenance, collection method, and **known biases** for UCF-Crime, plus [ShanghaiTech](data/cards/shanghaitech.md) & [VIRAT](data/cards/virat.md) as evaluated alternatives |
| **Sample** | a committed, media-free [example manifest](data/samples/ucf_crime/) shows the schema without any download |

**Honest framing:** UCF-Crime ships category labels + temporal windows, not prose,
so reports are **templated** from labels as weak supervision (frame → *which
incident type and when*). That limitation is documented in the
[data card](data/cards/ucf_crime.md) — and is exactly why event-overlap F1, not
BLEU, is the headline metric for this source. See [`data/README.md`](data/README.md)
for the full build workflow.

## Evaluation

Text-overlap metrics alone reward fluent boilerplate, so SENTRY reports several axes:

| metric | what it measures |
|---|---|
| BLEU-1..4 / ROUGE-L / METEOR | wording overlap with the reference report |
| **event-overlap F1** | does the report flag the **same incident events** (intrusion, loitering, weapon, theft, …) as the reference? |
| **hallucination rate** | how often the report asserts an event the reference never mentions (reference-grounded) |
| **calibration (ECE)** | does the confidence mean what it says? feeds the reliability score |

## Reliability & hallucination flagging

Every generated report carries a **reliability score** (raw confidence, or a
calibrated P(reliable) once a calibrator is fit). It maps to an operator-facing
risk level, and high-risk reports are **flagged**:

| reliability | risk | action |
|---|---|---|
| ≥ 0.66 | ✅ low | usable draft |
| 0.40–0.66 | ⚠️ elevated | review recommended |
| < 0.40 | 🛑 high — **flagged** | do not act on without human verification |

**Example (output format):**

| frame | drafted report | reliability | risk |
|---|---|---|---|
| entrance, daytime | "A person enters through the main door carrying a backpack. No anomalies observed." | 0.81 | ✅ low |
| loading bay, dusk | "An individual leaves a bag near the dock and walks out of frame, unattended." | 0.58 | ⚠️ elevated |
| parking lot, night | "Two people are fighting near a vehicle; one appears armed." | 0.27 | 🛑 **flagged** |

The third row is the point: a confident-sounding but **low-reliability** report is
flagged rather than surfaced as fact. The Grad-CAM overlay is the second check: if
"armed" doesn't light up a plausible region of the frame, the claim isn't grounded.

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m pytest -q                  # 35 tests

# Build the real-dataset manifest (after placing UCF-Crime under data/raw/ — see data/README.md)
PYTHONPATH=src python -m data.datasets.ucf_crime \
  --data-root data/raw/ucf_crime \
  --temporal  data/raw/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
  --out-dir   data/processed/ucf_crime

PYTHONPATH=src python -m model.train --config configs/default.yaml   # fine-tune (LoRA)
PYTHONPATH=src python app/app.py                                     # local demo
```

## Project layout

```
src/data/         UCF-Crime adapter + manifest writer, dataset/collator, frame extraction + letterbox + leakage-safe split
src/model/        BLIP-2 + LoRA loader, training loop, inference, prompts, multi-view
src/grounding/    Grad-CAM over ViT patches (letterbox-aware) + overlay
src/eval/         NLG + event-overlap metrics, hallucination analysis, calibration, reliability
data/cards/       dataset cards: provenance, collection, known biases
data/samples/     committed example manifest (schema demo, no media)
app/app.py        Gradio demo (frame -> report + reliability + grounding)
configs/          model / LoRA / training / eval config + incident prompt
tests/            35 unit tests (hermetic; no download, no GPU)
deploy/           Hugging Face Space
```

## Status & roadmap

The pipeline, **real-dataset integration** (UCF-Crime dataloader + manifest + data
cards), evaluation, reliability flagging, grounding, and a deployed demo are in
place. The live demo runs the prompt-conditioned base model so you can exercise the
full flow (report → reliability → grounding) today.

**Next:** fine-tune on UCF-Crime end-to-end and publish per-class metrics · add a
ShanghaiTech grounding-evaluation adapter · pair with a captioned source for true
free-text report supervision (see the [cards roadmap](data/cards/README.md#roadmap)).

## Ethics

Surveillance report generation is sensitive. SENTRY is a research project, not a
deployed security tool; its outputs are unverified model guesses, and it ships no
weights trained to identify real individuals. The reliability flag and grounding
overlay exist precisely to keep a human in the loop. The data cards document the
biases of the underlying footage so the model's blind spots are visible, not hidden.
