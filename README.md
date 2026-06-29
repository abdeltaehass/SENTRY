# 🛰️ SENTRY — Surveillance Incident Report Generator

SENTRY turns a **surveillance camera frame into a structured, factual incident
report** — what is happening, and any anomalies or notable events — and ships it
with a **reliability score** that flags low-confidence outputs for human review,
plus a **Grad-CAM overlay** showing where each sentence is grounded in the frame.

**Live demo:** https://huggingface.co/spaces/essencelinked/sentry-incident-reports
&nbsp;·&nbsp; **Stack:** PyTorch · Hugging Face transformers/PEFT · BLIP-2 + LoRA · Gradio

> ⚠️ Research / educational project — **not** for real security or surveillance
> decisions. Outputs are model-generated and may be wrong.

---

## Problem
Security operations centers drown in video: a handful of staff watch hundreds of
feeds. The useful signal — "someone left a bag by the entrance and walked away" —
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
- **Prompt-conditioned:** the model is trained and served with the same prompt
  ("Describe what is happening in this frame, including any anomalies or notable
  events, in a concise factual report."), masked out of the loss.
- **Wide-frame aware:** security footage (16:9 / 4:3) is **letterboxed** to square
  rather than squished, so geometry is preserved.

## Dataset
SENTRY consumes a JSONL of `{id, image_path, report, camera_id, location_id}`
(optionally `image_paths` for multi-camera scenes), produced by a pluggable
ingestion pipeline (`src/data/preprocess.py`): video → frame sampling →
letterbox, an annotation parser, and a **leakage-safe split by camera / location
id** (every frame from one camera stays in a single split). Bring your own
labelled surveillance-frame → incident-report source; the schema and split are
ready.

## Evaluation metrics
Text-overlap metrics alone reward fluent boilerplate, so SENTRY reports several axes:

| metric | what it measures |
|---|---|
| BLEU-1..4 / ROUGE-L / METEOR | wording overlap with the reference report |
| **event-overlap F1** | does the report flag the **same incident events** (intrusion, loitering, weapon, theft, …) as the reference? |
| **hallucination rate** | how often the report asserts an event the reference never mentions (reference-grounded) |
| **calibration (ECE)** | does the confidence mean what it says? feeds the reliability score |

## Reliability & hallucination flagging
Every generated report carries a **reliability score** (raw confidence, or a
calibrated P(reliable) when a calibrator is fit). It maps to an operator-facing
risk level, and high-risk reports are **flagged**:

| reliability | risk | action |
|---|---|---|
| ≥ 0.66 | ✅ low | usable draft |
| 0.40–0.66 | ⚠️ elevated | review recommended |
| < 0.40 | 🛑 high — **flagged** | do not act on without human verification |

## Example output (format)
*Illustrative of the output schema and the reliability flag — the model is
prompt-conditioned on the frame.*

| frame | drafted report | reliability | risk |
|---|---|---|---|
| entrance, daytime | "A person enters through the main door carrying a backpack. No anomalies observed." | 0.81 | ✅ low |
| loading bay, dusk | "An individual leaves a bag near the dock and walks out of frame, unattended." | 0.58 | ⚠️ elevated |
| parking lot, night | "Two people are fighting near a vehicle; one appears armed." | 0.27 | 🛑 **flagged — high hallucination risk** |

The third row is the point: a confident-sounding but **low-reliability** report is
flagged rather than surfaced as fact — exactly when a human should verify before
acting. The Grad-CAM overlay is the second check: if "armed" doesn't light up a
plausible region of the frame, the claim isn't grounded.

## Run it
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                         # tests
PYTHONPATH=src python -m data.preprocess --annotations raw/annotations.json --group-key camera_id
PYTHONPATH=src python -m model.train --config configs/default.yaml   # after data is ready
PYTHONPATH=src python app/app.py                           # demo
```

## Project layout
```
src/model/        BLIP-2 + LoRA loader, training loop, inference, prompts, multi-view
src/grounding/    Grad-CAM over ViT patches (letterbox-aware) + overlay
src/eval/         NLG + event-overlap metrics, hallucination analysis, calibration, reliability
src/data/         frame extraction + letterbox + camera/location split, dataset + collator
app/app.py        Gradio demo (frame -> report + reliability + grounding)
configs/          model / LoRA / training / eval config + incident prompt
tests/            unit tests
deploy/           Hugging Face Space
```

## Status & roadmap
Early-stage: the pipeline, evaluation, reliability flagging, grounding, and demo
are in place; the **next step is training on a labelled incident dataset**. The
live demo runs the prompt-conditioned base model so you can exercise the full
flow (report → reliability → grounding) today.

## Ethics
Surveillance report generation is sensitive. SENTRY is a research project, not a
deployed security tool; its outputs are unverified model guesses. The reliability
flag and grounding overlay exist precisely to keep a human in the loop.
