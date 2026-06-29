# 🛰️ SENTRY — Surveillance Incident Report Generator

A vision-language model that turns a surveillance image / camera frame into a
structured **incident report**, with a generation confidence score and a Grad-CAM
grounding overlay showing where each sentence is anchored.

> ⚠️ Educational / research project — not for real security or surveillance
> decisions.

This repo is scaffolded from a reusable VLM engine: BLIP-2 + LoRA fine-tuning,
Grad-CAM grounding, confidence calibration, a hallucination/reliability analysis,
a Gradio demo skeleton, and HF-Space-style deployment. The **domain-agnostic core
is in place**; the surveillance-specific layers are TODO.

## Structure
```
src/
  config.py            YAML -> attribute-access config
  data/preprocess.py   frame extraction + letterbox + camera/location split -> JSONL  (scaffold)
  data/dataset.py      IncidentReportDataset (aspect-preserving) + prompt-aware collator
  model/               BLIP-2 + LoRA loader, training loop, inference, prompts, multi-view
  grounding/           Grad-CAM over ViT patches + heatmap overlay
  eval/                NLG metrics, event-overlap metric, hallucination analysis, calibration
app/app.py             Gradio demo (upload image -> incident report + confidence + grounding)
configs/default.yaml   model / LoRA / training / eval config + incident prompt
tests/                 unit tests for the core
```

## Run (PYTHONPATH=src)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                 # tests
PYTHONPATH=src python -m model.train --config configs/default.yaml   # after data is ready
PYTHONPATH=src python app/app.py                   # demo
```

## Data pipeline
`src/data/preprocess.py` scaffolds the surveillance pipeline: optional video
**frame extraction**, **letterbox** aspect handling for wide footage, a leakage-safe
**split by camera/location id**, and a **placeholder annotation schema** (see its
docstring). Wire your dataset into `build_records`, then:
```bash
PYTHONPATH=src python -m data.preprocess --annotations raw/annotations.json --group-key camera_id
```
The model is **prompt-conditioned** — `model.prompt` in `configs/default.yaml` is
masked out of the training loss and passed to `generate` at inference, so training
and serving stay consistent.

## TODO (dataset-specific)
- **Plug in your dataset:** adapt `build_records` to your annotation format, and
  confirm a labelled surveillance-frame → incident-report source.
- **Event lexicon:** refine `EVENT_CATEGORIES` in `src/eval/metrics.py` for your
  incident taxonomy (intrusion, loitering, weapon, theft, …).
- **Branding / copy:** finalize the app text and a domain README.
