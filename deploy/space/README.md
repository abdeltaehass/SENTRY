---
title: SENTRY Incident Report Generator
emoji: 🛰️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
---

# SENTRY — Incident Report Generator

Upload a surveillance frame → a drafted incident report (what is happening, plus
anomalies / notable events) and a **reliability score** that flags low-confidence
outputs for human review.

⚠️ **Research demo — not for real security or surveillance decisions.** Outputs
are model-generated and may be wrong.

- Code: https://github.com/abdeltaehass/SENTRY
- This Space runs the prompt-conditioned **base** vision-language model on free
  CPU (int8-quantized). Grad-CAM grounding (which needs gradients) and the
  fine-tuned incident model run in the GitHub repo.
