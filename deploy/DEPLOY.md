# Deploying SENTRY to Hugging Face Spaces

A live, public demo on free HF hosting. The push needs **your** HF account.

## What gets deployed
`deploy/space/` + the vendored `src/` and `configs/`:
- `app.py` — CPU demo: frame → incident report + reliability score.
- `requirements.txt` — CPU torch + runtime deps.
- `README.md` — Space metadata (`sdk: gradio`).
- `src/`, `configs/` — vendored so the Space is self-contained.

## Deploy (one command)
```bash
hf auth login          # WRITE token from huggingface.co/settings/tokens
python scripts/deploy_space.py --repo-id <your-username>/sentry-incident-reports
```
Prints the live URL: `https://huggingface.co/spaces/<your-username>/sentry-incident-reports`.

## Notes
- With no fine-tuned adapter yet, the Space runs the **prompt-conditioned base
  model** (set `SENTRY_ADAPTER` once an adapter is trained and uploaded).
- The base VLM is large for free CPU (16 GB); int8 quantization helps, but if it
  hits memory limits, switch the Space hardware to **ZeroGPU** or a larger tier.
- Grad-CAM grounding needs gradients (disabled under quantization) — it runs in
  the local/GPU app.
