"""SENTRY interactive demo (Gradio).

Upload a surveillance image / camera frame → drafted incident report + a
**reliability score** (high-risk outputs are flagged for human review) + a
Grad-CAM grounding overlay, with a dropdown to ground any individual sentence.

Run:
    PYTHONPATH=src python app/app.py      # then open the printed local URL
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from config import load_config
from eval.reliability import assess_reliability, load_calibrator
from grounding import grounding_cam, overlay_heatmap, split_sentences
from model.inference import generate_with_confidence
from model.model import load_base, load_for_inference, pick_device
from model.prompts import incident_prompt

CONFIG = "configs/default.yaml"
ADAPTER = "outputs/incident_lora"        # trained adapter, if present
DECODE = dict(num_beams=4, repetition_penalty=1.5, no_repeat_ngram_size=3)

DISCLAIMER = (
    "**SENTRY** drafts an incident report from a surveillance frame, with a "
    "**reliability score** (high-risk outputs are flagged for human review) and a "
    "Grad-CAM overlay showing which region each sentence is grounded in.\n\n"
    "⚠️ *Research demo — not for real security or surveillance decisions. Outputs "
    "are a baseline and may be inaccurate.*"
)

# Lazy, cached model so importing this module (and building the UI) is cheap.
_STATE: dict = {}


def _model():
    if "model" not in _STATE:
        device = pick_device()
        cfg = load_config(CONFIG)
        if Path(ADAPTER).exists():
            model, processor = load_for_inference(cfg, ADAPTER, device=device)
        else:
            model, processor = load_base(cfg, device=device)   # base VLM; fine-tuning pending
        model.to(device).eval()
        _STATE.update(model=model, processor=processor, device=device,
                      prompt=incident_prompt(cfg), calibrator=load_calibrator())
    return _STATE


def _risk_banner(rel: dict) -> str:
    s = rel["reliability_score"]
    if rel["flagged"]:
        return (f"### 🛑 FLAGGED — high hallucination risk &nbsp; (reliability {s:.2f})\n"
                "Do **not** act on this report without human verification.")
    if rel["risk_level"] == "elevated":
        return f"### ⚠️ Elevated hallucination risk &nbsp; (reliability {s:.2f}) — review recommended"
    return f"### ✅ Low hallucination risk &nbsp; (reliability {s:.2f})"


def analyze(image):
    if image is None:
        return "Please upload a frame.", {}, "", None, gr.update(choices=[], value=None)
    s = _model()
    image = image.convert("RGB")
    report, conf = generate_with_confidence(
        s["model"], s["processor"], image, s["device"],
        max_new_tokens=96, prompt=s["prompt"], **DECODE,
    )
    rel = assess_reliability(conf, s["calibrator"])
    overlay = overlay_heatmap(
        image,
        grounding_cam(s["model"], s["processor"], image, report, s["device"], letterbox_input=True),
        letterbox_input=True,
    )
    sentences = split_sentences(report)
    return (
        report,
        {"reliability": rel["reliability_score"]},
        _risk_banner(rel),
        overlay,
        gr.update(choices=sentences, value=sentences[0] if sentences else None),
    )


def ground_sentence(image, sentence):
    if image is None or not sentence:
        return None
    s = _model()
    image = image.convert("RGB")
    cam = grounding_cam(s["model"], s["processor"], image, sentence, s["device"], letterbox_input=True)
    return overlay_heatmap(image, cam, letterbox_input=True)


def _examples() -> list[list[str]]:
    candidates = [
        "data/raw/incidents/examples/frame_0001.jpg",
        "data/raw/incidents/examples/frame_0002.jpg",
    ]
    return [[p] for p in candidates if Path(p).exists()]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SENTRY — Incident Report Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛰️ SENTRY — Incident Report Generator")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Surveillance frame", height=360)
                run = gr.Button("Generate incident report", variant="primary")
                examples = _examples()
                if examples:
                    gr.Examples(examples=examples, inputs=image_in, label="Example frames")
            with gr.Column(scale=1):
                report_out = gr.Textbox(label="Drafted incident report", lines=6)
                risk_out = gr.Markdown()
                conf_out = gr.Label(label="Reliability score (0 = unreliable, 1 = reliable)")
                overlay_out = gr.Image(label="Grad-CAM grounding", height=360)
                sentence_dd = gr.Dropdown(
                    label="Ground a specific sentence", choices=[], interactive=True
                )

        run.click(analyze, inputs=image_in,
                  outputs=[report_out, conf_out, risk_out, overlay_out, sentence_dd])
        sentence_dd.change(ground_sentence, inputs=[image_in, sentence_dd], outputs=overlay_out)
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
