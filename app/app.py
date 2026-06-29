"""SENTRY interactive demo (Gradio).

Upload a surveillance image / camera frame → generated incident report +
generation confidence + a Grad-CAM grounding overlay, with a dropdown to ground
any individual sentence.

Run:
    PYTHONPATH=src python app/app.py      # then open the printed local URL
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from config import load_config
from grounding import grounding_cam, overlay_heatmap, split_sentences
from model.inference import generate_with_confidence
from model.model import load_for_inference, pick_device
from model.prompts import incident_prompt

CONFIG = "configs/default.yaml"
ADAPTER = "outputs/incident_lora"
DECODE = dict(num_beams=4, repetition_penalty=1.5, no_repeat_ngram_size=3)

DISCLAIMER = (
    "**SENTRY** turns a surveillance image into a draft incident report, with a "
    "confidence score and a Grad-CAM overlay showing where each sentence is "
    "grounded.\n\n"
    "⚠️ *Educational / research demo — not for real security or surveillance "
    "decisions. Outputs are a baseline model and may be inaccurate.*"
)

# Lazy, cached model so importing this module (and building the UI) is cheap.
_STATE: dict = {}


def _model():
    if "model" not in _STATE:
        device = pick_device()
        cfg = load_config(CONFIG)
        model, processor = load_for_inference(cfg, ADAPTER, device=device)
        model.to(device).eval()
        _STATE.update(model=model, processor=processor, device=device, prompt=incident_prompt(cfg))
    return _STATE


def analyze(image):
    if image is None:
        return "Please upload an image.", {}, None, gr.update(choices=[], value=None)
    s = _model()
    image = image.convert("RGB")
    report, conf = generate_with_confidence(
        s["model"], s["processor"], image, s["device"],
        max_new_tokens=96, prompt=s["prompt"], **DECODE
    )
    overlay = overlay_heatmap(image, grounding_cam(s["model"], s["processor"], image, report, s["device"]))
    sentences = split_sentences(report)
    return (
        report,
        {"generation confidence": conf},
        overlay,
        gr.update(choices=sentences, value=sentences[0] if sentences else None),
    )


def ground_sentence(image, sentence):
    if image is None or not sentence:
        return None
    s = _model()
    image = image.convert("RGB")
    return overlay_heatmap(image, grounding_cam(s["model"], s["processor"], image, sentence, s["device"]))


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
                image_in = gr.Image(type="pil", label="Surveillance image / camera frame", height=360)
                run = gr.Button("Generate report", variant="primary")
                examples = _examples()
                if examples:
                    gr.Examples(examples=examples, inputs=image_in, label="Example frames")
            with gr.Column(scale=1):
                report_out = gr.Textbox(label="Generated incident report", lines=7)
                conf_out = gr.Label(label="Confidence (geometric-mean token probability)")
                overlay_out = gr.Image(label="Grad-CAM grounding", height=360)
                sentence_dd = gr.Dropdown(
                    label="Ground a specific sentence", choices=[], interactive=True
                )

        run.click(analyze, inputs=image_in,
                  outputs=[report_out, conf_out, overlay_out, sentence_dd])
        sentence_dd.change(ground_sentence, inputs=[image_in, sentence_dd], outputs=overlay_out)
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
