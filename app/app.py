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
from model.temporal import (
    AGGREGATIONS,
    generate_temporal_with_confidence,
    load_frames,
    temporal_prompt,
)
from schema.incident import IncidentReport, now_timestamp

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
                      prompt=incident_prompt(cfg), temporal_prompt=temporal_prompt(cfg),
                      num_frames=cfg.get("temporal.num_frames", 4),
                      default_aggregate=cfg.get("temporal.aggregate", "concat"),
                      calibrator=load_calibrator())
    return _STATE


def _risk_banner(rel: dict) -> str:
    s = rel["reliability_score"]
    if rel["flagged"]:
        return (f"### 🛑 FLAGGED — high hallucination risk &nbsp; (reliability {s:.2f})\n"
                "Do **not** act on this report without human verification.")
    if rel["risk_level"] == "elevated":
        return f"### ⚠️ Elevated hallucination risk &nbsp; (reliability {s:.2f}) — review recommended"
    return f"### ✅ Low hallucination risk &nbsp; (reliability {s:.2f})"


def analyze(image, location):
    if image is None:
        return "Please upload a frame.", {}, "", None, gr.update(choices=[], value=None), {}
    s = _model()
    image = image.convert("RGB")
    report, conf = generate_with_confidence(
        s["model"], s["processor"], image, s["device"],
        max_new_tokens=96, prompt=s["prompt"], **DECODE,
    )
    rel = assess_reliability(conf, s["calibrator"])
    cam = grounding_cam(s["model"], s["processor"], image, report, s["device"],
                        letterbox_input=True)
    overlay = overlay_heatmap(image, cam, letterbox_input=True)
    structured = IncidentReport.from_signals(
        report, confidence=conf, reliability=rel, cam=cam,
        timestamp=now_timestamp(), location=(location or None),
    ).to_dict()
    sentences = split_sentences(report)
    return (
        report,
        {"reliability": rel["reliability_score"]},
        _risk_banner(rel),
        overlay,
        gr.update(choices=sentences, value=sentences[0] if sentences else None),
        structured,
    )


def ground_sentence(image, sentence):
    if image is None or not sentence:
        return None
    s = _model()
    image = image.convert("RGB")
    cam = grounding_cam(s["model"], s["processor"], image, sentence, s["device"], letterbox_input=True)
    return overlay_heatmap(image, cam, letterbox_input=True)


def _file_paths(files) -> list[str]:
    """Normalize gr.File (filepath mode) output to a list of path strings."""
    if not files:
        return []
    files = files if isinstance(files, list) else [files]
    return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]


def analyze_clip(files, aggregate, location):
    """Temporal path: 3-5 ordered frames -> one report describing events over time."""
    paths = _file_paths(files)
    if len(paths) < 2:
        return ("Upload at least 2 ordered frames (3-5 works best) to read a "
                "sequence of events.", {}, "", "", {})
    s = _model()
    aggregate = aggregate or s["default_aggregate"]
    images = load_frames(paths, letterbox_input=True, num_frames=s["num_frames"])
    report, conf = generate_temporal_with_confidence(
        s["model"], s["processor"], images, s["device"],
        max_new_tokens=96, prompt=s["temporal_prompt"], strategy=aggregate, **DECODE,
    )
    rel = assess_reliability(conf, s["calibrator"])
    structured = IncidentReport.from_signals(
        report, confidence=conf, reliability=rel,
        timestamp=now_timestamp(), location=(location or None),
    ).to_dict()
    # Single-frame baseline on the last frame, to make the temporal gain visible.
    baseline, _ = generate_with_confidence(
        s["model"], s["processor"], images[-1], s["device"],
        max_new_tokens=96, prompt=s["prompt"], **DECODE,
    )
    return (report, {"reliability": rel["reliability_score"]}, _risk_banner(rel),
            baseline, structured)


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

        with gr.Tabs():
            with gr.Tab("Single frame"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(type="pil", label="Surveillance frame", height=360)
                        location_in = gr.Textbox(
                            label="Camera location (optional)", placeholder="e.g. east entrance"
                        )
                        run = gr.Button("Generate incident report", variant="primary")
                        examples = _examples()
                        if examples:
                            gr.Examples(examples=examples, inputs=image_in, label="Example frames")
                    with gr.Column(scale=1):
                        report_out = gr.Textbox(label="Drafted incident report", lines=6)
                        risk_out = gr.Markdown()
                        conf_out = gr.Label(label="Reliability (0 = unreliable, 1 = reliable)")
                        overlay_out = gr.Image(label="Grad-CAM grounding", height=360)
                        sentence_dd = gr.Dropdown(
                            label="Ground a specific sentence", choices=[], interactive=True
                        )
                        json_out = gr.JSON(label="Structured incident report (JSON)")

                run.click(analyze, inputs=[image_in, location_in],
                          outputs=[report_out, conf_out, risk_out, overlay_out, sentence_dd,
                                   json_out])
                sentence_dd.change(ground_sentence, inputs=[image_in, sentence_dd],
                                   outputs=overlay_out)

            with gr.Tab("🎞️ Multi-frame (temporal)"):
                gr.Markdown(
                    "Upload **3–5 ordered frames** from a short clip. SENTRY encodes "
                    "each frame, aggregates across the sequence, and reports the "
                    "**events over time** — not just one scene. The single-frame "
                    "baseline (last frame only) is shown for contrast."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        frames_in = gr.File(
                            file_count="multiple", type="filepath",
                            label="Ordered frames (3–5)", file_types=["image"],
                        )
                        aggregate_dd = gr.Dropdown(
                            choices=list(AGGREGATIONS), value="concat",
                            label="Cross-frame aggregation",
                        )
                        clip_location_in = gr.Textbox(
                            label="Camera location (optional)", placeholder="e.g. loading bay"
                        )
                        run_clip = gr.Button("Generate temporal report", variant="primary")
                    with gr.Column(scale=1):
                        clip_report_out = gr.Textbox(label="Temporal incident report", lines=6)
                        clip_risk_out = gr.Markdown()
                        clip_conf_out = gr.Label(label="Reliability (0 = unreliable, 1 = reliable)")
                        baseline_out = gr.Textbox(
                            label="Single-frame baseline (last frame only)", lines=4
                        )
                        clip_json_out = gr.JSON(label="Structured incident report (JSON)")

                run_clip.click(
                    analyze_clip, inputs=[frames_in, aggregate_dd, clip_location_in],
                    outputs=[clip_report_out, clip_conf_out, clip_risk_out, baseline_out,
                             clip_json_out],
                )
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
