"""Single-image inference: image -> generated incident report.

Loads the base VLM + a trained LoRA adapter and generates report text for an
image. Same code path the Gradio demo calls.

Usage:
    PYTHONPATH=src python -m model.inference --image path/to/frame.jpg \
        --config configs/default.yaml --adapter outputs/incident_lora
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image

from config import load_config

from .model import load_for_inference, pick_device
from .prompts import incident_prompt


def _strip_prompt(text: str, prompt: str) -> str:
    """Decoder-only models echo the text prompt; drop it from the output."""
    return text[len(prompt):].strip() if prompt and text.startswith(prompt) else text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an incident report for one image")
    p.add_argument("--image", required=True, help="Path to an image")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--adapter", required=True, help="Path to a trained adapter dir")
    p.add_argument("--json", action="store_true",
                   help="emit a structured incident-report JSON object instead of raw text")
    p.add_argument("--location", default=None, help="camera location tag for the JSON record")
    return p.parse_args()


@torch.no_grad()
def generate_report(image_path: str, cfg, adapter_path: str, device: str | None = None) -> str:
    device = device or pick_device()
    model, processor = load_for_inference(cfg, adapter_path, device=device)
    model.to(device).eval()

    prompt = incident_prompt(cfg)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.get("eval.max_new_tokens", 96),
        num_beams=cfg.get("eval.num_beams", 4),
        repetition_penalty=cfg.get("eval.repetition_penalty", 1.5),
        no_repeat_ngram_size=cfg.get("eval.no_repeat_ngram_size", 3),
    )
    return _strip_prompt(processor.batch_decode(out, skip_special_tokens=True)[0].strip(), prompt)


@torch.no_grad()
def generate_with_confidence(model, processor, image, device, max_new_tokens: int = 96,
                             prompt: str | None = None, **decode):
    """Generate a report and a confidence score (geometric-mean token probability).

    `prompt` conditions generation (prompt-template). Works for greedy/sampling and
    beam search (uses beam_indices when present). Returns (text, confidence in [0, 1]).
    """
    proc_kwargs = {"text": prompt} if prompt else {}
    inputs = processor(images=image, return_tensors="pt", **proc_kwargs).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         output_scores=True, return_dict_in_generate=True, **decode)
    beam_indices = getattr(out, "beam_indices", None)
    trans = model.compute_transition_scores(
        out.sequences, out.scores, beam_indices, normalize_logits=True
    )
    finite = torch.isfinite(trans)
    confidence = float(trans[finite].mean().exp()) if finite.any() else 0.0
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
    return _strip_prompt(text, prompt or ""), confidence


@torch.no_grad()
def generate_structured(model, processor, image, device, *, prompt, calibrator=None,
                        cam=None, timestamp=None, location=None,
                        max_new_tokens: int = 96, **decode):
    """Generate a validated structured incident report (``schema.IncidentReport``).

    Wraps ``generate_with_confidence`` and folds in the reliability assessment,
    so the returned record carries the description, incident type, confidence,
    hallucination flag, and (if a ``cam`` is supplied) grounding regions.
    """
    from eval.reliability import assess_reliability
    from schema.incident import IncidentReport

    text, conf = generate_with_confidence(
        model, processor, image, device, max_new_tokens=max_new_tokens, prompt=prompt, **decode
    )
    rel = assess_reliability(conf, calibrator)
    return IncidentReport.from_signals(
        text, confidence=conf, reliability=rel, cam=cam,
        timestamp=timestamp, location=location,
    )


@torch.no_grad()
def generate_texts(model, processor, image_paths, device, max_new_tokens: int = 128, **decode) -> list[str]:
    """Generate report text for many images with a given decoding config."""
    out_texts: list[str] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **decode)
        out_texts.append(processor.batch_decode(out, skip_special_tokens=True)[0].strip())
    return out_texts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if not args.json:
        print(generate_report(args.image, cfg, args.adapter))
        return

    from schema.incident import now_timestamp

    device = pick_device()
    model, processor = load_for_inference(cfg, args.adapter, device=device)
    model.to(device).eval()
    image = Image.open(args.image).convert("RGB")
    report = generate_structured(
        model, processor, image, device,
        prompt=incident_prompt(cfg), timestamp=now_timestamp(), location=args.location,
        max_new_tokens=cfg.get("eval.max_new_tokens", 96),
        num_beams=cfg.get("eval.num_beams", 4),
        repetition_penalty=cfg.get("eval.repetition_penalty", 1.5),
        no_repeat_ngram_size=cfg.get("eval.no_repeat_ngram_size", 3),
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
