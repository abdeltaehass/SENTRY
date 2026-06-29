"""Visual grounding — Grad-CAM over the ViT patch features.

For a given (image, sentence), we backprop the sentence's language-model loss to
the BLIP-2 vision encoder's patch features and build a Grad-CAM heatmap. The map
shows which image regions made that sentence more likely — i.e. whether the text
is actually grounded in the image (a sanity-check against hallucination).

BLIP-2 path: ViT patches -> Q-Former -> language_projection -> OPT. Gradient of
the OPT loss flows all the way back to the ViT patches even though the backbone
is frozen (we enable grad on pixel_values so the graph is built).

Security-camera frames are wide, so pass `letterbox_input=True` to pad them to
square (matching how the model is fed) before computing/overlaying the heatmap —
otherwise a 16:9 frame is squished and the heat lands in the wrong place.
"""

from __future__ import annotations

import re

import numpy as np
import torch
from PIL import Image

from data.preprocess import letterbox


def _unwrap(model):
    """Return the underlying conditional-generation model through any PEFT wrapper."""
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


def split_sentences(text: str) -> list[str]:
    """Naive sentence split for per-sentence grounding."""
    parts = re.split(r"(?<=[.])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def grounding_cam(model, processor, image: Image.Image, text: str, device: str,
                  letterbox_input: bool = False) -> np.ndarray:
    """Grad-CAM over ViT patches for `text`. Returns an (S, S) map in [0, 1]."""
    if letterbox_input:
        image = letterbox(image)
    vision = _unwrap(model).vision_model
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out.last_hidden_state
        hs.retain_grad()
        captured["hs"] = hs

    handle = vision.register_forward_hook(hook)
    try:
        enc = processor(images=image, text=text, return_tensors="pt").to(device)
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            enc["pixel_values"].requires_grad_(True)  # build graph back to the image
            loss = model(**enc, labels=labels).loss
            loss.backward()

        acts = captured["hs"].detach()[0, 1:, :]   # drop CLS -> [num_patches, hidden]
        grads = captured["hs"].grad[0, 1:, :]       # [num_patches, hidden]
    finally:
        handle.remove()

    weights = grads.mean(dim=0)                              # alpha_c, [hidden]
    cam = torch.relu((acts * weights).sum(dim=-1)).float()   # [num_patches]
    cam = cam / (cam.max() + 1e-8)

    side = int(round(cam.shape[0] ** 0.5))
    return cam[: side * side].reshape(side, side).cpu().numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, size: int = 224,
                    alpha: float = 0.5, letterbox_input: bool = False) -> Image.Image:
    """Blend a CAM (small grid) over the image as a jet heatmap."""
    import matplotlib

    if letterbox_input:
        image = letterbox(image)
    base = image.convert("RGB").resize((size, size))
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize((size, size), Image.BILINEAR)
    cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam_arr)[..., :3]    # RGBA -> RGB
    blended = (1 - alpha) * (np.asarray(base) / 255.0) + alpha * heat
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype("uint8"))


def ground_report(model, processor, image: Image.Image, report: str, device: str,
                  letterbox_input: bool = False):
    """Per-sentence grounding. Returns list of (sentence, overlay_image)."""
    out = []
    for sent in split_sentences(report):
        cam = grounding_cam(model, processor, image, sent, device, letterbox_input=letterbox_input)
        out.append((sent, overlay_heatmap(image, cam, letterbox_input=letterbox_input)))
    return out
