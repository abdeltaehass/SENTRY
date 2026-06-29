"""Hermetic tests for grounding helpers (no model required)."""

import numpy as np
from PIL import Image

from grounding import _unwrap, overlay_heatmap, split_sentences


class _Inner:
    pass


class _Lora:
    def __init__(self, inner):
        self.model = inner


class _Peft:
    def __init__(self, inner):
        self.base_model = _Lora(inner)


def test_unwrap_peft_and_plain():
    inner = _Inner()
    assert _unwrap(_Peft(inner)) is inner   # unwraps PEFT -> base model
    plain = _Inner()
    assert _unwrap(plain) is plain          # plain model returned as-is


def test_split_sentences():
    assert split_sentences("An intruder enters. No weapon seen. Two people present.") == [
        "An intruder enters.",
        "No weapon seen.",
        "Two people present.",
    ]
    assert split_sentences("   ") == []


def test_overlay_heatmap_shape():
    img = Image.new("RGB", (60, 40), color=(20, 20, 20))
    cam = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = overlay_heatmap(img, cam, size=224, alpha=0.5)
    assert out.size == (224, 224)
    assert out.mode == "RGB"
