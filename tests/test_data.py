"""Hermetic tests for the surveillance data pipeline."""

import torch
from PIL import Image

from data.dataset import ReportCollator
from data.preprocess import assert_no_leakage, clean_text, letterbox, split_by_group


# --- aspect handling --------------------------------------------------------

def test_letterbox_pads_to_square():
    wide = Image.new("RGB", (160, 90), (255, 0, 0))   # 16:9 frame
    assert letterbox(wide).size == (160, 160)         # padded to a square
    assert letterbox(wide, size=224).size == (224, 224)


def test_clean_text():
    assert clean_text("  a   b\n c ") == "a b c"
    assert clean_text(None) == ""


# --- leakage-safe split by camera/location ----------------------------------

def _recs(cam, n):
    return [{"id": f"{cam}_{i}", "camera_id": cam, "report": "x"} for i in range(n)]


def test_split_by_group_no_leakage():
    records = [r for c in range(50) for r in _recs(f"cam_{c}", 3)]
    splits = split_by_group(records, group_key="camera_id", seed=42)
    assert_no_leakage(splits, group_key="camera_id")   # raises on any shared camera
    cams = {k: {r["camera_id"] for r in v} for k, v in splits.items()}
    assert cams["train"].isdisjoint(cams["test"])
    assert cams["train"].isdisjoint(cams["val"])
    assert sum(len(v) for v in splits.values()) == len(records)


# --- prompt-masking collator (fake processor/tokenizer, no model) -----------

class _FakeTok:
    pad_token_id = 0
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        ids = [self.bos_token_id] if add_special_tokens else []
        ids += [10 + i for i, _ in enumerate(text.split())]   # 1 id per word
        return {"input_ids": ids}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTok()

    def __call__(self, images, text, **kwargs):
        rows = [self.tokenizer(t)["input_ids"] for t in text]
        maxlen = max(len(r) for r in rows)
        padded = [r + [self.tokenizer.pad_token_id] * (maxlen - len(r)) for r in rows]
        return {"input_ids": torch.tensor(padded),
                "pixel_values": torch.zeros(len(text), 3, 4, 4)}


def test_collator_masks_prompt_tokens():
    proc = _FakeProcessor()
    prompt = "describe the scene"          # 3 prompt tokens
    col = ReportCollator(proc, prompt=prompt)
    out = col([{"image": None, "text": "intruder at gate", "raw": {"id": "s1"}}])
    labels = out["labels"]
    # sequence = [bos] + 3 prompt + 3 report = 7; mask first 4 (bos + prompt)
    assert (labels[0, :4] == -100).all()
    assert (labels[0, 4:] != -100).all()   # report tokens are kept in the loss


def test_collator_no_processor_returns_raw():
    out = ReportCollator(None)([{"image": "img", "text": "t", "raw": {"id": "s1"}}])
    assert out == {"images": ["img"], "texts": ["t"], "ids": ["s1"]}
