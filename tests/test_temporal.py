"""Hermetic tests for multi-frame / temporal reasoning.

No model download and no transformers: a small FakeCore/FakeProcessor exercise the
real per-frame encoding + cross-frame aggregation + inputs_embeds assembly that
``model.temporal`` performs, and tiny on-disk frames exercise the clip dataset.
"""

import json

import pytest
import torch
from PIL import Image

from data.dataset import ClipReportCollator, ClipReportDataset
from data.preprocess import even_sample
from model.temporal import (
    AGGREGATIONS,
    TemporalAttentionPooler,
    aggregate_frame_tokens,
    build_temporal_inputs_embeds,
    encode_frame_tokens,
    sample_clip,
)

Q, QHID, HID, VOCAB = 2, 8, 4, 20


# --- even sampling ----------------------------------------------------------

def test_even_sample_keeps_endpoints_and_order():
    assert even_sample(list(range(10)), 3) == [0, 4, 9]   # endpoints included
    assert even_sample([1, 2, 3], 5) == [1, 2, 3]         # fewer than k -> all
    assert even_sample("abcde", 1) == ["a"]               # k == 1 -> first
    assert even_sample([], 3) == [] and even_sample([1], 0) == []


def test_sample_clip_is_even_sample():
    assert sample_clip(list("abcde"), 3) == ["a", "c", "e"]


# --- cross-frame aggregation ------------------------------------------------

def _frames(values, q=Q, h=HID):
    """One [1, q, h] token set per value, filled with that constant."""
    return [torch.full((1, q, h), float(v)) for v in values]


def test_concat_preserves_temporal_order():
    cat = aggregate_frame_tokens(_frames([0, 1, 2]), "concat")
    assert cat.shape == (1, 3 * Q, HID)
    assert (cat[0, 0:Q] == 0).all()        # frame 0 tokens first
    assert (cat[0, Q:2 * Q] == 1).all()    # then frame 1
    assert (cat[0, 2 * Q:3 * Q] == 2).all()


def test_mean_and_max_pool_to_single_set():
    assert aggregate_frame_tokens(_frames([0, 2]), "mean").shape == (1, Q, HID)
    assert (aggregate_frame_tokens(_frames([0, 2]), "mean") == 1).all()
    mx = aggregate_frame_tokens(_frames([0, 4, 1]), "max")
    assert mx.shape == (1, Q, HID) and (mx == 4).all()


def test_attn_falls_back_to_mean_without_pooler():
    out = aggregate_frame_tokens(_frames([0, 2]), "attn", pooler=None)
    assert out.shape == (1, Q, HID) and (out == 1).all()


def test_attn_uses_pooler_when_given():
    pooler = TemporalAttentionPooler(HID)
    out = aggregate_frame_tokens(_frames([0, 1, 2]), "attn", pooler=pooler)
    assert out.shape == (1, Q, HID)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        aggregate_frame_tokens(_frames([0]), "bogus")
    assert set(AGGREGATIONS) == {"concat", "mean", "max", "attn"}


# --- temporal attention pooler ----------------------------------------------

def test_pooler_shapes_and_normalized_weights():
    pooler = TemporalAttentionPooler(HID)
    pooled, weights = pooler(torch.randn(2, 3, Q, HID))   # [B, T, Q, H]
    assert pooled.shape == (2, Q, HID)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)


# --- fake BLIP-2 core for the assembly path ---------------------------------

class _Ret:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _Vision(torch.nn.Module):
    def __init__(self, hidden=QHID):
        super().__init__()
        self.proj = torch.nn.Linear(3 * 4 * 4, hidden)   # gives params (-> dtype)
        self.hidden = hidden

    def forward(self, pixel_values, return_dict=True):
        b = pixel_values.shape[0]
        return _Ret(torch.zeros(b, 5, self.hidden))      # 5 fake patches


class _QFormer(torch.nn.Module):
    def forward(self, query_embeds, encoder_hidden_states=None,
                encoder_attention_mask=None, return_dict=True):
        return _Ret(query_embeds)                        # [b, Q, QHID]


class _Batch(dict):
    def to(self, _device):
        return self


class _PixelValues:
    def __init__(self, pixel_values):
        self.pixel_values = pixel_values


class FakeProcessor:
    class _Tok:
        pad_token_id = 0

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [10 + i for i, _ in enumerate(text.split())] or [10]
            if return_tensors == "pt":
                return _Batch(input_ids=torch.tensor([ids]),
                              attention_mask=torch.ones(1, len(ids), dtype=torch.long))
            return {"input_ids": ids}

    def __init__(self):
        self.tokenizer = self._Tok()

    def __call__(self, images=None, return_tensors="pt", **_):
        return _PixelValues(torch.zeros(1, 3, 4, 4))


class FakeCore(torch.nn.Module):
    def __init__(self, q=Q, qhidden=QHID, hidden=HID, vocab=VOCAB):
        super().__init__()
        self.vision_model = _Vision(qhidden)
        self.qformer = _QFormer()
        self.query_tokens = torch.nn.Parameter(torch.randn(1, q, qhidden))
        self.language_projection = torch.nn.Linear(qhidden, hidden)
        self.embed = torch.nn.Embedding(vocab, hidden)

    def get_input_embeddings(self):
        return self.embed


def test_encode_frame_tokens_shape():
    core, proc = FakeCore(), FakeProcessor()
    out = encode_frame_tokens(core, proc, Image.new("RGB", (8, 8)), "cpu")
    assert out.shape == (1, Q, HID)


def test_build_temporal_inputs_embeds_concat_and_mean():
    core, proc = FakeCore(), FakeProcessor()
    images = [Image.new("RGB", (8, 8)) for _ in range(3)]
    prompt = "describe the events over time"            # 5 tokens
    n_text = 5

    emb, mask, ids, n_vis = build_temporal_inputs_embeds(
        core, proc, images, "cpu", prompt, strategy="concat")
    assert n_vis == 3 * Q                                # every frame's tokens kept
    assert emb.shape == (1, n_vis + n_text, HID)
    assert mask.shape == (1, n_vis + n_text)
    assert int(mask.sum()) == n_vis + n_text             # all-ones (no padding)
    assert ids.shape == (1, n_text)
    # text half of inputs_embeds is exactly the token embeddings of the prompt
    assert torch.allclose(emb[:, n_vis:], core.get_input_embeddings()(ids))

    emb_m, _, _, n_vis_m = build_temporal_inputs_embeds(
        core, proc, images, "cpu", prompt, strategy="mean")
    assert n_vis_m == Q                                  # pooled to a single set
    assert emb_m.shape == (1, Q + n_text, HID)


# --- clip dataset / collator ------------------------------------------------

@pytest.fixture
def clip_jsonl(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (160, 90), (i * 10, 0, 0)).save(p)   # wide 16:9 frames
        paths.append(str(p))
    rec = {"id": "clip_1", "image_path": paths[0], "image_paths": paths,
           "report": "A person approaches, stops, then leaves a bag unattended."}
    jp = tmp_path / "clips.jsonl"
    jp.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return jp


def test_clip_dataset_samples_and_letterboxes(clip_jsonl):
    ds = ClipReportDataset(clip_jsonl, num_frames=3)
    item = ds[0]
    assert len(item["frames"]) == 3                       # 5 frames -> 3 sampled
    assert all(im.size == (160, 160) for im in item["frames"])  # letterboxed square
    assert item["text"].startswith("A person approaches")


def test_clip_dataset_falls_back_to_single_image(tmp_path):
    p = tmp_path / "only.jpg"
    Image.new("RGB", (80, 80)).save(p)
    rec = {"id": "c", "image_path": str(p), "report": "x"}   # no image_paths
    jp = tmp_path / "one.jsonl"
    jp.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert len(ClipReportDataset(jp, num_frames=4)[0]["frames"]) == 1


def test_clip_collator_groups_clips(clip_jsonl):
    ds = ClipReportDataset(clip_jsonl, num_frames=3)
    out = ClipReportCollator(None)([ds[0]])
    assert out["ids"] == ["clip_1"]
    assert len(out["clips"]) == 1 and len(out["clips"][0]) == 3
    assert out["texts"][0].startswith("A person approaches")
