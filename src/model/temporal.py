"""Multi-frame / temporal reasoning over a short clip.

Single-frame captioning answers "what is in this frame"; surveillance needs
"what is *happening* over time" — a person *approached* the entrance, *stopped*,
and *left* a bag. That requires looking across several consecutive frames, not
one.

This module adds that without any model surgery to the frozen BLIP-2 backbone.
BLIP-2 maps one image to a fixed set of language-space query tokens
(``ViT -> Q-Former -> language_projection``). We run that encoder **per frame**,
then **aggregate the per-frame token sets across the sequence** before handing a
single visual context to the OPT language model:

    frame_1 ─┐
    frame_2 ─┤  ViT+Q-Former+proj  ─►  [T, Q, H] per-frame tokens
    ...      ┤  (per frame, frozen)        │
    frame_T ─┘                             ▼
                                  aggregate across T
                                  (concat | mean | max | attn)
                                           │
                                           ▼
                          [1, M, H] visual context  +  prompt  ─►  OPT  ─► report

Aggregation strategies:
  - ``concat`` (default): keep every frame's tokens in temporal order
    (``[1, T*Q, H]``) and let the LLM's self-attention reason across them — the
    same mechanism modern video-language models use. Zero new parameters, so it
    works with the base model or any LoRA adapter out of the box.
  - ``mean`` / ``max``: pool the per-frame tokens into one ``[1, Q, H]`` set —
    cheaper, order-agnostic.
  - ``attn``: a small learnable :class:`TemporalAttentionPooler` weights frames by
    salience (trainable; falls back to mean when no pooler is supplied).

The text path mirrors ``Blip2ForConditionalGeneration.generate``: we build the
visual tokens ourselves, concatenate the prompt's text embeddings, and call the
language model directly — so generation, confidence scoring, and the existing
reliability flow all carry over to clips unchanged.
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image

from data.preprocess import even_sample, letterbox

# Event-oriented prompt: ask for the sequence of events, not a single snapshot.
TEMPORAL_PROMPT = (
    "These are consecutive frames from a surveillance camera, in order. Describe "
    "the sequence of events over time — how the scene changes from the first frame "
    "to the last, and any anomalies or notable activity — in a concise factual report."
)

AGGREGATIONS = ("concat", "mean", "max", "attn")


def temporal_prompt(cfg=None) -> str:
    """Return the configured temporal prompt, falling back to the default."""
    if cfg is not None:
        return cfg.get("temporal.prompt") or TEMPORAL_PROMPT
    return TEMPORAL_PROMPT


def _unwrap(model):
    """Return the underlying conditional-generation model through any PEFT wrapper."""
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


# --- frame loading / sampling ----------------------------------------------


def load_frames(paths, letterbox_input: bool = True, image_size: int | None = None,
                num_frames: int | None = None) -> list[Image.Image]:
    """Open frame paths (optionally subsample + letterbox) into ordered PIL images."""
    if num_frames is not None:
        paths = even_sample(list(paths), num_frames)
    images = [Image.open(p).convert("RGB") for p in paths]
    if letterbox_input:
        images = [letterbox(im, size=image_size) for im in images]
    return images


def sample_clip(frames, num_frames: int = 4) -> list:
    """Evenly pick ``num_frames`` items (paths or images) from a longer sequence."""
    return even_sample(list(frames), num_frames)


# --- learnable temporal attention pooling ----------------------------------


class TemporalAttentionPooler(torch.nn.Module):
    """Attention-pool per-frame token sets into one set, weighting frames by salience.

    Each frame is summarized (mean over its query tokens), scored by a learnable
    linear head, and the per-frame token sets are combined with the resulting
    softmax weights. Reduces ``[B, T, Q, H] -> [B, Q, H]`` and also returns the
    per-frame attention weights (handy for "which frame mattered" introspection).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.score = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, frames: torch.Tensor):
        # frames: [B, T, Q, H]
        summary = self.dropout(frames.mean(dim=2))          # [B, T, H]
        logits = self.score(summary).squeeze(-1)            # [B, T]
        weights = logits.softmax(dim=1)                     # [B, T]
        pooled = torch.einsum("bt,btqh->bqh", weights, frames)  # [B, Q, H]
        return pooled, weights


# --- per-frame encoding + cross-frame aggregation ---------------------------


@torch.no_grad()
def encode_frame_tokens(core, processor, image: Image.Image, device: str) -> torch.Tensor:
    """Encode one frame into its language-space query tokens ``[1, Q, H]``.

    Mirrors the front half of BLIP-2: ``ViT -> Q-Former -> language_projection``.
    ``core`` is the unwrapped ``Blip2ForConditionalGeneration``.
    """
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    dtype = next(core.vision_model.parameters()).dtype
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    image_embeds = core.vision_model(pixel_values, return_dict=True).last_hidden_state
    image_attention = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=device)
    query_tokens = core.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = core.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention,
        return_dict=True,
    ).last_hidden_state
    return core.language_projection(query_output)           # [1, Q, H]


def aggregate_frame_tokens(frame_tokens, strategy: str = "concat",
                           pooler: TemporalAttentionPooler | None = None) -> torch.Tensor:
    """Aggregate a list of per-frame token sets ``[1, Q, H]`` into ``[1, M, H]``.

    ``concat`` keeps every frame (``M = T*Q``, temporal order preserved); ``mean`` /
    ``max`` / ``attn`` reduce to a single set (``M = Q``).
    """
    if not frame_tokens:
        raise ValueError("aggregate_frame_tokens: no frames given")
    stack = torch.stack([f.squeeze(0) for f in frame_tokens], dim=0)  # [T, Q, H]
    t, q, h = stack.shape

    if strategy == "concat":
        return stack.reshape(1, t * q, h)                   # temporal order preserved
    if strategy == "mean":
        return stack.mean(dim=0, keepdim=True)
    if strategy == "max":
        return stack.amax(dim=0, keepdim=True)
    if strategy == "attn":
        if pooler is None:
            return stack.mean(dim=0, keepdim=True)          # parameter-free fallback
        pooled, _ = pooler(stack.unsqueeze(0))              # [1, Q, H]
        return pooled
    raise ValueError(f"Unknown aggregation strategy {strategy!r}; expected one of {AGGREGATIONS}")


def build_temporal_inputs_embeds(core, processor, images, device, prompt: str | None,
                                 strategy: str = "concat",
                                 pooler: TemporalAttentionPooler | None = None):
    """Assemble ``inputs_embeds`` / ``attention_mask`` for the LLM from a clip.

    Visual tokens (aggregated across frames) come first, then the prompt's text
    embeddings — matching BLIP-2's own ordering. Returns
    ``(inputs_embeds, attention_mask, prompt_input_ids, n_visual)``.
    """
    frame_tokens = [encode_frame_tokens(core, processor, im, device) for im in images]
    visual = aggregate_frame_tokens(frame_tokens, strategy, pooler)      # [1, M, H]

    text = processor.tokenizer(prompt or "", return_tensors="pt").to(device)
    input_ids = text["input_ids"]
    text_embeds = core.get_input_embeddings()(input_ids)
    visual = visual.to(text_embeds.dtype)

    inputs_embeds = torch.cat([visual, text_embeds], dim=1)
    visual_mask = torch.ones(visual.shape[:-1], dtype=torch.long, device=device)
    attention_mask = torch.cat([visual_mask, text["attention_mask"]], dim=1)
    return inputs_embeds, attention_mask, input_ids, visual.shape[1]


# --- generation -------------------------------------------------------------


@torch.no_grad()
def generate_temporal_with_confidence(model, processor, images, device,
                                      max_new_tokens: int = 96, prompt: str | None = None,
                                      strategy: str = "concat",
                                      pooler: TemporalAttentionPooler | None = None,
                                      **decode):
    """Generate a clip-level report + confidence (geometric-mean token probability).

    Same confidence definition as the single-frame path, so the reliability /
    flagging flow is identical. Returns ``(text, confidence in [0, 1])``.
    """
    core = _unwrap(model)
    inputs_embeds, attention_mask, _, _ = build_temporal_inputs_embeds(
        core, processor, images, device, prompt, strategy, pooler
    )
    lm = core.language_model
    out = lm.generate(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, output_scores=True,
        return_dict_in_generate=True, **decode,
    )
    beam_indices = getattr(out, "beam_indices", None)
    trans = lm.compute_transition_scores(
        out.sequences, out.scores, beam_indices, normalize_logits=True
    )
    finite = torch.isfinite(trans)
    confidence = float(trans[finite].mean().exp()) if finite.any() else 0.0
    # inputs_embeds-only generation returns just the continuation (no prompt echo).
    text = processor.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
    return text, confidence


@torch.no_grad()
def generate_temporal_report(model, processor, images, device,
                             max_new_tokens: int = 96, prompt: str | None = None,
                             strategy: str = "concat",
                             pooler: TemporalAttentionPooler | None = None,
                             **decode) -> str:
    """Generate a clip-level incident report (text only)."""
    text, _ = generate_temporal_with_confidence(
        model, processor, images, device, max_new_tokens=max_new_tokens,
        prompt=prompt, strategy=strategy, pooler=pooler, **decode,
    )
    return text


# --- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a temporal incident report from a short clip of frames"
    )
    p.add_argument("--frames", nargs="+", help="ordered frame image paths (3-5 typical)")
    p.add_argument("--video", help="a video file to sample frames from (needs OpenCV)")
    p.add_argument("--num-frames", type=int, default=4, help="frames to sample from the clip")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--adapter", default=None, help="optional trained adapter dir")
    p.add_argument("--aggregate", default=None, choices=AGGREGATIONS,
                   help="cross-frame aggregation (default: temporal.aggregate or 'concat')")
    return p.parse_args()


def _resolve_frame_paths(args) -> list[str]:
    if args.frames:
        return args.frames
    if args.video:
        from data.preprocess import extract_frames

        out_dir = "data/processed/_temporal_clip"
        paths = extract_frames(args.video, out_dir, every_n_frames=15)
        return even_sample(paths, args.num_frames)
    raise SystemExit("Pass --frames <paths...> or --video <path>.")


def main() -> None:
    args = parse_args()
    from config import load_config

    from .model import load_base, load_for_inference, pick_device

    cfg = load_config(args.config)
    device = pick_device()
    if args.adapter:
        model, processor = load_for_inference(cfg, args.adapter, device=device)
    else:
        model, processor = load_base(cfg, device=device)
    model.to(device).eval()

    strategy = args.aggregate or cfg.get("temporal.aggregate", "concat")
    paths = even_sample(_resolve_frame_paths(args), args.num_frames)
    images = load_frames(paths, letterbox_input=cfg.get("data.letterbox", True))
    report = generate_temporal_report(
        model, processor, images, device,
        max_new_tokens=cfg.get("eval.max_new_tokens", 96),
        prompt=temporal_prompt(cfg), strategy=strategy,
        num_beams=cfg.get("eval.num_beams", 4),
        repetition_penalty=cfg.get("eval.repetition_penalty", 1.5),
        no_repeat_ngram_size=cfg.get("eval.no_repeat_ngram_size", 3),
    )
    print(f"[{len(images)} frames | aggregate={strategy}]\n{report}")


if __name__ == "__main__":
    main()
