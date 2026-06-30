"""Dataset + dataloader for image -> incident-report generation.

Consumes a processed JSONL (one record per line):
    {"id": "scene_0001",
     "image_path":  ".../frame_0001.jpg",
     "image_paths": [...],          # optional: multiple frames / camera views
     "report": "...",
     "camera_id": "...", "location_id": "..."}

`build_dataloader` returns a torch DataLoader. With a HF processor it yields
model-ready tensors (pixel_values / input_ids / labels); without one it yields
raw PIL images + text (handy for inspection and tests).

Security footage is wide, so `letterbox=True` pads frames to square (preserving
aspect ratio) before the model's square resize, instead of distorting them.
The surveillance ingestion/preprocessing lives in `data.preprocess`.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocess import even_sample, letterbox


class IncidentReportDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, text_field: str = "report",
                 letterbox_input: bool = True, image_size: int | None = None):
        """
        Args:
            jsonl_path: a processed split (train/val/test).
            text_field: record field to use as the generation target.
            letterbox_input: pad wide frames to square (preserve aspect ratio).
            image_size: optional square resize applied during letterboxing.
        """
        self.jsonl_path = Path(jsonl_path)
        self.text_field = text_field
        self.letterbox_input = letterbox_input
        self.image_size = image_size
        self.records = self._load(self.jsonl_path)

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Prepare your incident dataset first "
                "(see data/README.md and data.preprocess)."
            )
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def _image(self, record: dict) -> Image.Image:
        image = Image.open(record["image_path"]).convert("RGB")
        if self.letterbox_input:
            image = letterbox(image, size=self.image_size)
        return image

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        text = (record.get(self.text_field) or "").strip()
        return {"image": self._image(record), "text": text, "raw": record}


class ReportCollator:
    """Batch collator for incident-report fine-tuning.

    The image is the condition; the report text is the generation target.
    If `prompt` is set, the model is trained prompt-conditioned: the sequence is
    `prompt + report`, and the prompt tokens are masked out of the loss so only
    the report is learned. With no processor it returns raw PIL + text.
    """

    def __init__(self, processor=None, max_target_tokens: int = 128, prompt: str = ""):
        self.processor = processor
        self.max_target_tokens = max_target_tokens
        self.prompt = (prompt or "").strip()

    def _encode(self, images, texts):
        return self.processor(
            images=images, text=texts, return_tensors="pt",
            padding=True, truncation=True, max_length=self.max_target_tokens,
        )

    def __call__(self, batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]
        ids = [b["raw"].get("id") for b in batch]

        if self.processor is None:
            return {"images": images, "texts": texts, "ids": ids}

        tok = self.processor.tokenizer
        if self.prompt:
            enc = self._encode(images, [f"{self.prompt} {t}".strip() for t in texts])
            labels = enc["input_ids"].clone()
            labels[labels == tok.pad_token_id] = -100
            # mask the prompt portion so the loss is only on the report text
            n_prompt = len(tok(self.prompt, add_special_tokens=False)["input_ids"])
            bos = getattr(tok, "bos_token_id", None)
            offset = 1 if (bos is not None and enc["input_ids"][0, 0].item() == bos) else 0
            labels[:, : offset + n_prompt] = -100
        else:
            enc = self._encode(images, texts)
            labels = enc["input_ids"].clone()
            labels[labels == tok.pad_token_id] = -100
        enc["labels"] = labels
        return enc


def build_dataloader(
    jsonl_path: str | Path,
    text_field: str = "report",
    processor=None,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    max_target_tokens: int = 128,
    prompt: str = "",
    letterbox_input: bool = True,
) -> DataLoader:
    dataset = IncidentReportDataset(jsonl_path, text_field=text_field,
                                    letterbox_input=letterbox_input)
    collator = ReportCollator(processor=processor,
                              max_target_tokens=max_target_tokens, prompt=prompt)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


# --- multi-frame / temporal clips -------------------------------------------


class ClipReportDataset(Dataset):
    """Multi-frame variant: each record yields an ordered list of frames + report.

    Reads the same JSONL contract, using a record's ``image_paths`` (a list of
    frames, in temporal order) when present and falling back to the single
    ``image_path`` otherwise. Up to ``num_frames`` frames are evenly sampled so a
    long clip becomes a short, representative sequence the temporal model consumes
    (see ``model.temporal``).
    """

    def __init__(self, jsonl_path: str | Path, text_field: str = "report",
                 num_frames: int = 4, frames_field: str = "image_paths",
                 letterbox_input: bool = True, image_size: int | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.text_field = text_field
        self.num_frames = num_frames
        self.frames_field = frames_field
        self.letterbox_input = letterbox_input
        self.image_size = image_size
        self.records = IncidentReportDataset._load(self.jsonl_path)

    def _frame_paths(self, record: dict) -> list[str]:
        paths = record.get(self.frames_field) or [record["image_path"]]
        return even_sample(list(paths), self.num_frames)

    def _frames(self, record: dict) -> list[Image.Image]:
        images = [Image.open(p).convert("RGB") for p in self._frame_paths(record)]
        if self.letterbox_input:
            images = [letterbox(im, size=self.image_size) for im in images]
        return images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        text = (record.get(self.text_field) or "").strip()
        return {"frames": self._frames(record), "text": text, "raw": record}


class ClipReportCollator:
    """Collate clip records into per-clip frame lists + targets.

    Temporal forwards build ``inputs_embeds`` per clip (a clip's visual-token
    count depends on frame count and aggregation), so unlike the single-frame
    collator this keeps clips as Python lists rather than one stacked tensor; the
    temporal trainer encodes each clip in turn. With a processor it also returns
    tokenized ``target_ids`` for the report text.
    """

    def __init__(self, processor=None, prompt: str = ""):
        self.processor = processor
        self.prompt = (prompt or "").strip()

    def __call__(self, batch: list[dict]) -> dict:
        clips = [b["frames"] for b in batch]
        texts = [b["text"] for b in batch]
        ids = [b["raw"].get("id") for b in batch]
        out = {"clips": clips, "texts": texts, "ids": ids}
        if self.processor is not None:
            tok = self.processor.tokenizer
            out["target_ids"] = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        return out


def build_clip_dataloader(
    jsonl_path: str | Path,
    text_field: str = "report",
    processor=None,
    num_frames: int = 4,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    prompt: str = "",
    letterbox_input: bool = True,
) -> DataLoader:
    dataset = ClipReportDataset(jsonl_path, text_field=text_field,
                                num_frames=num_frames, letterbox_input=letterbox_input)
    collator = ClipReportCollator(processor=processor, prompt=prompt)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
