"""Model construction.

PRIMARY TRACK (`blip2_lora`): BLIP-2 = ViT image encoder -> Q-Former (the
"projection" that maps image features into the LLM's space) -> OPT (the LLM).
We freeze the ViT, the Q-Former, and the OPT base, and train only **LoRA adapters
on the OPT decoder + the `language_projection` layer**. PEFT saves exactly those.

ALTERNATIVE TRACK (`clip_llm`): assemble CLIP -> projection -> small LLM yourself
(LLaVA-style). Not yet implemented.

Weights are only downloaded/loaded when a builder is actually called.
"""

from __future__ import annotations


def pick_device() -> str:
    """Return the best available device string ('cuda' | 'mps' | 'cpu')."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str):
    """Pick a safe dtype. fp16/bf16 are flaky on MPS, so use fp32 locally."""
    import torch

    if device in ("mps", "cpu"):
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _auto_model_class():
    """Image-text-to-text auto class (renamed from Vision2Seq in transformers 5)."""
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    from transformers import Blip2ForConditionalGeneration

    return Blip2ForConditionalGeneration


def load_base(cfg, device: str | None = None):
    """Load the pretrained (model, processor) with no adapters attached."""
    from transformers import AutoProcessor

    device = device or pick_device()
    dtype = resolve_dtype(cfg.model.dtype, device)
    processor = AutoProcessor.from_pretrained(cfg.model.base_model)
    model = _auto_model_class().from_pretrained(cfg.model.base_model, torch_dtype=dtype)
    return model, processor


def build_model(cfg, device: str | None = None):
    """Build a trainable (model, processor): frozen backbone + LoRA + projection."""
    approach = cfg.model.approach

    if approach in ("blip2_lora", "smolvlm_lora"):
        from peft import LoraConfig, get_peft_model

        model, processor = load_base(cfg, device)
        lora = cfg.model.lora
        modules_to_save = cfg.get("model.lora.modules_to_save")  # e.g. ["language_projection"]

        peft_config = LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules),
            modules_to_save=list(modules_to_save) if modules_to_save else None,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, processor

    if approach == "clip_llm":
        raise NotImplementedError("clip_llm track is not yet implemented.")
    raise ValueError(f"Unknown model.approach: {approach!r}")


def load_for_inference(cfg, adapter_path: str, device: str | None = None):
    """Load the base model with a trained adapter attached for generation."""
    from peft import PeftModel
    from transformers import AutoProcessor

    model, processor = load_base(cfg, device)
    model = PeftModel.from_pretrained(model, adapter_path)
    try:
        processor = AutoProcessor.from_pretrained(adapter_path)
    except Exception:
        pass  # adapter dir may not carry the processor; keep the base one
    return model, processor


def load_for_demo(cfg, adapter_path: str, device: str = "cpu", quantize: bool = True):
    """Deployment loader: merge the adapter and (on CPU) int8-quantize for speed.

    Merging folds LoRA into the base weights (a plain model, no PEFT overhead);
    dynamic int8 quantization of the Linear layers shrinks RAM ~4x and speeds up
    CPU inference. NOTE: a quantized model cannot backprop, so Grad-CAM grounding
    is unavailable when quantize=True (generation + confidence still work).
    """
    import torch

    model, processor = load_for_inference(cfg, adapter_path, device=device)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.eval()
    if quantize and device == "cpu":
        engines = getattr(torch.backends.quantized, "supported_engines", [])
        if "qnnpack" in engines:          # ARM (Apple); Linux Spaces use fbgemm
            torch.backends.quantized.engine = "qnnpack"
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except (RuntimeError, NotImplementedError) as exc:
            print(f"[sentry] int8 quantization unavailable here ({exc}); using fp32 CPU.")
    return model.to(device), processor
