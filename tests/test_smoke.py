"""Phase 0 smoke tests — verify the scaffold imports and config loads."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    from config import load_config  # noqa: F401
    from data.dataset import IncidentReportDataset  # noqa: F401
    from eval.metrics import compute_text_metrics, event_overlap  # noqa: F401
    from grounding import grounding_cam  # noqa: F401
    from model.model import build_model, pick_device  # noqa: F401


def test_config_loads():
    from config import load_config

    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.project.name == "sentry"
    assert cfg.model.approach == "blip2_lora"
    assert cfg.model.lora.r == 16
    assert cfg.data.target == "report"
    assert cfg.get("wandb.project") == "sentry-incidents"


def test_device_resolves():
    from model.model import pick_device

    assert pick_device() in {"cuda", "mps", "cpu"}
