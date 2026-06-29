"""Prompt templates for incident-report generation.

The model is prompt-conditioned: training masks the prompt out of the loss (see
data.ReportCollator) and inference passes the same prompt to `generate`, so the
two stay consistent. The default lives in `configs/default.yaml` (`model.prompt`);
this module is the canonical fallback.
"""

from __future__ import annotations

INCIDENT_PROMPT = (
    "Describe what is happening in this frame, including any anomalies or notable "
    "events, in a concise factual report."
)


def incident_prompt(cfg=None) -> str:
    """Return the configured incident prompt, falling back to the default."""
    if cfg is not None:
        return cfg.get("model.prompt") or INCIDENT_PROMPT
    return INCIDENT_PROMPT
