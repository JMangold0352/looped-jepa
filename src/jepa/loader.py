"""Public API: load released I-JEPA checkpoints by registry name."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.weights import (
    ensure_checkpoint,
    get_released_weight,
    list_released_weights,
    load_checkpoint_state,
    load_state_into_model,
)

__all__ = ["load_ijepa", "list_released_weights", "get_released_weight"]


def load_ijepa(
    name: str,
    *,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    checkpoint: str | Path | None = None,
    map_location: Any | None = None,
) -> IJEPA:
    """Load a released I-JEPA model by registry key.

    Args:
        name: ``baseline_v3``, ``looped_v3``, or ``sandwich_rmsnorm``.
        pretrained: If True and the local checkpoint is missing, attempt download
            when URLs are configured in ``released_weights/urls.yaml``.
        device: Target device for the returned model.
        checkpoint: Optional explicit ``.pt`` path (skips registry default).
        map_location: Passed to ``torch.load`` (defaults to ``device``).

    Returns:
        Eval-mode ``IJEPA`` with weights loaded. Uses ``strict=False`` on
        ``load_state_dict``; only optimizer/probe/scaler key prefixes may be
        missing. Unexpected keys are logged and ignored.

    Raises:
        KeyError: Unknown registry name.
        FileNotFoundError: No local checkpoint and ``pretrained=False``.
        WeightDownloadError: Download URLs are placeholders or fetch failed.
    """
    spec = get_released_weight(name)
    ckpt_path = ensure_checkpoint(name, pretrained=pretrained, checkpoint=checkpoint)
    dev = get_device(device)
    loc = map_location if map_location is not None else dev

    cfg = load_config(spec.config_path)
    model = IJEPA.from_config(cfg)
    state = load_checkpoint_state(ckpt_path, map_location=loc)
    load_state_into_model(model, state)
    model.to(dev)
    model.eval()
    return model
