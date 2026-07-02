from __future__ import annotations

import os

import torch


def get_device(prefer: str = "auto") -> torch.device:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")
