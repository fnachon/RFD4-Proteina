"""Accelerator selection (CUDA > MPS > CPU) shared by inference, training and metrics."""

from __future__ import annotations

import logging
import os

import torch

__all__ = ["accelerator_name", "autocast_device_type", "get_device", "has_sdpa_backend", "resolve_precision"]

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available() and mps.is_built())


def accelerator_name() -> str:
    """Return ``"cuda"``, ``"mps"`` or ``"cpu"``. Override with ``RFP_DEVICE``."""
    forced = os.environ.get("RFP_DEVICE", "").strip().lower()
    if forced:
        if forced not in ("cuda", "mps", "cpu"):
            raise ValueError(f"RFP_DEVICE must be cuda, mps or cpu, got {forced!r}")
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def get_device(index: int | None = None) -> torch.device:
    """Default torch device for the current machine."""
    name = accelerator_name()
    if name == "cuda":
        return torch.device("cuda", torch.cuda.current_device() if index is None else index)
    return torch.device(name)


def has_sdpa_backend(device: torch.device | str | None = None) -> bool:
    """True where ``F.scaled_dot_product_attention`` is a usable (non-einsum) path."""
    return torch.device(device).type in ("cuda", "mps") if device is not None else accelerator_name() != "cpu"


def autocast_device_type(device: torch.device | str | None = None) -> str:
    return torch.device(device).type if device is not None else accelerator_name()


def resolve_precision(precision: str, accelerator: str | None = None) -> str:
    """Map a Lightning precision string to one the accelerator handles reliably.

    bf16 autocast is not dependable on MPS across macOS/PyTorch versions, so it is
    downgraded to fp32 unless ``RFP_MPS_ALLOW_BF16=1``.
    """
    accelerator = accelerator or accelerator_name()
    if accelerator == "mps" and "bf16" in str(precision) and os.environ.get("RFP_MPS_ALLOW_BF16", "") != "1":
        logger.warning(f"precision={precision!r} -> '32-true' on MPS (set RFP_MPS_ALLOW_BF16=1 to keep it)")
        return "32-true"
    if accelerator == "cpu" and "16" in str(precision):
        return "32-true"
    return precision
