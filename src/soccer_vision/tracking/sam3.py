"""SAM3 player segmentation/tracking adapter (optional GPU)."""

from __future__ import annotations


def is_available() -> bool:
    """Check if SAM3 dependencies are installed."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Stub — Phase 5 implementation
