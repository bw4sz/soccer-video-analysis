"""sn-gamestate / TrackLab adapter (optional [broadcast] extra)."""

from __future__ import annotations


def is_available() -> bool:
    """Check if TrackLab is installed."""
    try:
        import tracklab  # noqa: F401
        return True
    except ImportError:
        return False

# Stub — Phase 5 implementation
