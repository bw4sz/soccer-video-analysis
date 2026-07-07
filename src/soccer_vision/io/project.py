"""Project folder layout: runs/{match_id}/ structure."""

from __future__ import annotations

from pathlib import Path


class RunDir:
    """Manages the standard run directory layout for a processed match."""

    def __init__(self, base: Path, match_id: str):
        self.root = base / match_id
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def raw_video(self) -> Path:
        return self.root / "raw.mp4"

    @property
    def broadcast_proxy(self) -> Path:
        return self.root / "broadcast_proxy.mp4"

    @property
    def annotations(self) -> Path:
        return self.root / "annotations.json"

    @property
    def stats(self) -> Path:
        return self.root / "stats.json"

    @property
    def tracks(self) -> Path:
        """Per-frame player track boxes, used to draw player halos on clips."""
        return self.root / "tracks.json"

    @property
    def clips_dir(self) -> Path:
        d = self.root / "clips"
        d.mkdir(exist_ok=True)
        return d

    @property
    def sheets_dir(self) -> Path:
        d = self.root / "sheets"
        d.mkdir(exist_ok=True)
        return d

    @property
    def crop_metadata(self) -> Path:
        return self.root / "crop_metadata.json"
