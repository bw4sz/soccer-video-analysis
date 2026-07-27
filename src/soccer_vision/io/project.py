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
    def ball_track(self) -> Path:
        """Sampled ball positions, in the `events.deadball` ball-track schema.

        `process` computes these anyway; persisting them makes the ball
        trajectory inspectable and lets `trim-empty --track` reuse the run
        instead of re-detecting.
        """
        return self.root / "ball_track.json"

    @property
    def goals(self) -> Path:
        """Detected goal-mouth regions, written by `soccer-vision goals`.

        Separate from the run's events because the mouths are a property of the
        *camera setup*, not of play: detect once, then re-derive goal events as
        the ball track or thresholds change without touching the GPU again.
        """
        return self.root / "goals.json"

    @property
    def jerseys(self) -> Path:
        """Per-track voted jersey numbers, written by `soccer-vision identify`."""
        return self.root / "jerseys.json"

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
