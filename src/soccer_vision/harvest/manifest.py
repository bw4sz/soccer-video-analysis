"""Provenance manifest for harvested clips.

Each harvested clip appends one JSON line to ``manifest.jsonl``. The manifest
serves three jobs:

* **Attribution** — CC BY requires crediting the creator; :meth:`Manifest.write_attribution`
  renders a human-readable ``ATTRIBUTION.md`` from the records.
* **Dedup / resume** — :attr:`Manifest.seen_ids` lets a re-run skip videos we
  already have, so a 200-clip set can be built up over several sessions.
* **Diversity caps** — :attr:`Manifest.channel_counts` enforces a per-channel
  limit so one prolific uploader can't dominate the set.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ClipRecord:
    """One harvested clip's provenance. Fields map 1:1 to a JSONL line."""

    video_id: str
    url: str
    title: str
    channel: str
    channel_id: str
    license: str
    duration_s: float
    clip_start_s: float
    clip_len_s: float
    query: str
    path: str
    harvested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Manifest:
    """Append-only JSONL ledger of harvested clips.

    Loads any existing file on construction so counts and seen-ids reflect prior
    runs. Writes are flushed per-record, so an interrupted run keeps everything
    downloaded so far.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[dict] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                self.records.append(json.loads(line))

    @property
    def seen_ids(self) -> set[str]:
        return {r["video_id"] for r in self.records}

    @property
    def channel_counts(self) -> Counter:
        return Counter(r.get("channel_id") or r.get("channel") for r in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def append(self, record: ClipRecord) -> None:
        """Persist one record (creates the file/parents on first write)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")
        self.records.append(asdict(record))

    def write_attribution(self, out_path: str | Path | None = None) -> Path:
        """Render an ``ATTRIBUTION.md`` crediting every source (CC BY duty)."""
        out_path = Path(out_path) if out_path else self.path.with_name("ATTRIBUTION.md")
        lines = [
            "# Attribution",
            "",
            "Clips harvested from YouTube videos released under a Creative Commons",
            "Attribution (CC BY) licence. Credit to the original creators below.",
            "",
        ]
        for r in self.records:
            lines.append(
                f"- **{r['title']}** — {r['channel']} "
                f"([{r['video_id']}]({r['url']})), {r['license']}"
            )
        lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
