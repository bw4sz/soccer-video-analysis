"""SQLite database for matches, players, events, and clips."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, text

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class MatchDB:
    """Thin wrapper around SQLite for soccer-vision data."""

    def __init__(self, db_path: str | Path = "soccer_vision.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        self._init_schema()

    def _init_schema(self):
        schema_sql = SCHEMA_PATH.read_text()
        with self.engine.begin() as conn:
            for statement in schema_sql.split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))

    def add_match(
        self,
        match_id: str | None = None,
        *,
        raw_path: str | None = None,
        proxy_path: str | None = None,
        osl_path: str | None = None,
        stats_path: str | None = None,
    ) -> str:
        match_id = match_id or str(uuid.uuid4())[:8]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO matches "
                    "(id, raw_path, proxy_path, processed_at, osl_path, stats_path) "
                    "VALUES (:id, :raw, :proxy, :ts, :osl, :stats)"
                ),
                {
                    "id": match_id,
                    "raw": raw_path,
                    "proxy": proxy_path,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "osl": osl_path,
                    "stats": stats_path,
                },
            )
        return match_id

    def add_event(
        self,
        match_id: str,
        label: str,
        position_ms: int,
        *,
        frame: int | None = None,
        confidence: float | None = None,
        event_id: str | None = None,
    ) -> str:
        event_id = event_id or str(uuid.uuid4())[:8]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO events (id, match_id, label, position_ms, frame, confidence) "
                    "VALUES (:id, :mid, :label, :pos, :frame, :conf)"
                ),
                {
                    "id": event_id,
                    "mid": match_id,
                    "label": label,
                    "pos": position_ms,
                    "frame": frame,
                    "conf": confidence,
                },
            )
        return event_id

    def add_clip(
        self,
        match_id: str,
        path: str,
        *,
        event_id: str | None = None,
        player_id: str | None = None,
        pre_s: float = 5.0,
        post_s: float = 30.0,
        clip_id: str | None = None,
    ) -> str:
        clip_id = clip_id or str(uuid.uuid4())[:8]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO clips (id, match_id, event_id, player_id, path, pre_s, post_s) "
                    "VALUES (:id, :mid, :eid, :pid, :path, :pre, :post)"
                ),
                {
                    "id": clip_id,
                    "mid": match_id,
                    "eid": event_id,
                    "pid": player_id,
                    "path": path,
                    "pre": pre_s,
                    "post": post_s,
                },
            )
        return clip_id

    def get_events(self, match_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM events WHERE match_id = :mid ORDER BY position_ms"),
                {"mid": match_id},
            )
            return [dict(row._mapping) for row in result]

    def get_clips(self, match_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM clips WHERE match_id = :mid"),
                {"mid": match_id},
            )
            return [dict(row._mapping) for row in result]

    def get_match(self, match_id: str) -> dict | None:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM matches WHERE id = :id"),
                {"id": match_id},
            )
            row = result.fetchone()
            return dict(row._mapping) if row else None
