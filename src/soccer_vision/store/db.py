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
        team: str | None = None,
        track_id: int | None = None,
        player_id: str | None = None,
        source: str | None = None,
        event_id: str | None = None,
    ) -> str:
        event_id = event_id or str(uuid.uuid4())[:8]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO events "
                    "(id, match_id, label, position_ms, frame, confidence, "
                    "team, track_id, player_id, source) "
                    "VALUES (:id, :mid, :label, :pos, :frame, :conf, "
                    ":team, :track, :pid, :source)"
                ),
                {
                    "id": event_id,
                    "mid": match_id,
                    "label": label,
                    "pos": position_ms,
                    "frame": frame,
                    "conf": confidence,
                    "team": team,
                    "track": track_id,
                    "pid": player_id,
                    "source": source,
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
        track_id: int | None = None,
        team: str | None = None,
        pre_s: float = 5.0,
        post_s: float = 30.0,
        clip_id: str | None = None,
    ) -> str:
        clip_id = clip_id or str(uuid.uuid4())[:8]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO clips "
                    "(id, match_id, event_id, player_id, track_id, team, path, pre_s, post_s) "
                    "VALUES (:id, :mid, :eid, :pid, :track, :team, :path, :pre, :post)"
                ),
                {
                    "id": clip_id,
                    "mid": match_id,
                    "eid": event_id,
                    "pid": player_id,
                    "track": track_id,
                    "team": team,
                    "path": path,
                    "pre": pre_s,
                    "post": post_s,
                },
            )
        return clip_id

    def get_events(
        self,
        match_id: str,
        *,
        label: str | None = None,
        team: str | None = None,
        track_id: int | None = None,
    ) -> list[dict]:
        clauses = ["match_id = :mid"]
        params: dict = {"mid": match_id}
        if label is not None:
            clauses.append("label = :label")
            params["label"] = label
        if team is not None:
            clauses.append("team = :team")
            params["team"] = team
        if track_id is not None:
            clauses.append("track_id = :track")
            params["track"] = track_id
        where = " AND ".join(clauses)
        with self.engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT * FROM events WHERE {where} ORDER BY position_ms"),
                params,
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
