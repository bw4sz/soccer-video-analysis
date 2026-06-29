CREATE TABLE IF NOT EXISTS matches (
  id TEXT PRIMARY KEY,
  raw_path TEXT,
  proxy_path TEXT,
  processed_at TIMESTAMP,
  osl_path TEXT,
  stats_path TEXT
);

CREATE TABLE IF NOT EXISTS players (
  id TEXT PRIMARY KEY,
  name TEXT,
  jersey INTEGER,
  team TEXT,
  profile_id TEXT
);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  match_id TEXT,
  label TEXT,
  position_ms INTEGER,
  frame INTEGER,
  confidence REAL,
  verified BOOLEAN DEFAULT FALSE,
  FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS clips (
  id TEXT PRIMARY KEY,
  match_id TEXT,
  event_id TEXT,
  player_id TEXT,
  path TEXT,
  pre_s REAL,
  post_s REAL,
  FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS idp_links (
  player_id TEXT,
  focus_skill TEXT,
  clip_id TEXT,
  notes TEXT
);
