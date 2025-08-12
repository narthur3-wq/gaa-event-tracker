CREATE TABLE IF NOT EXISTS team (
  team_id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS player (
  player_id INTEGER PRIMARY KEY,
  team_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  shirt_number INTEGER,
  UNIQUE(team_id, name),
  FOREIGN KEY(team_id) REFERENCES team(team_id)
);

CREATE TABLE IF NOT EXISTS match (
  match_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  date TEXT,
  competition TEXT,
  venue TEXT
);

CREATE TABLE IF NOT EXISTS event (
  event_id INTEGER PRIMARY KEY,
  match_id INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  player_id INTEGER,
  event_type TEXT NOT NULL CHECK (event_type IN ('carry','kick','shot')),
  start_x REAL NOT NULL CHECK (start_x BETWEEN 0 AND 100),
  start_y REAL NOT NULL CHECK (start_y BETWEEN 0 AND 100),
  end_x REAL NOT NULL CHECK (end_x BETWEEN 0 AND 100),
  end_y REAL NOT NULL CHECK (end_y BETWEEN 0 AND 100),
  half INTEGER NOT NULL CHECK (half IN (1,2)),
  minute INTEGER CHECK (minute BETWEEN 0 AND 60),
  second INTEGER CHECK (second BETWEEN 0 AND 59),
  end_minute INTEGER,
  end_second INTEGER,
  carry_seconds REAL,
  outcome TEXT,
  outcome_class TEXT CHECK (outcome_class IN ('success','fail','other')),
  notes TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(match_id) REFERENCES match(match_id) ON DELETE CASCADE,
  FOREIGN KEY(team_id)  REFERENCES team(team_id),
  FOREIGN KEY(player_id) REFERENCES player(player_id)
);

CREATE INDEX IF NOT EXISTS idx_event_match     ON event(match_id);
CREATE INDEX IF NOT EXISTS idx_event_player    ON event(player_id);
CREATE INDEX IF NOT EXISTS idx_event_type      ON event(event_type);
CREATE INDEX IF NOT EXISTS idx_event_out_class ON event(outcome_class);
CREATE INDEX IF NOT EXISTS idx_event_time ON event(match_id, half, minute, second);
