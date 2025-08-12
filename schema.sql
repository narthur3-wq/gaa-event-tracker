PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS team (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS player (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS match (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT,
    competition TEXT,
    venue TEXT
);

CREATE TABLE IF NOT EXISTS event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    start_x REAL,
    start_y REAL,
    end_x REAL,
    end_y REAL,
    half INTEGER,
    minute INTEGER,
    second INTEGER,
    end_minute INTEGER,
    end_second INTEGER,
    carry_seconds REAL,
    outcome TEXT,
    outcome_class TEXT,
    notes TEXT,
    FOREIGN KEY (match_id) REFERENCES match(match_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES team(team_id),
    FOREIGN KEY (player_id) REFERENCES player(player_id)
);
