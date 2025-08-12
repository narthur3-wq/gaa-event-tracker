import io
import math
import re
import sqlite3
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

DB_PATH_DEFAULT = "events.db"

# ---------- Default outcome phrases (configurable in the sidebar) ----------
DEFAULT_PHRASES = {
    "carry": {"success": ["completed", "received", "won"], "fail": ["lost", "turnover", "foul"], "other": ["reset", "backwards"]},
    "kick":  {"success": ["completed", "received", "won"], "fail": ["lost", "turnover", "foul"], "other": ["reset", "backwards"]},
    "shot":  {"success": ["point", "goal"],               "fail": ["wide", "short", "post", "save"], "other": ["blocked", "45"]},
}

# ---------- Utility helpers ----------
def parse_mmss(s: str) -> tuple[int, int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", (s or "").strip())
    if not m: return (0, 0)
    mm, ss = int(m.group(1)), int(m.group(2))
    ss = max(0, min(59, ss))
    mm = max(0, min(45, mm))
    return (mm, ss)

def add_seconds(mm: int, ss: int, delta: float) -> tuple[int, int]:
    total = int(round(mm * 60 + ss + delta))
    if total < 0: total = 0
    return (total // 60, total % 60)

def clamp01(x: float) -> float:
    return max(0.0, min(100.0, float(x)))

def split_names(text: str) -> List[str]:
    # Accept commas, semicolons, slashes, newlines, and back slashes; clean+dedupe
    if not text: return []
    raw = re.split(r"[,;\n/\\]+", text)
    seen, out = set(), []
    for name in (n.strip() for n in raw):
        if not name: continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-","_") else "-" for c in s or "")

# ---------- Phrases (in-session configurable) ----------
def get_phrases():
    # Load current phrases from session, falling back to defaults
    out = {}
    for et in ("carry","kick","shot"):
        out[et] = {}
        for b in ("success","fail","other"):
            out[et][b] = st.session_state.get(f"phr_{et}_{b}", DEFAULT_PHRASES[et][b])
    return out

# PATCH: tolerant (substring) mapping instead of exact equality
def map_outcome_to_class(event_type: str, outcome_text: str) -> str:
    phrases = get_phrases()
    o = (outcome_text or "").strip().lower()
    for bucket in ("success","fail","other"):
        for p in phrases.get(event_type, {}).get(bucket, []):
            if p.lower() in o:
                return bucket
    return "other"

# ---------- DB helpers ----------
def get_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

# PATCH: require external schema.sql (no silent fallback)
def init_db(conn: sqlite3.Connection):
    try:
        schema = open("schema.sql", "r", encoding="utf-8").read()
    except FileNotFoundError:
        st.error("Schema file 'schema.sql' not found. Please include it with the app.")
        st.stop()
    conn.executescript(schema)
    conn.commit()

# --- CRUD helpers (subset) ---
def upsert_team(conn, name: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO team(name) VALUES (?)", (name,))
    cur.execute("SELECT team_id FROM team WHERE name=?", (name,))
    return int(cur.fetchone()[0])

def upsert_player(conn, team_id: int, name: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO player(team_id,name) VALUES (?,?)", (team_id, name))
    cur.execute("SELECT player_id FROM player WHERE team_id=? AND name=?", (team_id, name))
    return int(cur.fetchone()[0])

def insert_match(conn, name: str, date: str, competition: Optional[str], venue: Optional[str]) -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO match(name,date,competition,venue) VALUES (?,?,?,?)", (name, date, competition, venue))
    conn.commit(); return cur.lastrowid

def insert_event(conn, **evt) -> int:
    cols = [
        "match_id","team_id","player_id","event_type","start_x","start_y","end_x","end_y",
        "half","minute","second","end_minute","end_second","carry_seconds","outcome","outcome_class","notes"
    ]
    values = [evt.get(c) for c in cols]
    cur = conn.cursor()
    cur.execute(f"INSERT INTO event({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", values)
    conn.commit()
    return cur.lastrowid

def delete_event(conn, event_id: int):
    conn.execute("DELETE FROM event WHERE event_id=?", (event_id,))
    conn.commit()

# ---------- UI (excerpts showing patches) ----------
st.set_page_config(page_title="GAA Event Tracker", layout="wide")
st.title("GAA Event Tracker — match-by-match")

with st.sidebar:
    st.header("Data")
    db_path = st.text_input("Database file", DB_PATH_DEFAULT)
    seed_file = st.file_uploader("Import CSV (e.g., seed.csv)", type=["csv"], accept_multiple_files=False)

    if st.button("Initialise / Load schema"):
        with get_conn(db_path) as c:
            init_db(c)
        st.success("Database ready.")

    if seed_file is not None and st.button("Import CSV"):
        df = pd.read_csv(seed_file)
        with get_conn(db_path) as c:
            for _, r in df.iterrows():
                name = str(r.get("match_name") or r.get("name") or "Unnamed")
                date = str(r.get("date") or "")
                comp = str(r.get("competition") or None)
                venue = str(r.get("venue") or None)
                mid = insert_match(c, name, date, comp, venue)
                team = str(r.get("team") or "Team")
                tid = upsert_team(c, team)
                player = str(r.get("player") or "Player")
                pid = upsert_player(c, tid, player)

                et = str(r.get("event_type") or "").strip().lower()
                if et not in ("carry","kick","shot"): continue
                half = int(r.get("half") or 1)
                mm = int(r.get("minute") or 0)
                ss = int(r.get("second") or 0)
                sx = float(r.get("start_x") or 0.0)
                sy = float(r.get("start_y") or 0.0)
                ex = float(r.get("end_x") or sx)
                ey = float(r.get("end_y") or sy)
                # clamp to [0,100]
                sx = max(0.0, min(100.0, sx))
                sy = max(0.0, min(100.0, sy))
                ex = max(0.0, min(100.0, ex))
                ey = max(0.0, min(100.0, ey))
                outcome = str(r.get("outcome") or "")
                oc = map_outcome_to_class(et, outcome)
                cs = float(r.get("carry_seconds")) if pd.notnull(r.get("carry_seconds")) else None
                e_mm, e_ss = (None, None)
                if et == "carry" and cs is not None:
                    e_mm, e_ss = add_seconds(mm, ss, cs)
                insert_event(c,
                    match_id=mid, team_id=tid, player_id=pid, event_type=et,
                    start_x=sx, start_y=sy, end_x=ex, end_y=ey,
                    half=half, minute=mm, second=ss,
                    end_minute=e_mm, end_second=e_ss, carry_seconds=cs,
                    outcome=outcome, outcome_class=oc, notes=None)
        st.success("CSV imported.")
        st.rerun()

# Event entry (excerpt)
st.subheader("Add Event")
evt_type = st.selectbox("Category", ["carry","kick","shot"], index=0)
half = st.selectbox("Half", [1,2], index=0)
mtime = st.text_input("Match time (mm:ss)", "00:00")
mm, ss = parse_mmss(mtime)

phrases = get_phrases()
opts_success = phrases[evt_type]["success"]
opts_fail    = phrases[evt_type]["fail"]
opts_other   = phrases[evt_type]["other"]
choice_items = [(p, "success") for p in opts_success] + [(p, "fail") for p in opts_fail] + [(p, "other") for p in opts_other]
labels = [p for p,_ in choice_items] + ["(custom)"]
picked = st.selectbox("Outcome", labels, index=0, key="outcome_pick")

custom_outcome = ""
if picked == "(custom)":
    custom_outcome = st.text_input("Custom outcome")
    outcome_text = custom_outcome
    oc = st.selectbox("Classify custom outcome as", ["success","fail","other"], index=2)
else:
    outcome_text = picked
    oc = dict(choice_items).get(picked, "other")

st.caption(f"Outcome class → {oc}")

# Assume sx,sy,ex,ey provided via inputs bounded to 0..100
sx = st.number_input("start_x", 0.0, 100.0, 10.0)
sy = st.number_input("start_y", 0.0, 100.0, 50.0)
ex = st.number_input("end_x",   0.0, 100.0, 30.0)
ey = st.number_input("end_y",   0.0, 100.0, 55.0)

end_mm = end_ss = None
carry_seconds = None

if st.button("Save event"):
    with get_conn(DB_PATH_DEFAULT) as c:
        insert_event(c,
            match_id=1, team_id=1, player_id=1, event_type=evt_type,
            start_x=clamp01(sx), start_y=clamp01(sy), end_x=clamp01(ex), end_y=clamp01(ey),
            half=int(half), minute=int(mm), second=int(ss),
            end_minute=end_mm, end_second=end_ss, carry_seconds=carry_seconds,
            outcome=outcome_text, outcome_class=oc, notes=None)
    st.success("Event saved.")

def fig_to_png_bytes(fig: go.Figure) -> bytes:
    try:
        return fig.to_image(format="png", scale=2, engine="kaleido")
    except Exception as e:
        st.error(f"PNG export failed (is Kaleido installed on the server?): {e}")
        return b""
