import io
import re
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

DB_PATH_DEFAULT = "events.db"

# ------------------- Outcome phrases (user can tweak in-session) -------------------
DEFAULT_PHRASES = {
    "carry": {"success": ["completed", "received", "won"], "fail": ["lost", "turnover", "foul"], "other": ["reset", "backwards"]},
    "kick":  {"success": ["completed", "received", "won"], "fail": ["lost", "turnover", "foul"], "other": ["reset", "backwards"]},
    "shot":  {"success": ["point", "goal"],               "fail": ["wide", "short", "post", "save"], "other": ["blocked", "45"]},
}

# ------------------- Helpers -------------------
def parse_mmss(s: str) -> tuple[int, int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", (s or "").strip())
    if not m: return (0, 0)
    mm, ss = int(m.group(1)), int(m.group(2))
    ss = max(0, min(59, ss)); mm = max(0, min(45, mm))
    return (mm, ss)

def add_seconds(mm: int, ss: int, delta: float) -> tuple[int, int]:
    total = int(round(mm * 60 + ss + delta))
    if total < 0: total = 0
    return (total // 60, total % 60)

def clamp01(x: float) -> float:
    return max(0.0, min(100.0, float(x)))

def get_phrases():
    out = {}
    for et in ("carry","kick","shot"):
        out[et] = {}
        for b in ("success","fail","other"):
            out[et][b] = st.session_state.get(f"phr_{et}_{b}", DEFAULT_PHRASES[et][b])
    return out

def map_outcome_to_class(event_type: str, outcome_text: str) -> str:
    phrases = get_phrases(); o = (outcome_text or "").strip().lower()
    for bucket in ("success","fail","other"):
        for p in phrases.get(event_type, {}).get(bucket, []):
            if p.lower() in o:
                return bucket
    return "other"

# ------------------- DB helpers -------------------
def get_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(conn: sqlite3.Connection):
    try:
        schema = open("schema.sql", "r", encoding="utf-8").read()
    except FileNotFoundError:
        st.error("Schema file 'schema.sql' not found. Add it to your repo root.")
        st.stop()
    conn.executescript(schema); conn.commit()

# CRUD
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
    conn.commit(); return cur.lastrowid

# ------------------- UI -------------------
st.set_page_config(page_title="GAA Event Tracker", layout="wide")
st.title("GAA Event Tracker — match-by-match")

with st.sidebar:
    st.header("Data")
    db_path = st.text_input("Database file", DB_PATH_DEFAULT)
    if st.button("Initialise / Load schema"):
        with get_conn(db_path) as c: init_db(c)
        st.success("Database ready.")

    st.divider(); st.subheader("Setup")
    with st.form("setup"):
        m_name = st.text_input("Match name", value="Dublin vs Kerry")
        m_date = st.text_input("Date (YYYY-MM-DD)", value="2024-08-18")
        m_comp = st.text_input("Competition", value="All-Ireland")
        m_venue = st.text_input("Venue", value="Croke Park")
        t_name = st.text_input("Team", value="Dublin")
        p_name = st.text_input("Player", value="C. Kilkenny")
        submitted = st.form_submit_button("Save/Use setup")
        if submitted:
            with get_conn(db_path) as c:
                mid = insert_match(c, m_name, m_date, m_comp, m_venue)
                tid = upsert_team(c, t_name)
                pid = upsert_player(c, tid, p_name)
                st.session_state["current_ids"] = {"match":mid, "team":tid, "player":pid}
            st.success("Setup saved (IDs stored in session).")

# Guard for IDs
ids = st.session_state.get("current_ids")
if not ids:
    st.info("Use the Setup form in the sidebar to create/select match/team/player first.")
    st.stop()

# ------------------- Pitch -------------------
st.subheader("Pitch (click to add path)")

fig = go.Figure()
fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="#A3A3A3"))
for v in range(10, 100, 10):
    fig.add_shape(type="line", x0=v, y0=0, x1=v, y1=100, line=dict(color="#EFEFEF", width=1))
    fig.add_shape(type="line", x0=0, y0=v, x1=100, y1=v, line=dict(color="#EFEFEF", width=1))
fig.update_xaxes(range=[0,100], showgrid=False)
fig.update_yaxes(range=[0,100], showgrid=False, scaleanchor="x", scaleratio=1)
fig.update_layout(height=480, margin=dict(l=10,r=10,t=10,b=10))

pts = st.session_state.get("pts", [])
clicks = plotly_events(fig, click_event=True, select_event=False, hover_event=False)
if clicks:
    x = clamp01(clicks[0]["x"]) ; y = clamp01(clicks[0]["y"])
    if len(pts) >= 2: pts = []
    pts.append((x, y))
    st.session_state["pts"] = pts

col1, col2 = st.columns(2)
with col1: st.write("Start:", pts[0] if len(pts) >= 1 else (None, None))
with col2: st.write("End:",   pts[1] if len(pts) >= 2 else (None, None))

# ------------------- Event entry -------------------
st.subheader("Add Event")
colA, colB, colC = st.columns(3)
with colA:
    evt_type = st.selectbox("Type", ["carry","kick","shot"])
with colB:
    half = st.selectbox("Half", [1,2], index=0)
with colC:
    mtime = st.text_input("Time mm:ss", value="00:00")
mm, ss = parse_mmss(mtime)

phr = get_phrases()
choices = [(p, "success") for p in phr[evt_type]["success"]] + \
          [(p, "fail")    for p in phr[evt_type]["fail"]]    + \
          [(p, "other")   for p in phr[evt_type]["other"]]
labels = [p for p,_ in choices] + ["(custom)"]
picked = st.selectbox("Outcome", labels)
custom_outcome = ""
if picked == "(custom)":
    custom_outcome = st.text_input("Custom outcome")
    oc = st.selectbox("Classify custom outcome as", ["success","fail","other"], index=2)
    outcome_text = custom_outcome
else:
    oc = dict(choices).get(picked, "other")
    outcome_text = picked
st.caption(f"Outcome class → {oc}")

if len(pts) >= 1:
    sx, sy = pts[0]
else:
    sx = st.number_input("start_x", 0.0, 100.0, 10.0)
    sy = st.number_input("start_y", 0.0, 100.0, 50.0)
if len(pts) >= 2:
    ex, ey = pts[1]
else:
    ex = st.number_input("end_x",   0.0, 100.0, 30.0)
    ey = st.number_input("end_y",   0.0, 100.0, 55.0)

carry_seconds = None
end_mm = end_ss = None
if evt_type == "carry":
    carry_seconds = st.number_input("Carry seconds (optional)", value=0.0, step=0.5)
    if carry_seconds:
        end_mm, end_ss = add_seconds(mm, ss, carry_seconds)

if st.button("Save event"):
    with get_conn(DB_PATH_DEFAULT) as c:
        insert_event(
            c,
            match_id=ids["match"], team_id=ids["team"], player_id=ids["player"], event_type=evt_type,
            start_x=clamp01(sx), start_y=clamp01(sy), end_x=clamp01(ex), end_y=clamp01(ey),
            half=int(half), minute=int(mm), second=int(ss),
            end_minute=end_mm, end_second=end_ss, carry_seconds=carry_seconds,
            outcome=outcome_text, outcome_class=oc, notes=None,
        )
    st.success("Event saved.")
    st.session_state["pts"] = []

# ------------------- Recent events table -------------------
with get_conn(DB_PATH_DEFAULT) as c:
    df = pd.read_sql_query(
        "SELECT event_id, half, minute, second, event_type, outcome, outcome_class, start_x, start_y, end_x, end_y "
        "FROM event WHERE match_id=? ORDER BY event_id DESC LIMIT 150",
        c, params=(ids["match"],)
    )

st.dataframe(df, use_container_width=True)

# ------------------- Export -------------------
colx, coly = st.columns(2)
with colx:
    if st.button("Export events CSV"):
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="events.csv",
            mime="text/csv",
        )
with coly:
    if st.button("Export pitch PNG"):
        buf = io.BytesIO()
        go.Figure(fig).write_image(buf, format="png", engine="kaleido", scale=2)
        st.download_button("Download PNG", data=buf.getvalue(), file_name="pitch.png", mime="image/png")
import io
import re
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

DB_PATH_DEFAULT = "events.db"

# ------------------- Outcome phrases (user can tweak in-session) -------------------
DEFAULT_PHRASES = {
    "carry": {"success": ["completed", "received", "won"], "fail": ["lost", "turnover", "foul"], "other": ["reset", "backwards"]},
    "kick":  {"success": ["completed", "received", "won"], "fail": ["lost", "turnover", "foul"], "other": ["reset", "backwards"]},
    "shot":  {"success": ["point", "goal"],               "fail": ["wide", "short", "post", "save"], "other": ["blocked", "45"]},
}

# ------------------- Helpers -------------------
def parse_mmss(s: str) -> tuple[int, int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", (s or "").strip())
    if not m: return (0, 0)
    mm, ss = int(m.group(1)), int(m.group(2))
    ss = max(0, min(59, ss)); mm = max(0, min(45, mm))
    return (mm, ss)

def add_seconds(mm: int, ss: int, delta: float) -> tuple[int, int]:
    total = int(round(mm * 60 + ss + delta))
    if total < 0: total = 0
    return (total // 60, total % 60)

def clamp01(x: float) -> float:
    return max(0.0, min(100.0, float(x)))

def get_phrases():
    out = {}
    for et in ("carry","kick","shot"):
        out[et] = {}
        for b in ("success","fail","other"):
            out[et][b] = st.session_state.get(f"phr_{et}_{b}", DEFAULT_PHRASES[et][b])
    return out

def map_outcome_to_class(event_type: str, outcome_text: str) -> str:
    phrases = get_phrases(); o = (outcome_text or "").strip().lower()
    for bucket in ("success","fail","other"):
        for p in phrases.get(event_type, {}).get(bucket, []):
            if p.lower() in o:
                return bucket
    return "other"

# ------------------- DB helpers -------------------
def get_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(conn: sqlite3.Connection):
    try:
        schema = open("schema.sql", "r", encoding="utf-8").read()
    except FileNotFoundError:
        st.error("Schema file 'schema.sql' not found. Add it to your repo root.")
        st.stop()
    conn.executescript(schema); conn.commit()

# CRUD
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
    conn.commit(); return cur.lastrowid

# ------------------- UI -------------------
st.set_page_config(page_title="GAA Event Tracker", layout="wide")
st.title("GAA Event Tracker — match-by-match")

with st.sidebar:
    st.header("Data")
    db_path = st.text_input("Database file", DB_PATH_DEFAULT)
    if st.button("Initialise / Load schema"):
        with get_conn(db_path) as c: init_db(c)
        st.success("Database ready.")

    st.divider(); st.subheader("Setup")
    with st.form("setup"):
        m_name = st.text_input("Match name", value="Dublin vs Kerry")
        m_date = st.text_input("Date (YYYY-MM-DD)", value="2024-08-18")
        m_comp = st.text_input("Competition", value="All-Ireland")
        m_venue = st.text_input("Venue", value="Croke Park")
        t_name = st.text_input("Team", value="Dublin")
        p_name = st.text_input("Player", value="C. Kilkenny")
        submitted = st.form_submit_button("Save/Use setup")
        if submitted:
            with get_conn(db_path) as c:
                mid = insert_match(c, m_name, m_date, m_comp, m_venue)
                tid = upsert_team(c, t_name)
                pid = upsert_player(c, tid, p_name)
                st.session_state["current_ids"] = {"match":mid, "team":tid, "player":pid}
            st.success("Setup saved (IDs stored in session).")

# Guard for IDs
ids = st.session_state.get("current_ids")
if not ids:
    st.info("Use the Setup form in the sidebar to create/select match/team/player first.")
    st.stop()

# ------------------- Pitch -------------------
st.subheader("Pitch (click to add path)")

fig = go.Figure()
fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="#A3A3A3"))
for v in range(10, 100, 10):
    fig.add_shape(type="line", x0=v, y0=0, x1=v, y1=100, line=dict(color="#EFEFEF", width=1))
    fig.add_shape(type="line", x0=0, y0=v, x1=100, y1=v, line=dict(color="#EFEFEF", width=1))
fig.update_xaxes(range=[0,100], showgrid=False)
fig.update_yaxes(range=[0,100], showgrid=False, scaleanchor="x", scaleratio=1)
fig.update_layout(height=480, margin=dict(l=10,r=10,t=10,b=10))

pts = st.session_state.get("pts", [])
clicks = plotly_events(fig, click_event=True, select_event=False, hover_event=False)
if clicks:
    x = clamp01(clicks[0]["x"]) ; y = clamp01(clicks[0]["y"])
    if len(pts) >= 2: pts = []
    pts.append((x, y))
    st.session_state["pts"] = pts

col1, col2 = st.columns(2)
with col1: st.write("Start:", pts[0] if len(pts) >= 1 else (None, None))
with col2: st.write("End:",   pts[1] if len(pts) >= 2 else (None, None))

# ------------------- Event entry -------------------
st.subheader("Add Event")
colA, colB, colC = st.columns(3)
with colA:
    evt_type = st.selectbox("Type", ["carry","kick","shot"])
with colB:
    half = st.selectbox("Half", [1,2], index=0)
with colC:
    mtime = st.text_input("Time mm:ss", value="00:00")
mm, ss = parse_mmss(mtime)

phr = get_phrases()
choices = [(p, "success") for p in phr[evt_type]["success"]] + \
          [(p, "fail")    for p in phr[evt_type]["fail"]]    + \
          [(p, "other")   for p in phr[evt_type]["other"]]
labels = [p for p,_ in choices] + ["(custom)"]
picked = st.selectbox("Outcome", labels)
custom_outcome = ""
if picked == "(custom)":
    custom_outcome = st.text_input("Custom outcome")
    oc = st.selectbox("Classify custom outcome as", ["success","fail","other"], index=2)
    outcome_text = custom_outcome
else:
    oc = dict(choices).get(picked, "other")
    outcome_text = picked
st.caption(f"Outcome class → {oc}")

if len(pts) >= 1:
    sx, sy = pts[0]
else:
    sx = st.number_input("start_x", 0.0, 100.0, 10.0)
    sy = st.number_input("start_y", 0.0, 100.0, 50.0)
if len(pts) >= 2:
    ex, ey = pts[1]
else:
    ex = st.number_input("end_x",   0.0, 100.0, 30.0)
    ey = st.number_input("end_y",   0.0, 100.0, 55.0)

carry_seconds = None
end_mm = end_ss = None
if evt_type == "carry":
    carry_seconds = st.number_input("Carry seconds (optional)", value=0.0, step=0.5)
    if carry_seconds:
        end_mm, end_ss = add_seconds(mm, ss, carry_seconds)

if st.button("Save event"):
    with get_conn(DB_PATH_DEFAULT) as c:
        insert_event(
            c,
            match_id=ids["match"], team_id=ids["team"], player_id=ids["player"], event_type=evt_type,
            start_x=clamp01(sx), start_y=clamp01(sy), end_x=clamp01(ex), end_y=clamp01(ey),
            half=int(half), minute=int(mm), second=int(ss),
            end_minute=end_mm, end_second=end_ss, carry_seconds=carry_seconds,
            outcome=outcome_text, outcome_class=oc, notes=None,
        )
    st.success("Event saved.")
    st.session_state["pts"] = []

# ------------------- Recent events table -------------------
with get_conn(DB_PATH_DEFAULT) as c:
    df = pd.read_sql_query(
        "SELECT event_id, half, minute, second, event_type, outcome, outcome_class, start_x, start_y, end_x, end_y "
        "FROM event WHERE match_id=? ORDER BY event_id DESC LIMIT 150",
        c, params=(ids["match"],)
    )

st.dataframe(df, use_container_width=True)

# ------------------- Export -------------------
colx, coly = st.columns(2)
with colx:
    if st.button("Export events CSV"):
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="events.csv",
            mime="text/csv",
        )
with coly:
    if st.button("Export pitch PNG"):
        buf = io.BytesIO()
        go.Figure(fig).write_image(buf, format="png", engine="kaleido", scale=2)
        st.download_button("Download PNG", data=buf.getvalue(), file_name="pitch.png", mime="image/png")
