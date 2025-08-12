# app.py — GAA Event Tracker (full version)

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

# ------------------- Outcome phrases -------------------
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

# ------------------- Pitch helpers (GAA look + click mesh) -------------------
# Pick midpoints of allowed sizes; we only use these to scale features visually
PITCH_LEN_M = 140.0   # 130–145 m
PITCH_WID_M = 85.0    # 80–90  m

def _mx(m: float) -> float:  # metres -> % along length (x axis 0..100)
    return 100.0 * float(m) / PITCH_LEN_M

def _my(m: float) -> float:  # metres -> % along width  (y axis 0..100)
    return 100.0 * float(m) / PITCH_WID_M

def make_gaa_pitch() -> go.Figure:
    fig = go.Figure()

    # Grass (with mowing stripes)
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color="#14532d", width=2), fillcolor="#d1fae5")
    for v in range(10, 100, 10):
        fig.add_shape(type="rect", x0=v-10, y0=0, x1=v, y1=100, line=dict(width=0),
                      fillcolor="#ecfdf5" if (v//10) % 2 else "#d1fae5", layer="below")

    # Halfway
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="#64748b", width=2))

    # 13 / 20 / 45 from both ends
    for m in (13, 20, 45):
        x = _mx(m)
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=100, line=dict(color="#94a3b8"))
        fig.add_shape(type="line", x0=100.0 - x, y0=0, x1=100.0 - x, y1=100, line=dict(color="#94a3b8"))

    # Centre circle (~10 m radius)
    r_centre_m = 10.0
    t = np.linspace(0, 2*np.pi, 241)
    cx, cy = 50.0, 50.0
    rr = _mx(r_centre_m)  # scale by length
    fig.add_trace(go.Scatter(
        x=cx + rr*np.cos(t), y=cy + rr*np.sin(t),
        mode="lines", line=dict(color="#64748b"), hoverinfo="skip", showlegend=False
    ))

    # Goal areas (approx spec)
    small_depth = _mx(4.5)      # ~4.5 m deep
    small_width = _my(14.0)     # ~14 m wide
    large_depth = _mx(13.0)     # ~13 m deep
    large_width = _my(19.0)     # ~19 m wide

    def goal_boxes_left(x0: float):
        y0 = 50.0 - small_width/2.0
        y1 = 50.0 + small_width/2.0
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x0+small_depth, y1=y1,
                      line=dict(color="#0f172a", width=2))
        y0b = 50.0 - large_width/2.0
        y1b = 50.0 + large_width/2.0
        fig.add_shape(type="rect", x0=x0, y0=y0b, x1=x0+large_depth, y1=y1b,
                      line=dict(color="#0f172a"))

    # Left goal
    goal_boxes_left(0.0)
    # Right goal (mirror)
    goal_boxes_left(100.0 - large_depth)
    fig.add_shape(type="rect",
                  x0=100.0 - small_depth, y0=50.0 - small_width/2.0,
                  x1=100.0, y1=50.0 + small_width/2.0,
                  line=dict(color="#0f172a", width=2))

    # “D” arcs (~20 m radius)
    r_d = _mx(20.0)
    th = np.linspace(-np.pi/2, np.pi/2, 121)
    fig.add_trace(go.Scatter(
        x=(0 + r_d*np.cos(th)), y=(50.0 + r_d*np.sin(th)),
        mode="lines", line=dict(color="#94a3b8"), hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=(100.0 - r_d*np.cos(th)), y=(50.0 + r_d*np.sin(th)),
        mode="lines", line=dict(color="#94a3b8"), hoverinfo="skip", showlegend=False
    ))

    # Axes/layout
    fig.update_xaxes(range=[0,100], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0,100], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True)
    fig.update_layout(height=540, margin=dict(l=10,r=10,t=10,b=10), dragmode=False, clickmode="event+select")
    return fig

def add_click_mesh(fig: go.Figure, step: float = 2.0) -> None:
    """Invisible scatter grid so clicks register anywhere."""
    vs = np.arange(0, 100 + 1e-9, step)
    xx, yy = np.meshgrid(vs, vs)
    fig.add_trace(go.Scatter(
        x=xx.ravel(), y=yy.ravel(),
        mode="markers",
        marker=dict(size=18, opacity=0.001),
        hoverinfo="skip", showlegend=False, name="_clickmesh",
    ))

# ------------------- UI -------------------
st.set_page_config(page_title="GAA Event Tracker", layout="wide")
st.title("GAA Event Tracker — match-by-match")

with st.sidebar:
    st.header("Data")
    db_path = st.text_input("Database file", DB_PATH_DEFAULT, key="db_path")
    if st.button("Initialise / Load schema", key="btn_init_schema"):
        with get_conn(db_path) as c: init_db(c)
        st.success("Database ready.", icon="✅")

    st.divider(); st.subheader("Setup")
    with st.form("setup_form", clear_on_submit=False):
        m_name = st.text_input("Match name", value="Dublin vs Kerry", key="m_name")
        m_date = st.text_input("Date (YYYY-MM-DD)", value="2025-08-10", key="m_date")
        m_comp = st.text_input("Competition", value="All-Ireland", key="m_comp")
        m_venue = st.text_input("Venue", value="Croke Park", key="m_venue")
        t_name = st.text_input("Team", value="Dublin", key="t_name")
        p_name = st.text_input("Player", value="Player A", key="p_name")
        submitted = st.form_submit_button("Save/Use setup", use_container_width=True)
        if submitted:
            with get_conn(db_path) as c:
                mid = insert_match(c, m_name, m_date, m_comp, m_venue)
                tid = upsert_team(c, t_name)
                pid = upsert_player(c, tid, p_name)
                st.session_state["current_ids"] = {"match": mid, "team": tid, "player": pid}
            st.success("Setup saved (IDs stored in session).", icon="💾")

# Guard for IDs
ids = st.session_state.get("current_ids")
if not ids:
    st.info("Use the Setup form in the sidebar to create/select match/team/player first.")
    st.stop()

# ------------------- Pitch (click to add path) -------------------
st.subheader("Pitch (click to add path)")

fig = make_gaa_pitch()
add_click_mesh(fig, step=2.0)  # makes the whole canvas clickable

# IMPORTANT: unique key for the event listener
clicks = plotly_events(fig, click_event=True, select_event=False, hover_event=False, key="gaa_pitch")

# Keep two points (start, end) in session
pts = st.session_state.get("pts", [])
if clicks:
    x = clamp01(clicks[0]["x"]); y = clamp01(clicks[0]["y"])
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
    evt_type = st.selectbox("Type", ["carry", "kick", "shot"], key="evt_type")
with colB:
    half = st.selectbox("Half", [1, 2], index=0, key="half")
with colC:
    mtime = st.text_input("Time mm:ss", value="00:00", key="mtime")
mm, ss = parse_mmss(mtime)

phr = get_phrases()
choices = [(p, "success") for p in phr[evt_type]["success"]] + \
          [(p, "fail")    for p in phr[evt_type]["fail"]]    + \
          [(p, "other")   for p in phr[evt_type]["other"]]
labels = [p for p, _ in choices] + ["(custom)"]
picked = st.selectbox("Outcome", labels, key="outcome_pick")
custom_outcome = ""
if picked == "(custom)":
    custom_outcome = st.text_input("Custom outcome", key="custom_outcome")
    oc = st.selectbox("Classify custom outcome as", ["success", "fail", "other"], index=2, key="custom_class")
    outcome_text = custom_outcome
else:
    oc = dict(choices).get(picked, "other")
    outcome_text = picked
st.caption(f"Outcome class → {oc}")

# Coordinates from clicks (else fall back to numeric)
if len(pts) >= 1:
    sx, sy = pts[0]
else:
    sx = st.number_input("start_x", 0.0, 100.0, 10.0, key="start_x")
    sy = st.number_input("start_y", 0.0, 100.0, 50.0, key="start_y")
if len(pts) >= 2:
    ex, ey = pts[1]
else:
    ex = st.number_input("end_x",   0.0, 100.0, 30.0, key="end_x")
    ey = st.number_input("end_y",   0.0, 100.0, 55.0, key="end_y")

carry_seconds = None
end_mm = end_ss = None
if evt_type == "carry":
    carry_seconds = st.number_input("Carry seconds (optional)", value=0.0, step=0.5, key="carry_seconds")
    if carry_seconds:
        end_mm, end_ss = add_seconds(mm, ss, carry_seconds)

if st.button("Save event", key="save_event"):
    with get_conn(DB_PATH_DEFAULT) as c:
        insert_event(
            c,
            match_id=ids["match"], team_id=ids["team"], player_id=ids["player"], event_type=evt_type,
            start_x=clamp01(sx), start_y=clamp01(sy), end_x=clamp01(ex), end_y=clamp01(ey),
            half=int(half), minute=int(mm), second=int(ss),
            end_minute=end_mm, end_second=end_ss, carry_seconds=carry_seconds,
            outcome=outcome_text, outcome_class=oc, notes=None,
        )
    st.success("Event saved.", icon="✅")
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
    if st.button("Export events CSV", key="export_csv"):
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="events.csv",
            mime="text/csv",
            key="download_csv_btn"
        )
with coly:
    if st.button("Export pitch PNG", key="export_png"):
        buf = io.BytesIO()
        make_gaa_pitch().write_image(buf, format="png", engine="kaleido", scale=2)
        st.download_button("Download PNG", data=buf.getvalue(), file_name="pitch.png", mime="image/png", key="download_png_btn")
