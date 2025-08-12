# app.py — GAA Tracking (carries/kicks/shots) with pass mapping, DB + analytics
# ---------------------------------------------------------------------------------
# What you get
#  - Robust pitch click capture (transparent heatmap surface -> clicks always register)
#  - Start/End workflow for carries/kicks/shots; carry time auto-timer
#  - Passes are handled as a subtype of "kick" (per schema constraint)
#  - SQLite persistence (auto-inits from embedded schema), CSV import/export, simple seed loader
#  - Review + Analytics (tables, pass map, start/end heatmaps)
#  - Clean, responsive Streamlit UI
#
# Requirements (example):
#   streamlit==1.34.0
#   plotly==5.24.1
#   streamlit-plotly-events==0.0.6
#   pandas==2.2.2
#   numpy==1.26.4
#   kaleido==0.2.1
# ---------------------------------------------------------------------------------

from __future__ import annotations
import time
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

# -------------------------- Streamlit page config ------------------------------
st.set_page_config(page_title="GAA Tracker", layout="wide")
st.markdown(
    """
    <style>
    .metric-row {display:flex; gap:1rem; flex-wrap:wrap;}
    .metric {background:#0b1220; color:#e5e7eb; padding:12px 16px; border-radius:14px;}
    .metric .label {font-size:12px; color:#9ca3af}
    .metric .value {font-size:20px; font-weight:700}
    .ukit-card {background:#0f172a0f; border:1px solid #1f2937; border-radius:16px; padding:16px}
    .ukit-b {border:1px solid #1f2937; border-radius:10px; padding:8px 12px;}
    .small {font-size:12px; color:#94a3b8}
    .good {color:#22c55e}
    .bad {color:#ef4444}
    .pill {display:inline-block; padding:2px 8px; border-radius:999px; background:#111827; color:#cbd5e1; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- DB schema & helpers -------------------------------
SCHEMA_SQL = """
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
  name TEXT UNIQUE NOT NULL,
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
  end_x   REAL NOT NULL CHECK (end_x   BETWEEN 0 AND 100),
  end_y   REAL NOT NULL CHECK (end_y   BETWEEN 0 AND 100),
  half INTEGER NOT NULL CHECK (half IN (1,2)),
  minute INTEGER, second INTEGER CHECK (second BETWEEN 0 AND 59),
  end_minute INTEGER, end_second INTEGER CHECK (end_second BETWEEN 0 AND 59),
  carry_seconds REAL,
  outcome TEXT,
  outcome_class TEXT CHECK (outcome_class IN ('success','fail','other')),
  notes TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(match_id) REFERENCES match(match_id),
  FOREIGN KEY(team_id)  REFERENCES team(team_id),
  FOREIGN KEY(player_id) REFERENCES player(player_id)
);

CREATE INDEX IF NOT EXISTS idx_event_match     ON event(match_id);
CREATE INDEX IF NOT EXISTS idx_event_player    ON event(player_id);
CREATE INDEX IF NOT EXISTS idx_event_type      ON event(event_type);
CREATE INDEX IF NOT EXISTS idx_event_out_class ON event(outcome_class);
"""

DB_PATH = st.session_state.get("db_path", "gaa.db")
st.session_state["db_path"] = DB_PATH

@contextmanager
def get_conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield con
    finally:
        con.close()

# init DB
with get_conn() as con:
    con.executescript(SCHEMA_SQL)
    con.commit()

# -------------------------- Data helpers --------------------------------------

def upsert_team(con: sqlite3.Connection, name: str) -> int:
    cur = con.execute("INSERT OR IGNORE INTO team(name) VALUES (?)", (name.strip(),))
    if cur.lastrowid:
        return cur.lastrowid
    row = con.execute("SELECT team_id FROM team WHERE name=?", (name.strip(),)).fetchone()
    return row[0]

def upsert_player(con: sqlite3.Connection, team_id: int, player_name: str, shirt_number: Optional[int]=None) -> int:
    con.execute("INSERT OR IGNORE INTO player(team_id, name, shirt_number) VALUES (?,?,?)",
                (team_id, player_name.strip(), shirt_number))
    row = con.execute("SELECT player_id FROM player WHERE team_id=? AND name=?",
                      (team_id, player_name.strip())).fetchone()
    return row[0]

def upsert_match(con: sqlite3.Connection, name: str, date: str, competition: str, venue: str) -> int:
    con.execute("INSERT OR IGNORE INTO match(name, date, competition, venue) VALUES (?,?,?,?)",
                (name.strip(), date, competition, venue))
    row = con.execute("SELECT match_id FROM match WHERE name=?", (name.strip(),)).fetchone()
    return row[0]

# -------------------------- State & models ------------------------------------
@dataclass
class ClickState:
    start: Optional[Tuple[float,float]] = None
    end: Optional[Tuple[float,float]] = None

@dataclass
class TimerState:
    t0: Optional[float] = None

if "click" not in st.session_state:
    st.session_state["click"] = ClickState()
if "timer" not in st.session_state:
    st.session_state["timer"] = TimerState()
if "current_event_type" not in st.session_state:
    st.session_state["current_event_type"] = "carry"

# -------------------------- Pitch drawing -------------------------------------

def gaa_pitch_figure() -> go.Figure:
    fig = go.Figure()

    # grass
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color="#14532d", width=2), fillcolor="#d1fae5")
    # stripes
    for v in range(10, 100, 10):
        fig.add_shape(type="rect", x0=v-10, y0=0, x1=v, y1=100, line=dict(width=0),
                      fillcolor="#ecfdf5" if (v//10) % 2 else "#d1fae5", layer="below")

    # halfway + 13/20/45 from both ends
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="#64748b", width=2))
    for m in (13, 20, 45):
        x = 100 * m / 140.0
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=100, line=dict(color="#94a3b8"))
        fig.add_shape(type="line", x0=100-x, y0=0, x1=100-x, y1=100, line=dict(color="#94a3b8"))

    # centre circle
    t = np.linspace(0, 2*np.pi, 241)
    r = 100 * 10.0 / 140.0
    fig.add_trace(go.Scatter(x=50 + r*np.cos(t), y=50 + r*np.sin(t),
                             mode="lines", line=dict(color="#64748b"), hoverinfo="skip", showlegend=False))

    # small & large rectangles (approx)
    sd, sw = 100*4.5/140.0, 100*14.0/85.0
    ld, lw = 100*13.0/140.0, 100*19.0/85.0
    def goals(x0):
        fig.add_shape(type="rect", x0=x0, y0=50-sw/2, x1=x0+sd, y1=50+sw/2, line=dict(color="#0f172a", width=2))
        fig.add_shape(type="rect", x0=x0, y0=50-lw/2, x1=x0+ld, y1=50+lw/2, line=dict(color="#0f172a"))
    goals(0); goals(100-ld); fig.add_shape(type="rect", x0=100-sd, y0=50-sw/2, x1=100, y1=50+sw/2, line=dict(color="#0f172a", width=2))

    # D arcs
    rd = 100*20.0/140.0
    th = np.linspace(-np.pi/2, np.pi/2, 121)
    fig.add_trace(go.Scatter(x=0 + rd*np.cos(th),   y=50 + rd*np.sin(th),   mode="lines", line=dict(color="#94a3b8"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=100 - rd*np.cos(th), y=50 + rd*np.sin(th),   mode="lines", line=dict(color="#94a3b8"), hoverinfo="skip", showlegend=False))

    fig.update_xaxes(range=[0,100], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0,100], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True)
    fig.update_layout(height=580, margin=dict(l=10,r=10,t=10,b=10),
                      dragmode=False, clickmode="event+select", hovermode="closest")
    return fig

# transparent click surface so *shapes* also return clicks

def add_click_surface(fig: go.Figure, step: float = 1.0):
    n = int(round(100.0 / step))
    xs = np.linspace(0.0, 100.0, n + 1)
    ys = np.linspace(0.0, 100.0, n + 1)
    Z = np.zeros((len(ys), len(xs)))
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=Z,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        showscale=False, hoverinfo="skip", name="_CLICK_SURFACE", opacity=0.01
    ))


def add_arrow(fig: go.Figure, x0, y0, x1, y1):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="#111827")

# -------------------------- Sidebar: context ----------------------------------
with st.sidebar:
    st.header("Context")
    with st.expander("Match", True):
        m_name = st.text_input("Name", value="Dubs vs Kerry 2025-08-10")
        m_date = st.text_input("Date (YYYY-MM-DD)", value="2025-08-10")
        m_comp = st.text_input("Competition", value="All-Ireland")
        m_venue = st.text_input("Venue", value="Croke Park")
        if st.button("Save/Select match", use_container_width=True):
            with get_conn() as con:
                st.session_state["match_id"] = upsert_match(con, m_name, m_date, m_comp, m_venue)
                con.commit()
            st.success("Match ready")

    with st.expander("Teams & Players", True):
        t_name = st.text_input("Team", value="Dublin")
        p_name = st.text_input("Player", value="Player A")
        shirt = st.number_input("Shirt # (optional)", 0, 99, 0)
        if st.button("Save/Select team+player", use_container_width=True):
            with get_conn() as con:
                team_id = upsert_team(con, t_name)
                player_id = upsert_player(con, team_id, p_name, None if shirt == 0 else int(shirt))
                st.session_state["team_id"] = team_id
                st.session_state["player_id"] = player_id
                con.commit()
            st.success("Team & player ready")

    with st.expander("Event options", True):
        et_map = {"Carry": "carry", "Pass (kick)": "kick", "Shot": "shot"}
        label = st.radio("Event type", list(et_map.keys()), index=0)
        st.session_state["current_event_type"] = et_map[label]
        auto_timer = st.checkbox("Auto-timer for carries", value=True)
        outcome = st.selectbox("Outcome (optional)", ["", "success", "completed", "intercepted", "turnover", "goal", "wide", "save"])
        notes = st.text_input("Notes", value="")

        st.caption("Half/time (optional)")
        colh, colm, cols = st.columns([1,1,1])
        with colh:
            half = st.selectbox("Half", [1,2], index=0)
        with colm:
            minute = st.number_input("Min", 0, 99, 0)
        with cols:
            second = st.number_input("Sec", 0, 59, 0)

# -------------------------- Main: Tracker -------------------------------------
colL, colR = st.columns([7,5])

with colL:
    st.subheader("Pitch")

    fig = gaa_pitch_figure()
    add_click_surface(fig, step=1.0)  # dense grid -> every click hits

    # draw any in-progress selection
    if st.session_state["click"].start:
        sx, sy = st.session_state["click"].start
        fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers",
                                 marker=dict(size=10, color="#2563eb"), name="start"))
    if st.session_state["click"].start and st.session_state["click"].end:
        ex, ey = st.session_state["click"].end
        add_arrow(fig, *st.session_state["click"].start, ex, ey)

    clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="pitch_click_key")

    # Update selection
    if clicks:
        x = float(clicks[-1]["x"]); y = float(clicks[-1]["y"])
        if st.session_state["click"].start is None:
            st.session_state["click"].start = (x, y)
            # start timer if carry
            if st.session_state["current_event_type"] == "carry" and auto_timer:
                st.session_state["timer"].t0 = time.perf_counter()
        elif st.session_state["click"].end is None:
            st.session_state["click"].end = (x, y)
        else:
            st.session_state["click"].start = (x, y)
            st.session_state["click"].end = None
        st.rerun()

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Undo", use_container_width=True):
            if st.session_state["click"].end is not None:
                st.session_state["click"].end = None
            elif st.session_state["click"].start is not None:
                st.session_state["click"].start = None
            st.rerun()
    with c2:
        if st.button("Clear", use_container_width=True):
            st.session_state["click"] = ClickState()
            st.session_state["timer"].t0 = None
            st.rerun()
    with c3:
        ready = (st.session_state.get("match_id") and st.session_state.get("team_id") and
                 st.session_state.get("player_id") and st.session_state["click"].start and st.session_state["click"].end)
        if st.button("Save event", use_container_width=True, disabled=not ready):
            sx, sy = st.session_state["click"].start
            ex, ey = st.session_state["click"].end
            carry_seconds = None
            if st.session_state["current_event_type"] == "carry" and auto_timer and st.session_state["timer"].t0:
                carry_seconds = round(max(0.0, time.perf_counter() - st.session_state["timer"].t0), 2)
            with get_conn() as con:
                con.execute(
                    """
                    INSERT INTO event(match_id, team_id, player_id, event_type,
                                      start_x, start_y, end_x, end_y,
                                      half, minute, second, end_minute, end_second,
                                      carry_seconds, outcome, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        st.session_state["match_id"], st.session_state["team_id"], st.session_state["player_id"],
                        st.session_state["current_event_type"],
                        sx, sy, ex, ey,
                        int(half), int(minute), int(second), None, None,
                        carry_seconds, (outcome or None), (notes or None)
                    )
                )
                con.commit()
            st.session_state["click"] = ClickState()
            st.session_state["timer"].t0 = None
            st.success("Saved")
            st.rerun()

with colR:
    st.subheader("Quick metrics")
    with get_conn() as con:
        total = con.execute("SELECT COUNT(*) FROM event").fetchone()[0]
        carries = con.execute("SELECT COUNT(*) FROM event WHERE event_type='carry'").fetchone()[0]
        kicks = con.execute("SELECT COUNT(*) FROM event WHERE event_type='kick'").fetchone()[0]
        shots = con.execute("SELECT COUNT(*) FROM event WHERE event_type='shot'").fetchone()[0]
    st.markdown('<div class="metric-row">' + ''.join([
        f'<div class="metric"><div class="label">Total</div><div class="value">{total}</div></div>',
        f'<div class="metric"><div class="label">Carries</div><div class="value">{carries}</div></div>',
        f'<div class="metric"><div class="label">Kicks/Passes</div><div class="value">{kicks}</div></div>',
        f'<div class="metric"><div class="label">Shots</div><div class="value">{shots}</div></div>',
    ]) + '</div>', unsafe_allow_html=True)

    st.markdown("\n")
    st.subheader("Review")

    with get_conn() as con:
        df = pd.read_sql_query(
            """
            SELECT e.event_id, m.name AS match, t.name AS team, p.name AS player,
                   e.event_type, e.start_x, e.start_y, e.end_x, e.end_y,
                   e.half, e.minute, e.second, e.carry_seconds, e.outcome, e.notes, e.created_at
            FROM event e
            JOIN match m ON m.match_id=e.match_id
            JOIN team t  ON t.team_id=e.team_id
            LEFT JOIN player p ON p.player_id=e.player_id
            ORDER BY e.event_id DESC
            """,
            con,
        )
    st.dataframe(df, use_container_width=True, height=280)

    cexp = st.columns([1,1])
    with cexp[0]:
        if st.button("Export CSV", use_container_width=True, help="Download all events as CSV"):
            csv = df.to_csv(index=False).encode()
            st.download_button("Download events.csv", data=csv, file_name="events.csv", mime="text/csv", use_container_width=True)
    with cexp[1]:
        seed_up = st.file_uploader("Load seed CSV (optional)", type=["csv"], label_visibility="collapsed")
        if seed_up is not None:
            s = pd.read_csv(seed_up)
            with get_conn() as con:
                # Create minimal match/team/player rows and insert events
                m_ids = {}
                t_ids = {}
                p_ids = {}
                for _, r in s.iterrows():
                    mid = m_ids.get(r['match'])
                    if not mid:
                        mid = upsert_match(con, str(r['match']), str(r.get('date', '')), str(r.get('competition','')), str(r.get('venue','')))
                        m_ids[r['match']] = mid
                    tid = t_ids.get(r['team'])
                    if not tid:
                        tid = upsert_team(con, str(r['team']))
                        t_ids[r['team']] = tid
                    pid = p_ids.get((tid, r['player']))
                    if not pid:
                        pid = upsert_player(con, tid, str(r['player']))
                        p_ids[(tid, r['player'])] = pid
                    con.execute(
                        "INSERT INTO event(match_id, team_id, player_id, event_type, start_x, start_y, end_x, end_y, half, minute, second, carry_seconds, outcome)\n                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (mid, tid, pid, str(r['event_type']), float(r['start_x']), float(r['start_y']), float(r['end_x']), float(r['end_y']),
                         int(r['half']), int(r['minute']), int(r['second']), None if pd.isna(r.get('carry_seconds')) else float(r['carry_seconds']),
                         None if pd.isna(r.get('outcome')) else str(r['outcome']))
                    )
                con.commit()
            st.success("Seed loaded")
            st.rerun()

# -------------------------- Analytics -----------------------------------------
st.markdown("---")
st.subheader("Analytics")

with get_conn() as con:
    df_all = pd.read_sql_query("SELECT * FROM event", con)

if df_all.empty:
    st.info("Record some events to unlock charts.")
else:
    ac1, ac2 = st.columns([1,1])

    # Start heatmap
    with ac1:
        st.caption("Start locations heatmap")
        h = go.Figure()
        h.add_trace(go.Histogram2d(
            x=df_all['start_x'], y=df_all['start_y'], nbinsx=25, nbinsy=25, showscale=True
        ))
        h.update_xaxes(range=[0,100], visible=False, fixedrange=True)
        h.update_yaxes(range=[0,100], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True)
        h.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(h, use_container_width=True)

    # End heatmap
    with ac2:
        st.caption("End locations heatmap")
        h2 = go.Figure()
        h2.add_trace(go.Histogram2d(
            x=df_all['end_x'], y=df_all['end_y'], nbinsx=25, nbinsy=25, showscale=True
        ))
        h2.update_xaxes(range=[0,100], visible=False, fixedrange=True)
        h2.update_yaxes(range=[0,100], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True)
        h2.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(h2, use_container_width=True)

    # Pass map (kicks only)
    st.caption("Pass map (kicks)")
    pm = gaa_pitch_figure()
    add_click_surface(pm, step=5)  # not necessary but aligns visuals
    kicks = df_all[df_all['event_type']==='kick'] if 'event_type' in df_all.columns else df_all.iloc[0:0]
    if 'event_type' in df_all.columns:
        kicks = df_all[df_all['event_type'] == 'kick']
        for _, r in kicks.iterrows():
            add_arrow(pm, r['start_x'], r['start_y'], r['end_x'], r['end_y'])
        st.plotly_chart(pm, use_container_width=True)

# -------------------------- Footer --------------------------------------------
st.caption("Built with Streamlit + Plotly. Click the pitch to set start & end. For carries, the timer starts on first click if enabled.")
