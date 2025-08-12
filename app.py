# app.py — GAA Tracking (carries/kicks/shots) with pass mapping, DB + analytics
# ---------------------------------------------------------------------------------
# Fixed version: unique keys for all widgets to prevent StreamlitDuplicateElementId
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

st.set_page_config(page_title="GAA Tracker", layout="wide")

SCHEMA_SQL = """CREATE TABLE IF NOT EXISTS team (
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
  end_x REAL NOT NULL CHECK (end_x BETWEEN 0 AND 100),
  end_y REAL NOT NULL CHECK (end_y BETWEEN 0 AND 100),
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
"""

DB_PATH = "gaa.db"
@contextmanager
def get_conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield con
    finally:
        con.close()

with get_conn() as con:
    con.executescript(SCHEMA_SQL)
    con.commit()

def upsert_team(con, name):
    con.execute("INSERT OR IGNORE INTO team(name) VALUES (?)", (name.strip(),))
    return con.execute("SELECT team_id FROM team WHERE name=?", (name.strip(),)).fetchone()[0]

def upsert_player(con, team_id, player_name, shirt_number=None):
    con.execute("INSERT OR IGNORE INTO player(team_id, name, shirt_number) VALUES (?,?,?)", (team_id, player_name.strip(), shirt_number))
    return con.execute("SELECT player_id FROM player WHERE team_id=? AND name=?", (team_id, player_name.strip())).fetchone()[0]

def upsert_match(con, name, date, competition, venue):
    con.execute("INSERT OR IGNORE INTO match(name, date, competition, venue) VALUES (?,?,?,?)", (name.strip(), date, competition, venue))
    return con.execute("SELECT match_id FROM match WHERE name=?", (name.strip(),)).fetchone()[0]

@dataclass
class ClickState:
    start: Optional[Tuple[float, float]] = None
    end: Optional[Tuple[float, float]] = None

@dataclass
class TimerState:
    t0: Optional[float] = None

if "click" not in st.session_state:
    st.session_state["click"] = ClickState()
if "timer" not in st.session_state:
    st.session_state["timer"] = TimerState()
if "current_event_type" not in st.session_state:
    st.session_state["current_event_type"] = "carry"

def gaa_pitch_figure():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="#14532d"), fillcolor="#d1fae5")
    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(range=[0, 100], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=580, margin=dict(l=10, r=10, t=10, b=10), dragmode=False, clickmode="event+select")
    return fig

def add_click_surface(fig, step=1.0):
    xs = np.linspace(0, 100, int(100/step)+1)
    ys = np.linspace(0, 100, int(100/step)+1)
    Z = np.zeros((len(ys), len(xs)))
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=Z, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False, hoverinfo="skip", opacity=0.01))

def add_arrow(fig, x0, y0, x1, y1):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3)

with st.sidebar:
    st.header("Context")
    with st.expander("Match", True):
        m_name = st.text_input("Name", key="match_name")
        m_date = st.text_input("Date (YYYY-MM-DD)", key="match_date")
        m_comp = st.text_input("Competition", key="match_comp")
        m_venue = st.text_input("Venue", key="match_venue")
        if st.button("Save/Select match", key="btn_save_match"):
            with get_conn() as con:
                st.session_state["match_id"] = upsert_match(con, m_name, m_date, m_comp, m_venue)
                con.commit()
            st.success("Match ready")
    with st.expander("Teams & Players", True):
        t_name = st.text_input("Team", key="team_name")
        p_name = st.text_input("Player", key="player_name")
        shirt = st.number_input("Shirt # (optional)", 0, 99, 0, key="shirt_number")
        if st.button("Save/Select team+player", key="btn_save_team_player"):
            with get_conn() as con:
                team_id = upsert_team(con, t_name)
                player_id = upsert_player(con, team_id, p_name, None if shirt == 0 else int(shirt))
                st.session_state["team_id"] = team_id
                st.session_state["player_id"] = player_id
                con.commit()
            st.success("Team & player ready")
    with st.expander("Event options", True):
        et_map = {"Carry": "carry", "Pass (kick)": "kick", "Shot": "shot"}
        label = st.radio("Event type", list(et_map.keys()), index=0, key="event_type_radio")
        st.session_state["current_event_type"] = et_map[label]
        auto_timer = st.checkbox("Auto-timer for carries", value=True, key="auto_timer")
        outcome = st.selectbox("Outcome (optional)", ["", "success", "completed", "intercepted", "turnover", "goal", "wide", "save"], key="outcome")
        notes = st.text_input("Notes", key="notes")
        half = st.selectbox("Half", [1, 2], index=0, key="half")
        minute = st.number_input("Min", 0, 99, 0, key="minute")
        second = st.number_input("Sec", 0, 59, 0, key="second")

colL, colR = st.columns([7, 5])
with colL:
    st.subheader("Pitch")
    fig = gaa_pitch_figure()
    add_click_surface(fig, step=1.0)
    if st.session_state["click"].start:
        sx, sy = st.session_state["click"].start
        fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers"))
    if st.session_state["click"].start and st.session_state["click"].end:
        add_arrow(fig, *st.session_state["click"].start, *st.session_state["click"].end)
    clicks = plotly_events(fig, click_event=True, hover_event=False, key="pitch_click")
    if clicks:
        x = float(clicks[-1]["x"]); y = float(clicks[-1]["y"])
        if st.session_state["click"].start is None:
            st.session_state["click"].start = (x, y)
            if st.session_state["current_event_type"] == "carry" and auto_timer:
                st.session_state["timer"].t0 = time.perf_counter()
        elif st.session_state["click"].end is None:
            st.session_state["click"].end = (x, y)
        else:
            st.session_state["click"].start = (x, y)
            st.session_state["click"].end = None
        st.rerun()
    if st.button("Undo", key="btn_undo"):
        if st.session_state["click"].end is not None:
            st.session_state["click"].end = None
        elif st.session_state["click"].start is not None:
            st.session_state["click"].start = None
        st.rerun()
    if st.button("Clear", key="btn_clear"):
        st.session_state["click"] = ClickState()
        st.session_state["timer"].t0 = None
        st.rerun()
    ready = (st.session_state.get("match_id") and st.session_state.get("team_id") and st.session_state.get("player_id") and st.session_state["click"].start and st.session_state["click"].end)
    if st.button("Save event", key="btn_save_event", disabled=not ready):
        sx, sy = st.session_state["click"].start
        ex, ey = st.session_state["click"].end
        carry_seconds = None
        if st.session_state["current_event_type"] == "carry" and auto_timer and st.session_state["timer"].t0:
            carry_seconds = round(max(0.0, time.perf_counter() - st.session_state["timer"].t0), 2)
        with get_conn() as con:
            con.execute("INSERT INTO event(match_id, team_id, player_id, event_type, start_x, start_y, end_x, end_y, half, minute, second, carry_seconds, outcome, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (st.session_state["match_id"], st.session_state["team_id"], st.session_state["player_id"], st.session_state["current_event_type"], sx, sy, ex, ey, int(half), int(minute), int(second), carry_seconds, (outcome or None), (notes or None)))
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
    st.write({"Total": total, "Carries": carries, "Kicks": kicks, "Shots": shots})
    with get_conn() as con:
        df = pd.read_sql_query("SELECT * FROM event ORDER BY event_id DESC", con)
    st.dataframe(df)
