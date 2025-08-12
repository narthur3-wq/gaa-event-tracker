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
    "carry": {"success": ["success"], "fail": ["turnover"], "other": []},
    "kick":  {"success": ["completed","received","won"], "fail": ["turnover","intercepted","lost"], "other": []},
    "shot":  {"success": ["point","goal","two point"], "fail": ["wide","short","post","save"], "other": []},
}

def get_phrases():
    # Load current phrases from session, falling back to defaults
    out = {}
    for et in ("carry","kick","shot"):
        out[et] = {}
        for b in ("success","fail","other"):
            out[et][b] = st.session_state.get(f"phr_{et}_{b}", DEFAULT_PHRASES[et][b])
    return out

def map_outcome_to_class(event_type: str, outcome_text: str) -> str:
    phrases = get_phrases()
    o = (outcome_text or "").strip().lower()
    for bucket in ("success","fail","other"):
        for p in phrases.get(event_type, {}).get(bucket, []):
            if o == p.lower():
                return bucket
    return "other"

# ---------- DB helpers ----------
def get_conn(db_path: str):
    return sqlite3.connect(db_path, check_same_thread=False)

def init_db(conn: sqlite3.Connection):
    try:
        schema = open("schema.sql", "r", encoding="utf-8").read()
    except FileNotFoundError:
        schema = """
        CREATE TABLE IF NOT EXISTS team (team_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
        CREATE TABLE IF NOT EXISTS player (player_id INTEGER PRIMARY KEY, team_id INTEGER NOT NULL, name TEXT NOT NULL, shirt_number INTEGER, UNIQUE(team_id, name));
        CREATE TABLE IF NOT EXISTS match (match_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, date TEXT, competition TEXT, venue TEXT);
        CREATE TABLE IF NOT EXISTS event (
          event_id INTEGER PRIMARY KEY, match_id INTEGER NOT NULL, team_id INTEGER NOT NULL, player_id INTEGER,
          event_type TEXT NOT NULL, start_x REAL NOT NULL, start_y REAL NOT NULL, end_x REAL NOT NULL, end_y REAL NOT NULL,
          half INTEGER NOT NULL, minute INTEGER, second INTEGER, end_minute INTEGER, end_second INTEGER, carry_seconds REAL,
          outcome TEXT, outcome_class TEXT, notes TEXT, created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_event_match     ON event(match_id);
        CREATE INDEX IF NOT EXISTS idx_event_player    ON event(player_id);
        CREATE INDEX IF NOT EXISTS idx_event_type      ON event(event_type);
        CREATE INDEX IF NOT EXISTS idx_event_out_class ON event(outcome_class);
        """
    conn.executescript(schema)
    conn.commit()

def upsert_team(conn, name: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO team(name) VALUES (?)", (name,))
    cur.execute("SELECT team_id FROM team WHERE name=?", (name,))
    return cur.fetchone()[0]

def upsert_player(conn, team_id: int, name: str, shirt_number: Optional[int] = None) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO player(team_id,name,shirt_number) VALUES (?,?,?)", (team_id, name, shirt_number))
    cur.execute("SELECT player_id FROM player WHERE team_id=? AND name=?", (team_id, name))
    return cur.fetchone()[0]

def upsert_match(conn, name: str, date: str = None, competition: str = None, venue: str = None) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO match(name,date,competition,venue) VALUES (?,?,?,?)", (name, date, competition, venue))
    cur.execute("SELECT match_id FROM match WHERE name=?", (name,))
    return cur.fetchone()[0]

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

def get_matches(conn):
    return pd.read_sql_query("SELECT match_id,name,date,competition,venue FROM match ORDER BY date DESC, name DESC", conn)

def get_players_for_team(conn, team_id: int):
    return pd.read_sql_query("SELECT player_id,name FROM player WHERE team_id=? ORDER BY name", conn, params=(team_id,))

def get_events(conn, match_id: int):
    q = """
    SELECT e.*, p.name AS player_name, t.name AS team_name
    FROM event e
    LEFT JOIN player p ON p.player_id=e.player_id
    LEFT JOIN team t   ON t.team_id=e.team_id
    WHERE e.match_id=?
    ORDER BY e.half, e.minute, e.second, e.event_id
    """
    return pd.read_sql_query(q, conn, params=(match_id,))

# ---------- Pitch (Gaelic) ----------
def gaa_pitch_figure() -> go.Figure:
    def lx(m): return m/140*100
    def wy(m): return m/85*100
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100)
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100)
    for m in (13,20,45):
        x = lx(m); xr = 100-x
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=100, line=dict(dash="dot"))
        fig.add_shape(type="line", x0=xr, y0=0, x1=xr, y1=100, line=dict(dash="dot"))
    sm_len, sm_w = 14, 4.5
    lg_len, lg_w = 19, 13
    fig.add_shape(type="rect", x0=0, y0=50-wy(sm_w/2), x1=lx(sm_len), y1=50+wy(sm_w/2))
    fig.add_shape(type="rect", x0=0, y0=50-wy(lg_w/2), x1=lx(lg_len), y1=50+wy(lg_w/2))
    fig.add_shape(type="rect", x0=100-lx(sm_len), y0=50-wy(sm_w/2), x1=100, y1=50+wy(sm_w/2))
    fig.add_shape(type="rect", x0=100-lx(lg_len), y0=50-wy(lg_w/2), x1=100, y1=50+wy(lg_w/2))
    fig.update_xaxes(range=[0,100], visible=False)
    fig.update_yaxes(range=[0,100], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def add_arrow(fig: go.Figure, x0, y0, x1, y1, color="#444", width=2):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=width, arrowcolor=color)

# ---------- Styling / bins ----------
COLOR_SUCCESS = "#2ca02c"
COLOR_FAIL    = "#d62728"
COLOR_OTHER   = "#7f7f7f"
CARRY_BINS = [(0,3,2), (3,5,4), (5,999,6)]  # (min_sec, max_sec, arrow_width)

def seconds_or_distance(row) -> float:
    if pd.notnull(row.get("carry_seconds")):
        return float(row["carry_seconds"])
    dx = (row["end_x"] - row["start_x"]) / 100 * 140
    dy = (row["end_y"] - row["start_y"]) / 100 * 85
    metres = math.hypot(dx, dy)
    return metres / 5.5  # rough proxy

def carry_width_for_value(v: float) -> int:
    for lo, hi, w in CARRY_BINS:
        if lo <= v < hi:
            return w
    return CARRY_BINS[-1][2]

def color_for_class(oc: str) -> str:
    return COLOR_SUCCESS if oc=="success" else (COLOR_FAIL if oc=="fail" else COLOR_OTHER)

def draw_events_on_pitch(fig: go.Figure, df: pd.DataFrame, title: str):
    for _, r in df.iterrows():
        width = 2
        if r["event_type"] == "carry":
            width = carry_width_for_value(seconds_or_distance(r))
        add_arrow(fig, r["start_x"], r["start_y"], r["end_x"], r["end_y"], color=color_for_class(r.get("outcome_class","other")), width=width)
    fig.update_layout(title=title)
    return fig

# ---------- Small helpers ----------
def parse_mmss(mmss: str):
    try:
        mm, ss = [int(x) for x in (mmss or "00:00").split(":", 1)]
        ss = max(0, min(59, ss)); mm = max(0, mm)
        return mm, ss
    except Exception:
        return 0, 0

def add_seconds(mm: int, ss: int, add: float):
    total = int(round(mm*60 + ss + (add or 0)))
    if total < 0: total = 0
    return total // 60, total % 60

def parse_player_blob(text: str) -> List[str]:
    # Split on commas, semicolons, newlines, forward/back slashes; clean+dedupe
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

# ---------- UI ----------
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

    if seed_file is not None and st.button("Import CSV into DB"):
        with get_conn(db_path) as c:
            init_db(c)
            df = pd.read_csv(seed_file)
            df.columns = [c.strip().lower() for c in df.columns]
            for _, r in df.iterrows():
                mid = upsert_match(c, r.get("match"), str(r.get("date") or ""), r.get("competition"), r.get("venue"))
                tid = upsert_team(c, r.get("team"))
                pid = None
                if isinstance(r.get("player"), str) and r.get("player").strip():
                    pid = upsert_player(c, tid, r.get("player").strip())
                et = (r.get("event_type") or "").lower().strip()
                half = int(r.get("half") or 1)
                mm = int(r.get("minute") or 0)
                ss = int(r.get("second") or 0)
                sx = float(r.get("start_x") or 0.0)
                sy = float(r.get("start_y") or 0.0)
                ex = float(r.get("end_x") or sx)
                ey = float(r.get("end_y") or sy)
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

    # ----- Outcome phrases config -----
    st.header("Outcome phrases")
    st.caption("These power the Outcome dropdowns & colours.")
    def edit_list(label, key, default):
        txt = st.text_area(label, value=", ".join(st.session_state.get(key, default)), height=60)
        st.session_state[key] = [t.strip() for t in txt.replace("\n", ",").split(",") if t.strip()]

    for et in ("carry","kick","shot"):
        with st.expander(et.capitalize()):
            edit_list("Success", f"phr_{et}_success", DEFAULT_PHRASES[et]["success"])
            edit_list("Fail",    f"phr_{et}_fail",    DEFAULT_PHRASES[et]["fail"])
            edit_list("Other",   f"phr_{et}_other",   DEFAULT_PHRASES[et]["other"])

colL, colR = st.columns([2,3])

with colL:
    st.subheader("Select / Create Match")
    with get_conn(db_path) as c:
        try: init_db(c)
        except Exception: pass
        matches = get_matches(c)
    match_names = ["➕ Create new match …"] + matches["name"].tolist()
    pick = st.selectbox("Match", match_names)

    if pick == "➕ Create new match …":
        st.info("Create a match to start logging events.")
        m_name = st.text_input("Match name", "Dubs vs Kerry 2025-08-10")
        m_date = st.date_input("Date")
        m_comp = st.text_input("Competition", "All-Ireland")
        m_venue= st.text_input("Venue", "Croke Park")
        if st.button("Create match"):
            with get_conn(db_path) as c:
                upsert_match(c, m_name, str(m_date), m_comp, m_venue)
            st.success(f"Created match: {m_name}")
            st.rerun()
        st.stop()
    else:
        with get_conn(db_path) as c:
            mid = upsert_match(c, pick)

    st.subheader("Team & Player")
    team_name = st.text_input("Team", "Dublin")
    with get_conn(db_path) as c:
        tid = upsert_team(c, team_name)
        roster = get_players_for_team(c, tid)

    with st.expander("Bulk add players to this team"):
        st.caption("Paste names separated by commas, semicolons, new lines, / or \\ .")
        raw = st.text_area("Player names", height=120, placeholder="Player A\nPlayer B\nPlayer C")
        if st.button("Add players"):
            names = parse_player_blob(raw)
            added = 0
            if names:
                with get_conn(db_path) as c:
                    for n in names:
                        upsert_player(c, tid, n)
                        added += 1
                st.success(f"Added/ensured {added} players for '{team_name}'.")
                st.rerun()
            else:
                st.warning("No valid names found.")

    player_options = ["(none)"] + roster["name"].tolist()
    player_choice = st.selectbox("Player", player_options)
    if player_choice == "(none)":
        new_player = st.text_input("Add single player (quick)")
        if new_player:
            with get_conn(db_path) as c:
                upsert_player(c, tid, new_player)
            st.success(f"Added player: {new_player}")
            st.rerun()
        pid = None
    else:
        with get_conn(db_path) as c:
            pid = upsert_player(c, tid, player_choice)

    st.subheader("Add Event")
    evt_type = st.selectbox("Category", ["carry","kick","shot"], index=0)
    half = st.selectbox("Half", [1,2], index=0)
    mtime = st.text_input("Match time (mm:ss)", "00:00")
    mm, ss = parse_mmss(mtime)

    # Outcome dropdown (with configurable phrases)
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
        oc = "other"  # strict: custom defaults to "other"
    else:
        outcome_text = picked
        oc = dict(choice_items).get(picked, "other")

    st.caption(f"Outcome class → {oc}")

    carry_seconds = None
    end_mm, end_ss = (None, None)
    if evt_type == "carry":
        carry_seconds = st.number_input("Carry duration (seconds)", min_value=0.0, step=0.1, value=0.0)
        if carry_seconds == 0.0:
            carry_seconds = None
        if carry_seconds is not None:
            end_mm, end_ss = add_seconds(mm, ss, carry_seconds)
            st.caption(f"Ends at ≈ {end_mm:02d}:{end_ss:02d} (auto)")

    # Pitch — click to set start/end
    st.markdown("**Click the pitch:** first click sets start, second sets end.")
    if "start" not in st.session_state: st.session_state.start = None
    if "end" not in st.session_state: st.session_state.end = None

    fig = gaa_pitch_figure()
    if st.session_state.start is not None:
        fig.add_trace(go.Scatter(x=[st.session_state.start[0]], y=[st.session_state.start[1]], mode="markers", marker_size=10, name="start"))
    if st.session_state.end is not None and st.session_state.start is not None:
        add_arrow(fig, st.session_state.start[0], st.session_state.start[1], st.session_state.end[0], st.session_state.end[1], color="#1f77b4", width=3)

    clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="pitch")
    if clicks:
        last = clicks[-1]
        x = float(last.get("x")); y = float(last.get("y"))
        if st.session_state.start is None:
            st.session_state.start = (x, y)
        elif st.session_state.end is None:
            st.session_state.end = (x, y)
        else:
            st.session_state.start = (x, y)
            st.session_state.end = None
        st.rerun()

    cclear, _ = st.columns([1,3])
    if cclear.button("Clear selection"):
        st.session_state.start = None
        st.session_state.end = None
        st.rerun()

    def _def(val, fb): return float(val) if val is not None else fb
    c1, c2, c3, c4 = st.columns(4)
    sx = c1.number_input("start_x", 0.0, 100.0, value=_def(st.session_state.start[0] if st.session_state.start else None, 10.0))
    sy = c2.number_input("start_y", 0.0, 100.0, value=_def(st.session_state.start[1] if st.session_state.start else None, 50.0))
    ex = c3.number_input("end_x",   0.0, 100.0, value=_def(st.session_state.end[0]   if st.session_state.end   else None, 30.0))
    ey = c4.number_input("end_y",   0.0, 100.0, value=_def(st.session_state.end[1]   if st.session_state.end   else None, 55.0))

    typed_start = (sx, sy); typed_end = (ex, ey)
    if st.session_state.start != typed_start: st.session_state.start = typed_start
    if st.session_state.end   != typed_end:   st.session_state.end   = typed_end

    if st.button("Save event"):
        with get_conn(db_path) as c:
            insert_event(c,
                match_id=mid, team_id=tid, player_id=pid, event_type=evt_type,
                start_x=sx, start_y=sy, end_x=ex, end_y=ey,
                half=int(half), minute=int(mm), second=int(ss),
                end_minute=end_mm, end_second=end_ss, carry_seconds=carry_seconds,
                outcome=outcome_text, outcome_class=oc, notes=None)
        st.success("Event saved.")
        st.rerun()

with colR:
    st.subheader("Match view & tools")
    with get_conn(db_path) as c:
        events = get_events(c, mid)

    tab_charts, tab_manage = st.tabs(["Charts & Table", "Manage (delete)"])

    with tab_charts:
        fcol1, fcol2, fcol3 = st.columns(3)
        pl_opts = ["All"] + sorted([p for p in events["player_name"].dropna().unique().tolist()])
        chosen_player = fcol1.selectbox("Player", pl_opts)
        half_opts = ["Both", 1, 2]
        chosen_half = fcol2.selectbox("Half", half_opts)
        out_opts = ["All","success","fail","other"]
        chosen_out = fcol3.selectbox("Outcome class", out_opts)

        filt = events.copy()
        if chosen_player != "All": filt = filt[filt["player_name"] == chosen_player]
        if chosen_half in (1,2):   filt = filt[filt["half"] == chosen_half]
        if chosen_out != "All":    filt = filt[filt["outcome_class"] == chosen_out]

        carr = filt[filt.event_type=="carry"]
        kick = filt[filt.event_type=="kick"]
        shot = filt[filt.event_type=="shot"]

        c1, c2 = st.columns(2)
        fig1 = gaa_pitch_figure(); draw_events_on_pitch(fig1, carr, "Carries (width=binned duration, color=success/fail)")
        c1.plotly_chart(fig1, use_container_width=True)

        fig2 = gaa_pitch_figure(); draw_events_on_pitch(fig2, kick, "Kicks (color=success/fail)")
        c2.plotly_chart(fig2, use_container_width=True)

        fig3 = gaa_pitch_figure(); draw_events_on_pitch(fig3, shot, "Shots (color=success/fail)")
        st.plotly_chart(fig3, use_container_width=True)

        st.caption("Colors: success=green, fail=red, other=grey. Carry width bins: 0–3s thin, 3–5s medium, >5s thick (or distance proxy if no seconds).")

        # Export helpers
        def make_fname(chart: str) -> str:
            match_label = safe_name(pick) if pick else "match"
            player_label = chosen_player if chosen_player != "All" else "All"
            half_label = f"H{chosen_half}" if chosen_half in (1, 2) else "Both"
            return f"{match_label}_{safe_name(player_label)}_{half_label}_{chart}.png"

        def fig_to_png_bytes(fig: go.Figure) -> bytes:
            return fig.to_image(format="png", scale=2, engine="kaleido")

        ec1, ec2, ec3 = st.columns(3)
        ec1.download_button("Export Carries PNG", fig_to_png_bytes(fig1), file_name=make_fname("Carries"), mime="image/png")
        ec2.download_button("Export Kicks PNG",   fig_to_png_bytes(fig2), file_name=make_fname("Kicks"),   mime="image/png")
        ec3.download_button("Export Shots PNG",   fig_to_png_bytes(fig3), file_name=make_fname("Shots"),   mime="image/png")

        csv_name = make_fname("filtered").replace(".png", ".csv")
        st.download_button("Export filtered CSV", data=filt.to_csv(index=False).encode("utf-8"), file_name=csv_name, mime="text/csv")

    with tab_manage:
        st.write("Delete incorrect rows here, then re-add in the left panel if needed.")
        for r in events.itertuples(index=False):
            cols = st.columns([5,1])
            cols[0].markdown(f"**ID {r.event_id}** · {r.event_type} · {r.team_name or ''} · {r.player_name or ''} · H{r.half} {str(r.minute).zfill(2)}:{str(r.second).zfill(2)} · {r.outcome} ({r.outcome_class})")
            if cols[1].button("🗑 Delete", key=f"del_{r.event_id}"):
                with get_conn(db_path) as c:
                    delete_event(c, int(r.event_id))
                st.warning(f"Deleted event {r.event_id}")
                st.rerun()
