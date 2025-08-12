# app.py — MINIMAL pitch click test (no DB, no extras)

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Pitch Click Test", layout="wide")
st.title("Pitch Click Test")

# --- draw a simple GAA pitch (0..100 coords) ---
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

    # goal areas (approx)
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
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10),
                      dragmode=False, clickmode="event+select", hovermode="closest")
    return fig

# --- robust click-surface: transparent HEATMAP (works even when scatter clicks fail) ---
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

# --- arrow helper ---
def add_arrow(fig: go.Figure, x0, y0, x1, y1):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="#111827")

# session state
st.session_state.setdefault("start", None)
st.session_state.setdefault("end", None)

# build figure
fig = gaa_pitch_figure()
add_click_surface(fig, step=1.0)   # dense grid -> every click hits

# show current selection
if st.session_state.start:
    fig.add_trace(go.Scatter(x=[st.session_state.start[0]], y=[st.session_state.start[1]],
                             mode="markers", marker=dict(size=10, color="#2563eb"), name="start"))
if st.session_state.start and st.session_state.end:
    add_arrow(fig, *st.session_state.start, *st.session_state.end)

# listen for clicks
clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="pitch_test_key")

# basic debug
st.write("RAW_CLICKS:", clicks)

# update state on click
if clicks:
    x = float(clicks[-1]["x"]); y = float(clicks[-1]["y"])
    if st.session_state.start is None:
        st.session_state.start = (x, y)
    elif st.session_state.end is None:
        st.session_state.end = (x, y)
    else:
        st.session_state.start = (x, y)
        st.session_state.end = None
    st.rerun()

# clear
if st.button("Clear"):
    st.session_state.start = None
    st.session_state.end = None
    st.rerun()
