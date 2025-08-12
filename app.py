87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
import numpy as np
colL, colR = st.columns([7,5])
with colL:
    st.subheader("Pitch")
    fig = make_pitch()
    add_click_surface(fig)

    # draw current selection
    if st.session_state.clicks.start:
        sx, sy = st.session_state.clicks.start
        fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers", marker=dict(size=10)))
    if st.session_state.clicks.start and st.session_state.clicks.end:
        ex, ey = st.session_state.clicks.end
        arrow(fig, *st.session_state.clicks.start, ex, ey)

    clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="pitch_events")

    if clicks:
        x = float(clicks[-1]["x"]) ; y = float(clicks[-1]["y"]) 
        if st.session_state.clicks.start is None:
            st.session_state.clicks.start = (x, y)
            if etype == "carry" and auto_timer:
                st.session_state.t0 = time.perf_counter()
        elif st.session_state.clicks.end is None:
            st.session_state.clicks.end = (x, y)
        else:
            st.session_state.clicks.start = (x, y)
            st.session_state.clicks.end = None
        st.rerun()

    b1, b2, b3 = st.columns(3)
    if b1.button("Undo", key="undo"):
        if st.session_state.clicks.end is not None:
            st.session_state.clicks.end = None
        elif st.session_state.clicks.start is not None:
            st.session_state.clicks.start = None
        st.rerun()
    if b2.button("Clear", key="clear"):
        st.session_state.clicks = Clicks()
        st.session_state.t0 = None
        st.rerun()

    ready = (st.session_state.clicks.start and st.session_state.clicks.end and
             ((etype != "kick" and player.strip()) or (etype == "kick" and p_from.strip() and p_to.strip())))

    if b3.button("Save", key="save", disabled=not ready):
        sx, sy = st.session_state.clicks.start
        ex, ey = st.session_state.clicks.end
        duration = None
        if etype == "carry" and st.session_state.t0 is not None:
            duration = round(time.perf_counter() - st.session_state.t0, 2)
        event = dict(
            type=etype,
            player=player.strip() if etype != "kick" else None,
            pass_from=p_from.strip() if etype == "kick" else None,
            pass_to=p_to.strip() if etype == "kick" else None,
            start_x=round(sx,2), start_y=round(sy,2), end_x=round(ex,2), end_y=round(ey,2),
            duration_seconds=duration,
            notes=notes.strip() or None,
            ts=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        st.session_state.events.append(event)
        st.session_state.clicks = Clicks()
        st.session_state.t0 = None
        st.success("Saved")
        st.rerun()

# ----------------- Review + export --------------------------------------------

with colR:
    st.subheader("Events")
    df = pd.DataFrame(st.session_state.events)
    st.dataframe(df, use_container_width=True, height=380)

    if not df.empty:
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "gaa_events.csv", "text/csv", key="dl")

# ----------------- Done --------------------------------------------------------

st.caption("Simple mode: in‑memory only. Use the CSV download to persist data. Add DB later if needed.")
