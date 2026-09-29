"""Ripple — U.S. input-output shock propagator.

Pick a sector and a shock, compute the Leontief inverse L = (I - A)^-1 over
BEA's annual input-output accounts, and watch the ripple travel upstream
through the supply network. Nodes are colored/sized by cumulative impact;
edge width is the induced interindustry flow. Built with Streamlit,
networkx, and Plotly.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bea_io

st.set_page_config(
    page_title="Ripple · US Input-Output Shock Explorer",
    page_icon="🌊",
    layout="wide",
)

# Impact color ramp: deep navy (low) -> teal -> sand -> warm amber (high).
IMPACT_SCALE = [
    [0.0, "#12263f"],
    [0.25, "#1f6f8b"],
    [0.55, "#5fb0a8"],
    [0.8, "#e8c468"],
    [1.0, "#e2704b"],
]

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
      .ripple-title {
        font-size: 2.9rem; font-weight: 700; letter-spacing: -0.02em;
        background: linear-gradient(92deg, #e8c468 0%, #5fb0a8 55%, #6ea8d8 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        margin-bottom: 0.15rem;
      }
      .ripple-sub { color: #9aa7bd; font-size: 1.05rem; margin-top: 0; }
      div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px; padding: 15px 18px;
      }
      div[data-testid="stMetric"] label { color: #9aa7bd; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.07); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_year(year: int) -> tuple[bea_io.IOTables, bea_io.IOModel]:
    tables = bea_io.load_tables(year)
    return tables, bea_io.build_model(tables)


def short(name: str, limit: int = 26) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown("## 🎛️ Shock controls")
    year = st.selectbox("BEA table year", list(reversed(bea_io.YEARS)), index=0)
    st.markdown("---")

with st.sidebar:
    sector_idx = 0  # set after data loads
    shock_mode = st.radio(
        "Shock measured as",
        ["Dollar shock ($B)", "% of sector output"],
        horizontal=True,
    )
    if shock_mode.startswith("Dollar"):
        shock_value = st.slider(
            "Shock magnitude ($ billions)", 1.0, 250.0, 25.0, 5.0
        )
    else:
        shock_value = st.slider(
            "Shock magnitude (% of output)", 1.0, 50.0, 10.0, 1.0
        )
    rounds = st.slider("Supply-chain rounds to animate", 2, 12, 8)
    top_edges = st.slider("Supply links shown", 20, 150, 80, 10)
    st.caption(
        "Frames show cumulative impacts (I + A + … + A^r)·s; "
        "the final frame is the full Leontief total L·s."
    )

# ------------------------------------------------------------------- data ---
st.markdown(
    '<div class="ripple-title">Ripple</div>'
    '<p class="ripple-sub">How far does a shock to one sector travel through '
    'the U.S. economy? BEA input-output accounts × the Leontief inverse.</p>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading BEA input-output tables…"):
        tables, model = load_year(year)
except Exception as exc:  # network / layout failures surface clearly in-app
    st.error(
        f"Could not load the BEA input-output tables for {year}.\n\n{exc}\n\n"
        "The app downloads them live from apps.bea.gov — check the connection "
        "and rerun."
    )
    st.stop()

sector_options = list(range(len(tables.codes)))
with st.sidebar:
    sector_idx = st.selectbox(
        "Sector to shock",
        sector_options,
        index=sector_options[tables.codes.index("3361MV")]
        if "3361MV" in tables.codes
        else 0,
        format_func=lambda i: f"{tables.codes[i]} · {tables.names[i]}",
    )

n = len(tables.codes)
sector_name = tables.names[sector_idx]
sector_code = tables.codes[sector_idx]

# --------------------------------------------------------------- model run ---
if shock_mode.startswith("Dollar"):
    shock_m = shock_value * 1000.0  # $B -> $ millions
else:
    shock_m = shock_value / 100.0 * float(tables.industry_output[sector_idx])

s = bea_io.shock_vector(n, sector_idx, shock_m)
series = bea_io.ripple_series(model.A, s, rounds)
series[-1] = model.L @ s  # exact converged total for the final frame
total = model.L @ s

total_B = total.sum() / 1000.0
shock_B = shock_m / 1000.0
multiplier = total.sum() / shock_m
indirect_share = 1.0 - shock_m / total.sum()
own_output_B = float(tables.industry_output[sector_idx]) / 1000.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Shock to the sector", f"${shock_B:,.1f}B", f"{shock_m / tables.industry_output[sector_idx]:.1%} of its output")
m2.metric("Total output impact", f"${total_B:,.1f}B", "direct + all upstream rounds")
m3.metric("Output multiplier", f"{multiplier:.2f}×", f"{indirect_share:.0%} of it is indirect")
top_order = np.argsort(-total)
m4.metric(
    "Most affected besides the shock",
    short(tables.names[top_order[1] if top_order[0] == sector_idx else top_order[0]], 24),
    f"${(total[top_order[1] if top_order[0] == sector_idx else top_order[0]] / 1000.0):,.1f}B",
)

st.divider()

# ------------------------------------------------------------ network graph ---
A = model.A
y_final = total / 1000.0  # $B
flows_final = A * total[None, :]  # induced flow i->j in $M

pairs = np.dstack(np.unravel_index(np.argsort(-flows_final, axis=None), flows_final.shape))[0]
edges = [
    (int(i), int(j))
    for i, j in pairs
    if flows_final[i, j] > 0
][:top_edges]

G = nx.DiGraph()
G.add_nodes_from(range(n))
for i, j in edges:
    G.add_edge(i, j, weight=float(flows_final[i, j]))
pos = nx.spring_layout(G, weight="weight", seed=11)

xs = np.array([pos[i][0] for i in range(n)])
ys = np.array([pos[i][1] for i in range(n)])
xs = (xs - xs.mean()) / (np.ptp(xs) / 2.0 + 1e-9)
ys = (ys - ys.mean()) / (np.ptp(ys) / 2.0 + 1e-9)
for i in range(n):
    pos[i] = (float(xs[i]), float(ys[i]))

fmax = float(flows_final[edges[0]]) if edges else 1.0
ymax = float(y_final.max()) or 1.0

label_nodes = set(int(k) for k in np.argsort(-y_final)[:12]) | {sector_idx}
node_text = [
    short(tables.names[i], 22) if i in label_nodes else "" for i in range(n)
]

# Plotly does not accept array-valued line widths, so edges are grouped into
# width buckets; each bucket is one trace whose scalar width animates per round.
N_BUCKETS = 8
BUCKET_WIDTHS = [0.5 + i * 6.0 / (N_BUCKETS - 1) for i in range(N_BUCKETS)]


def edge_traces_for(round_idx: int) -> list[go.Scatter]:
    fl = A * series[round_idx][None, :]  # induced flow, $M
    buckets: dict[int, dict[str, list]] = {
        b: {"x": [], "y": [], "text": []} for b in range(N_BUCKETS)
    }
    for i, j in edges:
        norm = (float(fl[i, j]) / fmax) ** 0.7  # 0..1 across displayed edges
        bucket = buckets[min(N_BUCKETS - 1, int(norm * N_BUCKETS))]
        bucket["x"] += [pos[i][0], pos[j][0], None]
        bucket["y"] += [pos[i][1], pos[j][1], None]
        label = (
            f"{short(tables.names[i], 34)} → {short(tables.names[j], 34)}"
            f"<br>induced flow ${fl[i, j] / 1000.0:,.2f}B<extra></extra>"
        )
        bucket["text"] += [label, label, ""]

    traces = []
    for b in range(N_BUCKETS):
        bucket = buckets[b]
        traces.append(
            go.Scatter(
                x=bucket["x"],
                y=bucket["y"],
                mode="lines",
                line=dict(width=BUCKET_WIDTHS[b], color="rgba(154, 167, 189, 0.5)"),
                hoverinfo="text",
                hovertext=bucket["text"],
                showlegend=False,
            )
        )
    return traces


def node_colors(round_idx: int) -> np.ndarray:
    return series[round_idx] / 1000.0


def node_sizes(round_idx: int) -> np.ndarray:
    y = series[round_idx] / 1000.0
    return 9.0 + 44.0 * np.clip(y / ymax, 0.0, 1.0) ** 0.6


def node_hover_for(round_idx: int) -> list[str]:
    y = series[round_idx]
    out: list[str] = []
    for i in range(n):
        share = f"{y[i] / y_final[i]:.0%} of its total" if y_final[i] > 0 else "no effect"
        out.append(
            f"<b>{tables.names[i]}</b> ({tables.codes[i]})<br>"
            f"cumulative impact: ${y[i] / 1000.0:,.2f}B ({share})"
            f"<extra></extra>"
        )
    return out


NODE_TRACE = N_BUCKETS  # node trace sits after the edge-bucket traces
node_trace = go.Scatter(
    x=[pos[i][0] for i in range(n)],
    y=[pos[i][1] for i in range(n)],
    mode="markers+text",
    text=node_text,
    textposition="top center",
    textfont=dict(size=10, color="rgba(214, 222, 238, 0.85)"),
    hoverinfo="text",
    hovertext=node_hover_for(0),
    marker=dict(
        size=node_sizes(0),
        color=node_colors(0),
        colorscale=IMPACT_SCALE,
        cmin=0.0,
        cmax=ymax,
        showscale=True,
        colorbar=dict(
            title="Impact<br>($B)",
            thickness=14,
            len=0.62,
            tickfont=dict(color="#9aa7bd"),
            outlinewidth=0,
        ),
        line=dict(width=1.0, color="rgba(232, 236, 244, 0.35)"),
    ),
    name="sector impact",
)

frames = []
for r in range(rounds + 1):
    cum_B = float(series[r].sum() / 1000.0)
    title = (
        f"Round {r} — {'the shock hits' if r == 0 else f'{r} upstream wave' + ('s' if r > 1 else '')}"
        f" · cumulative impact ${cum_B:,.1f}B"
        + (" (≈ total L·s)" if r == rounds else "")
    )
    frames.append(
        go.Frame(
            name=str(r),
            data=edge_traces_for(r)
            + [
                go.Scatter(
                    hovertext=node_hover_for(r),
                    marker=dict(
                        size=node_sizes(r),
                        color=node_colors(r),
                    ),
                )
            ],
            traces=list(range(N_BUCKETS + 1)),
            layout=go.Layout(title_text=title),
        )
    )

slider_steps = []
for r in range(rounds + 1):
    slider_steps.append(
        dict(
            method="animate",
            args=[
                [str(r)],
                {
                    "frame": {"duration": 850, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 450},
                },
            ],
            label=str(r),
        )
    )

fig = go.Figure(data=edge_traces_for(0) + [node_trace], frames=frames)
fig.update_layout(
    title=dict(
        text="Round 0 — the shock hits · cumulative impact "
        f"${series[0].sum() / 1000.0:,.1f}B",
        font=dict(size=15, color="#c9d2e3"),
        x=0.01,
    ),
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    hovermode="closest",
    margin=dict(l=10, r=10, t=95, b=10),
    height=640,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.0,
            y=1.14,
            xanchor="left",
            yanchor="top",
            font=dict(color="#e8ecf4"),
            bgcolor="rgba(31, 111, 139, 0.55)",
            bordercolor="rgba(232, 196, 104, 0.5)",
            buttons=[
                dict(
                    label="▶  Play ripple",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 850, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 450},
                        },
                    ],
                ),
                dict(
                    label="⏸  Pause",
                    method="animate",
                    args=[
                        [None],
                        {"frame": {"duration": 0}, "mode": "immediate"},
                    ],
                ),
            ],
        )
    ],
    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="Round ", font=dict(color="#c9d2e3")),
            pad=dict(t=55),
            font=dict(color="#9aa7bd"),
            steps=slider_steps,
        )
    ],
)

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
st.caption(
    "Node color and size = cumulative output impact of the shock "
    "(L·s after the last round). Edge width = interindustry flow pulled "
    "through that supply link. Hover any node or link for details."
)

# --------------------------------------------------------- impact breakdown ---
bar_col, table_col = st.columns([1, 1])

top_n = 15
order = np.argsort(-total)[:top_n]
bar_fig = go.Figure(
    go.Bar(
        x=(total[order] / 1000.0)[::-1],
        y=[short(tables.names[i], 32) for i in order][::-1],
        orientation="h",
        marker=dict(
            color=(total[order] / 1000.0)[::-1],
            colorscale=IMPACT_SCALE,
            showscale=False,
        ),
        hovertemplate="%{y}<br>$%{x:,.2f}B<extra></extra>",
    )
)
bar_fig.update_layout(
    title=dict(text=f"Top {top_n} sectors by total impact ($B)", font=dict(size=15, color="#c9d2e3")),
    margin=dict(l=10, r=10, t=55, b=10),
    height=470,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="#9aa7bd"),
    yaxis=dict(color="#c9d2e3"),
)
with bar_col:
    st.plotly_chart(bar_fig, use_container_width=True, config={"displaylogo": False})

impact_df = pd.DataFrame(
    {
        "Code": [tables.codes[i] for i in order],
        "Sector": [tables.names[i] for i in order],
        "Impact ($B)": total[order] / 1000.0,
        "Share of total": total[order] / total.sum(),
        "Own output ($B)": tables.industry_output[order] / 1000.0,
        "Impact vs own output": total[order] / tables.industry_output[order],
    }
).round(4)
with table_col:
    st.markdown("#### Where the shock lands")
    st.dataframe(
        impact_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Impact ($B)": st.column_config.NumberColumn(format="%.2f"),
            "Share of total": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=1.0),
            "Own output ($B)": st.column_config.NumberColumn(format="%.1f"),
            "Impact vs own output": st.column_config.NumberColumn(format="%.1%%"),
        },
        height=432,
    )
    full_df = pd.DataFrame(
        {
            "code": tables.codes,
            "sector": tables.names,
            "impact_billions": total / 1000.0,
            "share_of_total": total / total.sum(),
            "own_output_billions": tables.industry_output / 1000.0,
        }
    )
    st.download_button(
        "Download full impact table (CSV)",
        data=full_df.to_csv(index=False).encode("utf-8"),
        file_name=f"ripple_{year}_{tables.codes[sector_idx]}_impacts.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ------------------------------------------------------------- methodology ---
with st.expander("Data source & methodology"):
    st.markdown(
        f"""
**Data.** BEA *Input-Output Accounts Data* — annual Use and Make tables,
Summary level (71 sectors), {bea_io.YEARS[0]}–{bea_io.YEARS[-1]}. BEA publishes
these free releases as XLSX workbooks with one sheet per year; this app
downloads them live from `apps.bea.gov` and parses the requested
**{year}** sheet (values in millions of current dollars).

- [BEA Input-Output Accounts Data](https://www.bea.gov/industry/input-output-accounts-data)
- [Use table (after redefinitions, producers' prices)]({bea_io.USE_URL})
- [Make table (after redefinitions)]({bea_io.MAKE_URL})

**Model.** Direct requirements are built with the market-share
(fixed-industry-sales-structure) technology:

`S[i,k] = Make[i,k] / commodity output k`, `Z[i,j] = Σ_k S[i,k]·Use[k,j]`,
`A[i,j] = Z[i,j] / industry output j`.

The Leontief inverse is `L = (I − A)⁻¹` and the total output impact of a
final-demand shock `s` is `L·s`. Animation frames show the cumulative partial
sums `(I + A + … + A^r)·s` — each frame is one more round of upstream
purchases rippling through the supply network.

*Validation:* `L·f` reproduces BEA's published industry outputs to within
~0.005% for every year.
"""
    )
