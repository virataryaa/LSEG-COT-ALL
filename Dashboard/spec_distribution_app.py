"""
spec_distribution_app.py — COT Distribution (LSEG)
Companion mini-app to cot_app.py, sharing the same Database/ parquets.
Two tabs:
  1. Z-Score Matrix — every single commodity x lookback window (1/3/5/10y),
     for the chosen category's Net level and its weekly change. Always uses
     the Disagg (Futures-only) report so all 7 commodities are on equal
     footing (RC/LCC have no CIT report).
  2. Distribution — one commodity's Net / Long / Short, level + weekly
     change, as histograms with the latest data point marked. Toggle
     between raw (k lots) and % of Total OI.

Combined commodities (KRC/CLC/SLS) are out of scope here — this tool is
about single-commodity positioning distributions, not the combined-leg
views cot_app.py builds for those.
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="COT DISTRIBUTION", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  :root { color-scheme: light !important; }
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background:#ffffff !important; color:#1a1a1a !important;
  }
  [data-testid="stSidebar"] { background:#f7f8fa !important; }
  [data-testid="stHeader"]  { background:transparent !important; }
  .block-container { padding-top:1.2rem !important; max-width:1600px; }
  div[data-testid="stTabs"] button { font-size:0.85rem !important; font-weight:500; }
</style>""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_DIR     = Path(__file__).resolve().parent.parent / "Database"
CIT_FILE   = DB_DIR / "cot_cit.parquet"
FO_FILE    = DB_DIR / "cot_disagg_futopt.parquet"
FUT_FILE   = DB_DIR / "cot_disagg_fut.parquet"
ROLLEX_DIR = DB_DIR / "Rollex"

# ── Commodity config — single commodities only (no KRC/CLC/SLS combined legs) ──
COMM_COLORS = {"KC":"#1a56db","CC":"#d97706","SB":"#059669","CT":"#7c3aed",
               "RC":"#dc2626","LCC":"#0891b2","LSU":"#ea580c"}
COMM_NAMES  = {"KC":"KC : Arabica Coffee","CC":"CC : NYC Cocoa","SB":"SB : Sugar #11",
               "CT":"CT : Cotton #2","RC":"RC : Robusta Coffee","LCC":"LCC : London Cocoa",
               "LSU":"LSU : London White Sugar"}
CIT_COMMS = {"KC", "CC", "SB", "CT"}

CIT_SPEC = {
    "Large Spec":    {"long":"Spec Long",    "short":"Spec Short",    "net":"Spec Net"},
    "Non-Rep":       {"long":"Non Rep Long", "short":"Non Rep Short", "net":"Non Rep Net"},
    "Index Traders": {"long":"Index Long",   "short":"Index Short",   "net":"Index Net"},
    "Large + Small":                 {"long":"Spec+NonRep Long",  "short":"Spec+NonRep Short",  "net":"Spec+NonRep Net"},
    "Large Spec + Index + Non-Rep":  {"long":"Combined Spec Long","short":"Combined Spec Short","net":"Combined Spec Net"},
    "Commercial":    {"long":"Comm Long",    "short":"Comm Short",    "net":"Comm Net"},
}
DISAGG_SPEC = {
    "Managed Money": {"long":"MM Long",       "short":"MM Short",       "net":"MM Net"},
    "Other Rept":    {"long":"Other Long",    "short":"Other Short",    "net":"Other Net"},
    "Non-Rep":       {"long":"Non Rep Long",  "short":"Non Rep Short",  "net":"Non Rep Net"},
    "Swap Dealers":  {"long":"Swap Long",     "short":"Swap Short",     "net":"Swap Net"},
    "MM + Other + Non-Rep":         {"long":"MM+Other+NonRep Long","short":"MM+Other+NonRep Short","net":"MM+Other+NonRep Net"},
    "MM + Non-Rep":                 {"long":"MM+NonRep Long",      "short":"MM+NonRep Short",      "net":"MM+NonRep Net"},
    "Commercial (Producer)": {"long":"Producer Long", "short":"Producer Short", "net":"Comm Net"},
}
LOOKBACKS = [1, 3, 5, 10]

C_LONG, C_SHORT, C_NET = "#16a34a", "#dc2626", "#1a56db"


# ── Data loading (same derivation logic as cot_app.py) ─────────────────────────
def _derive_nets(df):
    pairs = [
        ("Spec Long", "Spec Short", "Spec Net"), ("Comm Long", "Comm Short", "Comm Net"),
        ("MM Long", "MM Short", "MM Net"), ("Swap Long", "Swap Short", "Swap Net"),
        ("Other Long", "Other Short", "Other Net"), ("Producer Long", "Producer Short", "Comm Net"),
    ]
    for l, s, n in pairs:
        if l in df.columns and s in df.columns and n not in df.columns:
            df[n] = df[l] - df[s]
    return df

def _add_pct(df):
    for col in list(df.columns):
        pct_col = f"Pct OI {col}"
        if (pct_col not in df.columns and "Total OI" in df.columns
                and col not in ("Date", "Commodity", "Crop", "Px")
                and not col.startswith(("Traders", "Conc", "Pct OI"))):
            try:
                df[pct_col] = (df[col] / df["Total OI"] * 100).round(2)
            except Exception:
                pass
    return df

@st.cache_data(ttl=600)
def load_cit() -> pd.DataFrame:
    df = pd.read_parquet(CIT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date", "Commodity", "Crop")]
    df[num] = df[num].astype(float)
    df = _derive_nets(df)
    # Combined spec (CIT) = Large Spec + Non-Rep + Index
    for side in ("Long", "Short"):
        df[f"Combined Spec {side}"] = (
            df.get(f"Spec {side}", 0) + df.get(f"Non Rep {side}", 0) + df.get(f"Index {side}", 0)
        )
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    # Large Spec + Non-Rep (excl. Index)
    for side in ("Long", "Short"):
        df[f"Spec+NonRep {side}"] = df.get(f"Spec {side}", 0) + df.get(f"Non Rep {side}", 0)
    df["Spec+NonRep Net"] = df["Spec+NonRep Long"] - df["Spec+NonRep Short"]
    df = _add_pct(df)
    return df.sort_values(["Commodity", "Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_disagg(version: str) -> pd.DataFrame:
    path = FO_FILE if version == "F&O" else FUT_FILE
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date", "Commodity", "Crop")]
    df[num] = df[num].astype(float)
    df = _derive_nets(df)
    # Combined spec (Disagg) = MM + Other + Non-Rep + Swap
    for side in ("Long", "Short"):
        df[f"Combined Spec {side}"] = (
            df.get(f"MM {side}", 0) + df.get(f"Other {side}", 0)
            + df.get(f"Non Rep {side}", 0) + df.get(f"Swap {side}", 0)
        )
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    # MM + Other + Non-Rep (excl. Swap)
    for side in ("Long", "Short"):
        df[f"MM+Other+NonRep {side}"] = (
            df.get(f"MM {side}", 0) + df.get(f"Other {side}", 0) + df.get(f"Non Rep {side}", 0)
        )
    df["MM+Other+NonRep Net"] = df["MM+Other+NonRep Long"] - df["MM+Other+NonRep Short"]
    # MM + Non-Rep only (excl. Other, Swap)
    for side in ("Long", "Short"):
        df[f"MM+NonRep {side}"] = df.get(f"MM {side}", 0) + df.get(f"Non Rep {side}", 0)
    df["MM+NonRep Net"] = df["MM+NonRep Long"] - df["MM+NonRep Short"]
    df = _add_pct(df)
    return df.sort_values(["Commodity", "Crop", "Date"]).reset_index(drop=True)


@st.cache_data(ttl=600)
def load_rollex(commodity: str) -> pd.DataFrame:
    path = ROLLEX_DIR / f"rollex_{commodity}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["Date", "rollex_px"])
    r = pd.read_parquet(path, columns=["rollex_px"]).reset_index()
    r = r.rename(columns={r.columns[0]: "Date"})
    r["Date"] = pd.to_datetime(r["Date"])
    return r[["Date", "rollex_px"]]


def _series_for(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df.set_index("Date")[col], errors="coerce").dropna()
    return s

def _zscore(series: pd.Series, years: int) -> float:
    if series.empty:
        return np.nan
    cutoff = series.index.max() - pd.DateOffset(years=years)
    window = series[series.index >= cutoff]
    if len(window) < 5 or window.std(ddof=0) == 0 or pd.isna(window.std(ddof=0)):
        return np.nan
    return float((series.iloc[-1] - window.mean()) / window.std(ddof=0))

def _style_z(v):
    if pd.isna(v):
        return ""
    v = max(-3, min(3, v))
    if v >= 0:
        r, g, b = 255 - int(v/3*105), 235 - int(v/3*20), 130
    else:
        r, g, b = 250, 150 + int((v+3)/3*85), 120 + int((v+3)/3*60)
    return f"background-color:rgb({r},{g},{b});color:#1a1a2e"


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.05rem;font-weight:700;color:#1a56db;"
        "margin-bottom:16px;letter-spacing:.01em'>COT DISTRIBUTION</div>",
        unsafe_allow_html=True)

    commodity = st.selectbox("Commodity", list(COMM_NAMES.keys()),
                             format_func=lambda x: COMM_NAMES[x], key="sb_commodity")
    cit_ok = commodity in CIT_COMMS
    if cit_ok:
        report = st.radio("Report", ["CIT", "Disagg"], horizontal=True, key="rb_report")
    else:
        report = "Disagg"
        st.caption("RC/LCC — Disaggregated only")

    version_key = "Fut"
    if report == "Disagg":
        version = st.radio("Version", ["Fut only", "F&O combined"], horizontal=True, key="rb_version")
        version_key = "Fut" if "Fut" in version else "F&O"

    cat_map = CIT_SPEC if report == "CIT" else DISAGG_SPEC
    category = st.selectbox("Category", list(cat_map.keys()), key="sb_category")


# ── Load ─────────────────────────────────────────────────────────────────────
if report == "CIT":
    raw = load_cit()
    df = raw[raw["Commodity"] == commodity].sort_values("Date").reset_index(drop=True)
else:
    raw = load_disagg(version_key)
    df = raw[(raw["Commodity"] == commodity) & (raw["Crop"] == "All")].sort_values("Date").reset_index(drop=True)

st.title("COT Distribution")
st.caption(f"{COMM_NAMES[commodity]}  ·  {report}{'' if report == 'CIT' else ' (' + version_key + ')'}  ·  {category}")

cols = cat_map[category]
missing = [c for c in cols.values() if c not in df.columns]
if df.empty or missing:
    st.warning(f"No data available for {commodity} / {report} / {category}.")
    st.stop()

st.caption(
    f"**Latest COT date:** {df['Date'].max().strftime('%d %b %Y')}  |  "
    f"**Full history:** {df['Date'].min().strftime('%d %b %Y')} → {df['Date'].max().strftime('%d %b %Y')}"
)


tab_dist, tab_matrix = st.tabs(["Distribution", "Z-Score Matrix"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Z-SCORE MATRIX (all single commodities, Disagg Fut, chosen category)
# ══════════════════════════════════════════════════════════════════════════════
with tab_matrix:
    st.caption(
        "All 7 single commodities, Disaggregated (Futures-only) report — so RC/LCC/LSU "
        "(no CIT report) sit on the same basis as KC/CC/SB/CT. Category applies across the board."
    )
    _matrix_ref = load_disagg("Fut")
    st.caption(f"**Latest COT date:** {_matrix_ref['Date'].max().strftime('%d %b %Y')}")
    matrix_category = st.selectbox(
        "Category (matrix)", list(DISAGG_SPEC.keys()), key="sb_matrix_category"
    )
    mcols = DISAGG_SPEC[matrix_category]

    level_rows, chg_rows = {}, {}
    for cmm in COMM_NAMES:
        d = load_disagg("Fut")
        d = d[(d["Commodity"] == cmm) & (d["Crop"] == "All")].sort_values("Date")
        net_col = mcols["net"]
        if net_col not in d.columns or d.empty:
            level_rows[cmm] = {y: np.nan for y in LOOKBACKS}
            chg_rows[cmm]   = {y: np.nan for y in LOOKBACKS}
            continue
        level_s = _series_for(d, net_col)
        chg_s   = level_s.diff().dropna()
        level_rows[cmm] = {y: _zscore(level_s, y) for y in LOOKBACKS}
        chg_rows[cmm]   = {y: _zscore(chg_s, y) for y in LOOKBACKS}

    level_df = pd.DataFrame(level_rows).T[LOOKBACKS]
    chg_df   = pd.DataFrame(chg_rows).T[LOOKBACKS]
    level_df.index = [COMM_NAMES[c].split(" : ")[1] for c in level_df.index]
    chg_df.index   = [COMM_NAMES[c].split(" : ")[1] for c in chg_df.index]

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"**{matrix_category} Net — Z-score**")
        st.dataframe(level_df.style.map(_style_z).format("{:.2f}"), use_container_width=True)
    with mc2:
        st.markdown(f"**{matrix_category} Weekly Change — Z-score**")
        st.dataframe(chg_df.style.map(_style_z).format("{:.2f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTION (single commodity, sidebar category, level + weekly change)
# ══════════════════════════════════════════════════════════════════════════════
with tab_dist:
    uc1, uc2 = st.columns([1, 3])
    with uc1:
        unit = st.radio("Units", ["Raw (k lots)", "% of Total OI"], key="rb_unit")
    with uc2:
        lb_choice = st.radio("History window", ["All", "1y", "3y", "5y", "10y"],
                              horizontal=True, key="rb_lookback")

    d = df.copy()
    if lb_choice != "All":
        years = int(lb_choice.replace("y", ""))
        cutoff = d["Date"].max() - pd.DateOffset(years=years)
        d = d[d["Date"] >= cutoff]

    st.caption(
        f"Study period: **{d['Date'].min().strftime('%d %b %Y')} → {d['Date'].max().strftime('%d %b %Y')}** "
        f"({len(d)} weekly observations)"
    )

    use_pct = unit.startswith("%")
    def _col(base):
        return f"Pct OI {base}" if use_pct else base
    unit_label = "% of OI" if use_pct else "k lots"
    unit_div   = 1.0 if use_pct else 1000.0

    metrics = [("Net", cols["net"], C_NET), ("Long", cols["long"], C_LONG), ("Short", cols["short"], C_SHORT)]

    # Pre-compute every series first so subplot titles can bake in the
    # latest value — avoids stacking a separate annotation on top of the
    # title, which is what was colliding with the chart in the first version.
    panel_data = {}  # (row, col) -> (series, color)
    for i, (name, base_col, color) in enumerate(metrics, start=1):
        raw_col = _col(base_col)
        if raw_col not in d.columns:
            continue
        level_s = pd.to_numeric(d[raw_col], errors="coerce").dropna() / unit_div
        chg_s   = level_s.diff().dropna()
        panel_data[(1, i)] = (level_s, color, f"{name} — Level ({unit_label})")
        panel_data[(2, i)] = (chg_s,   color, f"{name} — Weekly Δ ({unit_label})")

    def _latest_date_str(series):
        # series keeps d's original row index through dropna/diff, so the
        # last surviving index maps straight back to that row's real Date —
        # not necessarily d["Date"].max(), since a column can be null on
        # the very latest report while an earlier column still has data.
        if series.empty:
            return None
        dt = d.loc[series.index[-1], "Date"]
        return pd.Timestamp(dt).strftime("%d %b %Y")

    def _auto_bin(series_list, target_bins=60):
        """Pick a round-ish bin width from the combined range of a row's series."""
        all_vals = pd.concat([s for s in series_list if not s.empty])
        if all_vals.empty:
            return 1.0
        span = all_vals.max() - all_vals.min()
        raw = span / target_bins if span > 0 else 1.0
        magnitude = 10 ** np.floor(np.log10(raw))
        for m in (1, 2, 2.5, 5, 10):
            if raw <= m * magnitude:
                return round(m * magnitude, 6)
        return round(10 * magnitude, 6)

    # Bin sizes are no longer user-editable — always the auto-computed
    # default (0.5% per bucket in %OI mode, or a round-ish width targeting
    # ~60 bins in raw k-lots mode).
    if use_pct:
        level_bin = chg_bin = 0.5
    else:
        level_bin = _auto_bin([panel_data[k][0] for k in panel_data if k[0] == 1])
        chg_bin   = _auto_bin([panel_data[k][0] for k in panel_data if k[0] == 2])

    y_mode = st.radio("Y-axis", ["% of weeks", "Raw count"], key="rb_y_mode", horizontal=True)
    bin_by_row = {1: level_bin, 2: chg_bin}
    hist_norm  = "percent" if y_mode == "% of weeks" else None
    y_axis_title = "% of weeks" if y_mode == "% of weeks" else "Weeks (count)"

    def _title_with_latest(key):
        if key not in panel_data or panel_data[key][0].empty:
            return panel_data.get(key, (None, None, ""))[2]
        series, _, base_title = panel_data[key]
        date_str = _latest_date_str(series)
        date_part = f" ({date_str})" if date_str else ""
        return f"{base_title}   ·   latest{date_part} {series.iloc[-1]:,.1f}"

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[_title_with_latest((1, i)) for i in range(1, 4)] +
                        [_title_with_latest((2, i)) for i in range(1, 4)],
        vertical_spacing=0.22, horizontal_spacing=0.07,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 12

    for (row, col), (series, color, _) in panel_data.items():
        if series.empty:
            continue
        fig.add_trace(go.Histogram(
            x=series.values, xbins=dict(size=bin_by_row[row]), marker_color=color,
            marker_line=dict(color="white", width=1),
            histnorm=hist_norm, opacity=0.85, showlegend=False,
        ), row=row, col=col)
        fig.add_vline(x=series.iloc[-1], line_dash="dash", line_color="#1a1a2e",
                      line_width=2, row=row, col=col)

    fig.update_layout(
        height=700, template="plotly_white", bargap=0.04,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
        margin=dict(l=30, r=30, t=70, b=40),
        font=dict(family="-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif", size=11),
    )
    fig.update_yaxes(title_text=y_axis_title, title_font_size=10)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Distribution of {category} Net/Long/Short — level and week-over-week change — "
        f"over the selected history window. Dashed line marks the latest data point "
        f"({d['Date'].max().strftime('%d %b %Y') if not d.empty else '—'})."
    )

    # ── Rollex Price — Weekly % Change histogram ────────────────────────────
    st.divider()
    st.markdown("**Rollex Price Weekly % Change**")
    rollex_df = load_rollex(commodity)
    if rollex_df.empty:
        st.info("Rollex price data not available for this commodity.")
    else:
        px_lvl = (d[["Date"]].merge(rollex_df, on="Date", how="inner")
                  .sort_values("Date").reset_index(drop=True))
        px_level_s = px_lvl["rollex_px"].dropna()
        px_chg_s   = (px_level_s.pct_change() * 100).dropna()  # weekly % change, not absolute

        if px_chg_s.empty:
            st.info("No overlapping weeks between COT dates and Rollex price data.")
        else:
            px_chg_bin = _auto_bin([px_chg_s])
            px_latest_date = px_lvl.loc[px_chg_s.index[-1], "Date"]
            px_date_str = pd.Timestamp(px_latest_date).strftime("%d %b %Y")
            title = f"Weekly Price Change %   ·   latest ({px_date_str}) {px_chg_s.iloc[-1]:+.2f}%"

            fig_px = go.Figure()
            fig_px.add_trace(go.Histogram(
                x=px_chg_s.values, xbins=dict(size=px_chg_bin), marker_color="#f59e0b",
                marker_line=dict(color="white", width=1), histnorm=hist_norm,
                opacity=0.85, showlegend=False,
            ))
            fig_px.add_vline(x=px_chg_s.iloc[-1], line_color="#1a1a2e", line_width=2)

            fig_px.update_layout(
                title=dict(text=title, font=dict(size=13)),
                height=340, template="plotly_white", bargap=0.04,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
                margin=dict(l=30, r=30, t=50, b=40),
                font=dict(family="-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif", size=11),
            )
            fig_px.update_yaxes(title_text=y_axis_title, title_font_size=10, showgrid=True, gridcolor="rgba(0,0,0,0.06)")
            fig_px.update_xaxes(title_text="Weekly Price Change %", showgrid=False)
            px_col, _spacer = st.columns([1, 1])
            with px_col:
                st.plotly_chart(fig_px, use_container_width=True)
            st.caption(
                "Distribution of the Rollex (roll-adjusted) price's week-over-week % change, "
                "over the same study period and date range as the positioning histograms above. "
                f"Solid line marks the latest value ({px_lvl['Date'].iloc[-1].strftime('%d %b %Y')})."
            )
