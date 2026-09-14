"""
cot_app.py — ICEBREAKER COT Dashboard
Run: streamlit run cot_app.py
"""

import datetime
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="COMPREHENSIVE COT", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  :root { color-scheme: light !important; }
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background:#ffffff !important; color:#1a1a1a !important;
  }
  [data-testid="stSidebar"] { background:#f7f8fa !important; }
  [data-testid="stHeader"]  { background:transparent !important; }
  .block-container { padding-top:1.2rem !important; max-width:1600px; }
  div[data-testid="stExpander"] {
    border:1px solid #e0e4ed !important; border-radius:7px !important;
  }
  div[data-testid="stTabs"] button { font-size:0.81rem !important; font-weight:500; }
  div[data-testid="stTabs"] button:nth-child(8),
  div[data-testid="stTabs"] button:nth-child(9),
  div[data-testid="stTabs"] button:nth-child(10),
  div[data-testid="stTabs"] button:nth-child(11) {
    background-color:#f3f4f6 !important;
    border-radius:6px 6px 0 0 !important;
  }
  div[data-testid="stTabs"] button:nth-child(12) {
    background-color:#fce7f3 !important;
    border-radius:6px 6px 0 0 !important;
  }
  div[data-testid="stTabs"] button:nth-child(12) p {
    color:#be185d !important; font-weight:600 !important;
  }
  div[data-testid="stTabs"] button:nth-child(13) {
    background-color:#ede9fe !important;
    border-radius:6px 6px 0 0 !important;
  }
  div[data-testid="stTabs"] button:nth-child(13) p {
    color:#6d28d9 !important; font-weight:600 !important;
  }
  hr { border:none !important; border-top:1px solid #e8e8ed !important; margin:.5rem 0 !important; }
  [data-testid="stRadio"] label { font-size:.82rem !important; }
</style>""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_DIR      = Path(__file__).resolve().parent.parent / "Database"
CIT_FILE    = DB_DIR / "cot_cit.parquet"
FO_FILE     = DB_DIR / "cot_disagg_futopt.parquet"
FUT_FILE    = DB_DIR / "cot_disagg_fut.parquet"
ROLLEX_DIR  = DB_DIR / "Rollex"
ROLLEX_MAP  = {
    "KC": "rollex_KC.parquet",
    "CC": "rollex_CC.parquet",
    "CT": "rollex_CT.parquet",
    "SB": "rollex_SB.parquet",
    "RC": "rollex_RC.parquet",
    "LCC":"rollex_LCC.parquet",
    "LSU":"rollex_LSU.parquet",
}
VAR_LOT_USD = {"KC":375, "CC":10, "SB":1120, "CT":500, "RC":10, "LCC":10, "LSU":50,
               "GC":100, "SI":5000, "HG":25000}  # = CONTRACT_SIZE, /100 already folded in for CENTS_QUOTED ones
_CONF_Z     = 2.3263

# ── Commodity config ──────────────────────────────────────────────────────────
COMM_COLORS = {
    "KC":"#1a56db","CC":"#d97706","SB":"#059669",
    "CT":"#7c3aed","RC":"#dc2626","LCC":"#0891b2",
    "LSU":"#ea580c","GC":"#ca8a04","SI":"#64748b","HG":"#b45309",
    "KRC":"#6d28d9","CLC":"#0f766e","SLS":"#a16207",
}
COMM_NAMES = {
    "KC":"KC : Arabica Coffee","CC":"CC : NYC Cocoa",
    "SB":"SB : Sugar #11","CT":"CT : Cotton #2",
    "RC":"RC : Robusta Coffee","LCC":"LCC : London Cocoa",
    "LSU":"LSU : London White Sugar",
    "GC":"GC : Gold","SI":"SI : Silver","HG":"HG : Copper",
    "KRC":"KC + RC : Combined Coffee",
    "CLC":"CC + LCC : Combined Cocoa",
    "SLS":"SB + LSU : Combined Sugar",
}
CONTRACT_SIZE = {"KC":37500,"CC":10,"SB":112000,"CT":50000,"RC":10,"LCC":10,"LSU":50,
                 "GC":100,"SI":5000,"HG":25000,"KRC":1,"CLC":1,"SLS":1}
CONTRACT_UNIT = {"KC":"lbs","CC":"MT","SB":"lbs","CT":"lbs","RC":"MT","LCC":"MT","LSU":"MT",
                 "GC":"oz","SI":"oz","HG":"lbs","KRC":"lots","CLC":"lots","SLS":"lots"}
CIT_COMMS     = {"KC","CC","SB","CT"}  # metals have no CIT/Index Traders category (verified), same as RC/LCC/LSU
# Commodities whose LSEG price feed is cents/lb (needs /100 for $ nominal) —
# deliberately NOT the same set as CONTRACT_UNIT=="lbs": Copper (HG) is
# physically "lbs" too but LSEG quotes HGc2 directly in $/lb already
# (verified 2026-08-28: HGc2 ~6.59, not ~659 — cents/lb would put copper
# under 7 cents a pound, which has never happened). Do not add HG here.
CENTS_QUOTED  = {"KC","SB","CT"}
COMBINED_COMMS = {"KRC","CLC","SLS"}
COMBINED_MAP   = {"KRC":("KC","RC"), "CLC":("CC","LCC"), "SLS":("SB","LSU")}

C_LONG  = "#16a34a"
C_SHORT = "#dc2626"
C_NET   = "#1a56db"
C_PRICE = "#f59e0b"
C_OLD   = "#e67e22"
C_NEW   = "#2980b9"
GRAY    = "#6e6e73"

CROP_START_MONTH = 9
_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
CROP_WEEK_TICKS  = {1:"Sep",5:"Oct",9:"Nov",14:"Dec",18:"Jan",22:"Feb",
                    26:"Mar",30:"Apr",35:"May",39:"Jun",43:"Jul",48:"Aug"}
MONTH_TICKS      = {1:"Jan",5:"Feb",9:"Mar",14:"Apr",18:"May",23:"Jun",
                    27:"Jul",32:"Aug",36:"Sep",40:"Oct",45:"Nov",49:"Dec"}

# ── Plot base ─────────────────────────────────────────────────────────────────
_BASE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif",
              color="#1a1a1a", size=11),
)

def _ax(x=False):
    b = dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
             zeroline=True, zerolinecolor="rgba(0,0,0,0.12)", zerolinewidth=1,
             showline=True, linecolor="rgba(0,0,0,0.08)", linewidth=1,
             tickfont=dict(size=10, color="#666"))
    if x:
        b.update(showgrid=False, tickangle=-35, nticks=20, hoverformat="%d %b %Y")
    return b


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _derive_nets(df, cols):
    pairs = [
        ("Spec Long",     "Spec Short",     "Spec Net"),
        ("Index Long",    "Index Short",    "Index Net"),
        ("Non Rep Long",  "Non Rep Short",  "Non Rep Net"),
        ("Comm Long",     "Comm Short",     "Comm Net"),
        ("MM Long",       "MM Short",       "MM Net"),
        ("Swap Long",     "Swap Short",     "Swap Net"),
        ("Other Long",    "Other Short",    "Other Net"),
        ("Producer Long", "Producer Short", "Comm Net"),
    ]
    for l, s, n in pairs:
        if l in df.columns and s in df.columns and n not in df.columns:
            df[n] = df[l] - df[s]
    return df

def _add_pct(df):
    for col in list(df.columns):
        pct_col = f"Pct OI {col}"
        if (pct_col not in df.columns and "Total OI" in df.columns
                and col not in ("Date","Commodity","Crop","Px")
                and not col.startswith("Traders") and not col.startswith("Conc")
                and not col.startswith("Pct OI")):
            try:
                df[pct_col] = (df[col] / df["Total OI"] * 100).round(2)
            except Exception:
                pass
    return df

@st.cache_data(ttl=600)
def load_cit() -> pd.DataFrame:
    df = pd.read_parquet(CIT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date","Commodity","Crop")]
    df[num] = df[num].astype(float)
    df = _derive_nets(df, df.columns)
    # Combined spec = Large Spec + Non Rep + Index
    for side in ("Long","Short"):
        df[f"Combined Spec {side}"] = (
            df.get(f"Spec {side}", 0) +
            df.get(f"Non Rep {side}", 0) +
            df.get(f"Index {side}", 0)
        )
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    # Large Spec + Non-Rep (excl. Index)
    for side in ("Long","Short"):
        df[f"Spec+NonRep {side}"] = (
            df.get(f"Spec {side}", 0) +
            df.get(f"Non Rep {side}", 0)
        )
    df["Spec+NonRep Net"] = df["Spec+NonRep Long"] - df["Spec+NonRep Short"]
    df = _add_pct(df)
    return df.sort_values(["Commodity","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_disagg(version: str) -> pd.DataFrame:
    path = FO_FILE if version == "F&O" else FUT_FILE
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date","Commodity","Crop")]
    df[num] = df[num].astype(float)
    df = _derive_nets(df, df.columns)
    # Combined spec (Disagg) = MM + Other + Non Rep + Swap
    for side in ("Long","Short"):
        df[f"Combined Spec {side}"] = (
            df.get(f"MM {side}", 0) +
            df.get(f"Other {side}", 0) +
            df.get(f"Non Rep {side}", 0) +
            df.get(f"Swap {side}", 0)
        )
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    # MM + Other + Non-Rep (excl. Swap)
    for side in ("Long","Short"):
        df[f"MM+Other+NonRep {side}"] = (
            df.get(f"MM {side}", 0) +
            df.get(f"Other {side}", 0) +
            df.get(f"Non Rep {side}", 0)
        )
    df["MM+Other+NonRep Net"] = df["MM+Other+NonRep Long"] - df["MM+Other+NonRep Short"]
    df = _add_pct(df)
    return df.sort_values(["Commodity","Crop","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_options_only() -> pd.DataFrame:
    """Options Only = F&O Combined minus Futures Only (numeric cols only)."""
    fo  = pd.read_parquet(FO_FILE)
    fut = pd.read_parquet(FUT_FILE)
    fo["Date"]  = pd.to_datetime(fo["Date"])
    fut["Date"] = pd.to_datetime(fut["Date"])
    id_cols = ["Date","Commodity","Crop"]
    num_fo  = [c for c in fo.columns  if c not in id_cols]
    num_fut = [c for c in fut.columns if c not in id_cols]
    num_both = [c for c in num_fo if c in num_fut]
    merged = fo.merge(fut[id_cols + num_both], on=id_cols, how="left", suffixes=("","_fut"))
    for c in num_both:
        merged[c] = pd.to_numeric(merged[c], errors="coerce") - pd.to_numeric(merged[f"{c}_fut"], errors="coerce")
        merged.drop(columns=[f"{c}_fut"], inplace=True)
    df = merged.copy()
    df = _derive_nets(df, df.columns)
    for side in ("Long","Short"):
        df[f"Combined Spec {side}"] = (
            df.get(f"MM {side}", 0) + df.get(f"Other {side}", 0) +
            df.get(f"Non Rep {side}", 0) + df.get(f"Swap {side}", 0))
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    for side in ("Long","Short"):
        df[f"MM+Other+NonRep {side}"] = (
            df.get(f"MM {side}", 0) + df.get(f"Other {side}", 0) + df.get(f"Non Rep {side}", 0))
    df["MM+Other+NonRep Net"] = df["MM+Other+NonRep Long"] - df["MM+Other+NonRep Short"]
    df = _add_pct(df)
    # Trader counts and concentration % are not valid for options-only
    # (trader overlap between fut/options means subtraction gives wrong counts;
    #  concentration % uses different OI bases so subtracting is meaningless)
    invalid_cols = [c for c in df.columns if
                    c.startswith("Traders") or c.startswith("Conc")]
    df[invalid_cols] = np.nan
    return df.sort_values(["Commodity","Crop","Date"]).reset_index(drop=True)


@st.cache_data(ttl=600)
def load_roll_yield() -> pd.DataFrame:
    path = DB_DIR / "RollYield" / "roll_yield_data.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["Date","Commodity","roll_yield_pct"])
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["roll_yield_pct"] = df["Roll_Yield_1yr"] * 100
    return df[["Date","Commodity","roll_yield_pct"]].sort_values(["Commodity","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_rollex(commodity: str) -> pd.DataFrame:
    fname = ROLLEX_MAP.get(commodity)
    if fname is None:
        return pd.DataFrame(columns=["Date","rollex_px","rollex_ret"])
    path = ROLLEX_DIR / fname
    if not path.exists():
        return pd.DataFrame(columns=["Date","rollex_px","rollex_ret"])
    _base = ["rollex_px", "rollex_ret"]
    try:
        df = pd.read_parquet(path, columns=_base + ["active_label"])
    except Exception:
        df = pd.read_parquet(path, columns=_base)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.reset_index()   # index named "Date" → column "Date"
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=600)
def load_rollex_ohlc(commodity: str) -> pd.DataFrame:
    """Daily OHLC continuous price, for the COT nowcast candlestick charts."""
    fname = ROLLEX_MAP.get(commodity)
    if fname is None:
        return pd.DataFrame()
    path = ROLLEX_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    _cols = ["rollex_open", "rollex_high", "rollex_low", "rollex_px"]
    df = pd.read_parquet(path, columns=_cols)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.reset_index()
    return df.dropna(subset=_cols).sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def _build_var_df(commodity: str) -> pd.DataFrame:
    rx = load_rollex(commodity)
    if rx.empty:
        return pd.DataFrame()
    rx = rx.sort_values("Date").reset_index(drop=True)
    for w in [20, 60, 120]:
        rx[f"vol_{w}"] = rx["rollex_ret"].rolling(w, min_periods=max(5, w // 4)).std()
    return rx.reset_index(drop=True)


def _inject_rollex(cot_df: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Replace Px with rollex_px via backward merge_asof on Date."""
    rx = load_rollex(commodity)
    if rx.empty:
        return cot_df
    cot_sorted = cot_df.sort_values("Date").copy()
    merged = pd.merge_asof(
        cot_sorted,
        rx[["Date","rollex_px","rollex_ret"]],
        on="Date",
        direction="backward"
    )
    merged["Px"]         = merged["rollex_px"]
    merged["Rollex_ret"] = merged["rollex_ret"]
    merged = merged.drop(columns=["rollex_px","c1","rollex_ret"], errors="ignore")
    return merged.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi_row(items: list, color: str):
    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 16px'>"
    for lbl, val, sub in items:
        sc = ("#16a34a" if sub and sub.startswith("▲") else
              "#dc2626" if sub and sub.startswith("▼") else "#888")
        sh = (f"<span style='font-size:.63rem;color:{sc};margin-left:4px'>{sub}</span>"
              if sub else "")
        html += (
            f"<div style='background:rgba({r},{g},{b},0.06);"
            f"border:1px solid rgba({r},{g},{b},0.18);border-radius:10px;"
            f"padding:8px 16px;min-width:115px;display:flex;flex-direction:column'>"
            f"<span style='font-size:.56rem;color:#888;text-transform:uppercase;"
            f"letter-spacing:.08em;margin-bottom:3px'>{lbl}</span>"
            f"<span style='font-size:.93rem;font-weight:700;color:{color}'>{val}{sh}</span>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _val(row, col, unit):
    v = row.get(col, np.nan)
    if pd.isna(v): return "—"
    if unit == "k lots":
        return f"{v/1000:.1f}k"
    pc = row.get(f"Pct OI {col}", np.nan)
    return f"{pc:.1f}%" if pd.notna(pc) else f"{v/1000:.1f}k"

def _chg(curr_row, prev_row, col, unit):
    v, p = curr_row.get(col, np.nan), prev_row.get(col, np.nan)
    if pd.isna(v) or pd.isna(p): return ""
    d = (v - p) / 1000
    return f"{'▲' if d>0 else '▼'}{abs(d):.1f}k"

def _px_kpi(curr_row, prev_row):
    v = curr_row.get("Px", np.nan)
    p = prev_row.get("Px", np.nan)
    if pd.isna(v): return "—", ""
    s = f"{v:.2f}"
    chg = "" if pd.isna(p) else (
        f"{'▲' if v>p else '▼'}{abs(v-p):.2f} ({abs((v-p)/p*100):.1f}%)" if p else "")
    return s, chg

def _get_y(d, col, unit):
    if unit == "k lots":
        return d[col] / 1000 if col in d.columns else pd.Series(dtype=float)
    pc = f"Pct OI {col}"
    if pc in d.columns: return d[pc]
    if "Total OI" in d.columns and col in d.columns:
        return (d[col] / d["Total OI"] * 100).round(2)
    return pd.Series(dtype=float)

def show_table(d: pd.DataFrame, pos_cols: list, chg_cols: list, label: str, n=60, scale=True):
    with _exp(label, expanded=False):
        avail_p = [c for c in pos_cols if c and c in d.columns]
        avail_c = [c for c in chg_cols if c and c in d.columns]
        src = d.sort_values("Date", ascending=False).head(n).copy()
        dates = pd.to_datetime(src["Date"]).dt.strftime("%d %b '%y").tolist()

        # build column data — positions and deltas in k lots (unless scale=False)
        col_data = {}
        for col in avail_p:
            if col == "Px" or not scale:
                col_data[col] = src[col].values
            else:
                col_data[col] = (src[col] / 1000).values
        for col in avail_c:
            if scale:
                col_data[f"Δ {col}"] = (src[col].diff(-1) / 1000).values
            else:
                col_data[f"Δ {col}"] = src[col].diff(-1).values
        if "Px" in avail_p:
            col_data["Px Δ%"] = src["Px"].pct_change(-1).mul(100).values
        if "Total OI" in d.columns and "Total OI" not in avail_p:
            col_data["OI (k)"] = (src["Total OI"] / 1000).values

        signed_cols = {c for c in col_data if c.startswith("Δ") or c == "Px Δ%"}
        px_cols     = {"Px"}
        pct_cols    = {"Px Δ%"}

        def _fmt(col, v):
            if pd.isna(v): return "—"
            if col in pct_cols:    return f"{v:+.2f}%"
            if col in signed_cols: return f"{v:+,.1f}"
            if col in px_cols:     return f"{v:.2f}"
            return f"{v:,.1f}"

        headers = list(col_data.keys())

        hdr_html = "<tr><th class='idx sub'>Date</th>"
        for h in headers:
            lbl = h if not scale or h in ("Px", "Px Δ%", "OI (k)") or h.startswith("Δ") else f"{h} (k)"
            hdr_html += f"<th class='sub'>{lbl}</th>"
        hdr_html += "</tr>"

        body_html = ""
        for i, date in enumerate(dates):
            body_html += f"<tr><td class='idx'>{date}</td>"
            for col in headers:
                v = col_data[col][i]
                txt = _fmt(col, v)
                if col in signed_cols or col in pct_cols:
                    try:
                        fv = float(v)
                        cls = "rpos" if fv > 0 else ("rneg" if fv < 0 else "")
                    except: cls = ""
                else:
                    cls = ""
                body_html += f"<td class='{cls}'>{txt}</td>"
            body_html += "</tr>"

        html = (f"{_RECAP_CSS}<div style='overflow-x:auto;overflow-y:auto;max-height:480px;margin-bottom:6px'>"
                f"<table class='rtbl'><thead>{hdr_html}</thead><tbody>{body_html}</tbody></table></div>")
        st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMMODITY-PREFIXED EXPANDERS / CHART TITLES
# The page header was removed; instead every expander label and chart title
# starts with the sidebar commodity code ("KC  ·  ..."). Labels that already
# name a commodity are left alone. Multi-commodity views call st.expander /
# st.plotly_chart directly so they aren't mislabelled.
# ══════════════════════════════════════════════════════════════════════════════
def _comm_prefix(text):
    code = st.session_state.get("sb_commodity")
    if not code or not text:
        return text
    plain = re.sub(r"<[^>]+>", "", str(text))
    if re.search(r"\b" + re.escape(code) + r"\b", plain) or COMM_NAMES[code].split(" : ")[1] in plain:
        return text
    return f"{code}  ·  {text}"

def _exp(label, *args, **kwargs):
    return st.expander(_comm_prefix(label), *args, **kwargs)

def _chart(fig, *args, **kwargs):
    title = getattr(getattr(getattr(fig, "layout", None), "title", None), "text", None)
    if title:
        fig.update_layout(title_text=_comm_prefix(title))
    return st.plotly_chart(fig, *args, **kwargs)

# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def rollex_toggle(key):
    """Per-tab switch for the Rollex Px overlay. Off = no price trace AND no
    right axis — a legend-hidden trace still leaves an empty secondary axis
    (default -1..4 range) that doesn't line up with the left scale."""
    return st.toggle("Show Rollex Px", value=False, key=key)

def _has_price(d):
    return "Px" in d.columns and not d["Px"].isna().all()

def _add_price(fig, d, secondary_y=True):
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["Px"], name="Rollex Px",
        line=dict(color=C_PRICE, width=1.2, dash="dot"), opacity=0.65,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Rollex Px: %{y:.2f}<extra></extra>",
    ), secondary_y=secondary_y)

def timeseries(d, series, title, ylabel, height=360, price=True):
    price = price and _has_price(d)
    fig = make_subplots(specs=[[{"secondary_y": price}]])
    for s in series:
        fig.add_trace(s["trace"], secondary_y=False)
    if price:
        _add_price(fig, d, secondary_y=True)
    fig.update_layout(
        **_BASE, height=height,
        title=dict(text=title, font=dict(size=12, color="#333"), x=0),
        margin=dict(l=52, r=55, t=42, b=72),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
    )
    fig.update_yaxes(title_text=ylabel, title_font_size=10, secondary_y=False, **_ax())
    if price:
        fig.update_yaxes(title_text="Rollex Px", title_font_size=10, secondary_y=True,
                         showgrid=False, tickfont=dict(size=10, color=C_PRICE))
    return fig

def bars_weekly(d, col, title, n=13):
    tail = d.tail(n + 1)
    chg  = tail[col].diff().iloc[1:] / 1000
    dates = tail["Date"].iloc[1:]
    fig = go.Figure(go.Bar(
        x=dates, y=chg,
        marker=dict(color=[C_LONG if v >= 0 else C_SHORT for v in chg],
                    opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{x|%d %b %y}</b><br>Δ: %{y:+.1f}k<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.14)")
    fig.update_layout(
        **_BASE, height=290,
        title=dict(text=title, font=dict(size=11, color="#444"), x=0),
        margin=dict(l=50, r=12, t=36, b=68),
        xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
        yaxis=dict(**_ax(), title_text="k lots", title_font_size=10),
        bargap=0.18, showlegend=False,
    )
    return fig

def bars_combined(d, lc, sc, nc, title, color, n=13, price=True):
    DARK_GREEN  = "#1a6b1a"
    LIGHT_GREEN = "#7dce7d"
    DARK_RED    = "#8b0000"
    LIGHT_RED   = "#f4a0a0"

    tail  = d.tail(n + 1)
    dates = tail["Date"].iloc[1:]

    if lc in d.columns and sc in d.columns:
        lchg = np.asarray(tail[lc].diff().iloc[1:] / 1000, dtype=float)
        schg = np.asarray(tail[sc].diff().iloc[1:] / 1000, dtype=float)
        long_add    = np.where(lchg > 0,  lchg, 0)
        long_liq    = np.where(lchg < 0,  lchg, 0)
        short_add   = np.where(schg > 0, -schg, 0)   # negated → goes below zero
        short_cover = np.where(schg < 0, -schg, 0)   # negated → goes above zero
    else:
        long_add = long_liq = short_add = short_cover = None

    price = price and "Px" in d.columns
    fig = make_subplots(specs=[[{"secondary_y": price}]])

    if long_add is not None:
        for arr, name, clr in [
            (long_add,    "Long Add",    DARK_GREEN),
            (long_liq,    "Long Liq.",   LIGHT_GREEN),
            (short_add,   "Short Add",   DARK_RED),
            (short_cover, "Short Cover", LIGHT_RED),
        ]:
            fig.add_trace(go.Bar(x=dates, y=arr, name=name,
                marker_color=clr, opacity=0.92,
                hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>{name}: %{{y:+.2f}}k<extra></extra>"),
                secondary_y=False)

    if price:
        px_vals = np.asarray(tail["Px"].iloc[1:], dtype=float)
        fig.add_trace(go.Scatter(x=dates, y=px_vals, name="Rollex Px", mode="lines",
            line=dict(color=C_PRICE, width=1.8),
            hovertemplate="<b>%{x|%d %b %y}</b><br>Rollex Px: %{y:.2f}<extra></extra>"),
            secondary_y=True)

    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.14)")
    fig.update_layout(**_BASE, height=340, barmode="relative",
        title=dict(text=title, font=dict(size=11, color="#444"), x=0),
        margin=dict(l=50, r=55, t=36, b=72),
        legend=dict(orientation="h", y=-0.26, x=0.5, xanchor="center", font_size=10),
        xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
        bargap=0.12)
    fig.update_yaxes(title_text="k lots", title_font_size=10, secondary_y=False, **_ax())
    if price:
        fig.update_yaxes(title_text="Rollex Px", title_font_size=10, secondary_y=True,
                         showgrid=False, tickfont=dict(size=10, color=C_PRICE))
    return fig


def histogram_dist(d, col, color, title):
    chg = (d[col] / 1000).diff().dropna()
    if chg.empty: return go.Figure().update_layout(**_BASE, height=260)
    lv = float(chg.iloc[-1])
    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    fig = go.Figure(go.Histogram(
        x=chg, nbinsx=30, showlegend=False,
        marker=dict(color=f"rgba({r},{g},{b},0.70)", line=dict(color="white", width=0.5)),
        hovertemplate="Δ: %{x:.1f}k<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=lv, line_dash="dash", line_color=C_SHORT, line_width=1.8,
                  annotation_text=f" {lv:+.1f}k", annotation_font_size=9,
                  annotation_font_color=C_SHORT)
    fig.update_layout(
        **_BASE, height=260,
        title=dict(text=f"Weekly Δ dist — {title}", font=dict(size=11, color="#444"), x=0),
        margin=dict(l=40, r=12, t=36, b=36),
        xaxis=dict(**_ax(x=True), title_text="k lots"),
        yaxis=dict(**_ax()), showlegend=False,
    )
    return fig

def seasonal(d, col, color, title):
    if d.empty or col not in d.columns:
        return go.Figure().update_layout(**_BASE, height=340,
            title=dict(text=f"Seasonality — {title}  (no data)", font=dict(size=11,color="#999"), x=0))
    s = d[["Date", col]].copy()
    s["v"]    = s[col] / 1000
    s["Week"] = s["Date"].dt.isocalendar().week.astype(int)
    s["Year"] = s["Date"].dt.year.astype(int)
    pivot = s.pivot_table(index="Week", columns="Year", values="v", aggfunc="mean")
    pivot = pivot[pivot.index <= 52]
    cur_year  = int(s["Year"].max())
    hist      = pivot[[c for c in pivot.columns if int(c) < cur_year]]
    if hist.empty or hist.shape[1] == 0:
        # Only current-year data — just show current year line, no band
        fig = go.Figure()
        if cur_year in pivot.columns:
            cy_raw = s[s["Year"] == cur_year].dropna(subset=["v"]).sort_values("Week")
            fig.add_trace(go.Scatter(x=cy_raw["Week"], y=cy_raw["v"], mode="lines+markers",
                name=str(cur_year), line=dict(color=color, width=2.5),
                marker=dict(size=4.5, color=color),
                customdata=cy_raw["Date"].dt.strftime("%d %b %Y"),
                hovertemplate="%{customdata}:  %{y:.1f}k<extra></extra>"))
        fig.update_layout(**_BASE, height=340,
            title=dict(text=f"Seasonality — {title}  ·  k lots  (current year only)",
                       font=dict(size=11,color="#999"), x=0),
            margin=dict(l=50,r=20,t=42,b=60),
            xaxis=dict(**_ax(), title_text="Week"),
            yaxis=dict(**_ax(), title_text="k lots", title_font_size=10))
        return fig
    p25, p75, med = hist.quantile(0.25,axis=1), hist.quantile(0.75,axis=1), hist.median(axis=1)
    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    fig = go.Figure()
    for yr in hist.columns:
        fig.add_trace(go.Scatter(x=pivot.index, y=hist[yr], mode="lines",
            line=dict(color="rgba(160,160,160,0.14)", width=1),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=list(p75.index)+list(p75.index[::-1]),
        y=list(p75.values)+list(p25.values[::-1]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.10)",
        line=dict(width=0), name="25–75th pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines", name="Median",
        line=dict(color=f"rgba({r},{g},{b},0.5)", width=1.6, dash="dash"),
        hovertemplate="Wk %{x}  Median: %{y:.1f}k<extra></extra>"))
    if cur_year in pivot.columns:
        cy_raw = s[s["Year"] == cur_year].dropna(subset=["v"]).sort_values("Week")
        fig.add_trace(go.Scatter(x=cy_raw["Week"], y=cy_raw["v"], mode="lines+markers",
            name=str(cur_year), line=dict(color=color, width=2.5),
            marker=dict(size=4.5, color=color),
            customdata=cy_raw["Date"].dt.strftime("%d %b %Y"),
            hovertemplate="%{customdata}:  %{y:.1f}k<extra></extra>"))

    MTICKS = {1:"Jan",5:"Feb",9:"Mar",14:"Apr",18:"May",23:"Jun",
              27:"Jul",32:"Aug",36:"Sep",40:"Oct",45:"Nov",49:"Dec"}
    fig.update_layout(
        **_BASE, height=340,
        title=dict(text=f"Seasonality — {title}  ·  k lots", font=dict(size=12,color="#333"), x=0),
        margin=dict(l=50,r=20,t=42,b=60),
        legend=dict(orientation="h",y=-0.18,x=0.5,xanchor="center",font_size=10),
        xaxis=dict(**_ax(), tickmode="array",
                   tickvals=list(MTICKS.keys()), ticktext=list(MTICKS.values()),
                   title_text="Week"),
        yaxis=dict(**_ax(), title_text="k lots", title_font_size=10),
    )
    return fig

def scatter_2d(d, x_col, y_col, color, title, xlabel, ylabel):
    xraw = d[x_col].pct_change()*100 if x_col=="Px" else d[x_col].diff()/1000
    x = np.asarray(xraw, dtype=float)
    y = np.asarray(d[y_col].diff()/1000, dtype=float)
    dates = np.asarray(d["Date"])
    mask = ~(np.isnan(x)|np.isnan(y))
    x,y,dates = x[mask],y[mask],dates[mask]
    if len(x)<5: return go.Figure().update_layout(**_BASE, height=340)
    r2 = float(np.corrcoef(x,y)[0,1]**2)
    sl,ic = np.polyfit(x,y,1)
    xl = np.linspace(x.min(),x.max(),200)
    rec = (dates-dates.min()).astype("timedelta64[D]").astype(float)
    nr  = rec/max(rec.max(),1)
    rv,gv,bv = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
        marker=dict(color=nr,
            colorscale=[[0,"rgba(200,210,230,0.5)"],[1,f"rgba({rv},{gv},{bv},0.85)"]],
            size=7, line=dict(width=0.5, color="white")),
        text=pd.to_datetime(dates).strftime("%Y-%m-%d"),
        hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        showlegend=False))
    fig.add_trace(go.Scatter(x=xl, y=sl*xl+ic, mode="lines",
        line=dict(color=color, width=1.6, dash="dash"), showlegend=False))
    fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", showlegend=False,
        marker=dict(symbol="star", size=14, color=C_SHORT,
                    line=dict(width=1.2, color="white"))))
    fig.update_layout(
        **_BASE, height=340,
        title=dict(text=f"{title}   <span style='font-size:10px;color:#888'>R²={r2:.2f}</span>",
                   font=dict(size=12,color="#333"), x=0),
        margin=dict(l=52,r=20,t=48,b=48),
        xaxis=dict(**_ax(x=True), title_text=xlabel),
        yaxis=dict(**_ax(), title_text=ylabel),
    )
    return fig


def scatter_3d(x, y, z, dates, color, title, xlabel, ylabel, zlabel, height=540):
    x, y, z = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    dates    = np.asarray(dates, dtype="datetime64[ns]")
    mask     = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z, dates = x[mask], y[mask], z[mask], dates[mask]
    if len(x) < 5:
        return go.Figure().update_layout(height=height,
            title=dict(text=title + "  [insufficient data]", font_size=12))
    corr_xy = float(np.corrcoef(x, y)[0, 1])
    corr_xz = float(np.corrcoef(x, z)[0, 1])
    corr_yz = float(np.corrcoef(y, z)[0, 1])
    rec  = (dates - dates.min()).astype("timedelta64[D]").astype(float)
    nr   = rec / max(rec.max(), 1)
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    def _ax3(label, bg):
        return dict(title=dict(text=label, font=dict(size=10, color="#555")),
                    tickfont=dict(size=9, color="#777"),
                    gridcolor="rgba(0,0,0,0.10)", gridwidth=1,
                    showbackground=True, backgroundcolor=bg,
                    zerolinecolor="rgba(0,0,0,0.20)", zerolinewidth=2)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
        marker=dict(color=nr,
            colorscale=[[0,"rgba(200,210,230,0.45)"],[1,f"rgba({r},{g},{b},0.9)"]],
            size=5, line=dict(width=0.6, color="white")),
        text=pd.to_datetime(dates).strftime("%Y-%m-%d"),
        hovertemplate=(f"<b>%{{text}}</b><br>{xlabel}: %{{x:.2f}}<br>"
                       f"{ylabel}: %{{y:.2f}}<br>{zlabel}: %{{z:.2f}}<extra></extra>"),
        showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers",
        showlegend=False,
        marker=dict(symbol="diamond", size=10, color=C_SHORT,
                    line=dict(width=1.5, color="white")),
        hovertemplate=(f"<b>Latest</b><br>{xlabel}: {x[-1]:.2f}<br>"
                       f"{ylabel}: {y[-1]:.2f}<br>{zlabel}: {z[-1]:.2f}<extra></extra>")))
    corr_str = f"r(X,Y)={corr_xy:+.2f}  ·  r(X,Z)={corr_xz:+.2f}  ·  r(Y,Z)={corr_yz:+.2f}"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", color="#2d2d2d", size=11),
        title=dict(text=f"{title}<br><span style='font-size:10px;color:#888'>{corr_str}</span>",
                   font=dict(size=12, color="#444"), x=0),
        height=height, margin=dict(l=0, r=0, t=70, b=20),
        scene=dict(
            aspectmode="cube",
            camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=-0.1),
                        eye=dict(x=1.6,y=1.6,z=0.8)),
            xaxis=_ax3(xlabel, "rgba(240,244,255,0.6)"),
            yaxis=_ax3(ylabel, "rgba(240,255,244,0.6)"),
            zaxis=_ax3(zlabel, "rgba(255,248,240,0.6)"),
        ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPEC
# ══════════════════════════════════════════════════════════════════════════════
CIT_SPEC = {
    "Large Spec":    {"long":"Spec Long",    "short":"Spec Short",    "net":"Spec Net",    "spread":None},
    "Non-Rep":       {"long":"Non Rep Long", "short":"Non Rep Short", "net":"Non Rep Net", "spread":None},
    "Index Traders": {"long":"Index Long",   "short":"Index Short",   "net":"Index Net",   "spread":None},
    "Large Spec + Non-Rep": {"long":"Spec+NonRep Long","short":"Spec+NonRep Short","net":"Spec+NonRep Net","spread":None},
    "Large Spec + Index + Non-Rep": {"long":"Combined Spec Long","short":"Combined Spec Short","net":"Combined Spec Net","spread":None},
}
DISAGG_SPEC = {
    "Managed Money":              {"long":"MM Long",    "short":"MM Short",    "net":"MM Net",    "spread":"MM Spread"},
    "Other Rept":                 {"long":"Other Long", "short":"Other Short", "net":"Other Net", "spread":"Other Spread"},
    "Non-Rep":                    {"long":"Non Rep Long","short":"Non Rep Short","net":"Non Rep Net","spread":None},
    "Swap Dealers":               {"long":"Swap Long",  "short":"Swap Short",  "net":"Swap Net",  "spread":"Swap Spread"},
    "MM + Other + Non-Rep":       {"long":"MM+Other+NonRep Long","short":"MM+Other+NonRep Short","net":"MM+Other+NonRep Net","spread":None},
    "MM + Other + Non-Rep + Swap":{"long":"Combined Spec Long","short":"Combined Spec Short","net":"Combined Spec Net","spread":None},
}

def render_spec(d, report, color):
    cats = CIT_SPEC if report == "CIT" else DISAGG_SPEC
    cat  = st.selectbox("Category", list(cats.keys()), key="spec_cat")
    unit = "k lots"

    cfg = cats[cat]
    lc, sc, nc, spc = cfg["long"], cfg["short"], cfg["net"], cfg["spread"]
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    with _exp("Seasonality", expanded=False):
        seas_items = [(lc, C_LONG, "Long"), (sc, C_SHORT, "Short"), (nc, C_NET, "Net")]
        avail_s = [(col, clr, lbl) for col, clr, lbl in seas_items if col in d.columns]
        scols = st.columns(len(avail_s))
        for ch, (col, clr, lbl) in zip(scols, avail_s):
            with ch:
                _chart(seasonal(d, col, clr, f"{cat} {lbl}"), width='stretch')

    show_px = rollex_toggle("spec_show_px")

    # Combined timeseries — Long, Short, Net all in selected unit + Price secondary
    ylabel = "k lots" if unit == "k lots" else "% of OI"
    suffix = "k" if unit == "k lots" else "%"
    traces = []
    if lc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, lc, unit), name="Long",
            line=dict(color=C_LONG, width=2.0),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Long: %{{y:.1f}}{suffix}<extra></extra>")})
    if sc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, sc, unit), name="Short",
            line=dict(color=C_SHORT, width=2.0),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Short: %{{y:.1f}}{suffix}<extra></extra>")})
    if nc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, nc, unit), name="Net",
            fill="tozeroy", fillcolor="rgba(26,86,219,0.09)",
            line=dict(color=C_NET, width=2.2),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Net: %{{y:.1f}}{suffix}<extra></extra>")})
    if spc and spc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, spc, unit), name="Spread",
            line=dict(color="#94a3b8", width=1.4, dash="dot"),
            visible="legendonly",
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Spread: %{{y:.1f}}{suffix}<extra></extra>")})
    _chart(timeseries(d, traces, f"{cat}  ·  {ylabel}", ylabel, price=show_px), width='stretch')

    # % of Total OI chart
    oi = d["Total OI"].replace(0, np.nan) if "Total OI" in d.columns else None
    if oi is not None:
        pct_traces = []
        for col, name, clr in [(lc, "Long", C_LONG), (sc, "Short", C_SHORT)]:
            if col in d.columns:
                pct_traces.append({"trace": go.Scatter(
                    x=d["Date"], y=(d[col] / oi * 100).round(2), name=f"{name} %",
                    line=dict(color=clr, width=2.0),
                    hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{name}: %{{y:.1f}}%<extra></extra>")})
        if nc in d.columns:
            pct_traces.append({"trace": go.Scatter(
                x=d["Date"], y=(d[nc] / oi * 100).round(2), name="Net %",
                fill="tozeroy", fillcolor="rgba(26,86,219,0.09)",
                line=dict(color=C_NET, width=2.2),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Net: %{y:.1f}%<extra></extra>")})
        if spc and spc in d.columns:
            pct_traces.append({"trace": go.Scatter(
                x=d["Date"], y=(d[spc] / oi * 100).round(2), name="Spread %",
                line=dict(color="#94a3b8", width=1.4, dash="dot"),
                visible="legendonly",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Spread: %{y:.1f}%<extra></extra>")})
        if pct_traces:
            st.caption("Denominator: All Crop Total OI")
            _chart(timeseries(d, pct_traces, f"{cat}  ·  % of Total OI", "% of OI", price=show_px), width='stretch')

    # Stacked Long Add/Liq + Short Add/Cover bars + Price
    _chart(bars_combined(d, lc, sc, nc, f"{cat} — weekly flow  ·  k lots", color, price=show_px),
                    width='stretch')

    show_table(d, [lc, sc, nc] + ([spc] if spc else []) + ["Px"],
               [lc, sc, nc], f"Data table — {cat}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMMERCIAL
# ══════════════════════════════════════════════════════════════════════════════
def render_commercial(d, report, color):
    is_disagg = report == "Disagg"
    lc  = "Producer Long"  if is_disagg else "Comm Long"
    sc  = "Producer Short" if is_disagg else "Comm Short"
    nc  = "Comm Net"
    lbl = "Producer/Merchant" if is_disagg else "Commercial"
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    unit = "k lots"

    with _exp("Seasonality", expanded=False):
        seas = [(lc, C_LONG, "Long"), (sc, C_SHORT, "Short"), (nc, C_NET, "Net")]
        avail_s = [(col, clr, name) for col, clr, name in seas if col in d.columns]
        scols = st.columns(len(avail_s))
        for ch, (col, clr, name) in zip(scols, avail_s):
            with ch:
                _chart(seasonal(d, col, clr, f"{lbl} {name}"), width='stretch')

    show_px = rollex_toggle("comm_show_px")

    # Combined timeseries — Long, Short, Net in selected unit + Price secondary
    ylabel = "k lots" if unit == "k lots" else "% of OI"
    suffix = "k" if unit == "k lots" else "%"
    traces = []
    for col, name, clr in [(lc, "Long", C_LONG), (sc, "Short", C_SHORT)]:
        if col in d.columns:
            traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, col, unit), name=name,
                line=dict(color=clr, width=2.0),
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{name}: %{{y:.1f}}{suffix}<extra></extra>")})
    if nc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, nc, unit), name="Net",
            fill="tozeroy", fillcolor="rgba(26,86,219,0.07)",
            line=dict(color=C_NET, width=2.2),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Net: %{{y:.1f}}{suffix}<extra></extra>")})
    _chart(timeseries(d, traces, f"{lbl}  ·  {ylabel}", ylabel, price=show_px), width='stretch')

    # % of Total OI chart
    oi = d["Total OI"].replace(0, np.nan) if "Total OI" in d.columns else None
    if oi is not None:
        pct_traces = []
        for col, name, clr in [(lc, "Long", C_LONG), (sc, "Short", C_SHORT)]:
            if col in d.columns:
                pct_traces.append({"trace": go.Scatter(
                    x=d["Date"], y=(d[col] / oi * 100).round(2), name=f"{name} %",
                    line=dict(color=clr, width=2.0),
                    hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{name}: %{{y:.1f}}%<extra></extra>")})
        if nc in d.columns:
            pct_traces.append({"trace": go.Scatter(
                x=d["Date"], y=(d[nc] / oi * 100).round(2), name="Net %",
                fill="tozeroy", fillcolor="rgba(26,86,219,0.07)",
                line=dict(color=C_NET, width=2.2),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Net: %{y:.1f}%<extra></extra>")})
        if pct_traces:
            _chart(timeseries(d, pct_traces, f"{lbl}  ·  % of Total OI", "% of OI", price=show_px), width='stretch')

    # Stacked Long Add/Liq + Short Add/Cover bars + Price
    _chart(bars_combined(d, lc, sc, nc, f"{lbl} — weekly flow  ·  k lots", color, price=show_px),
                    width='stretch')

    show_table(d, [lc, sc, nc, "Px"], [lc, sc, nc], f"Data table — {lbl}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SPREADING (Disagg only)
# ══════════════════════════════════════════════════════════════════════════════
SPREAD_COLS = {
    "Managed Money": ("MM Spread",    C_NET),
    "Swap Dealers":  ("Swap Spread",  "#7c3aed"),
    "Other Rept":    ("Other Spread", "#d97706"),
}

# OC/NC — same categories with a short key for column naming
_OCNC_CATS = {
    "Managed Money": ("MM Spread",    C_NET,    "MM"),
    "Swap Dealers":  ("Swap Spread",  "#7c3aed","Swap"),
    "Other Rept":    ("Other Spread", "#d97706","OR"),
}
_OCNC_C_CROSS = "#8b5cf6"   # cross-crop colour


def _build_ocnc_df(df_all_crops):
    """OC/NC = All Spreading − Old Spreading − Other Spreading per category.
    Cross-crop spreads land in All but CFTC never allocates them to Old or Other.
    Result is clipped to >= 0 (guaranteed by CFTC allocation rules).
    """
    all_ = df_all_crops[df_all_crops["Crop"] == "All"].set_index("Date").sort_index()
    old_ = df_all_crops[df_all_crops["Crop"] == "Old"].set_index("Date").sort_index()
    new_ = df_all_crops[df_all_crops["Crop"] == "Other"].set_index("Date").sort_index()
    common = all_.index.intersection(old_.index).intersection(new_.index)
    if common.empty:
        return pd.DataFrame()
    rows = {}
    for _lbl, (col, _clr, short) in _OCNC_CATS.items():
        if col not in all_.columns:
            continue
        a  = all_.loc[common, col].astype(float)
        o  = old_.loc[common, col].astype(float)  if col in old_.columns else pd.Series(0.0, index=common)
        n  = new_.loc[common, col].astype(float)  if col in new_.columns else pd.Series(0.0, index=common)
        oc = (a - o - n).clip(lower=0)
        rows[f"{short}_All"]      = a  / 1000
        rows[f"{short}_OldOld"]   = o  / 1000
        rows[f"{short}_NewNew"]   = n  / 1000
        rows[f"{short}_OCNC"]     = oc / 1000
        rows[f"{short}_OCNC_pct"] = (oc / a.replace(0, np.nan) * 100).round(1)
    return pd.DataFrame(rows, index=common).reset_index()

def render_spreading(d, color, df_all_crops=None, commodity=""):
    st.markdown(
        "<p style='font-size:.78rem;color:#666;margin-bottom:8px'>"
        "Spreading = offsetting long/short positions in different delivery months. "
        "Shown per category in lots and % of OI.</p>", unsafe_allow_html=True)

    unit = st.radio("Unit", ["k lots","% of OI"], horizontal=True, key="spread_unit")
    show_px = rollex_toggle("spread_show_px")
    if unit == "% of OI":
        st.caption("Denominator: All Crop Total OI")
    ylabel = "k lots" if unit=="k lots" else "% of OI"

    # All spreads on one timeseries
    traces = []
    for lbl,(col,clr) in SPREAD_COLS.items():
        if col not in d.columns: continue
        y = _get_y(d,col,unit)
        traces.append({"trace": go.Scatter(x=d["Date"],y=y,name=lbl,
            line=dict(color=clr,width=2.0),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{lbl}: %{{y:.1f}}<extra></extra>")})
    _chart(timeseries(d,traces,f"Spreading by Category  ·  {ylabel}",ylabel,price=show_px), width='stretch')

    avail = [(lbl,col,clr) for lbl,(col,clr) in SPREAD_COLS.items() if col in d.columns]

    # ── Pre-compute crop split data (used across multiple sections below) ──────
    _has_crops = df_all_crops is not None
    if _has_crops:
        old        = df_all_crops[df_all_crops["Crop"] == "Old"].set_index("Date").sort_index()
        other      = df_all_crops[df_all_crops["Crop"] == "Other"].set_index("Date").sort_index()
        all_       = df_all_crops[df_all_crops["Crop"] == "All"].set_index("Date").sort_index()
        _ocnc_on   = _build_ocnc_df(df_all_crops)
        _ocnc      = _ocnc_on  # same build, reuse
        spread_cols_avail = [col for _, (col, _) in SPREAD_COLS.items()
                             if col in old.columns or col in other.columns]
        _has_on = spread_cols_avail and not old.empty and not other.empty

    # ── 1. Old vs New Crop timeseries (always visible) ────────────────────────
    if _has_crops and _has_on:
        on_unit = st.radio("Unit", ["k lots", "% of OI"], horizontal=True, key="spread_on_unit")
        if on_unit == "% of OI":
            st.caption("Denominator: All Crop Total OI — same base for Old, New and OC/NC")
        _all_oi = all_["Total OI"].replace(0, np.nan) if "Total OI" in all_.columns else None
        def _ov(df, col):
            if col not in df.columns: return pd.Series(dtype=float)
            if on_unit == "k lots": return df[col] / 1000
            if _all_oi is not None:
                return (df[col].reindex(_all_oi.index) / _all_oi * 100).round(2)
            return df[col] / 1000
        c1, c2, c3 = st.columns(3)
        for ch, (lbl, col, clr) in zip([c1, c2, c3], avail):
            with ch:
                short = _OCNC_CATS.get(lbl, (None, None, None))[2]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=old.index, y=_ov(old, col), name="Old Crop",
                    line=dict(color=C_OLD, width=2.0),
                    hovertemplate="<b>%{x|%d %b %y}</b><br>Old: %{y:.1f}<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=other.index, y=_ov(other, col), name="New Crop",
                    line=dict(color=C_NEW, width=2.0),
                    hovertemplate="<b>%{x|%d %b %y}</b><br>New: %{y:.1f}<extra></extra>"))
                if short and not _ocnc_on.empty:
                    ocnc_col = f"{short}_OCNC"
                    if ocnc_col in _ocnc_on.columns:
                        if on_unit == "k lots":
                            ocnc_y = _ocnc_on.set_index("Date")[ocnc_col]
                        else:
                            ocnc_raw = _ocnc_on.set_index("Date")[ocnc_col] * 1000
                            ocnc_y   = (ocnc_raw.reindex(_all_oi.index) / _all_oi * 100).round(2) if _all_oi is not None else ocnc_raw / 1000
                        fig.add_trace(go.Scatter(
                            x=ocnc_y.index, y=ocnc_y.values,
                            name="OC/NC", line=dict(color=_OCNC_C_CROSS, width=2.0),
                            hovertemplate="<b>%{x|%d %b %y}</b><br>OC/NC: %{y:.1f}<extra></extra>"))
                fig.update_layout(
                    **_BASE, height=300,
                    title=dict(text=f"{lbl} Spread  ·  Old vs New  ·  {on_unit}",
                               font=dict(size=11, color="#374151"), x=0),
                    margin=dict(l=44, r=12, t=40, b=60),
                    legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font_size=10,
                                bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
                    yaxis=dict(**_ax()),
                )
                _chart(fig, width='stretch')

    # ── 2. OC/NC Decomposition (expander, open by default) ───────────────────
    if _has_crops and not _ocnc.empty:
        with _exp("OC/NC Cross-Crop Spreading  ·  Decomposition", expanded=True):
            st.markdown(
                "<p style='font-size:.76rem;color:#555;margin-bottom:10px'>"
                "<b>OC/NC = All Spreading − Old Spreading − Other Spreading</b></p>",
                unsafe_allow_html=True)
            latest_ocnc = _ocnc.sort_values("Date").iloc[-1]
            latest_dt   = pd.to_datetime(latest_ocnc["Date"]).strftime("%d %b %Y")
            st.markdown(
                f"<p style='font-size:.72rem;color:#888;margin-bottom:4px'>Latest — {latest_dt}</p>",
                unsafe_allow_html=True)
            snap_rows = ""
            for snap_lbl, (snap_col, snap_clr, short) in _OCNC_CATS.items():
                if f"{short}_All" not in _ocnc.columns:
                    continue
                a      = latest_ocnc.get(f"{short}_All",      np.nan)
                oo     = latest_ocnc.get(f"{short}_OldOld",   np.nan)
                nn     = latest_ocnc.get(f"{short}_NewNew",   np.nan)
                ocnc_v = latest_ocnc.get(f"{short}_OCNC",     np.nan)
                pct    = latest_ocnc.get(f"{short}_OCNC_pct", np.nan)
                fk = lambda v: f"{v:,.1f}k" if pd.notna(v) else "—"
                fp = (f"<b style='color:{_OCNC_C_CROSS}'>{pct:.1f}%</b>" if pd.notna(pct) else "—")
                snap_rows += (
                    f"<tr><td class='idx'>{snap_lbl}</td>"
                    f"<td>{fk(a)}</td>"
                    f"<td style='color:{C_OLD}'>{fk(oo)}</td>"
                    f"<td style='color:{C_NEW}'>{fk(nn)}</td>"
                    f"<td style='color:{_OCNC_C_CROSS};font-weight:700'>{fk(ocnc_v)}</td>"
                    f"<td>{fp}</td></tr>"
                )
            snap_hdr = (
                "<tr style='background:#f3f4f6'><th class='idx'>Category</th>"
                "<th>All (k)</th>"
                f"<th style='color:{C_OLD}'>Pure Old/Old (k)</th>"
                f"<th style='color:{C_NEW}'>Pure New/New (k)</th>"
                f"<th style='color:{_OCNC_C_CROSS}'>OC/NC Cross-crop (k)</th>"
                "<th>OC/NC %</th></tr>"
            )
            st.markdown(
                f"{_RECAP_CSS}<div style='overflow-x:auto;margin-bottom:14px'>"
                f"<table class='rtbl'><thead>{snap_hdr}</thead>"
                f"<tbody>{snap_rows}</tbody></table></div>",
                unsafe_allow_html=True)
            avail_ocnc = [(lbl2, col2, clr2, sh)
                          for lbl2, (col2, clr2, sh) in _OCNC_CATS.items()
                          if f"{sh}_All" in _ocnc.columns]
            sel_lbl = st.selectbox("Category", [x[0] for x in avail_ocnc], key="ocnc_cat_sel")
            _, _, _, short_sel = next(x for x in avail_ocnc if x[0] == sel_lbl)
            fig_stack = go.Figure()
            for sub_col, sub_name, fill_clr in [
                (f"{short_sel}_OldOld", "Pure Old/Old",     C_OLD),
                (f"{short_sel}_NewNew", "Pure New/New",     C_NEW),
                (f"{short_sel}_OCNC",   "OC/NC Cross-crop", _OCNC_C_CROSS),
            ]:
                if sub_col not in _ocnc.columns: continue
                r2, g2, b2 = int(fill_clr[1:3], 16), int(fill_clr[3:5], 16), int(fill_clr[5:7], 16)
                fig_stack.add_trace(go.Bar(
                    x=_ocnc["Date"], y=_ocnc[sub_col], name=sub_name,
                    marker=dict(color=f"rgba({r2},{g2},{b2},0.80)", line=dict(width=0)),
                    hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{sub_name}: %{{y:.1f}}k<extra></extra>"))
            if f"{short_sel}_All" in _ocnc.columns:
                fig_stack.add_trace(go.Scatter(
                    x=_ocnc["Date"], y=_ocnc[f"{short_sel}_All"],
                    name="All Spreading", mode="lines",
                    line=dict(color="#374151", width=1.6, dash="dot"),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>All: %{y:.1f}k<extra></extra>"))
            fig_stack.update_layout(
                **_BASE, height=340, barmode="stack",
                title=dict(text=f"{sel_lbl}  ·  Spreading Decomposition  ·  k lots",
                           font=dict(size=12, color="#374151"), x=0),
                margin=dict(l=50, r=20, t=42, b=70), bargap=0.15,
                legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                            font_size=10, bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
                yaxis=dict(**_ax(), title_text="k lots", title_font_size=10))
            _chart(fig_stack, width='stretch')
            fig_pct = go.Figure()
            for lbl3, (col3, clr3, short3) in _OCNC_CATS.items():
                pct_col = f"{short3}_OCNC_pct"
                if pct_col not in _ocnc.columns: continue
                fig_pct.add_trace(go.Scatter(
                    x=_ocnc["Date"], y=_ocnc[pct_col], name=lbl3,
                    line=dict(color=clr3, width=2.0),
                    hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{lbl3} OC/NC: %{{y:.1f}}%<extra></extra>"))
            fig_pct.update_layout(
                **_BASE, height=280,
                title=dict(text="OC/NC as % of All Spreading  ·  by Category",
                           font=dict(size=12, color="#374151"), x=0),
                margin=dict(l=50, r=20, t=42, b=70),
                legend=dict(orientation="h", y=-0.26, x=0.5, xanchor="center",
                            font_size=10, bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
                yaxis=dict(**_ax(), title_text="% of All Spreading",
                           title_font_size=10, ticksuffix="%", range=[0, 100]))
            _chart(fig_pct, width='stretch')

    # ── 3. Seasonality — calendar week (collapsed) ────────────────────────────
    with _exp("Seasonality", expanded=False):
        ch1,ch2,ch3 = st.columns(3)
        for ch,(lbl,col,clr) in zip([ch1,ch2,ch3], avail):
            with ch: _chart(seasonal(d,col,clr,f"{lbl} Spread"), width='stretch')

    # ── 4. Old/New/OC·NC Seasonality (collapsed) ─────────────────────────────
    if _has_crops and _has_on:
        with _exp("Old / New / OC·NC Spreading  ·  Seasonality", expanded=False):
            _forced_sm = {"CT": 7, "KC": 9, "CC": 9, "SB": 9}
            _def_sm    = _forced_sm.get(commodity, CROP_START_MONTH)
            if st.session_state.get("_spread_seas_comm") != commodity:
                st.session_state["spread_seas_sm"] = _def_sm
                st.session_state["_spread_seas_comm"] = commodity
            sm = st.selectbox(
                "Crop year starts in", list(range(1, 13)),
                index=_def_sm - 1, format_func=lambda m: _MONTHS[m - 1],
                key="spread_seas_sm")
            st.markdown(
                f"<p style='font-size:.75rem;color:#9ca3af;margin:0 0 10px'>"
                f"Each line = one crop year  ·  Bold = current ({_current_crop_year_label(sm)})  ·  Shaded = 25–75th pct</p>",
                unsafe_allow_html=True)
            _cidx    = old.index.intersection(other.index)
            _oidx    = _ocnc_on.set_index("Date") if not _ocnc_on.empty else pd.DataFrame()
            _sp_rows = {"Date": _cidx}
            for _l2, _c2, _ in avail:
                _s2 = _OCNC_CATS.get(_l2, (None, None, None))[2]
                if not _s2: continue
                _ov2 = old.loc[_cidx, _c2]   if _c2 in old.columns   else pd.Series(0.0, index=_cidx)
                _nv2 = other.loc[_cidx, _c2] if _c2 in other.columns else pd.Series(0.0, index=_cidx)
                _oc2 = (_oidx.loc[_cidx, f"{_s2}_OCNC"] * 1000
                        if not _oidx.empty and f"{_s2}_OCNC" in _oidx.columns
                        else pd.Series(0.0, index=_cidx))
                _sp_rows[f"{_s2}_Old"]  = _ov2
                _sp_rows[f"{_s2}_New"]  = _nv2
                _sp_rows[f"{_s2}_OCNC"] = _oc2.reindex(_cidx).fillna(0)
                _sp_rows[f"{_s2}_Diff"] = _ov2 - _nv2
            _wsp = pd.DataFrame(_sp_rows).reset_index(drop=True)
            _wsp["CropYear"] = pd.to_datetime(_wsp["Date"]).apply(lambda d: _crop_year_label(d, sm))
            _wsp["CropWeek"] = pd.to_datetime(_wsp["Date"]).apply(lambda d: _crop_week_num(d, sm))
            def _spsc(metric, title, accent):
                return _seas_chart(_wsp, metric, title, accent, ylabel="k lots", by_week=False, sm=sm)
            _CAT_ORD = [
                ("Managed Money", "MM",   C_NET),
                ("Other Rept",    "OR",   "#d97706"),
                ("Swap Dealers",  "Swap", "#7c3aed"),
            ]
            for _clbl, _sh, _cc in _CAT_ORD:
                if f"{_sh}_Old" not in _wsp.columns: continue
                st.markdown(
                    f"<div style='font-size:.78rem;font-weight:700;color:#374151;"
                    f"margin:14px 0 4px;letter-spacing:.03em'>{_clbl.upper()}</div>",
                    unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1: _chart(_spsc(f"{_sh}_Old",  f"{_clbl}  ·  Old Crop  ·  k lots",  C_OLD),         width='stretch')
                with c2: _chart(_spsc(f"{_sh}_New",  f"{_clbl}  ·  New Crop  ·  k lots",  C_NEW),         width='stretch')
                with c3: _chart(_spsc(f"{_sh}_OCNC", f"{_clbl}  ·  OC/NC  ·  k lots",     _OCNC_C_CROSS), width='stretch')
            st.markdown(
                "<div style='font-size:.78rem;font-weight:700;color:#374151;"
                "margin:14px 0 4px;letter-spacing:.03em'>OLD − NEW DIFFERENCE</div>",
                unsafe_allow_html=True)
            dc1, dc2, dc3 = st.columns(3)
            for _dc, (_clbl, _sh, _cc) in zip([dc1, dc2, dc3], _CAT_ORD):
                with _dc:
                    if f"{_sh}_Diff" in _wsp.columns:
                        _chart(_spsc(f"{_sh}_Diff", f"{_clbl}  ·  Old−New Diff  ·  k lots", "#6b7280"), width='stretch')

    # ── 5. Data table (collapsed, bottom) ────────────────────────────────────
    show_table(d, [col for _,(col,_) in SPREAD_COLS.items() if col in d.columns] + ["Total OI"],
               [col for _,(col,_) in SPREAD_COLS.items() if col in d.columns],
               "Data table — Spreading")

    # ── 6. Spread Intensity % — Spreading / (Long + Short + Spreading) ──────
    st.markdown(
        "<p style='font-size:.78rem;color:#666;margin:18px 0 4px'>"
        "<b>Spread Intensity %</b> = Spreading ÷ (Long + Short + Spreading)</p>",
        unsafe_allow_html=True)

    def _spread_intensity(legs):
        """legs: list of (long_col, short_col, spread_col) tuples — one for a
        single category, several summed together for the combined view."""
        long_s   = sum(d[lc].astype(float)  for lc, _, _   in legs if lc  in d.columns)
        short_s  = sum(d[sc].astype(float)  for _, sc, _   in legs if sc  in d.columns)
        spread_s = sum(d[spc].astype(float) for _, _, spc  in legs if spc in d.columns)
        denom = (long_s + short_s + spread_s)
        denom = denom.replace(0, np.nan) if hasattr(denom, "replace") else denom
        return spread_s / denom * 100

    si_cat_legs = {lbl: (f"{col.replace(' Spread', '')} Long",
                         f"{col.replace(' Spread', '')} Short", col)
                   for lbl, col, _ in avail}
    si_opts = list(si_cat_legs.keys()) + (["Combined (All)"] if len(si_cat_legs) > 1 else [])
    si_colors = {lbl: clr for lbl, _, clr in avail}
    si_colors["Combined (All)"] = "#111827"
    si_sel = st.multiselect("Category", si_opts, default=si_opts, key="spread_intensity_cats")

    if si_sel:
        fig_si = go.Figure()
        for lbl in si_sel:
            legs = list(si_cat_legs.values()) if lbl == "Combined (All)" else [si_cat_legs[lbl]]
            y = _spread_intensity(legs)
            fig_si.add_trace(go.Scatter(
                x=d["Date"], y=y, name=lbl,
                line=dict(color=si_colors.get(lbl, "#888"),
                          width=2.4 if lbl == "Combined (All)" else 1.8),
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{lbl}: %{{y:.1f}}%<extra></extra>"))
        fig_si.update_layout(
            **_BASE, height=360,
            title=dict(text="Spread Intensity %  ·  Spreading ÷ (Long + Short + Spreading)",
                       font=dict(size=12, color="#374151"), x=0),
            margin=dict(l=50, r=20, t=42, b=50),
            legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
            xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
            yaxis=dict(**_ax(), title_text="%", title_font_size=10))
        _chart(fig_si, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OLD / NEW CROP (Disagg only)
# ══════════════════════════════════════════════════════════════════════════════
# ── Old/New helpers ───────────────────────────────────────────────────────────
def _crop_year_label(dt, sm=CROP_START_MONTH):
    y, m = dt.year, dt.month
    return f"{str(y)[2:]}/{str(y+1)[2:]}" if m >= sm else f"{str(y-1)[2:]}/{str(y)[2:]}"

def _crop_week_num(dt, sm=CROP_START_MONTH):
    y = dt.year if dt.month >= sm else dt.year - 1
    return max(1, (dt - pd.Timestamp(y, sm, 1)).days // 7 + 1)

def _current_crop_year_label(sm=CROP_START_MONTH):
    return _crop_year_label(pd.Timestamp.now(), sm)

def _on_seasonal_wide(d_crops):
    old   = d_crops[d_crops["Crop"]=="Old"].set_index("Date").sort_index()
    other = d_crops[d_crops["Crop"]=="Other"].set_index("Date").sort_index()
    common = old.index.intersection(other.index)
    if common.empty: return pd.DataFrame()
    old, other = old.loc[common], other.loc[common]
    oi_sum = old["Total OI"] + other["Total OI"]
    def _col(df, name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    wide = pd.DataFrame({
        "OI Old %":         old["Total OI"] / oi_sum * 100,
        # Managed Money
        "MM Net Old":       old["MM Net"],       "MM Net New":       other["MM Net"],
        "MM Long Old":      old["MM Long"],       "MM Long New":      other["MM Long"],
        "MM Short Old":     old["MM Short"],      "MM Short New":     other["MM Short"],
        "MM Diff":          old["MM Net"] - other["MM Net"],
        # Commercial
        "Comm Net Old":     _col(old, "Comm Net"),    "Comm Net New":     _col(other, "Comm Net"),
        "Comm Long Old":    _col(old, "Producer Long"),  "Comm Long New":    _col(other, "Producer Long"),
        "Comm Short Old":   _col(old, "Producer Short"), "Comm Short New":   _col(other, "Producer Short"),
        "Comm Diff":        _col(old, "Comm Net") - _col(other, "Comm Net"),
        # Swap Dealers
        "Swap Net Old":     _col(old, "Swap Net"),    "Swap Net New":     _col(other, "Swap Net"),
        "Swap Long Old":    _col(old, "Swap Long"),   "Swap Long New":    _col(other, "Swap Long"),
        "Swap Short Old":   _col(old, "Swap Short"),  "Swap Short New":   _col(other, "Swap Short"),
        "Swap Diff":        _col(old, "Swap Net") - _col(other, "Swap Net"),
        # Other Reportables
        "Other Net Old":    _col(old, "Other Net"),   "Other Net New":    _col(other, "Other Net"),
        "Other Long Old":   _col(old, "Other Long"),  "Other Long New":   _col(other, "Other Long"),
        "Other Short Old":  _col(old, "Other Short"), "Other Short New":  _col(other, "Other Short"),
        "Other Diff":       _col(old, "Other Net") - _col(other, "Other Net"),
        "Week": pd.Series(common.isocalendar().week.astype(int).values, index=common),
        "Year": pd.Series(common.year, index=common),
    }, index=common)
    return wide.reset_index()

def _seas_chart(wide, metric, title, accent, ylabel="k lots", by_week=True, sm=CROP_START_MONTH):
    if wide.empty or metric not in wide.columns:
        return go.Figure().update_layout(**_BASE, height=340)
    grp = "Week" if by_week else "CropWeek"
    yr_col = "Year" if by_week else "CropYear"
    pivot = wide.pivot_table(index=grp, columns=yr_col, values=metric, aggfunc="mean")
    pivot = pivot[pivot.index <= 52]
    if by_week:
        cur = int(wide["Year"].max())
        hist_cols = [c for c in pivot.columns if c < cur]
    else:
        cur = _current_crop_year_label(sm)
        hist_cols = [c for c in pivot.columns if c != cur]
    hist = pivot[hist_cols] if hist_cols else pivot
    if hist.empty or hist.shape[1] == 0:
        p25 = p75 = med = pd.Series(dtype=float)
    else:
        p25, p75, med = hist.quantile(0.25, axis=1), hist.quantile(0.75, axis=1), hist.median(axis=1)
    r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
    fig = go.Figure()
    for yr in hist_cols:
        fig.add_trace(go.Scatter(x=pivot.index, y=hist[yr], mode="lines",
            line=dict(color="rgba(150,150,150,0.18)", width=1), showlegend=False, hoverinfo="skip"))
    xs = list(p75.index) + list(p75.index[::-1])
    fig.add_trace(go.Scatter(x=xs, y=list(p75.values)+list(p25.values[::-1]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.10)",
        line=dict(width=0), name="25–75th pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines", name="Median",
        line=dict(color=f"rgba({r},{g},{b},0.55)", width=1.6, dash="dash")))
    if cur in pivot.columns:
        cy = pivot[cur].dropna()
        fig.add_trace(go.Scatter(x=cy.index, y=cy.values, mode="lines+markers",
            name=str(cur), line=dict(color=accent, width=2.6), marker=dict(size=5, color=accent)))
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.12)")
    if by_week:
        xticks = dict(tickvals=list(MONTH_TICKS.keys()), ticktext=list(MONTH_TICKS.values()))
    else:
        xticks = dict(tickvals=list(CROP_WEEK_TICKS.keys()),
                      ticktext=[_MONTHS[(sm - 1 + i) % 12] for i in range(12)])
    fig.update_layout(**_BASE, height=340,
        title=dict(text=title, font=dict(size=12, color="#444"), x=0),
        margin=dict(l=50, r=20, t=40, b=60),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(x=True), title_text="", **xticks),
        yaxis=dict(**_ax(), title_text=ylabel, title_font_size=10))
    return fig



def _seas_chart_overlay(wide, metric_old, metric_new, title, ylabel, sm=CROP_START_MONTH):
    """Seasonality chart overlaying Old Crop and New Crop on the same axes."""
    if wide.empty:
        return go.Figure().update_layout(**_BASE, height=360)
    cur = _current_crop_year_label(sm)

    def _pivot(metric):
        if metric not in wide.columns:
            return None
        pv = wide.pivot_table(index="CropWeek", columns="CropYear", values=metric, aggfunc="mean")
        return pv[pv.index <= 52]

    pv_old = _pivot(metric_old)
    pv_new = _pivot(metric_new)

    fig = go.Figure()

    for pv, clr_hist, clr_cur, clr_band, label in [
        (pv_old, "rgba(230,126,34,0.15)", C_OLD, "rgba(230,126,34,0.12)", "Old Crop"),
        (pv_new, "rgba(41,128,185,0.15)",  C_NEW, "rgba(41,128,185,0.12)",  "New Crop"),
    ]:
        if pv is None:
            continue
        hist_cols = [c2 for c2 in pv.columns if c2 != cur]
        hist = pv[hist_cols] if hist_cols else pv

        # faint historical lines
        for yr in hist_cols:
            fig.add_trace(go.Scatter(
                x=pv.index, y=hist[yr], mode="lines",
                line=dict(color=clr_hist, width=1),
                showlegend=False, hoverinfo="skip"
            ))

        # IQR band
        if len(hist_cols) >= 4:
            p25 = hist.quantile(0.25, axis=1)
            p75 = hist.quantile(0.75, axis=1)
            xs  = list(p75.index) + list(p75.index[::-1])
            fig.add_trace(go.Scatter(
                x=xs, y=list(p75.values) + list(p25.values[::-1]),
                fill="toself", fillcolor=clr_band,
                line=dict(width=0), name=f"{label} 25-75%", hoverinfo="skip"
            ))

        # current crop year — bold
        if cur in pv.columns:
            cy = pv[cur].dropna()
            fig.add_trace(go.Scatter(
                x=cy.index, y=cy.values, mode="lines+markers",
                name=f"{label} {cur}",
                line=dict(color=clr_cur, width=2.6),
                marker=dict(size=4, color=clr_cur),
                hovertemplate=f"<b>{label} {cur}</b><br>Week %{{x}}: %{{y:.1f}}<extra></extra>"
            ))

    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.12)")
    xticks = dict(
        tickvals=list(CROP_WEEK_TICKS.keys()),
        ticktext=[_MONTHS[(sm - 1 + i) % 12] for i in range(12)]
    )
    fig.update_layout(
        **_BASE, height=360,
        title=dict(text=title, font=dict(size=12, color="#444"), x=0),
        margin=dict(l=50, r=20, t=40, b=70),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(x=True), **xticks),
        yaxis=dict(**_ax(), title_text=ylabel, title_font_size=10),
    )
    return fig


def render_old_new(d_crops, color, commodity=""):
    old   = d_crops[d_crops["Crop"]=="Old"].set_index("Date").sort_index()
    other = d_crops[d_crops["Crop"]=="Other"].set_index("Date").sort_index()
    alla  = d_crops[d_crops["Crop"]=="All"].set_index("Date").sort_index()

    if old.empty and other.empty:
        st.info("No Old/Other crop data in selected range."); return

    # Gate on a category column that's actually populated for Old/Other rows
    # (MM Long, charted right below), not Total OI — Total OI has no crop
    # split available from LSEG at all (a documented, permanent gap) and is
    # always NaN on these rows, so gating on it wrongly blocked this whole
    # tab even for commodities (KC/CC/SB/CT/GC/SI/HG) that DO have real
    # Old/Other category data.
    other_check = d_crops[d_crops["Crop"]=="Other"]["MM Long"].dropna()
    if other_check.empty:
        st.info("Old/New crop split not available for this commodity."); return

    # Default crop-start month per commodity (forced for key markets)
    _forced_months = {"CT": 7, "KC": 9, "CC": 9}
    _forced_names  = {"CT": "July", "KC": "September", "CC": "September"}
    _default_month = _forced_months.get(commodity, CROP_START_MONTH)
    _forced_help   = (
        f"Default forced to {_forced_names[commodity]} for {commodity} to match the standard crop calendar. "
        "You can change it manually if needed."
        if commodity in _forced_months else None
    )

    with _exp("Seasonality  ·  Old vs New Crop (adjustable start)", expanded=True):
        wide_full = _on_seasonal_wide(d_crops)
        if not wide_full.empty:
            if st.session_state.get("_cy_start_comm") != commodity:
                st.session_state["cy_start_on"] = _default_month
                st.session_state["_cy_start_comm"] = commodity
            sm = st.selectbox("Crop year starts in", list(range(1,13)),
                              index=_default_month-1, format_func=lambda m: _MONTHS[m-1],
                              key="cy_start_on", help=_forced_help)
            wide_cy = wide_full.copy()
            wide_cy["CropYear"] = pd.to_datetime(wide_cy["Date"]).apply(lambda d: _crop_year_label(d, sm))
            wide_cy["CropWeek"] = pd.to_datetime(wide_cy["Date"]).apply(lambda d: _crop_week_num(d, sm))
            cur_cy = _current_crop_year_label(sm)
            st.markdown(
                f"<p style='font-size:.75rem;color:{GRAY};margin-bottom:4px'>"
                f"Each line = one crop year · Bold = current ({cur_cy}) · Shaded = 25–75th pct</p>",
                unsafe_allow_html=True)

            def _sc(metric, title, ylabel="k lots"):
                return _seas_chart(wide_cy, metric, title, color, ylabel, by_week=False, sm=sm)

            # ── Managed Money ─────────────────────────────────────────────────
            st.markdown("<div style='font-size:.75rem;font-weight:700;color:#374151;"
                        "margin:14px 0 6px;letter-spacing:.04em'>MANAGED MONEY</div>",
                        unsafe_allow_html=True)
            mm1, mm2, mm3 = st.columns(3)
            with mm1: _chart(_sc("MM Net Old",   "MM Net (Old)  ·  k lots"),   width='stretch')
            with mm2: _chart(_sc("MM Net New",   "MM Net (New)  ·  k lots"),   width='stretch')
            with mm3: _chart(_sc("MM Diff",      "MM Net (Old − New)  ·  k lots"),    width='stretch')
            mm4, mm5, _ = st.columns(3)
            with mm4: _chart(_sc("MM Long Old",  "MM Long (Old)  ·  k lots"),  width='stretch')
            with mm5: _chart(_sc("MM Long New",  "MM Long (New)  ·  k lots"),  width='stretch')
            mm6, mm7, _ = st.columns(3)
            with mm6: _chart(_sc("MM Short Old", "MM Short (Old)  ·  k lots"), width='stretch')
            with mm7: _chart(_sc("MM Short New", "MM Short (New)  ·  k lots"), width='stretch')

            # ── Commercial ────────────────────────────────────────────────────
            st.markdown("<div style='font-size:.75rem;font-weight:700;color:#374151;"
                        "margin:14px 0 6px;letter-spacing:.04em'>COMMERCIAL</div>",
                        unsafe_allow_html=True)
            cm1, cm2, cm3 = st.columns(3)
            with cm1: _chart(_sc("Comm Net Old",   "Comm Net (Old)  ·  k lots"),           width='stretch')
            with cm2: _chart(_sc("Comm Net New",   "Comm Net (New)  ·  k lots"),           width='stretch')
            with cm3: _chart(_sc("Comm Diff",      "Comm Net (Old − New)  ·  k lots"),     width='stretch')
            cm4, cm5, _ = st.columns(3)
            with cm4: _chart(_sc("Comm Long Old",  "Comm Long (Old)  ·  k lots"),          width='stretch')
            with cm5: _chart(_sc("Comm Long New",  "Comm Long (New)  ·  k lots"),          width='stretch')
            cm6, cm7, _ = st.columns(3)
            with cm6: _chart(_sc("Comm Short Old", "Comm Short (Old)  ·  k lots"),         width='stretch')
            with cm7: _chart(_sc("Comm Short New", "Comm Short (New)  ·  k lots"),         width='stretch')

            # ── Swap Dealers ──────────────────────────────────────────────────
            st.markdown("<div style='font-size:.75rem;font-weight:700;color:#374151;"
                        "margin:14px 0 6px;letter-spacing:.04em'>SWAP DEALERS</div>",
                        unsafe_allow_html=True)
            sw1, sw2, sw3 = st.columns(3)
            with sw1: _chart(_sc("Swap Net Old",   "Swap Net (Old)  ·  k lots"),           width='stretch')
            with sw2: _chart(_sc("Swap Net New",   "Swap Net (New)  ·  k lots"),           width='stretch')
            with sw3: _chart(_sc("Swap Diff",      "Swap Net (Old − New)  ·  k lots"),     width='stretch')
            sw4, sw5, _ = st.columns(3)
            with sw4: _chart(_sc("Swap Long Old",  "Swap Long (Old)  ·  k lots"),          width='stretch')
            with sw5: _chart(_sc("Swap Long New",  "Swap Long (New)  ·  k lots"),          width='stretch')
            sw6, sw7, _ = st.columns(3)
            with sw6: _chart(_sc("Swap Short Old", "Swap Short (Old)  ·  k lots"),         width='stretch')
            with sw7: _chart(_sc("Swap Short New", "Swap Short (New)  ·  k lots"),         width='stretch')

            # ── Other Reportables ─────────────────────────────────────────────
            st.markdown("<div style='font-size:.75rem;font-weight:700;color:#374151;"
                        "margin:14px 0 6px;letter-spacing:.04em'>OTHER REPORTABLES</div>",
                        unsafe_allow_html=True)
            or1, or2, or3 = st.columns(3)
            with or1: _chart(_sc("Other Net Old",   "Other Net (Old)  ·  k lots"),         width='stretch')
            with or2: _chart(_sc("Other Net New",   "Other Net (New)  ·  k lots"),         width='stretch')
            with or3: _chart(_sc("Other Diff",      "Other Net (Old − New)  ·  k lots"),   width='stretch')
            or4, or5, _ = st.columns(3)
            with or4: _chart(_sc("Other Long Old",  "Other Long (Old)  ·  k lots"),        width='stretch')
            with or5: _chart(_sc("Other Long New",  "Other Long (New)  ·  k lots"),        width='stretch')
            or6, or7, _ = st.columns(3)
            with or6: _chart(_sc("Other Short Old", "Other Short (Old)  ·  k lots"),       width='stretch')
            with or7: _chart(_sc("Other Short New", "Other Short (New)  ·  k lots"),       width='stretch')

            # ── Open Interest ─────────────────────────────────────────────────
            st.markdown("<div style='font-size:.75rem;font-weight:700;color:#374151;"
                        "margin:14px 0 6px;letter-spacing:.04em'>OPEN INTEREST</div>",
                        unsafe_allow_html=True)
            oi1, oi2, _ = st.columns(3)
            with oi1: _chart(_sc("OI Old %", "OI % (Old)", "%"), width='stretch')

    with _exp("Data table  ·  Old Crop / New Crop", expanded=False):
        common_dates = old.index.union(other.index).sort_values()[::-1][:30]
        common_dates_ext = old.index.union(other.index).sort_values()[::-1][:31]

        def _get_s(df, col, dates):
            return (df.reindex(dates)[col] / 1000) if col in df.columns else pd.Series(np.nan, index=dates)

        def _build_tbl(dates):
            data = {}
            for src, lbl in [("MM Net","MM Net Old"),("Comm Net","Comm Net Old")]:
                data[("Net · k lots", lbl)] = _get_s(old, src, dates).values
            for src, lbl in [("MM Net","MM Net New"),("Comm Net","Comm Net New")]:
                data[("Net · k lots", lbl)] = _get_s(other, src, dates).values
            for src, lbl in [("MM Long","Old"),("MM Short","Old"),("Producer Long","Old"),("Producer Short","Old")]:
                data[(src.replace("Producer","Prod"), lbl)] = _get_s(old, src, dates).values
            for src, lbl in [("MM Long","New"),("MM Short","New"),("Producer Long","New"),("Producer Short","New")]:
                data[(src.replace("Producer","Prod"), lbl)] = _get_s(other, src, dates).values
            data[("OI · k lots", "Old")] = _get_s(old, "Total OI", dates).values
            data[("OI · k lots", "New")] = _get_s(other, "Total OI", dates).values
            return pd.DataFrame(data, index=dates)

        tbl_df = _build_tbl(common_dates)
        tbl_df.index = pd.to_datetime(tbl_df.index).strftime("%d %b '%y")
        tbl_df.index.name = None
        st.markdown(_recap_html(tbl_df, scroll=True), unsafe_allow_html=True)

        tbl_ext = _build_tbl(common_dates_ext)
        chg_df = tbl_ext.diff(-1).iloc[:len(common_dates)]
        chg_df.index = pd.to_datetime(common_dates).strftime("%d %b '%y")
        chg_df.index.name = None
        st.markdown(
            f"<p style='font-size:.72rem;color:{GRAY};margin:8px 0 2px'>Weekly change  ·  k lots</p>",
            unsafe_allow_html=True)
        all_groups = {g for g, _ in chg_df.columns}
        st.markdown(_recap_html(chg_df, signed_groups=all_groups, scroll=True), unsafe_allow_html=True)

    # ── OI split ──────────────────────────────────────────────────────────────
    with _exp("Open Interest  ·  Old vs New Crop", expanded=False):
        dates = old.index.union(other.index).sort_values()
        fig_oi = go.Figure([
            go.Bar(x=dates, y=old.reindex(dates)["Total OI"]/1000, name="Old Crop",
                   marker=dict(color=C_OLD,opacity=0.85,line=dict(width=0)),
                   hovertemplate="<b>%{x|%d %b %y}</b><br>Old OI: %{y:.1f}k<extra></extra>"),
            go.Bar(x=dates, y=other.reindex(dates)["Total OI"]/1000, name="New Crop",
                   marker=dict(color=C_NEW,opacity=0.85,line=dict(width=0)),
                   hovertemplate="<b>%{x|%d %b %y}</b><br>New OI: %{y:.1f}k<extra></extra>"),
        ])
        fig_oi.update_layout(**_BASE, barmode="stack", height=300,
            title=dict(text="Open Interest — Old vs New Crop  ·  k lots",font=dict(size=12,color="#444"),x=0),
            margin=dict(l=50,r=12,t=38,b=68), bargap=0.18,
            legend=dict(orientation="h",y=-0.24,x=0.5,xanchor="center",font_size=10),
            xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
            yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
        _chart(fig_oi, width='stretch')

    # ── Net positions ─────────────────────────────────────────────────────────
    with _exp("Net Positions  ·  MM & Commercial", expanded=False):
        c1, c2 = st.columns(2)
        for col_c, (col, title) in zip([c1,c2],[("MM Net","Managed Money Net"),("Comm Net","Commercial (Prod) Net")]):
            with col_c:
                fig = go.Figure()
                for crop_df, lbl, clr in [(old,"Old Crop",C_OLD),(other,"New Crop",C_NEW)]:
                    if col in crop_df.columns:
                        fig.add_trace(go.Scatter(x=crop_df.index, y=crop_df[col]/1000, name=lbl,
                            line=dict(color=clr, width=2.2, shape="spline", smoothing=0.6),
                            hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>{lbl}: %{{y:.1f}}k<extra></extra>"))
                fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.15)")
                fig.update_layout(**_BASE, height=340,
                    title=dict(text=f"{title}  ·  k lots",font=dict(size=12,color="#444"),x=0),
                    margin=dict(l=50,r=20,t=40,b=70),
                    legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
                _chart(fig, width='stretch')

    # ── Gross legs ────────────────────────────────────────────────────────────
    with _exp("Gross Legs  ·  Old vs New Crop", expanded=False):
        gross_legs = [(c,t) for c,t in [
            ("MM Long","MM Long"),("MM Short","MM Short"),
            ("Producer Long","Comm Long"),("Producer Short","Comm Short"),
        ] if c in old.columns or c in other.columns]
        for col, title in gross_legs:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                for crop_df, lbl, clr in [(old,"Old",C_OLD),(other,"New",C_NEW)]:
                    if col in crop_df.columns:
                        fig.add_trace(go.Scatter(x=crop_df.index, y=crop_df[col]/1000, name=lbl,
                            line=dict(color=clr,width=2.2,shape="spline",smoothing=0.6),
                            hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>{lbl}: %{{y:.1f}}k<extra></extra>"))
                fig.update_layout(**_BASE, height=300,
                    title=dict(text=f"{title}  ·  k lots",font=dict(size=12,color="#444"),x=0),
                    margin=dict(l=50,r=20,t=40,b=70),
                    legend=dict(orientation="h",y=-0.24,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
                _chart(fig, width='stretch')
            with c2:
                dates2 = old.index.union(other.index).sort_values()
                fig2 = go.Figure([
                    go.Bar(x=dates2, y=old.reindex(dates2)[col]/1000 if col in old.columns else None,
                           name="Old", marker=dict(color=C_OLD,opacity=0.85,line=dict(width=0)),
                           hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>Old: %{{y:.1f}}k<extra></extra>"),
                    go.Bar(x=dates2, y=other.reindex(dates2)[col]/1000 if col in other.columns else None,
                           name="New", marker=dict(color=C_NEW,opacity=0.85,line=dict(width=0)),
                           hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>New: %{{y:.1f}}k<extra></extra>"),
                ])
                fig2.update_layout(**_BASE, barmode="stack", height=280,
                    title=dict(text=f"{title} — Stacked  ·  k lots",font=dict(size=12,color="#444"),x=0),
                    margin=dict(l=50,r=20,t=40,b=70), bargap=0.12,
                    legend=dict(orientation="h",y=-0.26,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
                _chart(fig2, width='stretch')




# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRADERS
# ══════════════════════════════════════════════════════════════════════════════
CIT_TRADER_GROUPS = {
    "Spec":          ["Traders Spec Long","Traders Spec Short","Traders Spec Spread"],
    "Commercial":    ["Traders Comm Long","Traders Comm Short"],
    "Index":         ["Traders Index Long","Traders Index Short"],
    "All Reportable":["Traders Tot Rept Long","Traders Tot Rept Short"],
}
DISAGG_TRADER_GROUPS = {
    "Managed Money": ["Traders MM Long","Traders MM Short","Traders MM Spread"],
    "Swap Dealers":  ["Traders Swap Long","Traders Swap Short","Traders Swap Spread"],
    "Other Rept":    ["Traders Other Long","Traders Other Short","Traders Other Spread"],
    "Producer":      ["Traders Producer Long","Traders Producer Short"],
    "All Reportable":["Traders Tot Rept Long","Traders Tot Rept Short","Traders Total"],
}
TRADER_COLORS = [C_LONG, C_SHORT, "#94a3b8", C_NET, C_PRICE]

def render_traders(d, report, color):
    grp_map = CIT_TRADER_GROUPS if report=="CIT" else DISAGG_TRADER_GROUPS
    all_t   = [c for g in grp_map.values() for c in g if c in d.columns]

    latest = d.iloc[-1]
    kpi_items = [(c.replace("Traders ",""), f"{int(latest.get(c,0))}", "")
                 for c in all_t[:8] if pd.notna(latest.get(c))]
    kpi_row(kpi_items, color)

    group = st.selectbox("Group", list(grp_map.keys()), key="traders_grp")
    sel_cols = [c for c in grp_map[group] if c in d.columns]
    nice     = [c.replace("Traders ","") for c in sel_cols]

    fig = go.Figure()
    for col, name, clr in zip(sel_cols, nice, TRADER_COLORS):
        fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=name,
            line=dict(color=clr,width=2.0),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{name}: %{{y:.0f}}<extra></extra>"))
    fig.update_layout(
        **_BASE, height=360,
        title=dict(text=f"Traders in Each Category — {group}",font=dict(size=12,color="#333"),x=0),
        margin=dict(l=50,r=20,t=42,b=70),
        legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10),
        xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
        yaxis=dict(**_ax(),title_text="# traders",title_font_size=10))
    _chart(fig, width='stretch')

    with _exp("Weekly change — trader counts", expanded=False):
        cols_w = st.columns(min(len(sel_cols),3))
        for i,(col,name) in enumerate(zip(sel_cols,nice)):
            with cols_w[i%3]:
                chg = d[col].diff().tail(13)
                dates_b = d["Date"].tail(13)
                fb = go.Figure(go.Bar(x=dates_b, y=chg,
                    marker=dict(color=[C_LONG if v>=0 else C_SHORT for v in chg],
                                opacity=0.82,line=dict(width=0)),
                    hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>Δ: %{{y:+.0f}}<extra></extra>"))
                fb.add_hline(y=0,line_width=1,line_color="rgba(0,0,0,0.14)")
                fb.update_layout(**_BASE,height=240,
                    title=dict(text=f"{name} — Δ",font=dict(size=10,color="#444"),x=0),
                    margin=dict(l=40,r=8,t=32,b=60),showlegend=False,
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax()))
                _chart(fb, width='stretch')

    show_table(d, all_t, sel_cols, "Data table — trader counts", scale=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CONCENTRATION
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — RECAP
# ══════════════════════════════════════════════════════════════════════════════
_RECAP_GROUP_BG = {
    "Gross Positions": "#d1d5db",
    "NET":             "#bae6fd",
    "SPREAD":          "#fed7aa",
    "SP":              "#fed7aa",
    "MM+O+NR":         "#a7f3d0",
    "OI":              "#e5e7eb",
    "OI · k lots":     "#e5e7eb",
    "Δ 1w":            "#f9a8d4",
    "OI %":            "#1e3a8a",
    "Nominal (M USD)": "#d1fae5",
    "Nominal (M GBP)": "#fef9c3",
    "# Traders":       "#ede9fe",
    "k lots / Trader": "#fef08a",
    "Old Crop":        "#fde68a",
    "New Crop":        "#bbf7d0",
    "Net · k lots":    "#bae6fd",
    "MM Long":         "#d1fae5",
    "MM Short":        "#fee2e2",
    "Prod Long":       "#e0e7ff",
    "Prod Short":      "#fce7f3",
    "Rollex Px":       "#fef3c7",
    "Longs · k lots":  "#d1fae5",
    "Shorts · k lots": "#fee2e2",
    "Long % OI":       "#bbf7d0",
    "Short % OI":      "#fecaca",
}
_RECAP_GROUP_TEXT = {
    "OI %": "#ffffff",
}
_CHANGE_BG = "#f9a8d4"

_RECAP_CSS = """
<style>
.rtbl{border-collapse:collapse;font-size:.67rem;width:100%;font-family:-apple-system,sans-serif}
.rtbl th,.rtbl td{border:1px solid #e5e7eb;padding:2px 4px;text-align:center}
.rtbl td{white-space:nowrap}
.rtbl .grp{text-align:center;font-weight:700;font-size:.64rem;letter-spacing:.02em;white-space:normal;max-width:60px}
.rtbl .idx{text-align:left;font-weight:600;color:#374151;background:#f9fafb;min-width:52px;white-space:nowrap}
.rtbl .sub{background:#f9fafb;font-size:.60rem;color:#555;font-weight:600;text-align:center;white-space:normal;max-width:48px;line-height:1.25}
.rtbl tbody tr:hover td{background:#f0f9ff!important}
.rpos{color:#16a34a}.rneg{color:#dc2626}
.rtbl .gsep{box-shadow:inset 3px 0 0 #6b7280}
.rtbl .gsub{box-shadow:inset 1.5px 0 0 #b8c0cc}
.rtbl th.sub[data-tt]{position:relative;cursor:help}
.rtbl th.sub[data-tt]::after{
  content:attr(data-tt);
  position:absolute;
  top:calc(100% + 4px);
  left:50%;
  transform:translateX(-50%);
  background:#1e293b;
  color:#f1f5f9;
  padding:5px 10px;
  border-radius:5px;
  font-size:.69rem;
  font-weight:400;
  white-space:nowrap;
  z-index:9999;
  pointer-events:none;
  opacity:0;
  transition:opacity .15s ease;
  box-shadow:0 2px 8px rgba(0,0,0,.30);
  line-height:1.5;
}
.rtbl th.sub[data-tt]::before{
  content:'';
  position:absolute;
  top:100%;
  left:50%;
  transform:translateX(-50%);
  border:5px solid transparent;
  border-bottom-color:#1e293b;
  z-index:9999;
  opacity:0;
  transition:opacity .15s ease;
  pointer-events:none;
}
.rtbl th.sub[data-tt]:hover::after,.rtbl th.sub[data-tt]:hover::before{opacity:1}
</style>
"""

_COLUMN_TOOLTIPS = {
    ("NET", "Large+Small"):      "Large Spec Net + Non-Rep Net",
    ("NET", "Lrg+Sml+Idx"):     "Large Spec Net + Non-Rep Net + Index Net",
    ("MM+O+NR", "Long"):              "MM Long + Other Long + Non-Rep Long",
    ("MM+O+NR", "Short"):             "MM Short + Other Short + Non-Rep Short",
    ("NET", "MM+O+NR"):               "MM Net + Other Net + Non-Rep Net",
    ("NET", "Rest"):                  "Other Net + Non-Rep Net",
    ("NET", "MM"):                    "Managed Money Net",
    ("NET", "Comm"):                  "Producer/Commercial Net",
    ("Gross Positions", "L+S Long"):  "Large Long + Small Long (Spec + Non-Rep)",
    ("Gross Positions", "L+S Short"): "Large Short + Small Short (Spec + Non-Rep)",
    ("Gross Positions", "L+S+I Long"):  "Large + Small + Index Long (all ex-Commercial)",
    ("Gross Positions", "L+S+I Short"): "Large + Small + Index Short (all ex-Commercial)",
}

# Sub-group separators within Gross Positions (medium border before each new pair)
_RECAP_COL_SUBSEP = {
    ("Gross Positions", "L+S Long"),      # CIT: after Large pair
    ("Gross Positions", "Index Long"),    # CIT: after L+S pair
    ("Gross Positions", "L+S+I Long"),    # CIT: after Index pair
    ("Gross Positions", "Comm Long"),     # CIT + Disagg: start of commercial
    ("Gross Positions", "Other Long"),    # Disagg: after MM pair
    ("Gross Positions", "Non-Rep Long"),  # Disagg: after Other pair
    ("Gross Positions", "Swap Long"),     # Disagg: after Non-Rep pair
    ("MM+O+NR", "Long"),                  # Disagg: aggregate pair separator
}

def _recap_html(df, signed=False, change_table=False, scroll=False, signed_groups=None,
                pct_groups=None, pct_subcols=None, signed_rows=None, z_rows=None, max_height=None):
    if df.empty: return ""
    cols = list(df.columns)
    # Build group spans
    groups, prev = [], None
    for c in cols:
        g = c[0]
        if g == prev: groups[-1][1] += 1
        else: groups.append([g, 1]); prev = g

    # Pre-compute border class per column: gsep = major group start, gsub = sub-group within Gross
    col_sep = []
    ci = 0
    for g, span in groups:
        for j in range(span):
            c = cols[ci + j]
            if j == 0:
                col_sep.append("gsep")
            elif c in _RECAP_COL_SUBSEP:
                col_sep.append("gsub")
            else:
                col_sep.append("")
        ci += span

    # Header row 1 — merged group headers (bold left border on each group)
    h1 = '<tr><th class="idx sub"></th>'
    for g, span in groups:
        bg = _RECAP_GROUP_BG.get(g, "#f9fafb")
        fg = _RECAP_GROUP_TEXT.get(g, "#111827")
        h1 += (f'<th colspan="{span}" class="grp" '
               f'style="background:{bg};color:{fg};box-shadow:inset 3px 0 0 #6b7280">{g}</th>')
    h1 += '</tr>'

    # Header row 2 — sub-column names (with hover tooltips where defined)
    h2 = '<tr><th class="idx sub"></th>'
    for i, c in enumerate(cols):
        g = c[0]
        sep_cls = col_sep[i]
        tip = _COLUMN_TOOLTIPS.get(c)
        tip_attr = f' data-tt="{tip}"' if tip else ''
        label = f'{c[1]}&thinsp;<span style="font-size:.6rem;color:#9ca3af;font-weight:400">ⓘ</span>' if tip else c[1]
        fsz = ";font-size:.62rem" if len(c[1]) > 9 else ""
        cls_str = f"sub {sep_cls}".strip()
        if g in _RECAP_GROUP_TEXT:
            bg = _RECAP_GROUP_BG.get(g, "#f9fafb")
            fg = _RECAP_GROUP_TEXT[g]
            h2 += f'<th class="{cls_str}" style="background:{bg};color:{fg}{fsz}"{tip_attr}>{label}</th>'
        else:
            style_attr = f' style="font-size:.62rem"' if fsz else ''
            h2 += f'<th class="{cls_str}"{style_attr}{tip_attr}>{label}</th>'
    h2 += '</tr>'

    # Body rows
    body = ""
    for idx, row in df.iterrows():
        body += f'<tr><td class="idx">{idx}</td>'
        for i, c in enumerate(cols):
            sep_cls = col_sep[i]
            v = row[c]
            if pd.isna(v): body += f'<td class="{sep_cls}">—</td>'; continue
            is_z_row  = z_rows and idx in z_rows
            use_signed = (signed or change_table
                          or (signed_rows and idx in signed_rows)
                          or (signed_groups and isinstance(c, tuple) and c[0] in signed_groups))
            use_pct = ((pct_groups and isinstance(c, tuple) and c[0] in pct_groups) or
                       (pct_subcols and isinstance(c, tuple) and c in pct_subcols))
            fmt = ".2f" if is_z_row else ".1f"
            if use_signed:
                txt = f"{v:+{fmt}}"
                cls = "rpos" if v > 0 else ("rneg" if v < 0 else "")
            elif use_pct:
                txt = f"{v:.1f}%"; cls = ""
            else:
                txt = f"{v:{fmt}}"; cls = ""
            full_cls = f"{cls} {sep_cls}".strip()
            body += f'<td class="{full_cls}">{txt}</td>'
        body += '</tr>'

    if max_height is not None:
        scroll_style = f"overflow-x:auto;overflow-y:auto;max-height:{max_height}px;"
    elif scroll:
        scroll_style = "overflow-x:auto;overflow-y:auto;max-height:420px;"
    else:
        scroll_style = "overflow-x:auto;"
    return (f'{_RECAP_CSS}<div style="{scroll_style}margin-bottom:6px">'
            f'<table class="rtbl"><thead>{h1}{h2}</thead>'
            f'<tbody>{body}</tbody></table></div>')

def _build_recap_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    cols = {}

    if report == "CIT":
        for src, dst in [("Spec Long","Large Long"),("Spec Short","Large Short"),
                         ("Non Rep Long","Small Long"),("Non Rep Short","Small Short")]:
            if src in d.columns: cols[("Gross Positions", dst)] = gc(src) / 1000
        cols[("Gross Positions", "L+S Long")]    = (gc("Spec Long") + gc("Non Rep Long"))  / 1000
        cols[("Gross Positions", "L+S Short")]   = (gc("Spec Short")+ gc("Non Rep Short")) / 1000
        for src, dst in [("Index Long","Index Long"),("Index Short","Index Short")]:
            if src in d.columns: cols[("Gross Positions", dst)] = gc(src) / 1000
        cols[("Gross Positions", "L+S+I Long")]  = (gc("Spec Long") + gc("Non Rep Long")  + gc("Index Long"))  / 1000
        cols[("Gross Positions", "L+S+I Short")] = (gc("Spec Short")+ gc("Non Rep Short") + gc("Index Short")) / 1000
        for src, dst in [("Comm Long","Comm Long"),("Comm Short","Comm Short")]:
            if src in d.columns: cols[("Gross Positions", dst)] = gc(src) / 1000

        cols[("NET", "Large")]        = gc("Spec Net")   / 1000
        cols[("NET", "Small")]        = gc("Non Rep Net") / 1000
        cols[("NET", "Index")]        = gc("Index Net")   / 1000
        cols[("NET", "Comm")]         = gc("Comm Net")    / 1000
        cols[("NET", "Large+Small")]  = (gc("Spec Net") + gc("Non Rep Net")) / 1000
        cols[("NET", "Lrg+Sml+Idx")] = (gc("Spec Net") + gc("Non Rep Net") + gc("Index Net")) / 1000

        if "Spec Spread" in d.columns:
            cols[("SPREAD", "Spec Spread")] = gc("Spec Spread") / 1000

        cols[("OI", "Total OI")] = gc("Total OI") / 1000

    else:  # Disagg
        for src, dst in [
            ("MM Long",       "MM Long"),    ("MM Short",       "MM Short"),
            ("Other Long",    "Other Long"), ("Other Short",    "Other Short"),
            ("Non Rep Long",  "Non-Rep Long"),("Non Rep Short", "Non-Rep Short"),
            ("Swap Long",     "Swap Long"),  ("Swap Short",     "Swap Short"),
            ("Producer Long", "Comm Long"),  ("Producer Short", "Comm Short"),
        ]:
            if src in d.columns:
                cols[("Gross Positions", dst)] = gc(src) / 1000

        cols[("MM+O+NR", "Long")]  = (gc("MM Long")  + gc("Other Long")  + gc("Non Rep Long"))  / 1000
        cols[("MM+O+NR", "Short")] = (gc("MM Short") + gc("Other Short") + gc("Non Rep Short")) / 1000

        cols[("NET", "MM")]      = gc("MM Net")   / 1000
        cols[("NET", "Rest")]    = (gc("Other Net") + gc("Non Rep Net")) / 1000
        cols[("NET", "MM+O+NR")] = (gc("MM Net") + gc("Other Net") + gc("Non Rep Net")) / 1000
        cols[("NET", "Swap")]    = gc("Swap Net")  / 1000
        cols[("NET", "Comm")]    = gc("Comm Net")  / 1000

        for src, dst in [
            ("MM Spread",    "MM Spread"),
            ("Other Spread", "Other Spread"),
            ("Swap Spread",  "Swap Spread"),
        ]:
            if src in d.columns:
                cols[("SP", dst)] = gc(src) / 1000

        cols[("OI", "Total OI")] = gc("Total OI") / 1000

    # Add Rollex Px level — diff in summary gives absolute price change
    cols[("Rollex Px", "Level")] = gc("Px")

    body = pd.DataFrame(cols)
    body.index = pd.to_datetime(d["Date"])
    body = body.iloc[::-1]  # newest first

    row_1w, row_4w = {}, {}
    for c in body.columns:
        if len(body) >= 2:
            row_1w[c] = body.iloc[0][c] - body.iloc[1][c]
        if len(body) >= 5:
            row_4w[c] = body.iloc[0][c] - body.iloc[4][c]

    # Rollex Px Δ% 1w — body (newest first): pct_change(-1) = row vs the one after (older)
    px_lvl = body[("Rollex Px", "Level")]
    body[("Rollex Px", "Δ% 1w")] = px_lvl.pct_change(-1) * 100

    # Z-Score, Avg, Min, Max computed over full history (all columns incl. Δ% 1w)
    row_z, row_avg, row_min, row_max = {}, {}, {}, {}
    for c in body.columns:
        series = body[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) >= 4:
            mu, sigma = series.mean(), series.std()
            row_z[c]   = (series.iloc[0] - mu) / sigma if sigma > 0 else 0.0
            row_avg[c] = mu
            row_min[c] = series.min()
            row_max[c] = series.max()

    summary = pd.DataFrame(
        [row_1w, row_4w, row_z, row_avg, row_min, row_max],
        index=["Δ 1w", "Δ 1m", "Z-Score", "Avg", "Min", "Max"],
        columns=body.columns,
    )

    # Override summary Δ% 1w with proper cumulative % change (not diff of pct)
    summary[("Rollex Px", "Δ% 1w")] = np.nan
    if len(px_lvl) >= 2 and px_lvl.iloc[1] != 0:
        summary.loc["Δ 1w", ("Rollex Px", "Δ% 1w")] = (px_lvl.iloc[0] / px_lvl.iloc[1] - 1) * 100
    if len(px_lvl) >= 5 and px_lvl.iloc[4] != 0:
        summary.loc["Δ 1m", ("Rollex Px", "Δ% 1w")] = (px_lvl.iloc[0] / px_lvl.iloc[4] - 1) * 100

    body.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in body.index]
    return summary, body


def _build_oi_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    total_oi = gc("Total OI") / 1000

    if report == "CIT":
        oi_cols = {
            "Large Spec": (gc("Spec Long") + gc("Spec Short")) / 2 / 1000 + gc("Spec Spread") / 1000,
            "Small Spec": (gc("Non Rep Long") + gc("Non Rep Short")) / 2 / 1000,
            "Index":      (gc("Index Long") + gc("Index Short")) / 2 / 1000,
            "Commercial": (gc("Comm Long") + gc("Comm Short")) / 2 / 1000,
        }
    else:
        oi_cols = {
            "MM":         (gc("MM Long") + gc("MM Short")) / 2 / 1000 + gc("MM Spread") / 1000,
            "Other":      (gc("Other Long") + gc("Other Short")) / 2 / 1000 + gc("Other Spread") / 1000,
            "Swap":       (gc("Swap Long") + gc("Swap Short")) / 2 / 1000 + gc("Swap Spread") / 1000,
            "Commercial": (gc("Producer Long") + gc("Producer Short")) / 2 / 1000,
            "Non-Rep":    (gc("Non Rep Long") + gc("Non Rep Short")) / 2 / 1000,
        }

    oi_df = pd.DataFrame(oi_cols)
    oi_df.index = pd.to_datetime(d["Date"])
    oi_df = oi_df.iloc[::-1].iloc[:20]  # newest first, last 20

    total_s = pd.Series(total_oi.values, index=pd.to_datetime(d["Date"])).iloc[::-1].iloc[:20]
    pct_df  = oi_df.div(total_s.values, axis=0) * 100   # % before adding Total OI col

    oi_df["Total OI"] = total_s.values                  # add Total OI after pct calc
    chg_df = oi_df.diff(-1)

    cats     = list(oi_cols.keys())
    all_cats = cats + ["Total OI"]
    combined = pd.concat([oi_df[all_cats], pct_df[cats], chg_df[all_cats]], axis=1)
    combined.columns = pd.MultiIndex.from_tuples(
        [("OI · k lots", c) for c in all_cats] +
        [("OI %",        c) for c in cats] +
        [("Δ 1w",        c) for c in all_cats]
    )
    combined.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in combined.index]
    return combined


def _build_gross_legs_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    total_oi = gc("Total OI")

    if report == "CIT":
        cats = ["Large Spec", "Index", "Commercial", "Non-Rep"]
        longs = {
            "Large Spec": (gc("Spec Long")  + gc("Spec Spread")) / 1000,
            "Index":       gc("Index Long")  / 1000,
            "Commercial":  gc("Comm Long")   / 1000,
            "Non-Rep":     gc("Non Rep Long") / 1000,
        }
        shorts = {
            "Large Spec": (gc("Spec Short") + gc("Spec Spread")) / 1000,
            "Index":       gc("Index Short")  / 1000,
            "Commercial":  gc("Comm Short")   / 1000,
            "Non-Rep":     gc("Non Rep Short") / 1000,
        }
    else:
        cats = ["MM", "Other", "Swap", "Commercial", "Non-Rep"]
        longs = {
            "MM":         (gc("MM Long")    + gc("MM Spread"))    / 1000,
            "Other":      (gc("Other Long") + gc("Other Spread")) / 1000,
            "Swap":       (gc("Swap Long")  + gc("Swap Spread"))  / 1000,
            "Commercial":  gc("Producer Long") / 1000,
            "Non-Rep":     gc("Non Rep Long")  / 1000,
        }
        shorts = {
            "MM":         (gc("MM Short")    + gc("MM Spread"))    / 1000,
            "Other":      (gc("Other Short") + gc("Other Spread")) / 1000,
            "Swap":       (gc("Swap Short")  + gc("Swap Spread"))  / 1000,
            "Commercial":  gc("Producer Short") / 1000,
            "Non-Rep":     gc("Non Rep Short")  / 1000,
        }

    long_df  = pd.DataFrame(longs).iloc[::-1].iloc[:20]
    short_df = pd.DataFrame(shorts).iloc[::-1].iloc[:20]
    idx      = pd.to_datetime(d["Date"]).iloc[::-1].iloc[:20]
    tot_s    = (total_oi / 1000).iloc[::-1].iloc[:20].values

    long_pct  = long_df.div(tot_s, axis=0) * 100
    short_pct = short_df.div(tot_s, axis=0) * 100

    combined = pd.concat([long_df[cats], short_df[cats], long_pct[cats], short_pct[cats]], axis=1)
    combined.columns = pd.MultiIndex.from_tuples(
        [("Longs · k lots",  c) for c in cats] +
        [("Shorts · k lots", c) for c in cats] +
        [("Long % OI",       c) for c in cats] +
        [("Short % OI",      c) for c in cats]
    )
    combined.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in idx]
    return combined


def _build_nominal_df(d, commodity, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    size = CONTRACT_SIZE.get(commodity, 1)
    unit = CONTRACT_UNIT.get(commodity, "MT")
    ccy  = "GBP" if commodity == "LCC" else "USD"
    px   = gc("Px")

    # mult: M currency per 1 contract
    if commodity in CENTS_QUOTED:
        mult = px * size / 100 / 1_000_000   # cents/lb → USD
    else:
        mult = px * size / 1_000_000          # USD or GBP per MT/oz/lb (already whole-currency)

    grp = f"Nominal (M {ccy})"

    if report == "CIT":
        cols = {
            (grp, "Spec Net"):   (gc("Spec Net") + gc("Non Rep Net").fillna(0)) * mult,
            (grp, "Net Index"):  gc("Index Net") * mult,
            (grp, "Comm Long"):  gc("Comm Long") * mult,
            (grp, "Comm Short"): gc("Comm Short") * mult,
        }
    else:
        cols = {
            (grp, "Spec Net"):    (gc("MM Net") + gc("Other Net").fillna(0) + gc("Non Rep Net").fillna(0)) * mult,
            (grp, "Swap Net"):    gc("Swap Net")       * mult,
            (grp, "Comm Long"):   gc("Producer Long")  * mult,
            (grp, "Comm Short"):  gc("Producer Short") * mult,
        }

    body = pd.DataFrame(cols)
    body.index = pd.to_datetime(d["Date"])
    body = body.iloc[::-1].iloc[:20]

    row_1w, row_4w = {}, {}
    for c in body.columns:
        if len(body) >= 2: row_1w[c] = body.iloc[0][c] - body.iloc[1][c]
        if len(body) >= 5: row_4w[c] = body.iloc[0][c] - body.iloc[4][c]

    summary = pd.DataFrame([row_1w, row_4w], index=["+/-1w", "+/-4w"], columns=body.columns)
    body.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in body.index]
    return summary, body


def _summary_and_body(cols_dict, d_index, n=20):
    body = pd.DataFrame(cols_dict)
    body.index = pd.to_datetime(d_index)
    body = body.iloc[::-1].iloc[:n]
    row_1w, row_4w = {}, {}
    for c in body.columns:
        if len(body) >= 2: row_1w[c] = body.iloc[0][c] - body.iloc[1][c]
        if len(body) >= 5: row_4w[c] = body.iloc[0][c] - body.iloc[4][c]
    summary = pd.DataFrame([row_1w, row_4w], index=["+/-1w", "+/-4w"], columns=body.columns)
    body.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in body.index]
    return summary, body


def _build_traders_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty: return pd.DataFrame(), pd.DataFrame()
    def gc(name): return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    if report == "CIT":
        cols = {
            ("# Traders", "Lrg Long"):  gc("Traders Spec Long"),
            ("# Traders", "Lrg Short"): gc("Traders Spec Short"),
            ("# Traders", "Idx Long"):  gc("Traders Index Long"),
            ("# Traders", "Idx Short"): gc("Traders Index Short"),
            ("# Traders", "Lrg Spread"):gc("Traders Spec Spread"),
            ("# Traders", "Comm Long"): gc("Traders Comm Long"),
            ("# Traders", "Comm Short"):gc("Traders Comm Short"),
        }
    else:
        cols = {
            ("# Traders", "MM Long"):    gc("Traders MM Long"),
            ("# Traders", "MM Short"):   gc("Traders MM Short"),
            ("# Traders", "MM Spread"):  gc("Traders MM Spread"),
            ("# Traders", "Swap Long"):  gc("Traders Swap Long"),
            ("# Traders", "Swap Short"): gc("Traders Swap Short"),
            ("# Traders", "Swap Spread"):gc("Traders Swap Spread"),
            ("# Traders", "Other Long"): gc("Traders Other Long"),
            ("# Traders", "Other Short"):gc("Traders Other Short"),
            ("# Traders", "Prod Long"):  gc("Traders Producer Long"),
            ("# Traders", "Prod Short"): gc("Traders Producer Short"),
        }
    return _summary_and_body(cols, d["Date"])


def _build_lots_per_trader_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty: return pd.DataFrame(), pd.DataFrame()
    def gc(name): return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)
    def safe_div(num, den): return num.where(den == 0, num / den.replace(0, np.nan))

    if report == "CIT":
        cols = {
            ("k lots / Trader", "Lrg Long"):  safe_div(gc("Spec Long")  / 1000, gc("Traders Spec Long")),
            ("k lots / Trader", "Lrg Short"): safe_div(gc("Spec Short") / 1000, gc("Traders Spec Short")),
            ("k lots / Trader", "Idx Long"):  safe_div(gc("Index Long") / 1000, gc("Traders Index Long")),
            ("k lots / Trader", "Idx Short"): safe_div(gc("Index Short")/ 1000, gc("Traders Index Short")),
            ("k lots / Trader", "Lrg Spread"):safe_div(gc("Spec Spread")/ 1000, gc("Traders Spec Spread")),
            ("k lots / Trader", "Comm Long"): safe_div(gc("Comm Long")  / 1000, gc("Traders Comm Long")),
            ("k lots / Trader", "Comm Short"):safe_div(gc("Comm Short") / 1000, gc("Traders Comm Short")),
        }
    else:
        cols = {
            ("k lots / Trader", "MM Long"):    safe_div(gc("MM Long")       / 1000, gc("Traders MM Long")),
            ("k lots / Trader", "MM Short"):   safe_div(gc("MM Short")      / 1000, gc("Traders MM Short")),
            ("k lots / Trader", "MM Spread"):  safe_div(gc("MM Spread")     / 1000, gc("Traders MM Spread")),
            ("k lots / Trader", "Swap Long"):  safe_div(gc("Swap Long")     / 1000, gc("Traders Swap Long")),
            ("k lots / Trader", "Swap Short"): safe_div(gc("Swap Short")    / 1000, gc("Traders Swap Short")),
            ("k lots / Trader", "Swap Spread"):safe_div(gc("Swap Spread")   / 1000, gc("Traders Swap Spread")),
            ("k lots / Trader", "Other Long"): safe_div(gc("Other Long")    / 1000, gc("Traders Other Long")),
            ("k lots / Trader", "Other Short"):safe_div(gc("Other Short")   / 1000, gc("Traders Other Short")),
            ("k lots / Trader", "Prod Long"):  safe_div(gc("Producer Long") / 1000, gc("Traders Producer Long")),
            ("k lots / Trader", "Prod Short"): safe_div(gc("Producer Short")/ 1000, gc("Traders Producer Short")),
        }
    return _summary_and_body(cols, d["Date"])


def render_recap(d, report, color, commodity="KC", is_options=False):
    if d.empty:
        st.warning("No data for the selected filters."); return

    summary, body = _build_recap_df(d, report)
    if body.empty:
        st.warning("No data."); return

    view = body.iloc[:20]

    _PX_PCT = {("Rollex Px", "Δ% 1w")}

    with _exp("Change summary  ·  k lots", expanded=True):
        st.markdown(_recap_html(summary,
                                signed_rows={"Δ 1w", "Δ 1m", "Z-Score"},
                                z_rows={"Z-Score"},
                                pct_subcols=_PX_PCT,
                                max_height=148), unsafe_allow_html=True)

    with _exp("Historical positions  ·  k lots", expanded=True):
        st.markdown(_recap_html(view, scroll=True, pct_subcols=_PX_PCT), unsafe_allow_html=True)

    with _exp("Weekly change  ·  k lots", expanded=True):
        chg = view.diff(-1)
        st.markdown(_recap_html(chg, signed=True, change_table=True, scroll=True, pct_subcols=_PX_PCT), unsafe_allow_html=True)

        # Stats of weekly changes over full selected period
        chg_full = body.diff(-1).dropna()
        if not chg_full.empty:
            rz, ra, rn, rx = {}, {}, {}, {}
            for c in chg_full.columns:
                s = chg_full[c].replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) >= 4:
                    mu, sigma = s.mean(), s.std()
                    ra[c] = mu
                    rn[c] = s.min()
                    rx[c] = s.max()
                    if not chg.empty and c in chg.columns:
                        v = chg.iloc[0][c]
                        rz[c] = (v - mu) / sigma if pd.notna(v) and sigma > 0 else np.nan
            chg_stats = pd.DataFrame(
                [rz, ra, rn, rx],
                index=["Z-Score Δ", "Avg Δ", "Min Δ", "Max Δ"],
                columns=chg_full.columns,
            )
            st.markdown(
                "<p style='font-size:.72rem;color:#6e6e73;margin:10px 0 2px'>"
                "Weekly Δ stats  ·  selected period</p>",
                unsafe_allow_html=True,
            )
            st.markdown(_recap_html(chg_stats, signed=True, z_rows={"Z-Score Δ"}, pct_subcols=_PX_PCT), unsafe_allow_html=True)

    oi_tbl = _build_oi_df(d, report)
    with _exp("OI by category  ·  k lots  &  %", expanded=False):
        st.markdown(_recap_html(oi_tbl, signed_groups={"Δ 1w"}, pct_groups={"OI %"}, scroll=True), unsafe_allow_html=True)

    gross_tbl = _build_gross_legs_df(d, report)
    with _exp("Gross legs by category  ·  k lots  &  % OI", expanded=False):
        st.markdown(
            "<p style='font-size:.72rem;color:#6e6e73;margin:0 0 6px'>"
            "Long/Short include spreading positions. % columns are each leg divided by Total OI.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(_recap_html(gross_tbl, pct_groups={"Long % OI", "Short % OI"}, scroll=True), unsafe_allow_html=True)

    nom_summary, nom_body = _build_nominal_df(d, commodity, report)
    if not nom_body.empty:
        ccy = "GBP" if commodity == "LCC" else "USD"
        with _exp(f"Nominal Exposure  ·  M {ccy}", expanded=False):
            _spec_note = ("Spec Net = MM Net + Other Net + Non-Rep Net"
                          if report != "CIT" else
                          "Spec Net = Large Spec Net + Non-Rep Net")
            st.markdown(
                f"<div style='font-size:.72rem;color:#6b7280;margin-bottom:6px'>"
                f"{_spec_note}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(_recap_html(nom_summary, signed=True), unsafe_allow_html=True)
            st.markdown(_recap_html(nom_body, scroll=True), unsafe_allow_html=True)

    tr_summary, tr_body = _build_traders_df(d, report)
    if not tr_body.empty:
        with _exp("# of Traders", expanded=False):
            st.markdown(_recap_html(tr_summary, signed=True), unsafe_allow_html=True)
            st.markdown(_recap_html(tr_body, scroll=True), unsafe_allow_html=True)

    lpt_summary, lpt_body = _build_lots_per_trader_df(d, report)
    if not lpt_body.empty:
        with _exp("k lots / Trader  (avg position size per trader)", expanded=False):
            st.markdown(_recap_html(lpt_summary, signed=True), unsafe_allow_html=True)
            st.markdown(_recap_html(lpt_body, scroll=True), unsafe_allow_html=True)

    if report == "CIT":
        guide = """
**Large Long / Large Short** — Non-Commercial

**Small Long / Small Short** — Non-Reportable

**L+S Long / L+S Short** — Large + Small gross (Non-Commercial + Non-Reportable)

**L+S+I Long / L+S+I Short** — Large + Small + Index gross (all ex-Commercial)

**Large+Small** — Non-Commercial Net + Non-Reportable Net (total non-index speculative net)

**Lrg+Sml+Idx** — Non-Commercial Net + Non-Reportable Net + Index Traders Net (everything ex-Commercial)
"""
    else:
        guide = """
**MM+O+NR Long/Short** — MM Long/Short + Other Long/Short + Non-Rep Long/Short (all speculative ex swap dealers)

**MM+O+NR Net** — MM Net + Other Net + Non-Rep Net (combined speculative ex-swap net)

**Rest (NET)** — Other Net + Non-Reportable Net combined
"""
    with _exp("Column guide", expanded=False):
        st.markdown(guide)


# ── CIT vs Disagg crosswalk ───────────────────────────────────────────────────
CROSSWALK = {
    "Non-Comm vs Managed Money": {
        "cit_long":"Spec Long",    "cit_short":"Spec Short",    "cit_net":"Spec Net",
        "dag_long":"MM Long",      "dag_short":"MM Short",      "dag_net":"MM Net",
        "cit_label":"Non-Commercial (CIT)", "dag_label":"Managed Money (Disagg)",
        "desc":"",
    },
    "Index Traders vs Swap Dealers": {
        "cit_long":"Index Long",   "cit_short":"Index Short",   "cit_net":"Index Net",
        "dag_long":"Swap Long",    "dag_short":"Swap Short",    "dag_net":"Swap Net",
        "cit_label":"Index Traders (CIT)", "dag_label":"Swap Dealers (Disagg)",
        "desc":"",
    },
    "Commercial vs Producer/Merchant": {
        "cit_long":"Comm Long",     "cit_short":"Comm Short",     "cit_net":"Comm Net",
        "dag_long":"Producer Long", "dag_short":"Producer Short", "dag_net":"Comm Net",
        "cit_label":"CIT Commercial", "dag_label":"Producer/Merchant (Disagg)",
        "desc":"Physical hedgers — CIT Commercial includes swap-dealer activity that Disagg separates out.",
    },
    "CIT (NonComm + Index) vs Disagg (MM + Swap + Others)": {
        "cit_long":None, "cit_short":None, "cit_net":"Fin Net",
        "dag_long":None, "dag_short":None, "dag_net":"Fin Net",
        "cit_label":"CIT NonComm + Index", "dag_label":"Disagg MM + Swap + Other",
        "desc":"Total non-physical/financial side — CIT (NonComm+Index) vs Disagg (MM+Swap+Other).",
        "computed":True,
    },
}

CONC_COLS = ["Conc Gross 4 Long","Conc Gross 4 Short",
             "Conc Gross 8 Long","Conc Gross 8 Short",
             "Conc Net 4 Long","Conc Net 4 Short",
             "Conc Net 8 Long","Conc Net 8 Short"]

def render_concentration(d, color):
    avail = [c for c in CONC_COLS if c in d.columns]
    if not avail: st.info("No concentration data available."); return

    latest = d.iloc[-1]

    # Summary table
    st.markdown(f"**Latest — week of {pd.to_datetime(latest['Date']).strftime('%d %b %Y')}**")
    rows = {
        "":         ["4 Traders","8 Traders"],
        "Gross Long": [f"{latest.get('Conc Gross 4 Long',np.nan):.1f}%",
                       f"{latest.get('Conc Gross 8 Long',np.nan):.1f}%"],
        "Gross Short":[f"{latest.get('Conc Gross 4 Short',np.nan):.1f}%",
                       f"{latest.get('Conc Gross 8 Short',np.nan):.1f}%"],
        "Net Long":  [f"{latest.get('Conc Net 4 Long',np.nan):.1f}%",
                      f"{latest.get('Conc Net 8 Long',np.nan):.1f}%"],
        "Net Short": [f"{latest.get('Conc Net 4 Short',np.nan):.1f}%",
                      f"{latest.get('Conc Net 8 Short',np.nan):.1f}%"],
    }
    st.dataframe(pd.DataFrame(rows).set_index(""), width='content', height=130)
    st.markdown("")

    sel = st.multiselect("Series to chart", avail,
                         default=["Conc Gross 4 Long","Conc Gross 4 Short",
                                  "Conc Gross 8 Long","Conc Gross 8 Short"],
                         key="conc_sel")
    CONC_PALETTE = ["#1a56db","#dc2626","#1a56db","#dc2626",
                    "#7c3aed","#059669","#7c3aed","#059669"]
    CONC_DASH    = ["solid","solid","dash","dash","solid","solid","dash","dash"]

    if sel:
        fig = go.Figure()
        for i,col in enumerate(sel):
            fig.add_trace(go.Scatter(x=d["Date"], y=d[col],
                name=col.replace("Conc ",""),
                line=dict(color=CONC_PALETTE[i%8], width=2.0, dash=CONC_DASH[i%8]),
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{col}: %{{y:.1f}}%<extra></extra>"))
        fig.update_layout(
            **_BASE, height=360,
            title=dict(text="Concentration — % of OI held by largest traders",
                       font=dict(size=12,color="#333"),x=0),
            margin=dict(l=50,r=20,t=42,b=70),
            legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10),
            xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
            yaxis=dict(**_ax(),title_text="% of OI",title_font_size=10))
        _chart(fig, width='stretch')

    show_table(d, avail, avail[:4], "Data table — Concentration ratios", scale=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXPOSURE (Nominal; VaR placeholder)
# ══════════════════════════════════════════════════════════════════════════════
def render_exposure(d, commodity, color):
    cs  = CONTRACT_SIZE[commodity]
    cu  = CONTRACT_UNIT[commodity]
    px  = d["Px"].dropna().iloc[-1] if "Px" in d.columns and not d["Px"].dropna().empty else None

    st.markdown(
        f"<div style='background:#f0f4ff;border:1px solid #c7d7ff;border-radius:8px;"
        f"padding:12px 18px;margin-bottom:14px;font-size:.84rem'>"
        f"<b>Contract spec</b>: {commodity} — {cs:,} {cu} per contract &nbsp;|&nbsp; "
        f"Latest price: {'%.2f' % px if px else '—'}"
        f"</div>", unsafe_allow_html=True)

    if px is None:
        st.warning("Price data unavailable — cannot compute nominal.")
        return

    net_map = {}
    for c in ["Spec Net","MM Net","Combined Spec Net","Comm Net","Non Rep Net","Total OI"]:
        if c in d.columns:
            net_map[c] = d[c].iloc[-1]

    # mult: USD/GBP per 1 contract per unit price (cents/lb → USD needs /100)
    px_mult = cs / 100 if commodity in CENTS_QUOTED else cs

    rows = []
    for col, lots in net_map.items():
        if pd.isna(lots): continue
        nominal = abs(lots) * px_mult * px
        rows.append({"Position": col, "Contracts": f"{lots/1000:.1f}k",
                     "Nominal (USD)": f"${nominal/1e6:.1f}M"})
    if rows:
        st.markdown("**Nominal Exposure — latest week**")
        st.dataframe(pd.DataFrame(rows).set_index("Position"), width='content')

    # Timeseries of nominal net spec
    nc = next((c for c in ["Spec Net","MM Net","Combined Spec Net"] if c in d.columns), None)
    if nc:
        nom_ts = d[nc].abs() * px_mult * d["Px"].ffill() / 1e6
        fig = go.Figure(go.Scatter(
            x=d["Date"], y=nom_ts, name="Nominal |Net|",
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.09)",
            line=dict(color=color,width=2.0),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Nominal: $%{y:.0f}M<extra></extra>"))
        fig.update_layout(
            **_BASE, height=320,
            title=dict(text=f"{nc} — Nominal Exposure  ·  $M",font=dict(size=12,color="#333"),x=0),
            margin=dict(l=60,r=20,t=42,b=50), showlegend=False,
            xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
            yaxis=dict(**_ax(),title_text="$M",title_font_size=10))
        _chart(fig, width='stretch')

    st.markdown(
        "<div style='background:#fff8e8;border:1px solid #fde68a;border-radius:8px;"
        "padding:10px 16px;margin-top:10px;font-size:.82rem;color:#92400e'>"
        "VaR module — reserved for integration from the existing VaR project.</div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — SCATTER & CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def render_analysis(d, report, color, commodity="KC"):
    if d.empty or "Px" not in d.columns:
        st.warning("No data."); return

    # ── Build metrics dict ────────────────────────────────────────────────────
    def gc(col): return d[col].astype(float) if col in d.columns else pd.Series(np.nan, index=d.index)

    if report == "CIT":
        metrics = {
            "Spec Net":        gc("Spec Net") / 1000,
            "Index Net":       gc("Index Net") / 1000,
            "Comm Net":        gc("Comm Net") / 1000,
            "Non-Rep Net":     gc("Non Rep Net") / 1000,
            "Large+Small Net": (gc("Spec Net") + gc("Non Rep Net")) / 1000,
            "Large+Small+Index Net": (gc("Spec Net") + gc("Non Rep Net") + gc("Index Net")) / 1000,
            "Spec Long":       gc("Spec Long") / 1000,
            "Spec Short":      gc("Spec Short") / 1000,
            "Index Long":      gc("Index Long") / 1000,
            "Index Short":     gc("Index Short") / 1000,
            "Comm Long":       gc("Comm Long") / 1000,
            "Comm Short":      gc("Comm Short") / 1000,
        }
    else:
        metrics = {
            "MM Net":           gc("MM Net") / 1000,
            "Other Net":        gc("Other Net") / 1000,
            "Non-Rep Net":      gc("Non Rep Net") / 1000,
            "Swap Net":         gc("Swap Net") / 1000,
            "Comm Net":         gc("Comm Net") / 1000,
            "MM + Non Rep + Others": (gc("MM Net") + gc("Other Net") + gc("Non Rep Net")) / 1000,
            "MM+Other Net":     (gc("MM Net") + gc("Other Net")) / 1000,
            "MM Long":          gc("MM Long") / 1000,
            "MM Short":         gc("MM Short") / 1000,
            "Other Long":       gc("Other Long") / 1000,
            "Other Short":      gc("Other Short") / 1000,
            "Prod Long":        gc("Producer Long") / 1000,
            "Prod Short":       gc("Producer Short") / 1000,
        }

    px   = d["Px"].astype(float)
    px_chg = px.pct_change() * 100
    px_fwd = px_chg.shift(-1)

    # ── Top controls: Metric + Look-back Period (50/50) ──────────────────────
    st.markdown("""<style>
    div[data-testid="stSelectbox"] > div > div[data-baseweb="select"] > div {
        border-color: #d1d5db !important; border-radius: 7px !important;
    }
    </style>""", unsafe_allow_html=True)

    _reg_default = "Large+Small+Index Net" if report == "CIT" else "MM + Non Rep + Others"
    metric_keys  = list(metrics.keys())
    default_idx  = metric_keys.index(_reg_default) if _reg_default in metric_keys else 0

    sel = st.selectbox("Metric", metric_keys, index=default_idx, key="anal_col_v2",
                       help="COT element for regression & implied positioning")

    # Date range governed by sidebar — no separate lookback
    _d_last_date = d.iloc[-1]["Date"]
    _d_dates     = d["Date"].values
    _px_full     = px.reset_index(drop=True)
    _px_chg_full = px_chg.reset_index(drop=True)
    _sel_full    = metrics[sel].reset_index(drop=True)

    # ── Section 1: COT Elements Prediction ───────────────────────────────────
    st.markdown("<div style='font-size:.88rem;font-weight:700;color:#374151;"
                "margin:12px 0 10px;letter-spacing:.02em'>COT ELEMENTS PREDICTION AGAINST THE LATEST ROLLEX MOVE</div>",
                unsafe_allow_html=True)

    sel_series_win = metrics[sel].reset_index(drop=True)
    px_chg_r_win   = px_chg.reset_index(drop=True)

    ds_r   = sel_series_win.diff()
    common = ~(px_chg_r_win.isna() | ds_r.isna())
    x_hist = px_chg_r_win[common].values.astype(float)
    y_hist = ds_r[common].values.astype(float)

    # Strip CFTC data-entry errors (e.g. Non Rep Long = 2.7B lots on LCC)
    # Keep rows within 5 IQR of the median on the COT series
    if len(y_hist) >= 10:
        _iqr = np.percentile(y_hist, 75) - np.percentile(y_hist, 25)
        _med = np.median(y_hist)
        _clip = max(_iqr * 5, 1)
        _ok  = np.abs(y_hist - _med) <= _clip
        x_hist, y_hist = x_hist[_ok], y_hist[_ok]

    if len(x_hist) < 10:
        st.info(f"Not enough data for regression ({len(x_hist)} obs — need ≥ 10).")
    else:
        beta, alpha = np.polyfit(x_hist, y_hist, 1)
        y_hat  = beta * x_hist + alpha
        ss_res = np.sum((y_hist - y_hat) ** 2)
        ss_tot = np.sum((y_hist - y_hist.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        x_line = np.linspace(x_hist.min(), x_hist.max(), 200)
        y_line = beta * x_line + alpha

        # KPI values always from full series
        last_cot_date = pd.to_datetime(_d_last_date)
        last_cot_px   = float(_px_full.iloc[-1])
        last_cot_spec = float(_sel_full.iloc[-1])
        rp = ROLLEX_DIR / ROLLEX_MAP.get(commodity, f"rollex_{commodity}.parquet")
        latest_px, latest_date = last_cot_px, last_cot_date
        if rp.exists():
            try:
                rx = pd.read_parquet(rp, columns=["rollex_px"])
                if not rx.empty:
                    latest_px   = float(rx["rollex_px"].iloc[-1])
                    latest_date = rx.index[-1]
            except Exception:
                pass

        px_move_abs = latest_px - last_cot_px
        px_move_pct = px_move_abs / last_cot_px * 100 if last_cot_px != 0 else 0
        implied_chg = beta * px_move_pct + alpha
        implied_now = last_cot_spec + implied_chg
        arrow_px    = "▲" if px_move_abs >= 0 else "▼"
        arrow_sp    = "▲" if implied_chg >= 0 else "▼"
        clr_px      = C_LONG if px_move_abs >= 0 else C_SHORT
        _bg_chg     = "#16a34a" if implied_chg >= 0 else "#dc2626"

        _th = ("padding:5px 12px;font-size:.60rem;font-weight:700;color:#94a3b8;"
               "letter-spacing:.05em;border:1px solid #e5e7eb;background:#f9fafb;"
               "text-align:left;white-space:nowrap")
        _td = ("padding:6px 12px;font-size:.82rem;font-weight:700;color:#1e293b;"
               "border:1px solid #e5e7eb;white-space:nowrap;vertical-align:top")
        _sm = "font-size:.68rem;font-weight:400;color:#94a3b8;margin-top:2px"
        st.markdown(
            f"<table style='border-collapse:collapse;font-family:-apple-system,sans-serif;"
            f"margin:10px 0 18px;width:100%'>"
            f"<tr>"
            f"<th style='{_th}'>LAST COT DATE</th>"
            f"<th style='{_th}'>ROLLEX PX · LAST COT</th>"
            f"<th style='{_th}'>ROLLEX PX · LATEST · {latest_date.strftime('%d %b')}</th>"
            f"<th style='{_th}'>PRICE MOVE</th>"
            f"<th style='{_th}'>REGRESSION</th>"
            f"<th style='{_th}'>IMPLIED Δ {sel.upper()} · THIS WEEK</th>"
            f"<th style='{_th}'>IMPLIED CURRENT NET</th>"
            f"</tr><tr>"
            f"<td style='{_td}'>{last_cot_date.strftime('%a %d %b %Y')}</td>"
            f"<td style='{_td}'>{last_cot_px:.2f}</td>"
            f"<td style='{_td};color:#0c4a6e'>{latest_px:.2f}</td>"
            f"<td style='{_td};color:{clr_px}'>{arrow_px} {abs(px_move_abs):.2f} ({px_move_pct:+.1f}%)</td>"
            f"<td style='{_td}'>R² = {r2:.2f} · n={len(x_hist)}"
            f"<div style='{_sm}'>β = {beta:+.2f}k / 1%</div></td>"
            f"<td style='{_td};color:{_bg_chg};font-size:.92rem'>{arrow_sp} {implied_chg:+.1f}k lots</td>"
            f"<td style='{_td}'>{implied_now:+.1f}k lots"
            f"<div style='{_sm}'>{last_cot_spec:+.1f}k + ({implied_chg:+.1f}k)</div></td>"
            f"</tr></table>",
            unsafe_allow_html=True)

        win_label = ""
        dates_win    = pd.to_datetime(d["Date"].reset_index(drop=True))
        dates_common = dates_win[common]
        latest_pt_lbl = dates_common.iloc[-1].strftime("%d %b %Y")
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=x_hist, y=y_hist, mode="markers",
            marker=dict(color=color, size=6, opacity=0.55, line=dict(width=0.4, color="white")),
            hovertemplate="ΔPx%: %{x:.1f}%<br>Δ" + sel + ": %{y:.1f}k<extra></extra>",
            showlegend=False))
        fig_reg.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
            line=dict(color=color, width=2, dash="dash"), showlegend=False))
        fig_reg.add_trace(go.Scatter(
            x=[x_hist[-1]], y=[y_hist[-1]], mode="markers", name="Latest",
            marker=dict(symbol="star", size=14, color="#f59e0b",
                        line=dict(width=1.2, color="white")),
            hovertemplate=f"<b>{latest_pt_lbl}</b><br>ΔPx%: {x_hist[-1]:.1f}%<br>Δ{sel}: {y_hist[-1]:+.1f}k<extra></extra>"))
        fig_reg.add_annotation(
            x=x_hist[-1], y=y_hist[-1],
            text=f"<b>{latest_pt_lbl}</b>",
            showarrow=True, arrowhead=2, arrowwidth=1.2, arrowcolor="#f59e0b",
            font=dict(size=9, color="#92400e"),
            bgcolor="rgba(255,237,213,0.92)", borderpad=4,
            bordercolor="#f59e0b", borderwidth=1, ax=30, ay=-36)
        fig_reg.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"R² = {r2:.3f}  |  n = {len(x_hist)} obs{win_label}",
            showarrow=False, font=dict(size=10.5, color="#374151"),
            bgcolor="rgba(255,255,255,0.9)", borderpad=5,
            xanchor="left", yanchor="top",
            bordercolor="#e2e8f0", borderwidth=1)
        fig_reg.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper",
            text=f"<i>Δ{sel} = {beta:+.2f} × ΔPx% + {alpha:+.2f}</i>",
            showarrow=False, font=dict(size=10, color="#94a3b8"),
            bgcolor="rgba(255,255,255,0)", xanchor="right", yanchor="top")
        fig_reg.update_layout(**_BASE, height=420,
            title=dict(text=f"Δ{sel}  vs  ΔPx% 1w  ·  weekly changes",
                       font=dict(size=11, color="#374151"), x=0),
            margin=dict(l=60, r=24, t=48, b=56),
            xaxis=dict(**_ax(x=True), title_text="Price Δ% 1w", ticksuffix="%"),
            yaxis=dict(**_ax(), title_text=f"Δ{sel} (k lots)"))
        _chart(fig_reg, width='stretch')

        bar_win = st.radio(
            "History",
            ["1Y", "2Y", "5Y", "All"],
            index=0, horizontal=True,
            key="avp_window",
        )
        bar_n_map = {"1Y": 52, "2Y": 104, "5Y": 260}
        all_dates  = pd.to_datetime(_d_dates)
        ds_full    = _sel_full.diff()
        pred_full  = beta * _px_chg_full + alpha
        bar_mask   = ~(ds_full.isna() | _px_chg_full.isna())
        # Same CFTC data-entry-error strip used for the regression fit above,
        # so a garbage week doesn't dominate the Actual-vs-Predicted chart.
        bar_mask   = bar_mask & ((ds_full - _med).abs() <= _clip).fillna(False)
        _all_dates_v  = all_dates[bar_mask].values
        _all_actual   = ds_full[bar_mask].values
        _all_pred     = pred_full[bar_mask].values
        bar_n         = bar_n_map.get(bar_win, len(_all_dates_v))
        bar_dates     = _all_dates_v[-bar_n:]
        bar_actual    = _all_actual[-bar_n:]
        bar_pred      = _all_pred[-bar_n:]
        bar_labels    = pd.to_datetime(bar_dates).strftime("%d %b '%y")
        residuals     = bar_actual - bar_pred
        res_clr       = [C_LONG if v >= 0 else C_SHORT for v in residuals]
        win_txt       = bar_win if bar_win != "All" else "full history"
        bar_h         = max(340, min(520, 280 + len(bar_dates) * 2))
        cd = list(zip(bar_pred, residuals))
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Bar(
            x=bar_labels, y=bar_actual,
            customdata=cd, name="Actual", marker_color=color, opacity=0.75,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Actual: %{y:+.2f}k<br>"
                "Predicted: %{customdata[0]:+.2f}k<br>"
                "Error: %{customdata[1]:+.2f}k<extra></extra>")))
        fig_avp.add_trace(go.Bar(
            x=bar_labels, y=bar_pred,
            customdata=cd, name="Predicted", marker_color="#94a3b8", opacity=0.65,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Actual: %{customdata[0]:+.2f}k<br>"
                "Predicted: %{y:+.2f}k<br>"
                "Error: %{customdata[1]:+.2f}k<extra></extra>")))
        fig_avp.add_trace(go.Scatter(
            x=bar_labels, y=residuals,
            customdata=list(zip(bar_actual, bar_pred)),
            name="Error (Actual − Pred)",
            mode="markers", marker=dict(color=res_clr, size=5, symbol="diamond"),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Actual: %{customdata[0]:+.2f}k<br>"
                "Predicted: %{customdata[1]:+.2f}k<br>"
                "Error: %{y:+.2f}k<extra></extra>")))
        fig_avp.update_layout(**_BASE, height=bar_h, barmode="group",
            title=dict(text=f"Actual vs Predicted Δ{sel}  ·  {win_txt}",
                       font=dict(size=11, color="#374151"), x=0),
            margin=dict(l=56, r=24, t=44, b=80),
            xaxis={**_ax(x=True), "tickangle": -45, "tickfont": dict(size=8)},
            yaxis=dict(**_ax(), title_text="Δ (k lots)"),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right",
                        font=dict(size=9)))
        _chart(fig_avp, width='stretch')

    st.markdown("---")

    # ── Section 2: Pairwise COT Correlation ──────────────────────────────────
    st.markdown("<div style='font-size:.88rem;font-weight:700;color:#374151;"
                "margin:0 0 8px;letter-spacing:.02em'>PAIRWISE COT CORRELATION  ·  weekly Δ</div>",
                unsafe_allow_html=True)

    pw_series = {name: s.diff() for name, s in metrics.items()}
    pw_series["Rollex %Δ"] = px_chg
    pw_df = pd.DataFrame(pw_series).dropna(how="all")
    if len(pw_df) >= 4:
        pw_corr = pw_df.corr()
        pw_arr  = pw_corr.values.copy()
        np.fill_diagonal(pw_arr, np.nan)
        labels  = list(pw_corr.columns)
        n       = len(labels)

        # ── P-value matrix (scipy pearsonr per pair) ──────────────────────────
        pw_pval = np.full((n, n), np.nan)
        for i, ci in enumerate(labels):
            for j, cj in enumerate(labels):
                if i != j:
                    _v = pw_df[[ci, cj]].dropna()
                    if len(_v) >= 4:
                        _, pw_pval[i, j] = scipy_stats.pearsonr(_v[ci], _v[cj])
        insig_mask = pw_pval > 0.05   # True = not significant

        tick_y = [
            f"<b><span style='font-size:12px'>Rollex %Δ</span></b>"
            if l == "Rollex %Δ" else l
            for l in labels
        ]
        zv   = pw_arr.tolist()
        # Significant cells show value normally; insignificant cells show value in grey italic
        tv   = []
        for i, row in enumerate(pw_arr):
            tr = []
            for j, v in enumerate(row):
                if pd.notna(v):
                    tr.append(f"{v:+.2f}" if not insig_mask[i, j] else f"<i style='opacity:.35'>{v:+.2f}</i>")
                else:
                    tr.append("")
            tv.append(tr)

        # Grey overlay z-array: fill insignificant (non-diagonal) cells
        grey_z = np.where(insig_mask & ~np.eye(n, dtype=bool), 0.5, np.nan)

        cell_h = 34
        fig_pw = go.Figure(go.Heatmap(
            z=zv, x=labels, y=labels,
            colorscale=[[0,"#dc2626"],[0.45,"#fef2f2"],[0.5,"#f9fafb"],[0.55,"#f0fdf4"],[1,"#16a34a"]],
            zmid=0, zmin=-1, zmax=1,
            text=tv, texttemplate="%{text}", textfont=dict(size=9.5, color="#111"),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b>: r=%{z:.3f}<extra></extra>",
            colorbar=dict(title=dict(text="r", side="right"), thickness=12, len=0.72,
                          tickvals=[-1,-0.5,0,0.5,1], tickfont=dict(size=9)),
            xgap=2, ygap=2,
        ))
        # Semi-transparent grey overlay for insignificant cells
        fig_pw.add_trace(go.Heatmap(
            z=grey_z.tolist(), x=labels, y=labels,
            colorscale=[[0,"rgba(180,180,180,0.45)"],[1,"rgba(180,180,180,0.45)"]],
            zmin=0, zmax=1, showscale=False,
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b>: p > 0.05 (not significant)<extra></extra>",
            xgap=2, ygap=2,
        ))
        rollex_idx = labels.index("Rollex %Δ")
        fig_pw.add_shape(type="line", xref="paper", yref="y",
            x0=0, x1=1, y0=rollex_idx - 0.52, y1=rollex_idx - 0.52,
            line=dict(color="#374151", width=1.8, dash="dot"))
        fig_pw.update_layout(**_BASE,
            height=max(340, cell_h * n + 140),
            margin=dict(l=170, r=60, t=130, b=24),
            xaxis=dict(side="top", tickangle=-40, tickfont=dict(size=9),
                       showgrid=False, showline=False, zeroline=False),
            yaxis=dict(autorange="reversed", tickfont=dict(size=9),
                       tickvals=labels, ticktext=tick_y,
                       showgrid=False, showline=False, zeroline=False),
        )
        _chart(fig_pw, width='stretch')
        st.markdown(
            "<p style='font-size:.7rem;color:#9ca3af;margin-top:-8px'>"
            "Greyed cells: p > 0.05 (not statistically significant at 95% confidence)</p>",
            unsafe_allow_html=True,
        )

        # ── Correlation vs Rollex  |  Beta vs Rollex  (side by side) ────────────
        st.markdown(
            "<div style='font-size:.88rem;font-weight:700;color:#374151;"
            "margin:18px 0 6px;letter-spacing:.02em'>"
            "vs ROLLEX  ·  Correlation (r) &nbsp;&&nbsp; Beta (β)  ·  weekly Δ</div>",
            unsafe_allow_html=True,
        )
        _rx_col   = "Rollex %Δ"
        _cot_cols = [l for l in labels if l != _rx_col]
        _rx_idx   = labels.index(_rx_col)

        corrs, betas, pvals_rb, sig_rb = [], [], [], []
        for col in _cot_cols:
            _v = pw_df[[col, _rx_col]].dropna()
            if len(_v) >= 4:
                col_idx      = labels.index(col)
                r_val        = pw_arr[col_idx][_rx_idx]
                slope, _     = np.polyfit(_v[col].values, _v[_rx_col].values, 1)
                _, p_rb      = scipy_stats.pearsonr(_v[col], _v[_rx_col])
            else:
                r_val, slope, p_rb = np.nan, np.nan, np.nan
            corrs.append(r_val)
            betas.append(slope)
            pvals_rb.append(p_rb)
            sig_rb.append(p_rb <= 0.05 if pd.notna(p_rb) else False)

        rsqs = [v ** 2 if pd.notna(v) else np.nan for v in corrs]

        # Sort all series by R² descending (NaN treated as 0)
        _sort_idx  = sorted(range(len(_cot_cols)),
                            key=lambda i: rsqs[i] if pd.notna(rsqs[i]) else 0,
                            reverse=True)
        _cot_cols  = [_cot_cols[i] for i in _sort_idx]
        rsqs       = [rsqs[i]  for i in _sort_idx]
        corrs      = [corrs[i] for i in _sort_idx]
        betas      = [betas[i] for i in _sort_idx]
        sig_rb     = [sig_rb[i] for i in _sort_idx]

        def _bar_colors(vals, sigs, mode="signed"):
            if mode == "unsigned":
                return ["#0e7490" if s else "#d1d5db" for s in sigs]
            return [("#16a34a" if v > 0 else "#dc2626") if s else "#d1d5db"
                    for v, s in zip(vals, sigs)]

        def _bar_text(vals, fmt):
            return [f"{v:{fmt}}" if pd.notna(v) else "—" for v in vals]

        def _bar_chart(vals, sigs, title_x, fmt, hover_sfx, mode="signed"):
            fig = go.Figure(go.Bar(
                x=vals, y=_cot_cols, orientation="h",
                marker_color=_bar_colors(vals, sigs, mode),
                text=_bar_text(vals, fmt),
                textposition="outside", textfont=dict(size=9, color="#374151"),
                hovertemplate=f"<b>%{{y}}</b><br>{title_x} = %{{x:.4f}}<br>{hover_sfx}<extra></extra>",
                cliponaxis=False,
            ))
            if mode == "signed":
                fig.add_vline(x=0, line_color="#9ca3af", line_width=1)
            fig.update_layout(**_BASE,
                height=max(280, 26 * len(_cot_cols) + 70),
                margin=dict(l=130, r=70, t=28, b=36),
                xaxis=dict(title=title_x, tickfont=dict(size=9),
                           showgrid=True, gridcolor="#f3f4f6", zeroline=False),
                yaxis=dict(tickfont=dict(size=9), showgrid=False, autorange="reversed"),
            )
            return fig

        _col_r2, _col_r, _col_b = st.columns(3)
        with _col_r2:
            _chart(
                _bar_chart(rsqs, sig_rb, "R²", ".2f", "Variance explained vs Rollex %Δ",
                           mode="unsigned"),
                width='stretch',
            )
        with _col_r:
            _chart(
                _bar_chart(corrs, sig_rb, "Pearson Correlation  (r)", "+.2f", "Correlation with Rollex %Δ"),
                width='stretch',
            )
        with _col_b:
            _chart(
                _bar_chart(betas, sig_rb, "β  (Rollex %Δ per 1k lot)", "+.2f",
                           "% Rollex move per 1k lot weekly Δ"),
                width='stretch',
            )
        st.markdown(
            "<p style='font-size:.7rem;color:#9ca3af;margin-top:-10px'>"
            "Coloured bars = significant (p ≤ 0.05) · Grey = noisy / not significant</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Not enough overlapping data for pairwise correlation.")

    st.markdown("---")

    # ── Section 3: Fresh Flow Attribution ────────────────────────────────────
    # Does fresh buying/selling move price more than covering/liquidating?
    st.markdown("<div style='font-size:.88rem;font-weight:700;color:#374151;"
                "margin:0 0 8px;letter-spacing:.02em'>FRESH FLOW ATTRIBUTION  ·  "
                "does adding move price more than covering?</div>",
                unsafe_allow_html=True)

    # name -> (long cols to sum, short cols to sum)
    _pair_candidates = [
        ("Spec",               ["Spec Long"],              ["Spec Short"]),
        ("MM",                 ["MM Long"],                ["MM Short"]),
        ("Non Rep",            ["Non Rep Long"],           ["Non Rep Short"]),
        ("Comm",               ["Comm Long"],              ["Comm Short"]),
        ("Producer",           ["Producer Long"],          ["Producer Short"]),
        ("Other",              ["Other Long"],             ["Other Short"]),
        ("Index",              ["Index Long"],             ["Index Short"]),
        ("Swap",               ["Swap Long"],              ["Swap Short"]),
        ("Non Rep + Other",    ["Non Rep Long", "Other Long"],
                                ["Non Rep Short", "Other Short"]),
        ("Spec + Non Rep",     ["Spec Long", "Non Rep Long"],
                                ["Spec Short", "Non Rep Short"]),
        ("Spec + Non Rep + Index", ["Spec Long", "Non Rep Long", "Index Long"],
                                    ["Spec Short", "Non Rep Short", "Index Short"]),
        ("MM + Non Rep + Other",   ["MM Long", "Non Rep Long", "Other Long"],
                                    ["MM Short", "Non Rep Short", "Other Short"]),
    ]
    _avail_pairs = [
        (n, lc, sc) for n, lc, sc in _pair_candidates
        if all(c in d.columns for c in lc) and all(c in d.columns for c in sc)
    ]

    if not _avail_pairs:
        st.info("No gross Long/Short columns available for flow attribution.")
    else:
        _pair_names = [n for n, _, _ in _avail_pairs]
        _def_name   = "Spec" if report == "CIT" else "MM"
        _def_idx    = _pair_names.index(_def_name) if _def_name in _pair_names else 0
        cflow1, _ = st.columns([2, 5])
        with cflow1:
            flow_pick = st.selectbox(
                "Category", _pair_names, index=_def_idx, key="flow_attr_cat",
                help="Recommended: Spec for the CIT report, MM (Managed Money) for the Disaggregated "
                     "report — these are the pure directional speculative categories, uncontaminated "
                     "by hedging/spreading flow. Composite categories (e.g. Non Rep + Other) sum the "
                     "long and short legs before differencing.")
        _, long_cols, short_cols = next(t for t in _avail_pairs if t[0] == flow_pick)
        long_col  = " + ".join(long_cols)
        short_col = " + ".join(short_cols)

        dL  = sum(d[c].astype(float) for c in long_cols).diff() / 1000
        dS  = sum(d[c].astype(float) for c in short_cols).diff() / 1000
        dPx = px_chg

        fmask   = ~(dL.isna() | dS.isna() | dPx.isna())
        dL_v, dS_v, dPx_v = dL[fmask].values, dS[fmask].values, dPx[fmask].values
        dates_v = pd.to_datetime(d["Date"])[fmask].values

        if len(dL_v) < 10:
            st.info(f"Not enough data for flow attribution ({len(dL_v)} obs — need ≥ 10).")
        else:
            dnet    = dL_v - dS_v
            is_bull = dnet > 0
            is_bear = dnet < 0

            bull_long_contrib, bull_cover_contrib = dL_v, -dS_v
            bear_short_contrib, bear_liq_contrib  = dS_v, -dL_v

            group = np.full(len(dL_v), "", dtype=object)
            group[is_bull & (bull_long_contrib  >= bull_cover_contrib)] = "Long-Led Rally"
            group[is_bull & (bull_cover_contrib  >  bull_long_contrib)] = "Short-Cover Rally"
            group[is_bear & (bear_short_contrib >= bear_liq_contrib)]   = "Short-Led Selloff"
            group[is_bear & (bear_liq_contrib   >  bear_short_contrib)] = "Long-Liq Selloff"

            # Magnitude of whichever contribution was dominant that week — the
            # x-axis for the per-regime scatter below (e.g. for a Long-Led week,
            # this is simply how many k lots of fresh longs were added).
            dom_mag = np.full(len(dL_v), np.nan)
            _m1 = is_bull & (bull_long_contrib  >= bull_cover_contrib)
            _m2 = is_bull & (bull_cover_contrib  >  bull_long_contrib)
            _m3 = is_bear & (bear_short_contrib >= bear_liq_contrib)
            _m4 = is_bear & (bear_liq_contrib   >  bear_short_contrib)
            dom_mag[_m1] = bull_long_contrib[_m1]
            dom_mag[_m2] = bull_cover_contrib[_m2]
            dom_mag[_m3] = bear_short_contrib[_m3]
            dom_mag[_m4] = bear_liq_contrib[_m4]

            keep = group != ""
            group, dPx_g, dom_mag_g, dates_g = group[keep], dPx_v[keep], dom_mag[keep], dates_v[keep]

            _order  = ["Long-Led Rally", "Short-Cover Rally", "Short-Led Selloff", "Long-Liq Selloff"]
            _gcolor = {"Long-Led Rally": "#16a34a", "Short-Cover Rally": "#86efac",
                       "Short-Led Selloff": "#dc2626", "Long-Liq Selloff": "#fca5a5"}

            fig_box = go.Figure()
            for g in _order:
                yv = dPx_g[group == g]
                if len(yv) == 0: continue
                fig_box.add_trace(go.Box(
                    y=yv, name=f"{g}<br>(n={len(yv)})", marker_color=_gcolor[g],
                    boxmean=True, hovertemplate="ΔPx%: %{y:.2f}%<extra></extra>"))
            fig_box.update_layout(**_BASE, height=380, showlegend=False,
                title=dict(text=f"{flow_pick}: Weekly Price Δ% by Flow Regime  ·  {long_col} / {short_col}",
                           font=dict(size=11, color="#374151"), x=0),
                margin=dict(l=56, r=24, t=44, b=44),
                yaxis=dict(**_ax(), title_text="Price weekly Δ%"))
            _chart(fig_box, width='stretch')

            def _grp_stats(g):
                yv = dPx_g[group == g]
                if len(yv) == 0:
                    return dict(n=0, mean=np.nan, median=np.nan, std=np.nan)
                return dict(n=len(yv), mean=float(np.mean(yv)), median=float(np.median(yv)),
                            std=float(np.std(yv, ddof=1)) if len(yv) > 1 else np.nan)

            stats_rows = {g: _grp_stats(g) for g in _order}

            _th2 = ("padding:5px 10px;font-size:.60rem;font-weight:700;color:#94a3b8;"
                    "letter-spacing:.05em;border:1px solid #e5e7eb;background:#f9fafb;text-align:left")
            _td2 = "padding:6px 10px;font-size:.78rem;font-weight:600;color:#1e293b;border:1px solid #e5e7eb"
            rows_html = ""
            for g in _order:
                s = stats_rows[g]
                if s["n"] == 0: continue
                rows_html += (f"<tr><td style='{_td2}'>{g}</td>"
                              f"<td style='{_td2}'>{s['n']}</td>"
                              f"<td style='{_td2};color:{_gcolor[g]}'>{s['mean']:+.2f}%</td>"
                              f"<td style='{_td2}'>{s['median']:+.2f}%</td>"
                              f"<td style='{_td2}'>{s['std']:.2f}%</td></tr>")

            st.markdown(
                f"<table style='border-collapse:collapse;font-family:-apple-system,sans-serif;"
                f"margin:12px 0;width:100%'>"
                f"<tr><th style='{_th2}'>REGIME</th><th style='{_th2}'>N</th>"
                f"<th style='{_th2}'>MEAN ΔPX%</th><th style='{_th2}'>MEDIAN ΔPX%</th>"
                f"<th style='{_th2}'>STD</th></tr>{rows_html}</table>",
                unsafe_allow_html=True)

            st.markdown(
                "<p style='font-size:.68rem;color:#9ca3af;margin-top:10px'>"
                "Regime = dominant driver of the weekly net Δ: <b>Long-Led Rally</b> (Δnet&gt;0, ΔLong ≥ −ΔShort) · "
                "<b>Short-Cover Rally</b> (Δnet&gt;0, −ΔShort &gt; ΔLong) · "
                "<b>Short-Led Selloff</b> (Δnet&lt;0, ΔShort ≥ −ΔLong) · "
                "<b>Long-Liq Selloff</b> (Δnet&lt;0, −ΔLong &gt; ΔShort).</p>",
                unsafe_allow_html=True)

            # ── Per-regime scatter: flow magnitude vs price move (collapsed) ─
            with _exp("FLOW MAGNITUDE vs PRICE MOVE · by regime", expanded=False):
                st.markdown(
                    "<p style='font-size:.7rem;color:#9ca3af;margin:0 0 10px'>"
                    "X = size of that week's dominant driver (k lots) · Y = that week's price Δ% · "
                    "a positive slope means bigger flow weeks produced bigger price moves.</p>",
                    unsafe_allow_html=True)

                _reg_specs = [
                    ("Long-Led Rally",    "ΔLong added (k lots)"),
                    ("Short-Cover Rally", "−ΔShort covered (k lots)"),
                    ("Short-Led Selloff", "ΔShort added (k lots)"),
                    ("Long-Liq Selloff",  "−ΔLong liquidated (k lots)"),
                ]
                fig_grid = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[f"{g}  (n={int((group==g).sum())})" for g, _ in _reg_specs],
                    horizontal_spacing=0.09, vertical_spacing=0.16)

                for i, (g, xlabel) in enumerate(_reg_specs):
                    r, c = i // 2 + 1, i % 2 + 1
                    gx = dom_mag_g[group == g]
                    gy = dPx_g[group == g]
                    gd = dates_g[group == g]
                    gmask = ~(np.isnan(gx) | np.isnan(gy))
                    gx, gy, gd = gx[gmask], gy[gmask], gd[gmask]
                    if len(gx) == 0:
                        continue
                    fig_grid.add_trace(go.Scatter(
                        x=gx, y=gy, mode="markers",
                        marker=dict(color=_gcolor[g], size=6, opacity=0.65,
                                    line=dict(width=0.4, color="white")),
                        text=pd.to_datetime(gd).strftime("%d %b %Y"),
                        customdata=np.column_stack([
                            [f"{v:.1f}k" for v in gx],
                            [f"{v:+.2f}%" for v in gy],
                        ]),
                        hovertemplate=(f"<b>%{{text}}</b><br>{xlabel}: %{{customdata[0]}}<br>"
                                       "ΔPx%: %{customdata[1]}<extra></extra>"),
                        showlegend=False), row=r, col=c)
                    if len(gx) >= 4:
                        sl, ic = np.polyfit(gx, gy, 1)
                        r2v = float(np.corrcoef(gx, gy)[0, 1] ** 2)
                        xl = np.linspace(gx.min(), gx.max(), 50)
                        fig_grid.add_trace(go.Scatter(
                            x=xl, y=sl * xl + ic, mode="lines",
                            line=dict(color=_gcolor[g], width=1.8, dash="dash"),
                            showlegend=False), row=r, col=c)
                        fig_grid.add_annotation(
                            text=f"R²={r2v:.2f}", x=0.04, y=0.94, xref=f"x{i+1} domain" if i else "x domain",
                            yref=f"y{i+1} domain" if i else "y domain",
                            showarrow=False, font=dict(size=9, color="#6b7280"),
                            xanchor="left", yanchor="top")
                    _gx_ax = {**_ax(x=False), "tickfont": dict(size=8)}
                    _gy_ax = {**_ax(), "tickfont": dict(size=8)}
                    fig_grid.update_xaxes(title_text=xlabel, title_font=dict(size=9),
                                           row=r, col=c, **_gx_ax)
                    fig_grid.update_yaxes(title_text="Price Δ%" if c == 1 else "",
                                           title_font=dict(size=9),
                                           row=r, col=c, **_gy_ax)

                fig_grid.update_layout(**_BASE, height=560,
                    margin=dict(l=50, r=20, t=40, b=40))
                fig_grid.update_annotations(font=dict(size=10, color="#374151"))
                _chart(fig_grid, width='stretch')

            # ── Long vs Short flow map — every week, no regime bucketing ────
            st.markdown(
                "<div style='font-size:.82rem;font-weight:700;color:#374151;"
                "margin:20px 0 4px;letter-spacing:.02em'>"
                "LONG vs SHORT FLOW MAP  ·  every week, no bucketing</div>"
                "<p style='font-size:.7rem;color:#9ca3af;margin:0 0 10px'>"
                "X = that week's ΔLong · Y = that week's ΔShort · colour = that week's price Δ%. "
                "Bottom-right = longs adding + shorts covering (max bullish agreement) · "
                "top-left = longs liquidating + shorts adding (max bearish agreement) · "
                "the dotted diagonal is Δnet = 0.</p>",
                unsafe_allow_html=True)

            _clim = float(np.nanpercentile(np.abs(dPx_v), 95))
            _clim = _clim if _clim > 0 else 1.0
            fig_map = go.Figure(go.Scatter(
                x=dL_v, y=dS_v, mode="markers",
                marker=dict(
                    color=dPx_v, colorscale=[[0, "#dc2626"], [0.5, "#f4f4f5"], [1, "#16a34a"]],
                    cmin=-_clim, cmax=_clim, size=7, opacity=0.75,
                    line=dict(width=0.4, color="white"),
                    colorbar=dict(title=dict(text="Px Δ%", side="right"), thickness=12, len=0.75,
                                  tickfont=dict(size=9))),
                text=pd.to_datetime(dates_v).strftime("%d %b %Y"),
                customdata=np.column_stack([
                    [f"{v:+.1f}k" for v in dL_v],
                    [f"{v:+.1f}k" for v in dS_v],
                    [f"{v:+.2f}%" for v in dPx_v],
                ]),
                hovertemplate="<b>%{text}</b><br>ΔLong: %{customdata[0]}<br>ΔShort: %{customdata[1]}<br>"
                              "ΔPx%: %{customdata[2]}<extra></extra>",
            ))
            _lo = float(min(dL_v.min(), dS_v.min()))
            _hi = float(max(dL_v.max(), dS_v.max()))
            _pad = (_hi - _lo) * 0.04 or 1.0
            _rng = [_lo - _pad, _hi + _pad]
            fig_map.add_shape(type="line", x0=0, x1=0, y0=_lo, y1=_hi,
                               line=dict(color="#9ca3af", width=1))
            fig_map.add_shape(type="line", x0=_lo, x1=_hi, y0=0, y1=0,
                               line=dict(color="#9ca3af", width=1))
            fig_map.add_shape(type="line", x0=_lo, x1=_hi, y0=_lo, y1=_hi,
                               line=dict(color="#9ca3af", width=1, dash="dot"))
            fig_map.update_layout(**_BASE, height=460,
                title=dict(text=f"{flow_pick}: Δ{long_col}  vs  Δ{short_col}  ·  coloured by price move",
                           font=dict(size=11, color="#374151"), x=0),
                margin=dict(l=60, r=24, t=44, b=50),
                xaxis=dict(**_ax(), title_text=f"Δ{long_col} (k lots)", range=_rng,
                           constrain="domain"),
                yaxis=dict(**_ax(), title_text=f"Δ{short_col} (k lots)", range=_rng,
                           scaleanchor="x", scaleratio=1, constrain="domain"))
            _chart(fig_map, width='stretch')

            # ── Four-way regression: buying / liquidation / selling / covering ──
            # A single beta_long, beta_short model forces buying and liquidating
            # the SAME leg to have equal-and-opposite impact (it's one straight
            # line through both signs). To let each of the four flows have its
            # own independent price impact, split each leg into two >=0 variables.
            pos_L = np.clip(dL_v, 0, None)    # fresh buying (k lots)
            liq_L = np.clip(-dL_v, 0, None)   # long liquidation (k lots, magnitude)
            sell_S = np.clip(dS_v, 0, None)   # fresh short selling (k lots)
            cov_S  = np.clip(-dS_v, 0, None)  # short covering (k lots, magnitude)

            X4 = np.column_stack([pos_L, liq_L, sell_S, cov_S, np.ones_like(dL_v)])
            coef4, _, rank4, _ = np.linalg.lstsq(X4, dPx_v, rcond=None)
            b_buy, b_liq, b_sell, b_cov, a4 = coef4
            y_hat4  = X4 @ coef4
            ss_res4 = float(np.sum((dPx_v - y_hat4) ** 2))
            ss_tot4 = float(np.sum((dPx_v - dPx_v.mean()) ** 2))
            r2_4    = 1 - ss_res4 / ss_tot4 if ss_tot4 > 0 else 0

            n4, k4 = X4.shape
            dof4    = n4 - k4
            se4 = np.full(4, np.nan)
            p4  = np.full(4, np.nan)
            t_bull = p_bull = t_bear = p_bear = np.nan
            if dof4 > 0 and rank4 == k4:
                try:
                    sigma2  = ss_res4 / dof4
                    XtX_inv = np.linalg.inv(X4.T @ X4)
                    se4 = np.sqrt(np.diag(sigma2 * XtX_inv)[:4])
                    for i in range(4):
                        p4[i] = (2 * scipy_stats.t.sf(abs(coef4[i] / se4[i]), dof4)
                                  if se4[i] > 0 else np.nan)
                    # Bullish joint test: H0: b_buy = b_cov (both already same-sign)
                    var_bull = se4[0] ** 2 + se4[3] ** 2 - 2 * sigma2 * XtX_inv[0, 3]
                    if var_bull > 0:
                        t_bull = (b_buy - b_cov) / np.sqrt(var_bull)
                        p_bull = float(2 * scipy_stats.t.sf(abs(t_bull), dof4))
                    # Bearish joint test: H0: b_liq = b_sell (both already same-sign)
                    var_bear = se4[1] ** 2 + se4[2] ** 2 - 2 * sigma2 * XtX_inv[1, 2]
                    if var_bear > 0:
                        t_bear = (b_liq - b_sell) / np.sqrt(var_bear)
                        p_bear = float(2 * scipy_stats.t.sf(abs(t_bear), dof4))
                except np.linalg.LinAlgError:
                    pass

            st.markdown(
                "<div style='font-size:.82rem;font-weight:700;color:#374151;"
                "margin:18px 0 4px;letter-spacing:.02em'>"
                "PRICE IMPACT PER K-LOT  ·  all four flows, each controlling for the other three</div>",
                unsafe_allow_html=True)

            def _impact_card(col, label, beta, se, p, positive_is_bullish):
                sig = pd.notna(p) and p < 0.05
                _clr = ("#16a34a" if positive_is_bullish else "#dc2626") if sig else "#94a3b8"
                with col:
                    st.markdown(
                        (f"<div style='font-size:.75rem;color:#374151'><b>{label}</b><br>"
                         f"{beta:+.3f}% per k-lot<br>"
                         f"<span style='color:{_clr}'>p = {p:.3f}{' (significant)' if sig else ''}</span></div>"
                         ) if pd.notna(beta) else "<i>Not enough data</i>",
                        unsafe_allow_html=True)

            _ic1, _ic2, _ic3, _ic4 = st.columns(4)
            _impact_card(_ic1, "Fresh buying",      b_buy,  se4[0], p4[0], True)
            _impact_card(_ic2, "Short covering",     b_cov,  se4[3], p4[3], True)
            _impact_card(_ic3, "Fresh short selling", b_sell, se4[2], p4[2], False)
            _impact_card(_ic4, "Long liquidation",   b_liq,  se4[1], p4[1], False)

            _jc1, _jc2 = st.columns(2)
            with _jc1:
                if pd.notna(p_bull):
                    _v_bull = ("Fresh buying moves price more" if b_buy > b_cov
                               else "Short covering moves price more")
                    st.markdown(
                        f"<div style='font-size:.75rem;color:#374151'><b>Bullish: Buying vs Covering</b><br>"
                        f"t = {t_bull:+.2f}, p = {p_bull:.3f}<br>"
                        f"<span style='color:{'#16a34a' if p_bull < 0.05 else '#94a3b8'}'>"
                        f"{_v_bull}{' (significant)' if p_bull < 0.05 else ' (not significant)'}</span></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown("<i>Not enough data for joint test</i>", unsafe_allow_html=True)
            with _jc2:
                if pd.notna(p_bear):
                    _v_bear = ("Long liquidation moves price more" if b_liq < b_sell
                               else "Fresh short selling moves price more")
                    st.markdown(
                        f"<div style='font-size:.75rem;color:#374151'><b>Bearish: Selling vs Liquidation</b><br>"
                        f"t = {t_bear:+.2f}, p = {p_bear:.3f}<br>"
                        f"<span style='color:{'#dc2626' if p_bear < 0.05 else '#94a3b8'}'>"
                        f"{_v_bear}{' (significant)' if p_bear < 0.05 else ' (not significant)'}</span></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown("<i>Not enough data for joint test</i>", unsafe_allow_html=True)

            st.markdown(
                f"<p style='font-size:.68rem;color:#9ca3af;margin-top:10px'>"
                f"Model R² = {r2_4:.3f} · n = {n4} weeks (all weeks used, not just regime-dominant ones). "
                f"Buying and covering are expected positive (bullish per k-lot); selling and liquidation "
                f"are expected negative (bearish per k-lot) — a sign flipped from expectation usually "
                f"means that flow has no real price impact once the other three are controlled for.</p>",
                unsafe_allow_html=True)

            # ── Reverse regression: how much flow follows price ─────────────
            def _simple_ols(x, y):
                mask = ~(np.isnan(x) | np.isnan(y))
                x, y = x[mask], y[mask]
                n = len(x)
                if n < 8:
                    return dict(beta=np.nan, se=np.nan, p=np.nan, r2=np.nan, n=n)
                X = np.column_stack([x, np.ones_like(x)])
                coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
                beta, _alpha = coef
                y_hat  = X @ coef
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                dof = n - 2
                se = p = np.nan
                if dof > 0 and rank == 2:
                    sigma2  = ss_res / dof
                    XtX_inv = np.linalg.inv(X.T @ X)
                    se = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
                    if se > 0:
                        p = float(2 * scipy_stats.t.sf(abs(beta / se), dof))
                return dict(beta=float(beta), se=se, p=p, r2=r2, n=n)

            same_week = _simple_ols(dPx_v, dnet)

            st.markdown(
                "<div style='font-size:.82rem;font-weight:700;color:#374151;"
                "margin:18px 0 4px;letter-spacing:.02em'>"
                "FLOW SENSITIVITY TO PRICE  ·  k-lots of net flow per 1% price move</div>",
                unsafe_allow_html=True)

            def _flow_card(col, label, res, note):
                with col:
                    if pd.isna(res["beta"]):
                        st.markdown(f"<div style='font-size:.75rem;color:#374151'><b>{label}</b><br>"
                                     f"<i>Not enough data</i></div>", unsafe_allow_html=True)
                        return
                    sig = pd.notna(res["p"]) and res["p"] < 0.05
                    clr = "#1a56db" if sig else "#94a3b8"
                    st.markdown(
                        f"<div style='font-size:.75rem;color:#374151'><b>{label}</b><br>"
                        f"{res['beta']:+.2f}k lots per 1% ΔPx<br>"
                        f"<span style='color:{clr}'>p = {res['p']:.3f}{' (significant)' if sig else ''}</span><br>"
                        f"<span style='color:#9ca3af;font-size:.68rem'>R² = {res['r2']:.3f} · n = {res['n']}"
                        f" · {note}</span></div>",
                        unsafe_allow_html=True)

            _fc1, _ = st.columns(2)
            _flow_card(_fc1, "Same-week  (Δnet(t) ~ ΔPx(t))", same_week, "nowcast-style")

            # ── COT nowcast: project current-week position from price since cutoff ──
            if commodity in ROLLEX_MAP:
                st.markdown(
                    "<div style='font-size:.82rem;font-weight:700;color:#374151;"
                    "margin:18px 0 4px;letter-spacing:.02em'>"
                    f"{flow_pick.upper()} NOWCAST  ·  projected position from price action since last report</div>",
                    unsafe_allow_html=True)

                rx_daily = load_rollex_ohlc(commodity)
                if rx_daily.empty or pd.isna(same_week["beta"]):
                    st.info("Not enough Rollex/regression data to build a nowcast for this selection.")
                else:
                    last_cot_date  = pd.to_datetime(d["Date"].iloc[-1])
                    cutoff_price   = float(d["Px"].iloc[-1])
                    grossL_last    = float(sum(d[c].iloc[-1] for c in long_cols))
                    grossS_last    = float(sum(d[c].iloc[-1] for c in short_cols))
                    net_last       = (grossL_last - grossS_last) / 1000

                    since = rx_daily[rx_daily["Date"] > last_cot_date].reset_index(drop=True)
                    live_price = float(since["rollex_px"].iloc[-1]) if len(since) else cutoff_price
                    live_date  = since["Date"].iloc[-1] if len(since) else last_cot_date
                    live_chg   = (live_price / cutoff_price - 1) * 100

                    beta = same_week["beta"]
                    sim_prices = sorted({round(live_price * (1 + k / 100), 2) for k in range(-3, 4)}, reverse=True)
                    sim_rows = []
                    for sp in sim_prices:
                        chg = (sp / cutoff_price - 1) * 100
                        impact = beta * chg
                        sim_rows.append(dict(**{"Sim. Price": sp, "Chg vs cutoff": f"{chg:+.2f}%",
                                                 "COT Impact (k lots)": f"{impact:+.1f}k",
                                                 "Proj. Net (k lots)": f"{net_last + impact:+.1f}k"}))
                    sim_df = pd.DataFrame(sim_rows)

                    _live_clr = "#16a34a" if live_chg >= 0 else "#dc2626"
                    st.markdown(
                        f"<p style='font-size:.75rem;color:#374151'>Last published: <b>{last_cot_date.date()}</b> "
                        f"net {flow_pick} <b>{net_last:+.1f}k lots</b> @ cutoff price <b>{cutoff_price:.2f}</b>. "
                        f"Live: <b>{live_date.date()}</b> @ <b>{live_price:.2f}</b> "
                        f"(<span style='color:{_live_clr}'>{live_chg:+.2f}%</span> "
                        f"since cutoff).</p>", unsafe_allow_html=True)
                    st.dataframe(sim_df, width='stretch', hide_index=True)

                    _cndl = rx_daily[rx_daily["Date"] >= last_cot_date - pd.Timedelta(days=30)]
                    fig_now = go.Figure(go.Candlestick(
                        x=_cndl["Date"], open=_cndl["rollex_open"], high=_cndl["rollex_high"],
                        low=_cndl["rollex_low"], close=_cndl["rollex_px"],
                        increasing_line_color="#16a34a", decreasing_line_color="#dc2626", name="Price"))
                    fig_now.add_hline(y=cutoff_price, line=dict(color="#1a56db", width=1, dash="dash"),
                                       annotation_text="Last COT cutoff", annotation_font_size=9)
                    fig_now.update_layout(**_BASE, height=300, showlegend=False,
                        title=dict(text=f"{commodity}: price since last COT cutoff",
                                   font=dict(size=11, color="#374151"), x=0),
                        margin=dict(l=50, r=20, t=40, b=30), xaxis_rangeslider_visible=False,
                        xaxis=dict(**_ax(x=True)), yaxis=dict(**_ax(), title_text="Price"))
                    _chart(fig_now, width='stretch')

                    _hist_px = rx_daily.tail(260).reset_index(drop=True)
                    net_hist = pd.DataFrame({
                        "Date": d["Date"],
                        "Net": (sum(d[c].astype(float) for c in long_cols)
                                - sum(d[c].astype(float) for c in short_cols)) / 1000})
                    net_hist = net_hist[net_hist["Date"] >= _hist_px["Date"].min()]

                    fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_ov.add_trace(go.Candlestick(
                        x=_hist_px["Date"], open=_hist_px["rollex_open"], high=_hist_px["rollex_high"],
                        low=_hist_px["rollex_low"], close=_hist_px["rollex_px"], name="Price",
                        increasing_line_color="#a7f3d0", decreasing_line_color="#fecaca", showlegend=False),
                        secondary_y=False)
                    fig_ov.add_trace(go.Scatter(
                        x=net_hist["Date"], y=net_hist["Net"], mode="lines",
                        line=dict(color="#1a56db", width=1.4, shape="hv"), name="Published Net"),
                        secondary_y=True)
                    fig_ov.add_trace(go.Scatter(
                        x=[last_cot_date, live_date], y=[net_last, net_last + beta * live_chg],
                        mode="lines+markers", line=dict(color="#7c3aed", width=2, dash="dot"),
                        marker=dict(size=6), name="Projected"), secondary_y=True)
                    fig_ov.update_layout(**_BASE, height=380,
                        title=dict(text=f"{commodity}: 250-day price & {flow_pick} net  (published & projected)",
                                   font=dict(size=11, color="#374151"), x=0),
                        margin=dict(l=60, r=60, t=44, b=30), xaxis_rangeslider_visible=False,
                        legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=9)))
                    fig_ov.update_xaxes(**_ax(x=True))
                    fig_ov.update_yaxes(title_text="Price", secondary_y=False, **_ax())
                    fig_ov.update_yaxes(title_text="Net (k lots)", secondary_y=True,
                                         showgrid=False, tickfont=dict(size=9))
                    _chart(fig_ov, width='stretch')

    # scatter sections moved to dedicated Correlation tab (render_correlation)


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION TAB — Price vs Positioning, COT cross-scatter, 3D scatters
# ══════════════════════════════════════════════════════════════════════════════
def render_correlation(d, report, color):
    all_opts = [c for c in (
        ["Spec Net","Comm Net","Index Net","Non Rep Net"] if report=="CIT"
        else ["MM Net","Comm Net","Swap Net","Other Net","Non Rep Net"]
    ) + ["Spec Long","Spec Short","Comm Long","Comm Short",
         "MM Long","MM Short","Producer Long","Producer Short",
         "Other Long","Other Short","Non Rep Long","Non Rep Short"]
        if c in d.columns]

    with _exp("Price vs Positioning — scatter", expanded=True):
        c1, _ = st.columns([2,5])
        with c1:
            sel2_list = st.multiselect("COT element (summed if multiple)", all_opts,
                                        default=[all_opts[0]] if all_opts else [], key="anal_col_leg")
        sel2_avail = [c for c in sel2_list if c in d.columns]
        if sel2_avail and "Px" in d.columns:
            sel2  = " + ".join(sel2_avail)
            d_tmp = d.copy()
            d_tmp["_SEL"] = sum(d_tmp[c] for c in sel2_avail)
            ch1, ch2 = st.columns(2)
            with ch1:
                _chart(scatter_2d(d_tmp,"Px","_SEL",color,
                    f"Price Δ%  vs  {sel2} Δ","Price weekly Δ%",f"{sel2} Δ (k lots)"),
                    width='stretch')
            with ch2:
                x = np.asarray(d_tmp["Px"], dtype=float)
                y = np.asarray(d_tmp["_SEL"], dtype=float) / 1000
                dates = np.asarray(d_tmp["Date"])
                mask = ~(np.isnan(x)|np.isnan(y))
                if mask.sum() >= 5:
                    r2v = float(np.corrcoef(x[mask],y[mask])[0,1]**2)
                    sl, ic = np.polyfit(x[mask],y[mask],1)
                    xl = np.linspace(x[mask].min(),x[mask].max(),200)
                    rec = (dates[mask]-dates[mask].min()).astype("timedelta64[D]").astype(float)
                    nr  = rec/max(rec.max(),1)
                    rv, gv, bv = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=x[mask],y=y[mask],mode="markers",
                        marker=dict(color=nr,
                            colorscale=[[0,"rgba(200,210,230,0.5)"],[1,f"rgba({rv},{gv},{bv},0.85)"]],
                            size=7,line=dict(width=0.5,color="white")),
                        text=pd.to_datetime(dates[mask]).strftime("%Y-%m-%d"),
                        hovertemplate=f"<b>%{{text}}</b><br>Px: %{{x:.2f}}<br>{sel2}: %{{y:.1f}}k<extra></extra>",
                        showlegend=False))
                    fig2.add_trace(go.Scatter(x=xl,y=sl*xl+ic,mode="lines",
                        line=dict(color=color,width=1.6,dash="dash"),showlegend=False))
                    fig2.add_trace(go.Scatter(x=[x[mask][-1]],y=[y[mask][-1]],mode="markers",
                        marker=dict(symbol="star",size=14,color=C_SHORT,line=dict(width=1.2,color="white")),
                        showlegend=False))
                    fig2.update_layout(**_BASE, height=340,
                        title=dict(text=f"Price Level vs {sel2}  R²={r2v:.2f}",font=dict(size=12,color="#333"),x=0),
                        margin=dict(l=52,r=20,t=48,b=48),
                        xaxis=dict(**_ax(x=True),title_text="Rollex Px"),
                        yaxis=dict(**_ax(),title_text=f"{sel2} (k lots)"))
                    _chart(fig2, width='stretch')

    with _exp("COT vs COT Cross-Scatter", expanded=True):
        c1, c2 = st.columns(2)
        with c1: xs_sel = st.multiselect("X axis (summed if multiple)", all_opts, default=[all_opts[0]], key="xs_x")
        with c2: ys_sel = st.multiselect("Y axis (summed if multiple)", all_opts, default=[all_opts[min(1,len(all_opts)-1)]], key="xs_y")
        if xs_sel and ys_sel:
            xs_avail = [c for c in xs_sel if c in d.columns]
            ys_avail = [c for c in ys_sel if c in d.columns]
            if xs_avail and ys_avail:
                d_tmp = d.copy()
                d_tmp["_X"] = sum(d_tmp[c] for c in xs_avail)
                d_tmp["_Y"] = sum(d_tmp[c] for c in ys_avail)
                _chart(scatter_2d(d_tmp,"_X","_Y",color,
                    f"{' + '.join(xs_avail)}  vs  {' + '.join(ys_avail)}",
                    f"X Δ (k lots)","Y Δ (k lots)"), width='stretch')

    px_opt = ["Rollex Px"]
    all_3d = px_opt + all_opts

    def _build_series(col_list, mode):
        avail = [c for c in col_list if c in d.columns or c == "Rollex Px"]
        if not avail: return pd.Series(dtype=float), ""
        if "Rollex Px" in avail and len(avail) > 1:
            st.warning("Rollex Px can't be summed with COT columns — different units "
                       "(% price vs k lots). Pick Rollex Px alone or only COT columns for this axis.")
            return pd.Series(dtype=float), ""
        parts = []
        for c in avail:
            s = d["Px"] if c == "Rollex Px" else d[c]
            parts.append(s.pct_change()*100 if (c=="Rollex Px" and mode=="chg") else
                         s.diff()/1000 if mode=="chg" else
                         s if c=="Rollex Px" else s/1000)
        return sum(parts), " + ".join(avail)

    with _exp("3D Scatter — Weekly Change", expanded=False):
        c1,c2,c3 = st.columns(3)
        with c1: x3c = st.multiselect("X",all_3d,default=[all_3d[0]],key="3dc_x")
        with c2: y3c = st.multiselect("Y",all_3d,default=[all_3d[1]] if len(all_3d)>1 else [all_3d[0]],key="3dc_y")
        with c3: z3c = st.multiselect("Z",all_3d,default=[all_3d[2]] if len(all_3d)>2 else [all_3d[0]],key="3dc_z")
        if x3c and y3c and z3c:
            xs,xl = _build_series(x3c,"chg"); ys,yl = _build_series(y3c,"chg"); zs,zl = _build_series(z3c,"chg")
            if not xs.empty:
                _chart(scatter_3d(xs,ys,zs,d["Date"],color,f"{xl} × {yl} × {zl} — Weekly Δ",f"{xl} Δ",f"{yl} Δ",f"{zl} Δ"), width='stretch')

    with _exp("3D Scatter — Position Levels", expanded=False):
        c1,c2,c3 = st.columns(3)
        with c1: x3l = st.multiselect("X",all_3d,default=[all_3d[0]],key="3dl_x")
        with c2: y3l = st.multiselect("Y",all_3d,default=[all_3d[1]] if len(all_3d)>1 else [all_3d[0]],key="3dl_y")
        with c3: z3l = st.multiselect("Z",all_3d,default=[all_3d[2]] if len(all_3d)>2 else [all_3d[0]],key="3dl_z")
        if x3l and y3l and z3l:
            xs,xl = _build_series(x3l,"lvl"); ys,yl = _build_series(y3l,"lvl"); zs,zl = _build_series(z3l,"lvl")
            if not xs.empty:
                _chart(scatter_3d(xs,ys,zs,d["Date"],color,f"{xl} × {yl} × {zl} — Levels",xl,yl,zl), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — CIT vs DISAGG COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def render_comparison(commodity, start_date, end_date, color):
    if commodity not in CIT_COMMS:
        st.info("CIT vs Disagg comparison is only available for KC, CC, SB, and CT.")
        return

    cit_raw = load_cit()
    dag_raw = load_disagg("F&O")

    cit = cit_raw[
        (cit_raw["Commodity"] == commodity) &
        (cit_raw["Date"] >= pd.Timestamp(start_date)) &
        (cit_raw["Date"] <= pd.Timestamp(end_date))
    ].sort_values("Date").reset_index(drop=True).copy()

    dag = dag_raw[
        (dag_raw["Commodity"] == commodity) &
        (dag_raw["Crop"] == "All") &
        (dag_raw["Date"] >= pd.Timestamp(start_date)) &
        (dag_raw["Date"] <= pd.Timestamp(end_date))
    ].sort_values("Date").reset_index(drop=True).copy()

    if cit.empty or dag.empty:
        st.warning("Insufficient data for the selected range."); return

    # Compute Financial Net
    cit["Fin Net"] = (
        cit.get("Spec Net",  pd.Series(0, index=cit.index)).fillna(0) +
        cit.get("Index Net", pd.Series(0, index=cit.index)).fillna(0)
    )
    dag["Fin Net"] = (
        dag.get("MM Net",    pd.Series(0, index=dag.index)).fillna(0) +
        dag.get("Swap Net",  pd.Series(0, index=dag.index)).fillna(0) +
        dag.get("Other Net", pd.Series(0, index=dag.index)).fillna(0)
    )

    # Controls
    pair = st.selectbox("Pair", list(CROSSWALK.keys()), key="cmp_pair")
    unit = "k lots"
    cfg  = CROSSWALK[pair]
    if cfg.get("desc"):
        st.markdown(
            f"<div style='font-size:.77rem;color:#666;background:#f7f8fa;"
            f"border-radius:6px;padding:7px 12px;margin-bottom:12px'>{cfg['desc']}</div>",
            unsafe_allow_html=True)

    cit_nc, dag_nc = cfg["cit_net"], cfg["dag_net"]
    suffix = "k" if unit == "k lots" else "%"
    ylabel = "k lots" if unit == "k lots" else "% of OI"
    DAG_COLOR = "#64748b"
    r, g, b   = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    # Merge on Date for gap + correlation
    merge_cit = cit[["Date", cit_nc]].rename(columns={cit_nc: "cit_net"}) if cit_nc in cit.columns else None
    merge_dag = dag[["Date", dag_nc]].rename(columns={dag_nc: "dag_net"}) if dag_nc in dag.columns else None
    merged = None
    if merge_cit is not None and merge_dag is not None:
        merged = pd.merge(merge_cit, merge_dag, on="Date", how="inner").sort_values("Date")
        merged["gap"] = (merged["cit_net"] - merged["dag_net"]) / 1000

    # KPI row
    def _net_kpi_cmp(df, col, label):
        if col not in df.columns or df.empty: return (label, "—", "")
        v, p = df[col].iloc[-1], (df[col].iloc[-2] if len(df)>1 else df[col].iloc[-1])
        val  = f"{v/1000:.1f}k" if unit=="k lots" else (
               f"{v/df['Total OI'].iloc[-1]*100:.1f}%" if "Total OI" in df.columns else f"{v/1000:.1f}k")
        chg  = f"{'▲' if v>p else '▼'}{abs(v-p)/1000:.1f}k"
        return (label, val, chg)

    kpi_items = [
        _net_kpi_cmp(cit, cit_nc, cfg["cit_label"]),
        _net_kpi_cmp(dag, dag_nc, cfg["dag_label"]),
    ]
    if merged is not None and not merged.empty:
        gap_latest = merged["gap"].iloc[-1]
        ca  = np.asarray(merged["cit_net"], dtype=float)
        da  = np.asarray(merged["dag_net"], dtype=float)
        msk = ~(np.isnan(ca)|np.isnan(da))
        corr = float(np.corrcoef(ca[msk], da[msk])[0,1]) if msk.sum()>4 else np.nan
        kpi_items += [
            ("Gap (CIT−Disagg)", f"{gap_latest:+.1f}k", ""),
            ("Corr (full period)", f"{corr:.2f}" if pd.notna(corr) else "—", ""),
        ]
    kpi_row(kpi_items, color)

    # ── 1. Overlay Net timeseries ──────────────────────────────────────────────
    traces = []
    if cit_nc in cit.columns:
        traces.append({"trace": go.Scatter(
            x=cit["Date"], y=_get_y(cit, cit_nc, unit), name=cfg["cit_label"],
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{cfg['cit_label']}: %{{y:.1f}}{suffix}<extra></extra>")})
    if dag_nc in dag.columns:
        traces.append({"trace": go.Scatter(
            x=dag["Date"], y=_get_y(dag, dag_nc, unit), name=cfg["dag_label"],
            line=dict(color=DAG_COLOR, width=2.2, dash="dash"),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{cfg['dag_label']}: %{{y:.1f}}{suffix}<extra></extra>")})

    fig_ts = go.Figure()
    for s in traces:
        fig_ts.add_trace(s["trace"])
    fig_ts.update_layout(
        **_BASE, height=380,
        title=dict(text=f"{cfg['cit_label']}  vs  {cfg['dag_label']}  ·  Net  ·  {ylabel}",
                   font=dict(size=12,color="#333"), x=0),
        margin=dict(l=52,r=20,t=42,b=72),
        legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10),
        xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
    )
    fig_ts.update_yaxes(title_text=ylabel, title_font_size=10, **_ax())
    _chart(fig_ts, width='stretch')

    # ── 2. Gap bar chart ───────────────────────────────────────────────────────
    if merged is not None and not merged.empty:
        gap = np.asarray(merged["gap"], dtype=float)
        fig_gap = go.Figure(go.Bar(
            x=merged["Date"], y=gap,
            marker=dict(color=[C_LONG if v>=0 else C_SHORT for v in gap],
                        opacity=0.82, line=dict(width=0)),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>CIT−Disagg: %{y:+.1f}k<extra></extra>",
        ))
        fig_gap.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.14)")
        fig_gap.update_layout(
            **_BASE, height=260,
            title=dict(text=f"Gap: {cfg['cit_label']} minus {cfg['dag_label']}  ·  k lots",
                       font=dict(size=11,color="#444"), x=0),
            margin=dict(l=50,r=12,t=36,b=68), showlegend=False,
            xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
            yaxis=dict(**_ax(),title_text="k lots",title_font_size=10),
        )
        _chart(fig_gap, width='stretch')

    # ── 3. Long / Short breakdown ──────────────────────────────────────────────
    cit_lc, cit_sc = cfg.get("cit_long"), cfg.get("cit_short")
    dag_lc, dag_sc = cfg.get("dag_long"), cfg.get("dag_short")
    if cit_lc and dag_lc and cit_lc in cit.columns and dag_lc in dag.columns:
        st.markdown("**Long & Short breakdown**")
        ch1, ch2 = st.columns(2)
        for ch, (cit_col, dag_col, lbl, clr) in zip(
            [ch1, ch2],
            [(cit_lc, dag_lc, "Long", C_LONG), (cit_sc, dag_sc, "Short", C_SHORT)]
        ):
            with ch:
                fig_ls = go.Figure()
                fig_ls.add_trace(go.Scatter(
                    x=cit["Date"], y=_get_y(cit, cit_col, unit), name=f"CIT {lbl}",
                    line=dict(color=clr, width=2.0),
                    hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>CIT {lbl}: %{{y:.1f}}{suffix}<extra></extra>"))
                if dag_col and dag_col in dag.columns:
                    fig_ls.add_trace(go.Scatter(
                        x=dag["Date"], y=_get_y(dag, dag_col, unit), name=f"Disagg {lbl}",
                        line=dict(color=clr, width=2.0, dash="dash"),
                        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Disagg {lbl}: %{{y:.1f}}{suffix}<extra></extra>"))
                fig_ls.update_layout(
                    **_BASE, height=280,
                    title=dict(text=f"{lbl} positions  ·  {ylabel}", font=dict(size=11,color="#444"), x=0),
                    margin=dict(l=50,r=12,t=36,b=68), showlegend=True,
                    legend=dict(orientation="h",y=-0.32,x=0.5,xanchor="center",font_size=10),
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax(),title_text=ylabel,title_font_size=10))
                _chart(fig_ls, width='stretch')

    # ── 4. Rolling correlation ─────────────────────────────────────────────────
    if merged is not None and len(merged) > 12:
        with _exp("Rolling 52-week Correlation", expanded=False):
            s_cit = pd.Series(np.asarray(merged["cit_net"],dtype=float), index=merged["Date"])
            s_dag = pd.Series(np.asarray(merged["dag_net"],dtype=float), index=merged["Date"])
            roll  = s_cit.rolling(52).corr(s_dag).dropna()
            fig_rc = go.Figure(go.Scatter(
                x=roll.index, y=roll.values, mode="lines",
                line=dict(color=color, width=2.0),
                fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.08)",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Corr: %{y:.2f}<extra></extra>"))
            fig_rc.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.12)")
            fig_rc.update_layout(
                **_BASE, height=270,
                title=dict(text="Rolling 52-week Correlation — CIT vs Disagg",
                           font=dict(size=11,color="#444"), x=0),
                margin=dict(l=50,r=12,t=36,b=50),
                xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                yaxis=dict(**_ax(),title_text="Correlation",title_font_size=10,range=[-1.1,1.1]))
            _chart(fig_rc, width='stretch')

    # ── 5. Data table ──────────────────────────────────────────────────────────
    with _exp("Data table", expanded=False):
        cols_cit = [c for c in [cit_nc, cfg.get("cit_long"), cfg.get("cit_short")] if c and c in cit.columns]
        cols_dag = [c for c in [dag_nc, cfg.get("dag_long"), cfg.get("dag_short")] if c and c in dag.columns]
        tbl_cit  = cit[["Date"]+cols_cit].rename(columns={c: f"CIT {c}" for c in cols_cit}).sort_values("Date",ascending=False).head(60)
        tbl_dag  = dag[["Date"]+cols_dag].rename(columns={c: f"Dag {c}" for c in cols_dag}).sort_values("Date",ascending=False).head(60)
        tbl = pd.merge(tbl_cit, tbl_dag, on="Date", how="outer").sort_values("Date",ascending=False).reset_index(drop=True)
        tbl["Date"] = pd.to_datetime(tbl["Date"]).dt.strftime("%d %b '%y")
        num_c = [c for c in tbl.columns if c!="Date"]
        styled = (tbl.style.format({c:"{:,.0f}" for c in num_c}, na_rep="—").hide(axis="index"))
        st.dataframe(styled, width='stretch', height=380)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED COMMODITY RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def render_combined(commodity, start_date, end_date, color):
    comm_a, comm_b = COMBINED_MAP[commodity]
    color_a, color_b = COMM_COLORS[comm_a], COMM_COLORS[comm_b]
    t0, t1 = pd.Timestamp(start_date), pd.Timestamp(end_date)

    cit_raw = load_cit()
    dag_raw = load_disagg("F&O")

    cit = cit_raw[
        (cit_raw["Commodity"] == comm_a) &
        (cit_raw["Date"] >= t0) & (cit_raw["Date"] <= t1)
    ].sort_values("Date").reset_index(drop=True)
    dag = dag_raw[
        (dag_raw["Commodity"] == comm_b) & (dag_raw["Crop"] == "All") &
        (dag_raw["Date"] >= t0) & (dag_raw["Date"] <= t1)
    ].sort_values("Date").reset_index(drop=True)

    cit = _inject_rollex(cit, comm_a)
    dag = _inject_rollex(dag, comm_b)

    if cit.empty or dag.empty:
        st.warning(f"No data for {'CIT' if cit.empty else 'Disagg'} leg."); return

    def gc(d, col):
        return d[col].astype(float) if col in d.columns else pd.Series(0., index=d.index)

    def _clabel(text):
        st.markdown(
            f"<p style='font-size:.71rem;color:#4b5563;margin:6px 0 6px;"
            f"padding:4px 10px;background:#f3f4f6;border-left:3px solid {color};"
            f"border-radius:0 4px 4px 0;line-height:1.5'>{text}</p>",
            unsafe_allow_html=True)

    _spec_a  = f"<b>{comm_a}</b> CIT: Large Spec + Non-Rep"
    _spec_b  = f"<b>{comm_b}</b> Disagg F&amp;O: MM + Other + Non-Rep"
    _oi_a    = f"<b>{comm_a}</b> CIT Total OI"
    _oi_b    = f"<b>{comm_b}</b> Disagg F&amp;O Total OI"
    _idx_a   = f"<b>{comm_a}</b> CIT Index Net"
    _comm_a  = f"<b>{comm_a}</b> CIT Comm Long / Short"
    _comm_b  = f"<b>{comm_b}</b> Disagg Producer Long / Short"

    # ── Spec extraction ───────────────────────────────────────────────────────
    # CIT leg: Large Spec + Non-Rep + extra columns for charts tab
    a = pd.DataFrame({
        "Date":   pd.to_datetime(cit["Date"]),
        "Long":   (gc(cit,"Spec Long")  + gc(cit,"Non Rep Long"))  / 1000,
        "Short":  (gc(cit,"Spec Short") + gc(cit,"Non Rep Short")) / 1000,
        "Net":    (gc(cit,"Spec Net")   + gc(cit,"Non Rep Net"))   / 1000,
        "IdxNet": gc(cit,"Index Net")   / 1000,
        "CommL":  gc(cit,"Comm Long")   / 1000,
        "CommS":  gc(cit,"Comm Short")  / 1000,
        "OI":     gc(cit,"Total OI")    / 1000,
        "Px":     cit["Px"].values if "Px" in cit.columns else np.nan,
    })
    # Disagg leg: MM + Other + Non-Rep + extra columns for charts tab
    b = pd.DataFrame({
        "Date":   pd.to_datetime(dag["Date"]),
        "Long":   (gc(dag,"MM Long")  + gc(dag,"Other Long")  + gc(dag,"Non Rep Long"))  / 1000,
        "Short":  (gc(dag,"MM Short") + gc(dag,"Other Short") + gc(dag,"Non Rep Short")) / 1000,
        "Net":    (gc(dag,"MM Net")   + gc(dag,"Other Net")   + gc(dag,"Non Rep Net"))   / 1000,
        "CommL":  gc(dag,"Producer Long")  / 1000,
        "CommS":  gc(dag,"Producer Short") / 1000,
        "OI":     gc(dag,"Total OI")       / 1000,
        "Px":     dag["Px"].values if "Px" in dag.columns else np.nan,
    })

    merged = pd.merge(a, b, on="Date", suffixes=("_a","_b"), how="inner").sort_values("Date").reset_index(drop=True)
    if merged.empty:
        st.warning("No overlapping dates between the two legs."); return

    # comm_a and comm_b can have different contract sizes (e.g. KC = 37,500 lbs
    # vs RC = 10 MT) — a raw "1 lot + 1 lot" sum would misweight the smaller
    # contract. Scale comm_a's lot counts into comm_b-equivalent lots (by MT)
    # before combining so "Combined" figures reflect true physical exposure.
    def _mt_size(comm):
        sz = CONTRACT_SIZE.get(comm, 1)
        return sz * 0.00045359237 if CONTRACT_UNIT.get(comm) == "lbs" else sz
    _scale_a = _mt_size(comm_a) / _mt_size(comm_b)

    merged["Comb Long"]     = merged["Long_a"] * _scale_a  + merged["Long_b"]
    merged["Comb Short"]    = merged["Short_a"] * _scale_a + merged["Short_b"]
    merged["Comb Net"]      = merged["Net_a"] * _scale_a   + merged["Net_b"]
    merged["Comb OI"]       = merged["OI_a"] * _scale_a    + merged["OI_b"]
    merged["Comb Net+Idx"]  = merged["Comb Net"] + merged["IdxNet"].fillna(0) * _scale_a
    merged["Rel Spec"]      = merged["Net_a"] * _scale_a   - merged["Net_b"]
    merged["Comb CommL"]    = merged["CommL_a"] * _scale_a + merged["CommL_b"]
    merged["Comb CommS"]    = merged["CommS_a"] * _scale_a + merged["CommS_b"]
    merged["CommNet_a"]     = merged["CommL_a"] - merged["CommS_a"]
    merged["CommNet_b"]     = merged["CommL_b"] - merged["CommS_b"]
    merged["Comb CommNet"]  = merged["CommNet_a"] * _scale_a + merged["CommNet_b"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>"
        f"<div style='width:5px;height:38px;background:{color};border-radius:3px'></div>"
        f"<div><div style='font-size:1.2rem;font-weight:700;color:{color}'>{COMM_NAMES[commodity]}</div>"
        f"<div style='font-size:.73rem;color:#888'>"
        f"{comm_a} CIT · Large+Small &nbsp;+&nbsp; {comm_b} Disagg · MM+Other+Non-Rep"
        f" &nbsp;·&nbsp; {start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}"
        f"</div></div></div>", unsafe_allow_html=True)
    st.markdown("---")



    # ── Tabs ──────────────────────────────────────────────────────────────────
    c_tabs = st.tabs(["Recap", "Recap (Charts)", "Combined Net", "Gross Legs", "Weekly Flow"])

    # ── Tab 0: Recap ──────────────────────────────────────────────────────────
    with c_tabs[0]:
        _clabel(f"Spec Net / Long / Short = ({_spec_a}) + ({_spec_b})")
        _clabel(f"Commercial Net / Long / Short = ({_comm_a}) + ({_comm_b})")
        col_map = {
            (f"{comm_a} Net", "Net"):    merged["Net_a"].values,
            (f"{comm_b} Net", "Net"):    merged["Net_b"].values,
            ("Combined",      "Net"):    merged["Comb Net"].values,
            (f"{comm_a} Gross","Long"):  merged["Long_a"].values,
            (f"{comm_a} Gross","Short"): merged["Short_a"].values,
            (f"{comm_b} Gross","Long"):  merged["Long_b"].values,
            (f"{comm_b} Gross","Short"): merged["Short_b"].values,
            ("Combined",      "Long"):   merged["Comb Long"].values,
            ("Combined",      "Short"):  merged["Comb Short"].values,
            (f"{comm_a} Commercial", "Net"):   merged["CommNet_a"].values,
            (f"{comm_b} Commercial", "Net"):   merged["CommNet_b"].values,
            ("Combined Commercial",  "Net"):   merged["Comb CommNet"].values,
            (f"{comm_a} Commercial", "Long"):  merged["CommL_a"].values,
            (f"{comm_a} Commercial", "Short"): merged["CommS_a"].values,
            (f"{comm_b} Commercial", "Long"):  merged["CommL_b"].values,
            (f"{comm_b} Commercial", "Short"): merged["CommS_b"].values,
            ("Combined Commercial",  "Long"):  merged["Comb CommL"].values,
            ("Combined Commercial",  "Short"): merged["Comb CommS"].values,
        }
        body_df = pd.DataFrame(col_map, index=merged["Date"])
        body_df.columns = pd.MultiIndex.from_tuples(body_df.columns)
        body_df = body_df.iloc[::-1]

        row_1w, row_4w, row_z, row_avg, row_min, row_max = {}, {}, {}, {}, {}, {}
        for c in body_df.columns:
            s = body_df[c].replace([np.inf,-np.inf],np.nan).dropna()
            if len(body_df) >= 2: row_1w[c] = body_df.iloc[0][c] - body_df.iloc[1][c]
            if len(body_df) >= 5: row_4w[c] = body_df.iloc[0][c] - body_df.iloc[4][c]
            if len(s) >= 4:
                mu, sigma = s.mean(), s.std()
                row_z[c] = (s.iloc[0]-mu)/sigma if sigma>0 else 0.
                row_avg[c] = mu; row_min[c] = s.min(); row_max[c] = s.max()

        summary_df = pd.DataFrame(
            [row_1w, row_4w, row_z, row_avg, row_min, row_max],
            index=["Δ 1w","Δ 1m","Z-Score","Avg","Min","Max"],
            columns=body_df.columns)

        with _exp("Change summary  ·  k lots", expanded=True):
            st.markdown(_recap_html(summary_df, signed_rows={"Δ 1w","Δ 1m","Z-Score"}, z_rows={"Z-Score"}, max_height=148), unsafe_allow_html=True)

        with _exp("Historical positions  ·  k lots", expanded=True):
            disp = body_df.iloc[:20].copy()
            disp.index = [f"{dt.day}-{dt.strftime('%b-%y')}" if hasattr(dt,'day') else str(dt) for dt in disp.index]
            st.markdown(_recap_html(disp, scroll=True), unsafe_allow_html=True)

        with _exp("Weekly change  ·  k lots", expanded=True):
            chg = body_df.diff(-1).iloc[:20].copy()
            chg.index = disp.index
            st.markdown(_recap_html(chg, signed=True, change_table=True, scroll=True), unsafe_allow_html=True)

            # Weekly Δ stats over selected period
            chg_full = body_df.diff(-1).dropna()
            if not chg_full.empty:
                rz, ra, rn, rx = {}, {}, {}, {}
                for c in chg_full.columns:
                    s2 = chg_full[c].replace([np.inf,-np.inf], np.nan).dropna()
                    if len(s2) >= 4:
                        mu2, sigma2 = s2.mean(), s2.std()
                        ra[c] = mu2; rn[c] = s2.min(); rx[c] = s2.max()
                        if not chg.empty and c in chg.columns:
                            v2 = chg.iloc[0][c]
                            rz[c] = (v2 - mu2) / sigma2 if pd.notna(v2) and sigma2 > 0 else np.nan
                chg_stats = pd.DataFrame(
                    [rz, ra, rn, rx],
                    index=["Z-Score Δ", "Avg Δ", "Min Δ", "Max Δ"],
                    columns=chg_full.columns)
                st.markdown(
                    f"<p style='font-size:.72rem;color:{GRAY};margin:10px 0 2px'>Weekly Δ stats  ·  selected period</p>",
                    unsafe_allow_html=True)
                st.markdown(_recap_html(chg_stats, signed=True, z_rows={"Z-Score Δ"}), unsafe_allow_html=True)

    # ── Tab 1: Recap (Charts) ─────────────────────────────────────────────────
    with c_tabs[1]:
        r_c, g_c, b_c2 = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

        def _cline(title, traces, height=360):
            fig = go.Figure()
            for name, y, clr, dash, width, fill, yax in traces:
                kw = dict(line=dict(color=clr, width=width, dash=dash))
                if fill:
                    kw["fill"] = "tozeroy"
                    kw["fillcolor"] = f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.09)"
                if yax == "y2":
                    kw["yaxis"] = "y2"
                fig.add_trace(go.Scatter(x=merged["Date"], y=y, name=name,
                    hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{name}: %{{y:.1f}}<extra></extra>", **kw))
            fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.15)")
            fig.update_layout(**_BASE, height=height,
                title=dict(text=title, font=dict(size=11, color="#374151"), x=0),
                margin=dict(l=52, r=60, t=40, b=72),
                legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                            font_size=10, bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
                yaxis=dict(**_ax(), title_text="k lots", title_font_size=10),
                yaxis2={**_ax(), "title_text": "Price", "title_font_size": 10,
                        "overlaying": "y", "side": "right", "showgrid": False})
            _chart(fig, width='stretch')

        # 1. Combined OI
        _clabel(f"Combined OI = {_oi_a} + {_oi_b}")
        _cline(
            f"{COMM_NAMES[commodity]}  ·  Combined Open Interest  ·  k lots",
            [
                (f"{comm_a} OI",   merged["OI_a"],    color_a, "dot",  1.6, False, "y"),
                (f"{comm_b} OI",   merged["OI_b"],    color_b, "dot",  1.6, False, "y"),
                ("Combined OI",    merged["Comb OI"], color,   "solid",2.4, True,  "y"),
            ])

        # 2. Combined Net Specs
        _clabel(f"Combined Net = ({_spec_a}) + ({_spec_b})")
        _cline(
            f"{COMM_NAMES[commodity]}  ·  Combined Net Specs  ·  k lots",
            [
                (f"{comm_a} Net",  merged["Net_a"],    color_a, "dot",  1.6, False, "y"),
                (f"{comm_b} Net",  merged["Net_b"],    color_b, "dot",  1.6, False, "y"),
                ("Combined Net",   merged["Comb Net"], color,   "solid",2.4, True,  "y"),
            ])

        # 2b. Combined Gross Spec Legs
        _clabel(f"Combined Gross Spec = ({_spec_a}) + ({_spec_b})  ·  Long / Short")
        _cline(
            f"{COMM_NAMES[commodity]}  ·  Combined Gross Spec Legs  ·  k lots",
            [
                ("Combined Spec Long",  merged["Comb Long"],  C_LONG,  "solid", 2.2, False, "y"),
                ("Combined Spec Short", merged["Comb Short"], C_SHORT, "solid", 2.2, False, "y"),
            ])

        # 3. Combined Net + Index
        _clabel(f"Combined Net + Index = ({_spec_a}) + ({_spec_b}) + {_idx_a}")
        _cline(
            f"{COMM_NAMES[commodity]}  ·  Combined Net + Index  ·  k lots",
            [
                ("Combined Net",        merged["Comb Net"],     color,     "dot",  1.8, False, "y"),
                ("Combined Net + Index", merged["Comb Net+Idx"], "#f59e0b", "solid",2.4, True,  "y"),
            ])

        # 4. Relative Spec: CIT Net minus Disagg Net (bar)
        _clabel(f"Relative Spec = ({_spec_a} Net) − ({_spec_b} Net)")
        rel = merged["Rel Spec"]
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Bar(
            x=merged["Date"], y=rel,
            marker=dict(color=[color_a if v >= 0 else color_b for v in rel],
                        opacity=0.75, line=dict(width=0)),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>CIT − Disagg Net: %{y:+.1f}k<extra></extra>",
            name=f"{comm_a} CIT − {comm_b} Disagg"))
        fig_rel.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.2)")
        fig_rel.update_layout(**_BASE, height=300,
            title=dict(text=f"{COMM_NAMES[commodity]}  ·  Relative Spec  ·  CIT Net − Disagg Net  ·  k lots",
                       font=dict(size=11, color="#374151"), x=0),
            margin=dict(l=52, r=12, t=40, b=60), showlegend=True,
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                        font_size=10, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
            yaxis=dict(**_ax(), title_text="k lots", title_font_size=10))
        _chart(fig_rel, width='stretch')

        # 5. Combined Gross Commercial Legs
        _clabel(f"Combined Commercial = {_comm_a} + {_comm_b}")
        _cline(
            f"{COMM_NAMES[commodity]}  ·  Combined Gross Commercial Legs  ·  k lots"
            f"  ({comm_a} Comm + {comm_b} Producer)",
            [
                ("Combined Comm Long",  merged["Comb CommL"], C_LONG,  "solid", 2.2, False, "y"),
                ("Combined Comm Short", merged["Comb CommS"], C_SHORT, "solid", 2.2, False, "y"),
            ])

    # ── Tab 2: Combined Net ───────────────────────────────────────────────────
    with c_tabs[2]:
        _clabel(f"Combined Net = ({_spec_a}) + ({_spec_b})")
        r, g, b_c = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Net_a"], name=f"{comm_a} Spec Net",
            line=dict(color=color_a, width=1.8, dash="dot"),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{comm_a} Net: %{{y:.1f}}k<extra></extra>"))
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Net_b"], name=f"{comm_b} Spec Net",
            line=dict(color=color_b, width=1.8, dash="dot"),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{comm_b} Net: %{{y:.1f}}k<extra></extra>"))
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Comb Net"], name="Combined Net",
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b_c},0.09)",
            line=dict(color=color, width=2.4),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Combined: %{y:.1f}k<extra></extra>"))
        fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.15)")
        fig.update_layout(**_BASE, height=420,
            title=dict(text=f"{COMM_NAMES[commodity]}  ·  Combined Spec Net  ·  k lots",
                       font=dict(size=12,color="#374151"), x=0),
            margin=dict(l=52,r=12,t=44,b=72),
            legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
            yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
        _chart(fig, width='stretch')

        # Combined Gross Spec Legs — Long / Short
        _clabel(f"Combined Gross Spec = ({_spec_a}) + ({_spec_b})  ·  Long / Short")
        fig_gross = go.Figure()
        fig_gross.add_trace(go.Scatter(x=merged["Date"], y=merged["Comb Long"], name="Combined Spec Long",
            line=dict(color=C_LONG, width=2.2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Combined Long: %{y:.1f}k<extra></extra>"))
        fig_gross.add_trace(go.Scatter(x=merged["Date"], y=merged["Comb Short"], name="Combined Spec Short",
            line=dict(color=C_SHORT, width=2.2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Combined Short: %{y:.1f}k<extra></extra>"))
        fig_gross.update_layout(**_BASE, height=320,
            title=dict(text=f"{COMM_NAMES[commodity]}  ·  Combined Gross Spec Legs  ·  k lots",
                       font=dict(size=11,color="#374151"), x=0),
            margin=dict(l=52,r=12,t=40,b=60),
            legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
            yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
        _chart(fig_gross, width='stretch')

        delta = merged["Comb Net"].diff().fillna(0)
        fig2 = go.Figure(go.Bar(x=merged["Date"], y=delta,
            marker=dict(color=[C_LONG if v>=0 else C_SHORT for v in delta], opacity=0.8, line=dict(width=0)),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Δ Combined Net: %{y:+.1f}k<extra></extra>"))
        fig2.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.14)")
        fig2.update_layout(**_BASE, height=240,
            title=dict(text="Combined Net — weekly Δ  ·  k lots",font=dict(size=11,color="#444"),x=0),
            margin=dict(l=50,r=12,t=36,b=60), showlegend=False,
            xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
            yaxis=dict(**_ax(),title_text="Δ k lots",title_font_size=10))
        _chart(fig2, width='stretch')

    # ── Tab 3: Gross Legs ────────────────────────────────────────────────────
    with c_tabs[3]:
        _clabel(f"Left: {_spec_a} (Long / Short / Net) &nbsp;·&nbsp; Right: {_spec_b} (Long / Short / Net)")
        col1, col2 = st.columns(2)
        for ch, (comm, sfx, clr, leg_lbl) in zip([col1, col2], [
            (comm_a, "_a", color_a, f"Large+Small (CIT)"),
            (comm_b, "_b", color_b, f"MM+Other+Non-Rep (Disagg)"),
        ]):
            with ch:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=merged["Date"], y=merged[f"Long{sfx}"], name="Long",
                    line=dict(color=C_LONG, width=2.0),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Long: %{y:.1f}k<extra></extra>"))
                fig.add_trace(go.Scatter(x=merged["Date"], y=merged[f"Short{sfx}"], name="Short",
                    line=dict(color=C_SHORT, width=2.0),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Short: %{y:.1f}k<extra></extra>"))
                fig.add_trace(go.Scatter(x=merged["Date"], y=merged[f"Net{sfx}"], name="Net",
                    fill="tozeroy", fillcolor="rgba(26,86,219,0.09)",
                    line=dict(color=C_NET, width=2.2),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Net: %{y:.1f}k<extra></extra>"))
                fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.15)")
                fig.update_layout(**_BASE, height=340,
                    title=dict(text=f"{comm} Spec · {leg_lbl}  ·  k lots",font=dict(size=11,color="#374151"),x=0),
                    margin=dict(l=50,r=12,t=40,b=72),
                    legend=dict(orientation="h",y=-0.24,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
                _chart(fig, width='stretch')

    # ── Tab 4: Weekly Flow ────────────────────────────────────────────────────
    with c_tabs[4]:
        _clabel(f"Weekly Δ: {_spec_a} &nbsp;|&nbsp; {_spec_b} &nbsp;|&nbsp; Combined = sum of both")
        wf1, wf2, wf3 = st.columns(3)
        for ch, (col, title) in zip([wf1,wf2,wf3],[
            ("Net_a",   f"{comm_a} Net Δ"),
            ("Net_b",   f"{comm_b} Net Δ"),
            ("Comb Net","Combined Net Δ"),
        ]):
            delta = merged[col].diff().fillna(0)
            fig = go.Figure(go.Bar(x=merged["Date"], y=delta,
                marker=dict(color=[C_LONG if v>=0 else C_SHORT for v in delta],opacity=0.8,line=dict(width=0)),
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Δ: %{{y:+.1f}}k<extra></extra>"))
            fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.14)")
            fig.update_layout(**_BASE, height=280,
                title=dict(text=f"{title}  ·  k lots",font=dict(size=10,color="#444"),x=0),
                margin=dict(l=44,r=8,t=36,b=52), showlegend=False,
                xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                yaxis=dict(**_ax(),title_text="Δ k lots",title_font_size=9))
            with ch: _chart(fig, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# SPEC PROXIMITY TAB — all commodities at once, sequential date table
# ══════════════════════════════════════════════════════════════════════════════
def _spec_proximity_table(df_comm: pd.DataFrame, threshold_k: float) -> pd.DataFrame:
    """For each row, walk backward and find the most recent prior date where
    |Spec_now - Spec_prior| ≤ threshold. Returns the full table including
    unmatched rows (blanks for Prev/Px2/Perf/Spec2)."""
    df_comm = df_comm.sort_values("Date").reset_index(drop=True)
    n   = len(df_comm)
    dts = df_comm["Date"].values
    sps = df_comm["Spec"].values
    pxs = df_comm["Px"].values

    rows = []
    for i in range(n):
        row = {
            "Date 1": dts[i],
            "Prev Date 2": pd.NaT,
            "Weeks Between": np.nan,
            "Px (Date 1)": pxs[i],
            "Px2": np.nan,
            "Performance %": np.nan,
            "Spec 1": sps[i],
            "Spec 2": np.nan,
        }
        for j in range(i - 1, -1, -1):
            if pd.notna(sps[j]) and abs(sps[i] - sps[j]) <= threshold_k:
                row["Prev Date 2"]   = dts[j]
                row["Weeks Between"] = int((pd.Timestamp(dts[i]) - pd.Timestamp(dts[j])).days // 7)
                row["Px2"]           = pxs[j]
                row["Spec 2"]        = sps[j]
                if pd.notna(pxs[i]) and pd.notna(pxs[j]) and pxs[j] != 0:
                    row["Performance %"] = (pxs[i] / pxs[j] - 1) * 100
                break
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Date 1", ascending=False).reset_index(drop=True)


_DEFAULT_THRESH = {"KC": 2.0, "CC": 2.0, "SB": 5.0, "CT": 1.0,
                   "RC": 2.0, "LCC": 1.0, "LSU": 2.0}
_GRID_ROWS = [["KC", "RC", "SB"], ["CC", "LCC", "CT"], ["LSU", None, None]]


def _render_one_proximity_table(comm, study_weeks, cit_df, dag_df, start_date, end_date):
    """Render a single ultra-compact proximity table for one commodity."""
    # Pick data + spec formula
    if comm in CIT_COMMS:
        src = cit_df[(cit_df["Commodity"] == comm) &
                     (cit_df["Date"] >= pd.Timestamp(start_date)) &
                     (cit_df["Date"] <= pd.Timestamp(end_date))].copy()
        if "Crop" in src.columns: src = src[src["Crop"] == "All"]
        for c in ("Spec Net", "Non Rep Net", "Index Net"):
            if c not in src.columns: src[c] = 0
        src["Spec"] = (src["Spec Net"] + src["Non Rep Net"] + src["Index Net"]) / 1000
    else:
        src = dag_df[(dag_df["Commodity"] == comm) &
                     (dag_df["Date"] >= pd.Timestamp(start_date)) &
                     (dag_df["Date"] <= pd.Timestamp(end_date))].copy()
        if "Crop" in src.columns: src = src[src["Crop"] == "All"]
        for c in ("MM Net", "Other Net", "Non Rep Net"):
            if c not in src.columns: src[c] = 0
        src["Spec"] = (src["MM Net"] + src["Other Net"] + src["Non Rep Net"]) / 1000

    src = _inject_rollex(src.reset_index(drop=True), comm)
    df_comm = src[["Date", "Spec", "Px"]].dropna(subset=["Spec"]).reset_index(drop=True)
    if df_comm.empty:
        st.markdown(f"<p style='font-size:.7rem;color:#999'>{COMM_NAMES[comm]} — no data</p>",
                    unsafe_allow_html=True)
        return

    # Trim to study window (last N weeks of COT data)
    df_window = df_comm.tail(int(study_weeks)).reset_index(drop=True)

    color  = COMM_COLORS.get(comm, "#444")
    thresh = st.number_input(
        f"{COMM_NAMES[comm].split(' : ')[0]} · Spec Proximity (k lots)",
        min_value=0.1, max_value=200.0,
        value=float(_DEFAULT_THRESH.get(comm, 2.0)),
        step=0.5,
        key=f"sp_thr_{comm}",
        label_visibility="visible",
    )

    table_df = _spec_proximity_table(df_window, thresh).head(15)

    if table_df.empty:
        st.markdown(f"<p style='font-size:.7rem;color:#999'>No rows.</p>", unsafe_allow_html=True)
        return

    # Compact HTML table
    _max_abs_perf = max(
        abs(table_df["Performance %"].dropna().max() or 0),
        abs(table_df["Performance %"].dropna().min() or 0),
        1.0,
    )

    def _cell_perf(v):
        if pd.isna(v):
            return '<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;color:#cbd5e1;text-align:right">—</td>'
        bar_pct = min(abs(v) / _max_abs_perf * 100, 100)
        if v > 0:   bg, tc, bar = "#dcfce7", "#166534", "#86efac"
        elif v < 0: bg, tc, bar = "#fee2e2", "#991b1b", "#fca5a5"
        else:       bg, tc, bar = "#f1f5f9", "#475569", "#cbd5e1"
        return (f'<td style="padding:0;border-bottom:1px solid #eef0f4;background:{bg};position:relative">'
                f'<div style="position:absolute;left:0;top:0;bottom:0;width:{bar_pct}%;background:{bar};opacity:.55"></div>'
                f'<div style="position:relative;padding:3px 6px;font-weight:700;color:{tc};text-align:right;font-size:.66rem">{v:+.1f}%</div>'
                f'</td>')

    def _cell_date(v, primary=False):
        if pd.isna(v):
            return '<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;color:#cbd5e1;font-size:.66rem">—</td>'
        w = "700" if primary else "500"
        c = "#0f172a" if primary else "#64748b"
        return (f'<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;'
                f'font-weight:{w};color:{c};font-variant-numeric:tabular-nums;font-size:.66rem">'
                f'{pd.Timestamp(v).strftime("%d/%m/%y")}</td>')

    def _cell_num(v, fmt="{:.1f}", color_hex="#334155", bold=False):
        if pd.isna(v):
            return '<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;color:#cbd5e1;text-align:right;font-size:.66rem">—</td>'
        w = "700" if bold else "500"
        return (f'<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;'
                f'color:{color_hex};font-weight:{w};text-align:right;'
                f'font-variant-numeric:tabular-nums;font-size:.66rem">{fmt.format(v)}</td>')

    def _cell_weeks(v):
        if pd.isna(v):
            return '<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;color:#cbd5e1;text-align:center;font-size:.66rem">—</td>'
        iv = int(v)
        if iv <= 2:    pill_bg, pill_tc = "#dbeafe", "#1e40af"
        elif iv <= 8:  pill_bg, pill_tc = "#fef3c7", "#92400e"
        else:          pill_bg, pill_tc = "#fce7f3", "#9d174d"
        return (f'<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;text-align:center">'
                f'<span style="background:{pill_bg};color:{pill_tc};padding:1px 6px;'
                f'border-radius:8px;font-size:.62rem;font-weight:700">{iv}w</span></td>')

    # Header — unified dark navy for all commodities
    _HEAD_BG = "#1e3a8a"
    ths = [("Date",       "left"),   ("Prev",       "left"),   ("Wks",         "center"),
           ("Px (New)",   "right"),  ("Px (Old)",   "right"),  ("Perf",        "right"),
           ("Spec (New)", "right"),  ("Spec (Old)", "right")]
    head_html = "".join(
        f'<th style="background:{_HEAD_BG};color:white;font-weight:600;font-size:.62rem;'
        f'padding:5px 6px;text-align:{ta};letter-spacing:.04em">{t.upper()}</th>'
        for t, ta in ths
    )

    # Body
    body_rows = []
    for _, r in table_df.iterrows():
        cells = (
            _cell_date(r["Date 1"], primary=True),
            _cell_date(r["Prev Date 2"]),
            _cell_weeks(r["Weeks Between"]),
            _cell_num(r["Px (Date 1)"], "{:.1f}", "#0f172a", bold=True),
            _cell_num(r["Px2"],         "{:.1f}", "#64748b"),
            _cell_perf(r["Performance %"]),
            _cell_num(r["Spec 1"], "{:+.1f}", "#0c4a6e", bold=True),
            _cell_num(r["Spec 2"], "{:+.1f}", "#64748b"),
        )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        f'<div style="border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;'
        f'box-shadow:0 1px 2px rgba(0,0,0,.04);margin-top:6px">'
        f'<table style="width:100%;border-collapse:collapse;'
        f'font-family:-apple-system,BlinkMacSystemFont,Helvetica Neue,sans-serif">'
        f'<thead><tr>{head_html}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # ── Dynamic Spec Proximity (vs latest) — uses same study window ──────────
    _render_dynamic_proximity(comm, df_window)


def _render_dynamic_proximity(comm, df_comm):
    """Collapsible per-commodity section: pin latest spec, list all historical
    dates within ±X k lots of it, and compute return-from-then-to-now."""
    if df_comm.empty:
        return

    df_full = df_comm.sort_values("Date").reset_index(drop=True)
    latest      = df_full.iloc[-1]
    latest_spec = float(latest["Spec"])
    latest_date = pd.Timestamp(latest["Date"])
    latest_px   = float(latest["Px"]) if pd.notna(latest["Px"]) else None

    with st.expander(f"{comm}  ·  Dynamic Spec Proximity  ·  vs latest spec", expanded=False):
        # KPI strip
        _px_str = f"{latest_px:.2f}" if latest_px is not None else "—"
        st.markdown(
            f"<div style='display:flex;gap:6px;margin:2px 0 8px'>"
            f"<div style='flex:1;padding:5px 8px;background:#f1f5f9;border-radius:4px'>"
            f"<div style='font-size:.55rem;color:#94a3b8;letter-spacing:.05em'>LATEST SPEC</div>"
            f"<div style='font-size:.78rem;font-weight:700;color:#0c4a6e'>{latest_spec:+.1f}k</div></div>"
            f"<div style='flex:1;padding:5px 8px;background:#f1f5f9;border-radius:4px'>"
            f"<div style='font-size:.55rem;color:#94a3b8;letter-spacing:.05em'>LATEST DATE</div>"
            f"<div style='font-size:.78rem;font-weight:700;color:#0f172a'>{latest_date.strftime('%d/%m/%y')}</div></div>"
            f"<div style='flex:1;padding:5px 8px;background:#f1f5f9;border-radius:4px'>"
            f"<div style='font-size:.55rem;color:#94a3b8;letter-spacing:.05em'>LATEST PX</div>"
            f"<div style='font-size:.78rem;font-weight:700;color:#1e3a8a'>{_px_str}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        dyn_thresh = st.number_input(
            f"Proximity around {latest_spec:+.1f}k  (±k lots)",
            min_value=0.1, max_value=500.0, value=10.0, step=1.0,
            key=f"sp_dyn_thr_{comm}",
        )

        # All historical matches within sidebar date range
        m = df_full[abs(df_full["Spec"] - latest_spec) <= dyn_thresh].copy()
        m = m.sort_values("Date", ascending=False).reset_index(drop=True)
        if m.empty:
            st.markdown(
                f"<p style='font-size:.7rem;color:#999;margin-top:6px'>"
                f"No matches within ±{dyn_thresh:.1f}k of latest spec.</p>",
                unsafe_allow_html=True,
            )
            return

        # Build cells
        if latest_px is None or latest_px == 0:
            m["Perf"] = np.nan
        else:
            m["Perf"] = (latest_px / m["Px"] - 1) * 100
        m["Days"] = (latest_date - pd.to_datetime(m["Date"])).dt.days

        _max_abs_perf = max(
            abs(m["Perf"].dropna().max() or 0),
            abs(m["Perf"].dropna().min() or 0),
            1.0,
        )

        def _c_perf(v):
            if pd.isna(v):
                return '<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;color:#cbd5e1;text-align:right;font-size:.66rem">—</td>'
            bar_pct = min(abs(v) / _max_abs_perf * 100, 100)
            if v > 0:   bg, tc, bar = "#dcfce7", "#166534", "#86efac"
            elif v < 0: bg, tc, bar = "#fee2e2", "#991b1b", "#fca5a5"
            else:       bg, tc, bar = "#f1f5f9", "#475569", "#cbd5e1"
            return (f'<td style="padding:0;border-bottom:1px solid #eef0f4;background:{bg};position:relative">'
                    f'<div style="position:absolute;left:0;top:0;bottom:0;width:{bar_pct}%;background:{bar};opacity:.55"></div>'
                    f'<div style="position:relative;padding:3px 6px;font-weight:700;color:{tc};text-align:right;font-size:.66rem">{v:+.1f}%</div>'
                    f'</td>')

        def _c_date(v, primary=False):
            w = "700" if primary else "500"
            c = "#0f172a" if primary else "#475569"
            return (f'<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;'
                    f'font-weight:{w};color:{c};font-variant-numeric:tabular-nums;font-size:.66rem">'
                    f'{pd.Timestamp(v).strftime("%d/%m/%y")}</td>')

        def _c_num(v, fmt="{:.1f}", color_hex="#334155", bold=False, align="right"):
            if pd.isna(v):
                return f'<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;color:#cbd5e1;text-align:{align};font-size:.66rem">—</td>'
            w = "700" if bold else "500"
            return (f'<td style="padding:3px 6px;border-bottom:1px solid #eef0f4;'
                    f'color:{color_hex};font-weight:{w};text-align:{align};'
                    f'font-variant-numeric:tabular-nums;font-size:.66rem">{fmt.format(v)}</td>')

        _HEAD_BG = "#1e3a8a"
        ths = [("Date","left"), ("Spec","right"), ("Rollex","right"),
               ("Perf","right"), ("Days","right")]
        head_html = "".join(
            f'<th style="background:{_HEAD_BG};color:white;font-weight:600;font-size:.62rem;'
            f'padding:5px 6px;text-align:{ta};letter-spacing:.04em">{t.upper()}</th>'
            for t, ta in ths
        )

        body_rows = []
        for _, r in m.iterrows():
            is_latest = (pd.Timestamp(r["Date"]) == latest_date)
            cells = (
                _c_date(r["Date"], primary=True),
                _c_num(r["Spec"], "{:+.1f}", "#0c4a6e", bold=True),
                _c_num(r["Px"],   "{:.2f}",  "#1e3a8a", bold=is_latest),
                _c_perf(r["Perf"]),
                _c_num(r["Days"], "{:.0f}",  "#64748b"),
            )
            body_rows.append("<tr>" + "".join(cells) + "</tr>")

        table_html = (
            f'<div style="border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;'
            f'box-shadow:0 1px 2px rgba(0,0,0,.04);margin-top:4px">'
            f'<table style="width:100%;border-collapse:collapse;'
            f'font-family:-apple-system,BlinkMacSystemFont,Helvetica Neue,sans-serif">'
            f'<thead><tr>{head_html}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody>'
            f'</table></div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:.62rem;color:#94a3b8;margin-top:4px'>"
            f"{len(m)} matches within ±{dyn_thresh:.1f}k  ·  "
            f"Perf = % move from that date's Rollex to latest ({_px_str})</p>",
            unsafe_allow_html=True,
        )


@st.fragment
def render_spec_proximity(start_date, end_date, commodity=None):
    """Spec Proximity — the sidebar commodity's table by default, or the 3x3
    grid of all softs when 'All commodities' is chosen."""
    # ── Title bar with both formulas ──────────────────────────────────────────
    st.markdown(
        "<div style='background:#581c54;color:white;padding:10px 16px;border-radius:6px;"
        "margin-bottom:12px;font-size:.85rem;font-weight:600;letter-spacing:.02em'>"
        "SPEC PROXIMITY  &nbsp;|&nbsp;  "
        "<span style='font-weight:400;font-size:.78rem;color:#fcd5e9'>"
        "Spec for US (KC / CC / SB / CT) = Spec + Index + Non Rep&nbsp;&nbsp;·&nbsp;&nbsp;"
        "Spec for European (RC / LCC / LSU) = Managed Money + Others + Non Rep"
        "</span></div>",
        unsafe_allow_html=True,
    )

    # ── Global study-window radio ─────────────────────────────────────────────
    _c1, _c2 = st.columns([0.25, 0.75])
    with _c1:
        window_lbl = st.radio(
            "Study window",
            ["24w", "52w"], index=0, horizontal=True,
            key="sp_window",
            help="How far back to search for proximity matches. Each table shows last 15 weeks from this window.",
        )
    with _c2:
        scope = st.radio("Show", ["Selected commodity", "All commodities"],
                         index=0, horizontal=True, key="sp_scope") if commodity else "All commodities"
    study_weeks = int(window_lbl.replace("w", ""))

    # Load both reports once
    cit_df = load_cit()
    dag_df = load_disagg("F&O")

    if scope.startswith("Selected"):
        _render_one_proximity_table(commodity, study_weeks, cit_df, dag_df, start_date, end_date)
        return

    # ── 3 x 3 grid ────────────────────────────────────────────────────────────
    for row_comms in _GRID_ROWS:
        cols = st.columns(3, gap="small")
        for col, comm in zip(cols, row_comms):
            if comm is None:
                continue
            with col:
                _render_one_proximity_table(comm, study_weeks, cit_df, dag_df, start_date, end_date)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.05rem;font-weight:700;color:#1a56db;"
        "margin-bottom:16px;letter-spacing:.01em'>COMPREHENSIVE COT</div>",
        unsafe_allow_html=True)

    commodity = st.selectbox("Commodity", list(COMM_NAMES.keys()),
                             format_func=lambda x: COMM_NAMES[x], key="sb_commodity")
    color = COMM_COLORS[commodity]
    is_combined = commodity in COMBINED_COMMS

    if is_combined:
        report = "Combined"
        version_key = None
        st.markdown("<div style='font-size:.73rem;color:#999;margin:-6px 0 8px'>"
                    "Combined view · CIT leg + Disagg F&amp;O leg</div>", unsafe_allow_html=True)
    else:
        cit_ok = commodity in CIT_COMMS
        if cit_ok:
            report = st.radio("Report", ["CIT","Disagg"], horizontal=True, key="rb_report")
        else:
            report = "Disagg"
            st.markdown("<div style='font-size:.73rem;color:#999;margin:-6px 0 8px'>"
                        "RC/LCC — Disaggregated only</div>", unsafe_allow_html=True)

        version_key = None
        if report == "Disagg":
            version = st.radio("Version", ["F&O combined","Fut only","Options only"], horizontal=True, key="rb_version")
            version_key = "F&O" if "F&O" in version else ("Opt" if "Options" in version else "Fut")

    st.markdown("---")
    st.markdown("<div style='font-size:.78rem;font-weight:600;color:#444;"
                "margin-bottom:6px'>Date range</div>", unsafe_allow_html=True)
    if is_combined:
        _max_date = load_cit()["Date"].max().date()
    elif report == "CIT":
        _cit_df = load_cit()
        _max_date = _cit_df.loc[_cit_df["Commodity"] == commodity, "Date"].max().date()
    else:
        _dag_df = load_disagg("F&O")
        _max_date = _dag_df.loc[_dag_df["Commodity"] == commodity, "Date"].max().date()
    _date_key_suffix = f"{commodity}_{report}"
    start_date = st.date_input("From", value=datetime.date(2020,1,1),
                               min_value=datetime.date(2010,1,1), max_value=_max_date,
                               key=f"dt_from_{_date_key_suffix}")
    end_date   = st.date_input("To",   value=_max_date,
                               min_value=datetime.date(2010,1,1), max_value=_max_date,
                               key=f"dt_to_{_date_key_suffix}")



# ══════════════════════════════════════════════════════════════════════════════
# COMBINED COMMODITY — early exit
# ══════════════════════════════════════════════════════════════════════════════
if is_combined:
    render_combined(commodity, start_date, end_date, color)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD + FILTER
# ══════════════════════════════════════════════════════════════════════════════
if report == "CIT":
    raw = load_cit()
    df_all_crops = None
    df = raw[
        (raw["Commodity"]==commodity) &
        (raw["Date"]>=pd.Timestamp(start_date)) &
        (raw["Date"]<=pd.Timestamp(end_date))
    ].sort_values("Date").reset_index(drop=True)
else:
    raw = load_options_only() if version_key == "Opt" else load_disagg(version_key)
    df_all_crops = raw[
        (raw["Commodity"]==commodity) &
        (raw["Date"]>=pd.Timestamp(start_date)) &
        (raw["Date"]<=pd.Timestamp(end_date))
    ].sort_values(["Crop","Date"]).reset_index(drop=True)
    df = df_all_crops[df_all_crops["Crop"]=="All"].sort_values("Date").reset_index(drop=True)

is_options = (report == "Disagg" and version_key == "Opt")

# ── Inject Rollex price (replaces static Px with roll-adjusted daily series) ──
df = _inject_rollex(df, commodity)
if df_all_crops is not None:
    df_all_crops = _inject_rollex(df_all_crops, commodity)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
# Page header removed — commodity is shown on each expander / chart title (see _comm_prefix).

if df.empty:
    st.warning("No data for the selected filters."); st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TAB — RECAP (CHARTS)
# ══════════════════════════════════════════════════════════════════════════════
def render_recap_charts(d, report, color, commodity):
    if d.empty:
        st.warning("No data for the selected filters."); return

    d   = d.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(d["Date"])

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(np.nan, index=d.index)

    # Nominal multiplier
    size = CONTRACT_SIZE.get(commodity, 1)
    unit = CONTRACT_UNIT.get(commodity, "MT")
    ccy  = "GBP" if commodity == "LCC" else "USD"
    px   = gc("Px")
    mult = (px * size / 100 / 1_000_000) if commodity in CENTS_QUOTED else (px * size / 1_000_000)
    oi   = gc("Total OI").replace(0, np.nan)

    def _line(title, series_dict, clrs=None):
        dflt = [C_LONG, C_SHORT, C_NET, "#f59e0b", "#7c3aed"]
        if clrs is None: clrs = dflt
        fig = go.Figure()
        fig.update_layout(
            **_BASE,
            title=dict(text=f"{commodity} — {title}", font=dict(size=10, color="#374151")),
            height=260,
            margin=dict(l=40, r=8, t=36, b=48),
            showlegend=True,
            legend=dict(orientation="h", y=-0.28, font=dict(size=9)),
            xaxis=dict(**_ax(x=True)),
            yaxis=dict(**_ax()),
        )
        for i, (name, y) in enumerate(series_dict.items()):
            fig.add_trace(go.Scatter(
                x=dates, y=y, name=name,
                line=dict(color=clrs[i % len(clrs)], width=1.5)
            ))
        return fig

    # Same column shape for both report types (3 cols x 4 rows) so Streamlit's
    # layout tree stays consistent across reruns when toggling CIT/Disagg —
    # mismatched container shapes cause stale charts to flash during rerender.
    c1, c2, c3    = st.columns(3)
    c4, c5, c6    = st.columns(3)
    c7, c8, c9    = st.columns(3)
    c10, c11, c12 = st.columns(3)

    if report == "CIT":
        large_net = gc("Spec Net")       # Large Spec only (Spec Long - Spec Short)
        small_net = gc("Non Rep Net")    # Small Spec / non-reportables only
        spec_net  = large_net + small_net.fillna(0)  # Net Spec = Large + Small (classic COT convention)
        idx_net   = gc("Index Net")

        # Col 1 — Net positioning. Title spells out the composition.
        with c1:
            _chart(_line(
                "Net Spec (Large + Small) & Net Index k lots",
                {"Net Spec": spec_net / 1000, "Net Index": idx_net / 1000},
                [C_NET, C_LONG]
            ), width='stretch')

        # Col 2 — Large Net vs Small Net breakdown, right beside it, so the
        # "Net Spec = Large + Small" composition is visible at a glance.
        with c2:
            _chart(_line(
                "Large Net & Small Net k lots",
                {"Large Net": large_net / 1000, "Small Net": small_net / 1000},
                ["#93c5fd", "#c4b5fd"]
            ), width='stretch')

        # Col 9 (was empty in CIT layout) — Spec gross k lots, moved here
        # to make room for the Large/Small breakdown chart above in c2.
        with c9:
            _chart(_line(
                "Spec Gross k lots",
                {"Large Long": gc("Spec Long") / 1000, "Large Short": gc("Spec Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Col 3 — Spec gross % of OI
        with c3:
            _chart(_line(
                "Spec Gross % of OI",
                {"Lrg+Sml Long %":  (gc("Spec Long")  + gc("Non Rep Long"))  / oi * 100,
                 "Lrg+Sml Short %": (gc("Spec Short") + gc("Non Rep Short")) / oi * 100},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Col 4 — Spec nominal M USD (gross, Large + Small — same Lrg+Sml
        # convention as the "Spec Gross % of OI" chart beside it)
        with c4:
            _chart(_line(
                f"Spec Nominal M {ccy}",
                {"Lrg+Sml Long":  (gc("Spec Long")  + gc("Non Rep Long").fillna(0))  * mult,
                 "Lrg+Sml Short": (gc("Spec Short") + gc("Non Rep Short").fillna(0)) * mult},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Col 1 bottom — # of Traders
        with c5:
            _chart(_line(
                "# of Traders",
                {"Large Long": gc("Traders Spec Long"), "Large Short": gc("Traders Spec Short")},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Col 2 bottom — Commercial gross k lots
        with c6:
            _chart(_line(
                "Commercial Gross k lots",
                {"Comm Long": gc("Comm Long") / 1000, "Comm Short": gc("Comm Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Col 3 bottom — Commercial % of OI
        with c7:
            _chart(_line(
                "Commercial Gross % of OI",
                {"Comm Long %":  gc("Comm Long")  / oi * 100,
                 "Comm Short %": gc("Comm Short") / oi * 100},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Col 4 bottom — Commercial nominal M USD
        with c8:
            _chart(_line(
                f"Commercial Nominal M {ccy}",
                {"Gross Long": gc("Comm Long") * mult, "Gross Short": gc("Comm Short") * mult},
                [C_LONG, C_SHORT]
            ), width='stretch')

    else:  # Disagg
        mm_net   = gc("MM Net")
        swap_net = gc("Swap Net")

        # Row 1 — MM
        with c1:
            _chart(_line(
                "MM Gross k lots",
                {"MM Long": gc("MM Long") / 1000, "MM Short": gc("MM Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')

        with c2:
            _chart(_line(
                "MM Gross % of OI",
                {"MM Long %":  gc("MM Long")  / oi * 100,
                 "MM Short %": gc("MM Short") / oi * 100},
                [C_LONG, C_SHORT]
            ), width='stretch')

        with c3:
            _chart(_line(
                f"MM Nominal M {ccy}",
                {"MM Long": gc("MM Long") * mult, "MM Short": gc("MM Short") * mult},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Row 2 — Commercial
        with c4:
            _chart(_line(
                "Commercial Gross k lots",
                {"Prod Long": gc("Producer Long") / 1000, "Prod Short": gc("Producer Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')

        with c5:
            _chart(_line(
                "Commercial Gross % of OI",
                {"Prod Long %":  gc("Producer Long")  / oi * 100,
                 "Prod Short %": gc("Producer Short") / oi * 100},
                [C_LONG, C_SHORT]
            ), width='stretch')

        with c6:
            _chart(_line(
                f"Commercial Nominal M {ccy}",
                {"Prod Long": gc("Producer Long") * mult, "Prod Short": gc("Producer Short") * mult},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Row 3 — Other Reportables
        with c7:
            _chart(_line(
                "Other Gross k lots",
                {"Other Long": gc("Other Long") / 1000, "Other Short": gc("Other Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')

        with c8:
            _chart(_line(
                "Other Gross % of OI",
                {"Other Long %":  gc("Other Long")  / oi * 100,
                 "Other Short %": gc("Other Short") / oi * 100},
                [C_LONG, C_SHORT]
            ), width='stretch')

        with c9:
            _chart(_line(
                f"Other Nominal M {ccy}",
                {"Other Long": gc("Other Long") * mult, "Other Short": gc("Other Short") * mult},
                [C_LONG, C_SHORT]
            ), width='stretch')

        # Row 4 — Cross-category
        with c10:
            _chart(_line(
                "MM Net & Swap Net & Other Net k lots",
                {"MM Net": mm_net / 1000, "Swap Net": swap_net / 1000, "Other Net": gc("Other Net") / 1000},
                [C_NET, C_LONG, "#f59e0b"]
            ), width='stretch')

        with c11:
            _chart(_line(
                "# of Traders",
                {"MM Long": gc("Traders MM Long"), "MM Short": gc("Traders MM Short"),
                 "Other Long": gc("Traders Other Long"), "Other Short": gc("Traders Other Short")},
                [C_LONG, C_SHORT, "#f59e0b", "#7c3aed"]
            ), width='stretch')

        with c12:
            _chart(_line(
                "Other Spread k lots",
                {"Other Spread": gc("Other Spread") / 1000},
                ["#f59e0b"]
            ), width='stretch')

    # ── Roll Yield vs Positioning ──────────────────────────────────────────────
    with _exp("Roll Yield vs Positioning", expanded=True):
        ry_all = load_roll_yield()
        _ry_key = {"LSU": "W"}.get(commodity, commodity)
        ry_comm = ry_all[ry_all["Commodity"] == _ry_key].copy()
        if ry_comm.empty:
            st.info("Roll yield data not available for this commodity.")
        else:
            cot_sorted = d.sort_values("Date").copy()
            ry_sorted  = ry_comm.sort_values("Date")
            merged_ry  = pd.merge_asof(cot_sorted, ry_sorted[["Date","roll_yield_pct"]], on="Date", direction="backward")

            ry_vals = merged_ry["roll_yield_pct"].values.astype(float)

            if report == "CIT":
                short_vals = ((gc("Spec Short") + gc("Non Rep Short")) / 1000).values.astype(float)
                net_vals   = (gc("Spec Net") / 1000).values.astype(float)
                short_lbl  = "Spec + Non-Rep Short (k lots)"
                net_lbl    = "Spec Net (k lots)"
            else:
                short_vals = ((gc("MM Short") + gc("Other Short") + gc("Non Rep Short")) / 1000).values.astype(float)
                net_vals   = (gc("MM Net") / 1000).values.astype(float)
                short_lbl  = "MM + Other + Non-Rep Short (k lots)"
                net_lbl    = "MM Net (k lots)"

            def _ry_scatter(x, y, ylabel, title):
                mask = ~(np.isnan(x) | np.isnan(y))
                xm, ym = x[mask], y[mask]
                if len(xm) < 5:
                    return go.Figure()
                # polynomial trendline degree 2
                coeffs = np.polyfit(xm, ym, 2)
                x_line = np.linspace(xm.min(), xm.max(), 200)
                y_line = np.polyval(coeffs, x_line)
                # R²
                y_hat  = np.polyval(coeffs, xm)
                ss_res = np.sum((ym - y_hat) ** 2)
                ss_tot = np.sum((ym - ym.mean()) ** 2)
                r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                fig = go.Figure()
                # all points
                fig.add_trace(go.Scatter(
                    x=xm, y=ym, mode="markers",
                    marker=dict(color=color, size=6, opacity=0.65, line=dict(width=0.4, color="white")),
                    hovertemplate="Roll Yield: %{x:.1f}%<br>" + ylabel.split(" (")[0] + ": %{y:.1f}k<extra></extra>",
                    showlegend=False,
                ))
                # trendline
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    showlegend=False,
                ))
                # latest point
                fig.add_trace(go.Scatter(
                    x=[x[~np.isnan(x) & ~np.isnan(y)][-1]],
                    y=[y[~np.isnan(x) & ~np.isnan(y)][-1]],
                    mode="markers",
                    marker=dict(color="#f97316", size=10, symbol="circle", line=dict(width=1.5, color="white")),
                    showlegend=False,
                    hovertemplate="<b>Latest</b><br>Roll Yield: %{x:.1f}%<br>" + ylabel.split(" (")[0] + ": %{y:.1f}k<extra></extra>",
                ))
                fig.add_annotation(
                    x=0.98, y=0.98, xref="paper", yref="paper",
                    text=f"R² = {r2:.4f}", showarrow=False,
                    font=dict(size=13, color="#111"), bgcolor="rgba(255,255,255,0.7)",
                    borderpad=4, xanchor="right", yanchor="top",
                )
                fig.update_layout(
                    **_BASE,
                    title=dict(text=f"{commodity} — {title}", font=dict(size=11, color="#374151")),
                    height=380,
                    margin=dict(l=48, r=16, t=44, b=48),
                    xaxis=dict(**_ax(), title=dict(text="Roll Yield 1yr (%)", font=dict(size=9)), tickformat=".1f", ticksuffix="%"),
                    yaxis=dict(**_ax(), title=dict(text=ylabel, font=dict(size=9))),
                    showlegend=False,
                )
                return fig

            col_a, col_b = st.columns(2)
            with col_a:
                _chart(_ry_scatter(ry_vals, short_vals, short_lbl, "Large Spec Short vs Roll Yield"), width='stretch')
            with col_b:
                _chart(_ry_scatter(ry_vals, net_vals, net_lbl, "Net Spec vs Roll Yield"), width='stretch')

            st.markdown("---")
            if report == "CIT":
                _ry_opts = [c for c in [
                    "Spec Net","Spec Long","Spec Short",
                    "Index Net","Index Long","Index Short",
                    "Non Rep Net","Non Rep Long","Non Rep Short",
                    "Comm Net","Comm Long","Comm Short","Total OI",
                ] if c in merged_ry.columns]
            else:
                _ry_opts = [c for c in [
                    "MM Net","MM Long","MM Short",
                    "Other Net","Other Long","Other Short",
                    "Non Rep Net","Non Rep Long","Non Rep Short",
                    "Comm Net","Combined Spec Net",
                    "Swap Net","Swap Long","Swap Short",
                    "Comm Long","Comm Short","Total OI",
                ] if c in merged_ry.columns]

            _ry_sel = st.selectbox("Y-axis element", _ry_opts, key=f"ry_custom_{commodity}")
            _ry_yvals = (merged_ry[_ry_sel] / 1000).values.astype(float)
            _chart(
                _ry_scatter(ry_vals, _ry_yvals, f"{_ry_sel} (k lots)", f"{_ry_sel} vs Roll Yield"),
                width='stretch',
            )


# ══════════════════════════════════════════════════════════════════════════════
# SPEC VAR TAB
# ══════════════════════════════════════════════════════════════════════════════
def render_spec_var(commodity: str, df_cot: pd.DataFrame, report: str, color: str,
                    start_date=None, end_date=None):
    var_df = _build_var_df(commodity)
    lot    = VAR_LOT_USD.get(commodity, 10)
    t0 = pd.Timestamp(start_date) if start_date is not None else var_df["Date"].min() if not var_df.empty else pd.Timestamp("2000-01-01")
    t1 = pd.Timestamp(end_date)   if end_date   is not None else var_df["Date"].max() if not var_df.empty else pd.Timestamp.now()
    if not var_df.empty:
        var_df = var_df[(var_df["Date"] >= t0) & (var_df["Date"] <= t1)]

    # colour palette for cross-commodity (KC/RC = blue family, CC/LCC = red/orange family)
    _VAR_COLORS = {
        "KC": "#1d4ed8", "RC": "#38bdf8",
        "CC": "#b91c1c", "LCC":"#fb923c",
        "SB": "#059669", "CT": "#7c3aed",
    }
    _VAR_DASH = {
        "KC": "solid",   "RC": "dash",
        "CC": "solid",   "LCC":"dash",
        "SB": "solid",   "CT": "solid",
    }
    _VAR_WIDTH = {
        "KC": 2.0, "RC": 1.6,
        "CC": 2.0, "LCC":1.6,
        "SB": 1.6, "CT": 1.6,
    }

    # ── build predefined combination columns ─────────────────────────────────
    df_c = df_cot.copy()
    if report == "CIT":
        for side in ("Long", "Short", "Net"):
            if f"Spec {side}" in df_c.columns and f"Non Rep {side}" in df_c.columns:
                df_c[f"Spec + Non Rep {side}"] = df_c[f"Spec {side}"] + df_c[f"Non Rep {side}"].fillna(0)
            if f"Spec + Non Rep {side}" in df_c.columns and f"Index {side}" in df_c.columns:
                df_c[f"Spec + Non Rep + Index {side}"] = df_c[f"Spec + Non Rep {side}"] + df_c[f"Index {side}"].fillna(0)
        spec_opts = [c for c in ["Spec Net", "Spec + Non Rep Net", "Spec + Non Rep + Index Net"]
                     if c in df_c.columns]
    else:
        for side in ("Long", "Short", "Net"):
            if f"MM {side}" in df_c.columns and f"Non Rep {side}" in df_c.columns:
                df_c[f"MM + Non Rep {side}"] = df_c[f"MM {side}"] + df_c[f"Non Rep {side}"].fillna(0)
            if f"MM + Non Rep {side}" in df_c.columns and f"Other {side}" in df_c.columns:
                df_c[f"MM + Non Rep + Other {side}"] = df_c[f"MM + Non Rep {side}"] + df_c[f"Other {side}"].fillna(0)
        spec_opts = [c for c in ["MM Net", "MM + Non Rep Net", "MM + Non Rep + Other Net"]
                     if c in df_c.columns]
    if not spec_opts:
        st.warning("No spec Net columns found for this report type.")
        return

    # ── controls ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        spec_sel = st.selectbox("Spec Position", spec_opts, key=f"var_spec_{commodity}")
    with c2:
        win_sel = st.radio("Vol Window", [20, 60, 120], horizontal=True,
                           format_func=lambda x: f"{x}D", key=f"var_win_{commodity}")

    if var_df.empty:
        st.warning(f"No rollex / VaR data for {commodity}.")
        return

    base_name = spec_sel.rsplit(" ", 1)[0]
    long_col  = f"{base_name} Long"
    short_col = f"{base_name} Short"
    _ver_lbl  = f"{report} {version_key}" if report == "Disagg" and version_key else report

    # ── Expander 2: Spec Book VaR — Net/Long/Short + WoW change ──────────────
    with _exp(f"Spec Book VaR — {spec_sel}  ·  {win_sel}D  [{_ver_lbl}]", expanded=True):
        vcol = f"vol_{win_sel}"
        if vcol not in var_df.columns or spec_sel not in df_c.columns:
            st.info("Data not available.")
        else:
            vol_s = var_df[["Date", vcol]].dropna().copy()
            extra_cols = [c for c in [long_col, short_col] if c in df_c.columns]
            cols_sel = ["Date", spec_sel, "Px"] + extra_cols
            df_m = pd.merge_asof(
                df_c.sort_values("Date")[cols_sel].dropna(subset=[spec_sel]),
                vol_s.sort_values("Date"),
                on="Date", direction="backward",
            ).dropna(subset=[vcol])
            df_m["vpl"] = df_m["Px"] * lot * df_m[vcol] * _CONF_Z

            df_m = df_m.sort_values("Date")
            df_m["Net VaR ($M)"]   = df_m[spec_sel] * df_m["vpl"] / 1e6
            if long_col  in df_m.columns: df_m["Long VaR ($M)"]  = df_m[long_col]  * df_m["vpl"] / 1e6
            if short_col in df_m.columns: df_m["Short VaR ($M)"] = df_m[short_col] * df_m["vpl"] / 1e6
            df_m["Δ WoW"] = df_m["Net VaR ($M)"].diff()
            df_m = df_m.sort_values("Date", ascending=False).head(52)

            def _mfmt(v, invert=False):
                if pd.isna(v): return "—"
                pos = (v > 0) if not invert else (v < 0)
                cls = "rpos" if pos else "rneg"
                return f"<span class='{cls}'>${v:.1f}M</span>"
            def _dfmt(v):
                if pd.isna(v): return "—"
                cls = "rpos" if v > 0 else "rneg"
                arrow = "▲" if v > 0 else "▼"
                return f"<span class='{cls}'>{arrow}${abs(v):.1f}M</span>"

            rows_e2 = ""
            for _, r in df_m.iterrows():
                rows_e2 += (
                    f"<tr><td class='idx'>{r['Date'].strftime('%d-%b-%y')}</td>"
                    f"<td>{_mfmt(r.get('Net VaR ($M)'))}</td>"
                    f"<td>{_dfmt(r.get('Δ WoW'))}</td>"
                    f"<td>{_mfmt(r.get('Long VaR ($M)'))}</td>"
                    f"<td>{_mfmt(r.get('Short VaR ($M)'), invert=True)}</td></tr>"
                )
            hdr_e2 = ("<tr style='background:#f3f4f6'>"
                      "<th class='idx'>Date</th>"
                      "<th>Net VaR ($M)</th><th>Δ WoW</th>"
                      "<th>Long VaR ($M)</th><th>Short VaR ($M)</th></tr>")
            st.markdown(
                f"{_RECAP_CSS}<div style='overflow-x:auto;overflow-y:auto;max-height:480px'>"
                f"<table class='rtbl'><thead>{hdr_e2}</thead><tbody>{rows_e2}</tbody></table></div>",
                unsafe_allow_html=True,
            )

    # ── Expander 3: Book VaR timeseries — selected window only ───────────────
    with _exp(f"Book VaR — {win_sel}D timeseries  ({spec_sel}) [{_ver_lbl}]", expanded=True):
        vcol = f"vol_{win_sel}"
        if spec_sel not in df_c.columns or vcol not in var_df.columns:
            st.info("Data not available.")
        else:
            vol_s = var_df[["Date", vcol]].dropna().copy()
            dm = pd.merge_asof(
                df_c.sort_values("Date")[["Date", spec_sel, "Px"]].dropna(subset=[spec_sel]),
                vol_s.sort_values("Date"), on="Date", direction="backward",
            ).dropna(subset=[vcol])
            dm["vpl"] = dm["Px"] * lot * dm[vcol] * _CONF_Z
            dm["VaR"] = dm[spec_sel] * dm["vpl"] / 1e6
            dm = dm.sort_values("Date")

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=dm["Date"], y=dm["VaR"],
                mode="lines", name=f"{win_sel}D Net VaR ($M)",
                line=dict(color=color, width=1.8),
            ))
            fig3.update_layout(
                **_BASE, height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                yaxis=dict(**_ax(), title=dict(text="Net VaR ($M)", font=dict(size=9))),
                xaxis=dict(**_ax(x=True)),
            )
            _chart(fig3, use_container_width=True)

    # ── Expander 4: Long / Short VaR decomposition ────────────────────────────
    with _exp(f"Long / Short VaR — {win_sel}D  ({base_name}) [{_ver_lbl}]", expanded=False):
        vcol = f"vol_{win_sel}"
        has_legs = any(c in df_c.columns for c in [long_col, short_col])
        if not has_legs or vcol not in var_df.columns:
            st.info("Long / Short columns not available for this report type.")
        else:
            vol_ls = var_df[["Date", vcol]].dropna().copy()
            ls_cols = ["Date", "Px"] + [c for c in [long_col, short_col] if c in df_c.columns]
            dm_ls = pd.merge_asof(
                df_c.sort_values("Date")[ls_cols],
                vol_ls.sort_values("Date"), on="Date", direction="backward",
            ).dropna(subset=[vcol]).sort_values("Date")
            dm_ls["vpl"] = dm_ls["Px"] * lot * dm_ls[vcol] * _CONF_Z
            if long_col  in dm_ls.columns: dm_ls["Long VaR ($M)"]  = dm_ls[long_col]  * dm_ls["vpl"] / 1e6
            if short_col in dm_ls.columns: dm_ls["Short VaR ($M)"] = dm_ls[short_col] * dm_ls["vpl"] / 1e6

            fig4 = go.Figure()
            if "Long VaR ($M)" in dm_ls.columns:
                fig4.add_trace(go.Scatter(
                    x=dm_ls["Date"], y=dm_ls["Long VaR ($M)"],
                    mode="lines", name="Long VaR ($M)",
                    line=dict(color=C_LONG, width=1.6),
                ))
            if "Short VaR ($M)" in dm_ls.columns:
                fig4.add_trace(go.Scatter(
                    x=dm_ls["Date"], y=dm_ls["Short VaR ($M)"],
                    mode="lines", name="Short VaR ($M)",
                    line=dict(color=C_SHORT, width=1.6),
                ))
            fig4.update_layout(
                **_BASE, height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                yaxis=dict(**_ax(), title=dict(text="VaR ($M)", font=dict(size=9))),
                xaxis=dict(**_ax(x=True)),
            )
            _chart(fig4, use_container_width=True)

    # ── Expander 5: Cross-Commodity Comparison ────────────────────────────────
    with st.expander("Cross-Commodity Comparison", expanded=False):
        c_win = st.radio(
            "Window", [20, 60, 120], horizontal=True,
            format_func=lambda x: f"{x}D", key=f"var_cross_win_{commodity}",
        )

        # helper: build NetVaR timeseries for one commodity + one spec column
        def _cross_ts(comm, df_source, spec_col):
            sub = df_source[df_source["Commodity"] == comm].copy()
            if "Crop" in sub.columns:
                sub = sub[sub["Crop"] == "All"]
            if sub.empty or spec_col not in sub.columns:
                return None, None
            cv_df = _build_var_df(comm)
            if cv_df.empty:
                return None, None
            vc = f"vol_{c_win}"
            if vc not in cv_df.columns:
                return None, None
            vol_s2 = cv_df[["Date", vc]].dropna().copy()
            mc = pd.merge_asof(
                sub.sort_values("Date")[["Date", spec_col, "Px"]].dropna(subset=[spec_col]),
                vol_s2.sort_values("Date"), on="Date", direction="backward",
            ).dropna(subset=[vc])
            mc["vpl"] = mc["Px"] * VAR_LOT_USD.get(comm, 10) * mc[vc] * _CONF_Z
            mc = mc[(mc["Date"] >= t0) & (mc["Date"] <= t1)]
            if mc.empty:
                return None, None
            mc["NetVaR"] = mc[spec_col] * mc["vpl"] / 1e6
            snap = mc.sort_values("Date").iloc[-1]
            snap_info = {
                "name": COMM_NAMES[comm],
                "date": snap["Date"].strftime("%d-%b-%y"),
                "pos_k": f"{snap[spec_col]/1000:.1f}k",
                "vpl": f"${snap['vpl']:,.0f}",
                "net_var": snap["NetVaR"],
            }
            return mc[["Date","NetVaR"]].copy(), snap_info

        def _render_cross_section(comms, df_source, spec_opts_list, sel_key, chart_key):
            c_spec = st.selectbox("Spec", spec_opts_list, key=sel_key)
            # derive MM+Other if needed
            df_src = df_source.copy()
            if "MM + Other" in c_spec:
                side = c_spec.split()[-1]
                df_src["MM + Other Net"] = (
                    df_src.get("MM Net", pd.Series(0, index=df_src.index)) +
                    df_src.get("Other Net", pd.Series(0, index=df_src.index))
                )
            snaps, fig_x = [], go.Figure()
            for comm in comms:
                ts, snap = _cross_ts(comm, df_src, c_spec)
                if ts is None:
                    continue
                snaps.append(snap)
                ts = ts.sort_values("Date").copy()
                fig_x.add_trace(go.Scatter(
                    x=ts["Date"], y=ts["NetVaR"],
                    mode="lines", name=COMM_NAMES[comm],
                    line=dict(
                        color=_VAR_COLORS.get(comm, "#888"),
                        width=_VAR_WIDTH.get(comm, 1.5),
                        dash=_VAR_DASH.get(comm, "solid"),
                    ),
                ))
            if snaps:
                snaps.sort(key=lambda x: abs(x["net_var"]) if pd.notna(x["net_var"]) else 0, reverse=True)
                sr = ""
                for r in snaps:
                    nv = r["net_var"]
                    nv_cls = "rpos" if nv > 0 else "rneg"
                    nv_s = f"<span class='{nv_cls}'>${nv:.1f}M</span>" if pd.notna(nv) else "—"
                    sr += (f"<tr><td class='idx'>{r['name']}</td><td>{r['date']}</td>"
                           f"<td>{r['pos_k']}</td><td>{r['vpl']}</td><td>{nv_s}</td></tr>")
                hdr_x = ("<tr style='background:#f3f4f6'><th class='idx'>Commodity</th>"
                         f"<th>Date</th><th>{c_spec} (k)</th>"
                         "<th>VaR/lot($)</th><th>Net VaR($M)</th></tr>")
                st.markdown(
                    f"{_RECAP_CSS}<div style='overflow-x:auto'><table class='rtbl'>"
                    f"<thead>{hdr_x}</thead><tbody>{sr}</tbody></table></div>",
                    unsafe_allow_html=True,
                )
            fig_x.update_layout(
                **_BASE, height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                yaxis=dict(**_ax(), title=dict(text="Net VaR ($M)", font=dict(size=9))),
                xaxis=dict(**_ax(x=True)),
            )
            st.plotly_chart(fig_x, use_container_width=True, key=chart_key)

        # ── Section A: Disagg F&O — all 6 commodities ─────────────────────────
        st.markdown("**Disagg F&O — all commodities**  ·  <span style='font-size:.75rem;color:#888'>always Disagg F&O regardless of sidebar selection</span>", unsafe_allow_html=True)
        disagg_fo = load_disagg("F&O")
        if "MM Net" in disagg_fo.columns and "Non Rep Net" in disagg_fo.columns:
            disagg_fo["MM + Non Rep Net"] = disagg_fo["MM Net"] + disagg_fo["Non Rep Net"].fillna(0)
        if "MM + Non Rep Net" in disagg_fo.columns and "Other Net" in disagg_fo.columns:
            disagg_fo["MM + Non Rep + Other Net"] = disagg_fo["MM + Non Rep Net"] + disagg_fo["Other Net"].fillna(0)
        _render_cross_section(
            list(COMM_NAMES.keys()), disagg_fo,
            ["MM Net", "MM + Non Rep Net", "MM + Non Rep + Other Net"],
            f"var_cross_disagg_{commodity}",
            f"var_cross_chart_disagg_{commodity}",
        )

        st.markdown("---")

        # ── Section B: CIT — US markets only ─────────────────────────────────
        st.markdown("**CIT — US markets (KC · CC · SB · CT)**  ·  <span style='font-size:.75rem;color:#888'>always CIT regardless of sidebar selection</span>", unsafe_allow_html=True)
        cit_all = load_cit()
        if "Spec Net" in cit_all.columns and "Non Rep Net" in cit_all.columns:
            cit_all["Spec + Non Rep Net"] = cit_all["Spec Net"] + cit_all["Non Rep Net"].fillna(0)
        if "Spec + Non Rep Net" in cit_all.columns and "Index Net" in cit_all.columns:
            cit_all["Spec + Non Rep + Index Net"] = cit_all["Spec + Non Rep Net"] + cit_all["Index Net"].fillna(0)
        _render_cross_section(
            ["KC", "CC", "SB", "CT"], cit_all,
            ["Spec Net", "Spec + Non Rep Net", "Spec + Non Rep + Index Net"],
            f"var_cross_cit_{commodity}",
            f"var_cross_chart_cit_{commodity}",
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAIRS TAB  (KC+RC  and  CC+LCC  — always Disagg F&O)
# ══════════════════════════════════════════════════════════════════════════════
def render_pairs(start_date=None, end_date=None, commodity=None):
    if commodity in {"KC", "RC"}:
        pair = "KC + RC"
    elif commodity in {"CC", "LCC"}:
        pair = "CC + LCC"
    else:
        pair = "SB + LSU"

    st.markdown(
        f"<div style='font-size:.75rem;color:#555;margin-bottom:14px;padding:7px 14px;"
        f"background:#f8f9fb;border-radius:6px;border:1px solid #e5e7eb'>"
        f"Showing pair <b>{pair}</b> · Always uses <b>Disaggregated F&O</b> data regardless of sidebar report selection.</div>",
        unsafe_allow_html=True,
    )

    view = st.radio("View", ["Combined Net", "Individual Legs"], horizontal=True, key="pair_view")

    if "KC" in pair:
        comm_a, comm_b = "KC", "RC"
    elif "CC" in pair:
        comm_a, comm_b = "CC", "LCC"
    else:
        comm_a, comm_b = "SB", "LSU"
    clr_a = COMM_COLORS[comm_a]
    clr_b = COMM_COLORS[comm_b]

    fo   = load_disagg("F&O")
    t0   = pd.Timestamp(start_date) if start_date else pd.Timestamp("2010-01-01")
    t1   = pd.Timestamp(end_date)   if end_date   else pd.Timestamp.now()

    def _get(comm):
        sub = fo[(fo["Commodity"] == comm) & (fo["Crop"] == "All")].sort_values("Date").reset_index(drop=True)
        return sub[(sub["Date"] >= t0) & (sub["Date"] <= t1)].reset_index(drop=True)

    da = _get(comm_a)
    db = _get(comm_b)

    if da.empty or db.empty:
        st.warning("Not enough data for this pair in the selected date range.")
        return

    suf_a, suf_b = f"_{comm_a}", f"_{comm_b}"
    mg = pd.merge(da, db, on="Date", suffixes=(suf_a, suf_b), how="inner")
    if mg.empty:
        st.warning("No overlapping dates for this pair.")
        return

    ALL_LEGS = ["MM Net", "MM Long", "MM Short",
                "Comm Net", "Comm Long", "Comm Short",
                "Other Net", "Non Rep Net", "Combined Spec Net"]
    NET_OPTS = [c for c in ALL_LEGS if f"{c}{suf_a}" in mg.columns]
    LEG_OPTS = NET_OPTS  # same list for individual view

    def _col(comm, base):
        key = f"{base}_{comm}"
        return mg[key] if key in mg.columns else pd.Series(np.nan, index=mg.index)

    st.markdown("---")

    # ── COMBINED NET ──────────────────────────────────────────────────────────
    if view == "Combined Net":
        defaults = [n for n in ["MM Net", "Comm Net"] if n in NET_OPTS]
        sel_nets = st.multiselect(
            "COT Elements", NET_OPTS, default=defaults, key="pair_nets",
        )
        if not sel_nets:
            st.info("Select at least one element above.")
            return

        palette = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#d97706"]
        fig = go.Figure()
        for i, net in enumerate(sel_nets):
            sa = _col(comm_a, net)
            sb = _col(comm_b, net)
            combined = sa.fillna(0) + sb.fillna(0)
            clr = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=mg["Date"], y=combined, mode="lines",
                name=f"{net} (combined)",
                line=dict(color=clr, width=2.0),
            ))
            fig.add_trace(go.Scatter(
                x=mg["Date"], y=sa, mode="lines",
                name=f"{net} ({comm_a})",
                line=dict(color=clr, width=1.2, dash="dot"),
                visible="legendonly",
            ))
            fig.add_trace(go.Scatter(
                x=mg["Date"], y=sb, mode="lines",
                name=f"{net} ({comm_b})",
                line=dict(color=clr, width=1.2, dash="dash"),
                visible="legendonly",
            ))
        fig.update_layout(
            **_BASE, height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
            yaxis=dict(**_ax(), title=dict(text="k lots", font=dict(size=9))),
            xaxis=dict(**_ax(x=True)),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"pair_net_{comm_a}{comm_b}")

    # ── INDIVIDUAL LEGS ───────────────────────────────────────────────────────
    else:
        sel_leg = st.selectbox("COT Element", LEG_OPTS, key="pair_leg")

        sa = _col(comm_a, sel_leg)
        sb = _col(comm_b, sel_leg)

        sa_chg = sa.diff().dropna()
        sb_chg = sb.diff().dropna()
        shared = sa_chg.index.intersection(sb_chg.index)
        r_val  = None
        if len(shared) > 5:
            r_val = float(np.corrcoef(sa_chg.loc[shared].values, sb_chg.loc[shared].values)[0, 1])

        # Dual-axis timeseries
        st.markdown(
            f"<div style='font-size:.78rem;font-weight:600;color:#374151;margin-bottom:4px'>"
            f"{sel_leg}  —  {COMM_NAMES[comm_a]}  vs  {COMM_NAMES[comm_b]}</div>",
            unsafe_allow_html=True,
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=mg["Date"], y=sa, mode="lines",
            name=comm_a,
            line=dict(color=clr_a, width=1.9),
        ))
        fig2.add_trace(go.Scatter(
            x=mg["Date"], y=sb, mode="lines",
            name=comm_b,
            line=dict(color=clr_b, width=1.9),
        ))
        fig2.update_layout(
            **_BASE, height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
            yaxis=dict(**_ax(), title=dict(text="k lots", font=dict(size=9))),
            xaxis=dict(**_ax(x=True)),
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"pair_leg_{comm_a}{comm_b}")

        # Correlation scatter
        if len(shared) > 5:
            xa = sa_chg.loc[shared].values
            xb = sb_chg.loc[shared].values
            dates_s = mg.loc[shared, "Date"] if "Date" in mg.columns else pd.Series(shared)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=xa, y=xb, mode="markers",
                marker=dict(
                    color=list(range(len(xa))), colorscale="Blues",
                    size=6, opacity=0.75, showscale=False,
                ),
                text=[d.strftime("%d-%b-%y") if hasattr(d, "strftime") else str(d) for d in dates_s],
                hovertemplate=(f"{comm_a} delta: %{{x:.1f}}k<br>"
                               f"{comm_b} delta: %{{y:.1f}}k<br>%{{text}}<extra></extra>"),
                showlegend=False,
            ))
            fig3.add_trace(go.Scatter(
                x=[xa[-1]], y=[xb[-1]], mode="markers",
                marker=dict(color="#f97316", size=10, symbol="star"),
                showlegend=False,
            ))
            fig3.update_layout(
                **_BASE, height=320,
                margin=dict(l=48, r=16, t=16, b=48),
                xaxis=dict(**_ax(), title=dict(text=f"{comm_a} weekly change (k lots)", font=dict(size=9))),
                yaxis=dict(**_ax(), title=dict(text=f"{comm_b} weekly change (k lots)", font=dict(size=9))),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True, key=f"pair_scatter_{comm_a}{comm_b}")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
show_pairs = commodity in {"KC", "RC", "CC", "LCC", "SB", "LSU"}


# ══════════════════════════════════════════════════════════════════════════════
# TAB — PAIN TRADE (positioning flow + Rollex price-action)
# ══════════════════════════════════════════════════════════════════════════════
_PT_DARK_GREEN, _PT_LIGHT_GREEN = "#1a6b1a", "#7dce7d"
_PT_DARK_RED,   _PT_LIGHT_RED   = "#8b0000", "#f4a0a0"
_PT_AMBER, _PT_BLACK, _PT_NAVY  = "#e8a020", "#1d1d1f", "#0a2463"

def _pt_label(text):
    return (f"<div style='background:{_PT_NAVY};padding:5px 13px;border-radius:5px;"
            f"margin-bottom:8px'><span style='font-size:.78rem;font-weight:500;"
            f"letter-spacing:.07em;text-transform:uppercase;color:#dde4f0'>{text}</span></div>")

def _pt_nice_step(price_range):
    if price_range <= 0:
        return 1
    raw = price_range / 12
    mag = 10 ** np.floor(np.log10(raw))
    for mult in [1, 2, 2.5, 5, 10]:
        cand = mag * mult
        if cand >= raw:
            return max(1, int(cand))
    return max(1, int(mag * 10))

def _pt_hbar(values, prices, color, name):
    xs, ys = [], []
    for v, p in zip(values, prices):
        if pd.notna(v) and pd.notna(p) and v != 0:
            xs += [0, float(v), None]
            ys += [float(p), float(p), None]
    return go.Scatter(x=xs, y=ys, mode="lines", name=name,
                      line=dict(color=color, width=4))

def render_pain_trade(d, commodity, report, color, is_options):
    if commodity not in {"KC", "RC", "CC", "LCC", "SB", "CT"}:
        _na("Pain Trade is available for KC, RC, CC, LCC, SB, and CT.")
        return
    if is_options:
        _na("Pain Trade is not available for Options-only data.")
        return
    if d.empty:
        st.warning("No data for the selected filters.")
        return

    if report == "CIT":
        spec_l, spec_s, long3, short3, third_label = \
            "Spec Long", "Spec Short", "Index Long", "Index Short", "Index"
    else:
        spec_l, spec_s, long3, short3, third_label = \
            "MM Long", "MM Short", "Other Long", "Other Short", "Other Rep."

    # Daily Rollex for the post-COT dotted extension
    _rx_raw  = load_rollex(commodity).rename(columns={"rollex_px": "Rollex"})
    rx_daily = (
        _rx_raw[["Date", "Rollex"]].dropna(subset=["Rollex"])
        if not _rx_raw.empty else pd.DataFrame(columns=["Date", "Rollex"])
    )
    # Full version with active_label for the threshold-selector display
    _rx_labelled = (
        _rx_raw[["Date", "Rollex", "active_label"]].dropna(subset=["Rollex"])
        if not _rx_raw.empty and "active_label" in _rx_raw.columns
        else pd.DataFrame()
    )

    # Controls
    _radio_key = f"pt_radio_{commodity}_{report}"
    incl = st.radio(
        f"Include {third_label} in spec legs?",
        [f"Yes — Spec + Non Rep + {third_label}", "No — Spec + Non Rep only"],
        index=0, horizontal=True, key=_radio_key,
    )
    use_third = incl.startswith("Yes")

    # Compute legs (use full unfiltered series first so .diff is correct, then trim)
    df_pt = d.copy().sort_values("Date").reset_index(drop=True)
    df_pt["Rollex"] = pd.to_numeric(df_pt["Px"], errors="coerce")

    # Clip each position column to Total OI — guards against CFTC data entry errors
    # where a single row has an absurdly large value (e.g. LCC Non Rep Long 2.6bn in 2019)
    if "Total OI" in df_pt.columns:
        oi_cap = pd.to_numeric(df_pt["Total OI"], errors="coerce").clip(lower=1)
        for _pc in [spec_l, spec_s, long3, short3, "Non Rep Long", "Non Rep Short"]:
            if _pc in df_pt.columns:
                df_pt[_pc] = pd.to_numeric(df_pt[_pc], errors="coerce").clip(upper=oi_cap)

    if use_third:
        gross_long  = (df_pt[spec_l] + df_pt["Non Rep Long"]  + df_pt[long3])  / 1000
        gross_short = (df_pt[spec_s] + df_pt["Non Rep Short"] + df_pt[short3]) / 1000
        leg_label   = f"Spec + Non Rep + {third_label}"
    else:
        gross_long  = (df_pt[spec_l] + df_pt["Non Rep Long"])  / 1000
        gross_short = (df_pt[spec_s] + df_pt["Non Rep Short"]) / 1000
        leg_label   = "Spec + Non Rep"

    long_chg, short_chg = gross_long.diff(), gross_short.diff()
    df_pt["Long Add"]    =  long_chg.clip(lower=0)
    df_pt["Long Liq"]    =  long_chg.clip(upper=0)
    df_pt["Short Add"]   = -short_chg.clip(lower=0)
    df_pt["Short Cover"] = -short_chg.clip(upper=0)

    # ── Last N weeks selector ─────────────────────────────────────────────────
    _pt_max = df_pt["Date"].max()
    _pt_min = df_pt["Date"].min()
    _nw_opts = {"13w": 13, "26w": 26, "52w": 52, "Custom": None}
    _nw_sel  = st.radio("Show last", list(_nw_opts.keys()), index=0,
                        horizontal=True, key=f"pt_nw_{commodity}_{report}")
    _n_weeks = _nw_opts[_nw_sel]

    # diff already computed on full history — only slice for display
    if _nw_sel == "Custom":
        _cust_def = (pd.Timestamp(_pt_max) - pd.Timedelta(weeks=52)).date()
        _cust_min = _pt_min.date()
        _cust_max = _pt_max.date()
        _c_from, _c_to = st.slider(
            "Custom range",
            min_value=_cust_min, max_value=_cust_max,
            value=(max(_cust_min, _cust_def), _cust_max),
            format="DD MMM YYYY",
            key=f"pt_cust_{commodity}_{report}",
        )
        dff = df_pt[
            (df_pt["Date"] >= pd.Timestamp(_c_from)) &
            (df_pt["Date"] <= pd.Timestamp(_c_to))
        ].copy()
    else:
        dff = df_pt[df_pt["Date"] >= _pt_max - pd.Timedelta(weeks=_n_weeks)].copy()

    last_cot_date = dff["Date"].max()
    last_cot_str  = last_cot_date.strftime("%d/%m/%Y") if pd.notna(last_cot_date) else "—"
    latest_rx_str = rx_daily["Date"].max().strftime("%d/%m/%Y") if not rx_daily.empty else last_cot_str

    _rx_upto    = rx_daily[rx_daily["Date"] <= last_cot_date] if not rx_daily.empty else pd.DataFrame()
    if not _rx_upto.empty:
        window_px   = float(_rx_upto["Rollex"].iloc[-1])
        window_date = _rx_upto["Date"].iloc[-1].strftime("%d/%m/%Y")
        _lbl_upto   = _rx_labelled[_rx_labelled["Date"] <= last_cot_date] if not _rx_labelled.empty else pd.DataFrame()
        window_active_lbl = str(_lbl_upto["active_label"].iloc[-1]) if not _lbl_upto.empty else ""
    else:
        _cot_rx = dff.dropna(subset=["Rollex"])
        window_px   = float(_cot_rx["Rollex"].iloc[-1]) if not _cot_rx.empty else np.nan
        window_date = _cot_rx["Date"].iloc[-1].strftime("%d/%m/%Y") if not _cot_rx.empty else "—"
        window_active_lbl = ""

    # Latest Rollex — most recent daily value (may be ahead of latest COT date)
    if not rx_daily.empty:
        px_latest_rx         = float(rx_daily["Rollex"].iloc[-1])
        date_latest_rx       = rx_daily["Date"].iloc[-1].strftime("%d/%m/%Y")
        latest_rx_active_lbl = str(_rx_labelled["active_label"].iloc[-1]) if not _rx_labelled.empty else ""
    else:
        px_latest_rx, date_latest_rx = window_px, window_date
        latest_rx_active_lbl = window_active_lbl

    # ── VISUAL 1 — Spec legs bars + Rollex ───────────────────────────────────
    st.markdown(
        _pt_label(f"{commodity} — Spec Legs Weekly Change ({leg_label}) · Rollex (Right) "
                  f"| COT as of {last_cot_str} · Rollex as of {latest_rx_str}"),
        unsafe_allow_html=True,
    )

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    for col, c, name in [
        ("Long Add",    _PT_DARK_GREEN,  "Long Add"),
        ("Long Liq",    _PT_LIGHT_GREEN, "Long Liq."),
        ("Short Add",   _PT_DARK_RED,    "Short Add"),
        ("Short Cover", _PT_LIGHT_RED,   "Short Cover"),
    ]:
        fig1.add_trace(go.Bar(x=dff["Date"], y=dff[col], name=name,
                              marker_color=c, opacity=0.92), secondary_y=False)

    rx_solid = dff.dropna(subset=["Rollex"])
    fig1.add_trace(go.Scatter(x=rx_solid["Date"], y=rx_solid["Rollex"],
                              name="Rollex (COT period)", mode="lines",
                              line=dict(color=_PT_BLACK, width=2)),
                   secondary_y=True)

    if not rx_solid.empty and not rx_daily.empty:
        last_solid = rx_solid.iloc[-1:][["Date", "Rollex"]]
        rx_after   = rx_daily[rx_daily["Date"] > last_cot_date][["Date", "Rollex"]]
        rx_ext     = pd.concat([last_solid, rx_after]).sort_values("Date")
        if len(rx_ext) > 1:
            fig1.add_trace(go.Scatter(x=rx_ext["Date"], y=rx_ext["Rollex"],
                                      name=f"Rollex post-COT ({latest_rx_str})",
                                      mode="lines",
                                      line=dict(color=_PT_AMBER, width=2, dash="dot")),
                           secondary_y=True)
            # Endpoint marker + label for the latest post-COT Rollex value
            _last_pt = rx_ext.iloc[-1]
            fig1.add_trace(go.Scatter(
                x=[_last_pt["Date"]], y=[_last_pt["Rollex"]],
                mode="markers+text",
                marker=dict(color=_PT_AMBER, size=11, symbol="diamond",
                            line=dict(color=_PT_BLACK, width=1)),
                text=[f"  {_last_pt['Rollex']:.1f}"],
                textposition="middle right",
                textfont=dict(size=10, color=_PT_AMBER,
                              family="-apple-system,Helvetica Neue,sans-serif"),
                showlegend=False,
                hovertemplate=f"Latest Rollex %{{y:.2f}}<br>{latest_rx_str}<extra></extra>",
            ), secondary_y=True)

    # X-axis padding: extend ~5 days past the latest Rollex date so the
    # post-COT dotted segment + endpoint marker sit clear of the last bar.
    _x_left  = dff["Date"].min() - pd.Timedelta(days=2)
    _x_right_anchor = (
        rx_daily["Date"].max() if not rx_daily.empty else dff["Date"].max()
    )
    _x_right = _x_right_anchor + pd.Timedelta(days=5)

    fig1.update_layout(
        barmode="relative", height=420,
        margin=dict(t=10, b=10, l=4, r=4),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=9)),
        xaxis=dict(showgrid=False, tickfont=dict(size=9),
                   range=[_x_left, _x_right]),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system,Helvetica Neue,sans-serif", color=_PT_BLACK, size=10),
    )
    fig1.update_yaxes(title_text="k Contracts", secondary_y=False,
                      showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9))
    fig1.update_yaxes(title_text="Rollex Price", secondary_y=True,
                      showgrid=False, tickfont=dict(size=9))

    _chart(fig1, width='stretch')
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── VISUAL 2 — Rollex (Y) vs COT breakdown (X) ───────────────────────────
    scatter_df = dff.dropna(subset=["Rollex"]).copy()
    window_px_str = f"{window_px:.1f}" if pd.notna(window_px) else "—"
    st.markdown(
        _pt_label(f"{commodity} — Long Add · Liq | Short Add · Cover | Rollex (Y-axis) "
                  f"· Rollex {window_px_str} as of {window_date} · COT as of {last_cot_str}"),
        unsafe_allow_html=True,
    )

    _rx_range_v2 = (scatter_df["Rollex"].max() - scatter_df["Rollex"].min()
                    if not scatter_df.empty else 1)
    _auto_step_v2 = _pt_nice_step(_rx_range_v2)
    _ptm_c, _ = st.columns([0.18, 0.82])
    with _ptm_c:
        ptm_step = st.number_input(
            f"Bucket size (auto {_auto_step_v2})",
            min_value=0.1, value=float(_auto_step_v2), step=0.5,
            format="%.1f", key=f"ptm_step_{commodity}_{report}",
        )

    # ── Bucket by Rollex price (PTM Controller) ───────────────────────────────
    _bkt = scatter_df.copy()
    _bkt["_bin"]   = ((_bkt["Rollex"] / ptm_step).apply(np.floor) * ptm_step).round(4)
    _bkt["_label"] = _bkt["_bin"].apply(lambda x: f"{x:.1f}–{x + ptm_step:.1f}")
    _agg = (
        _bkt.groupby(["_bin", "_label"])[
            ["Long Add", "Long Liq", "Short Add", "Short Cover"]
        ].sum().reset_index().sort_values("_bin")
    )

    # Attach COT dates for each bucket (used in hover)
    _bkt_dates = (
        _bkt.groupby("_label")["Date"]
        .apply(lambda d: " · ".join(d.sort_values().dt.strftime("%d/%m/%Y").tolist()))
        .reset_index(name="_dates")
    )
    _agg = _agg.merge(_bkt_dates, on="_label", how="left")
    _agg["_dates"] = _agg["_dates"].fillna("")

    _la   = _agg["Long Add"].fillna(0)
    _ll   = _agg["Long Liq"].fillna(0)
    _sa   = _agg["Short Add"].fillna(0)
    _sc   = _agg["Short Cover"].fillna(0)
    _zero = [0.0] * len(_agg)
    _dates = _agg["_dates"]

    # Pass actual flow value + dates as customdata so hover shows the bar's
    # own contribution, not %{x} which Plotly resolves to base+x (end position).
    def _cd(vals):
        return [[round(float(v), 3), d] for v, d in zip(vals, _dates)]

    _htpl = (
        "<b>%{fullData.name}</b>: %{customdata[0]:.2f}k contracts<br>"
        "Bucket: %{y}<br>"
        "COT: %{customdata[1]}"
        "<extra></extra>"
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=_agg["_label"], x=_la, base=_zero,
        name="Long Add", orientation="h",
        marker_color=_PT_DARK_GREEN, opacity=0.9,
        customdata=_cd(_la), hovertemplate=_htpl,
    ))
    fig2.add_trace(go.Bar(
        y=_agg["_label"], x=_sc, base=_la,
        name="Short Cover", orientation="h",
        marker_color=_PT_LIGHT_RED, opacity=0.9,
        customdata=_cd(_sc), hovertemplate=_htpl,
    ))
    fig2.add_trace(go.Bar(
        y=_agg["_label"], x=_ll, base=_zero,
        name="Long Liq.", orientation="h",
        marker_color=_PT_LIGHT_GREEN, opacity=0.9,
        customdata=_cd(_ll), hovertemplate=_htpl,
    ))
    fig2.add_trace(go.Bar(
        y=_agg["_label"], x=_sa, base=_ll,
        name="Short Add", orientation="h",
        marker_color=_PT_DARK_RED, opacity=0.9,
        customdata=_cd(_sa), hovertemplate=_htpl,
    ))

    fig2.add_vline(x=0, line_color="#cccccc", line_width=1)

    # ── Week tags (W-4 … Latest) mapped to their bucket ──────────────────────
    recent5 = scatter_df.tail(5).reset_index(drop=True)
    _week_label_list = ["W-4", "W-3", "W-2", "W-1", "Latest"]
    _bucket_weeks: dict = {}
    for _wi, _wrow in recent5.iterrows():
        _wl   = _week_label_list[_wi] if _wi < len(_week_label_list) else f"W-{4 - _wi}"
        _wb   = round(np.floor(float(_wrow["Rollex"]) / ptm_step) * ptm_step, 4)
        _wlbl = f"{_wb:.1f}–{_wb + ptm_step:.1f}"
        _bucket_weeks.setdefault(_wlbl, []).append(
            (_wl, _wrow["Date"].strftime("%d/%m"), _wl == "Latest")
        )
    for _blbl, _wks in _bucket_weeks.items():
        if _blbl in _agg["_label"].values:
            _tag   = " · ".join(w for w, _, _ in _wks)
            _dates = " · ".join(d for _, d, _ in _wks)
            _is_lat = any(il for _, _, il in _wks)
            fig2.add_annotation(
                x=1.01, xref="paper", y=_blbl, yref="y",
                text=(f"<b style='color:{'#8b0000' if _is_lat else _PT_NAVY}'>{_tag}</b>"
                      f"<i style='font-size:7px;color:#888888'> {_dates}</i>"),
                showarrow=False, xanchor="left", align="left",
                font=dict(size=8, color=_PT_NAVY, family="-apple-system,sans-serif"),
                bgcolor="rgba(255,255,255,0.88)",
            )

    # ── COT-date and latest Rollex bucket indicators ──────────────────────────
    def _nearest_bucket_label(price, agg_df, step):
        row = agg_df[(agg_df["_bin"] <= price) & (price < agg_df["_bin"] + step)]
        if row.empty:
            row = agg_df.iloc[(agg_df["_bin"] - price).abs().argsort()[:1]]
        return row.iloc[0]["_label"] if not row.empty else None

    if pd.notna(window_px) and not _agg.empty:
        _cur_lbl = _nearest_bucket_label(window_px, _agg, ptm_step)
        if _cur_lbl:
            fig2.add_shape(
                type="line", x0=0, x1=1, xref="paper",
                y0=_cur_lbl, y1=_cur_lbl, yref="y",
                line=dict(color="#888888", width=1.5, dash="dot"),
                layer="below",
            )
            fig2.add_annotation(
                x=1.01, xref="paper", y=_cur_lbl, yref="y",
                text=f"<b>Price at COT</b> ({window_px:.1f}, {window_date})",
                showarrow=False, xanchor="left", align="left",
                font=dict(size=8.5, color="#4a5568", family="-apple-system,sans-serif"),
                bgcolor="rgba(230,230,245,0.92)",
            )
    if pd.notna(px_latest_rx) and not _agg.empty and date_latest_rx != window_date:
        _rx_lbl = _nearest_bucket_label(px_latest_rx, _agg, ptm_step)
        if _rx_lbl:
            fig2.add_shape(
                type="line", x0=0, x1=1, xref="paper",
                y0=_rx_lbl, y1=_rx_lbl, yref="y",
                line=dict(color=_PT_AMBER, width=1.5, dash="dot"),
                layer="below",
            )
            fig2.add_annotation(
                x=1.27, xref="paper", y=_rx_lbl, yref="y",
                text=f"<b>Latest Rollex Price</b> ({px_latest_rx:.1f}, {date_latest_rx})",
                showarrow=False, xanchor="left", align="left",
                font=dict(size=8.5, color=_PT_AMBER, family="-apple-system,sans-serif"),
                bgcolor="rgba(255,248,220,0.92)",
            )

    fig2.update_layout(
        barmode="overlay",
        height=max(380, len(_agg) * 62 + 80),
        margin=dict(t=10, b=10, l=90, r=480),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=9)),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9),
                   title="k Contracts", zeroline=False, autorange=True),
        yaxis=dict(showgrid=False, tickfont=dict(size=9),
                   title="Rollex Bucket",
                   categoryorder="array", categoryarray=list(_agg["_label"])),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system,Helvetica Neue,sans-serif",
                  color=_PT_BLACK, size=10),
    )
    _l, _ch, _r = st.columns([0.125, 0.75, 0.125])
    with _ch:
        _chart(fig2, width='stretch')

    # ── Rollex bucket table ───────────────────────────────────────────────────
    with _exp("Positioning by Rollex Level", expanded=False):
        # Compact controls row: Step | Weeks | Threshold Rollex radio
        _c_step, _c_wks, _c_radio = st.columns([0.12, 0.12, 0.76])
        with _c_wks:
            n_weeks = st.number_input("Weeks", min_value=1, max_value=500, value=13,
                                       step=1, key=f"pt_tbl_wks_{commodity}_{report}")

        tbl_full = df_pt[["Date", "Rollex", "Long Add", "Long Liq", "Short Add", "Short Cover"]] \
                    .dropna(subset=["Rollex"]).copy()
        tbl_df = tbl_full.sort_values("Date").tail(int(n_weeks)).copy()

        _rx_range  = tbl_df["Rollex"].max() - tbl_df["Rollex"].min() if not tbl_df.empty else 1
        _auto_step = _pt_nice_step(_rx_range)
        with _c_step:
            rx_step = st.number_input(f"Step (auto: {_auto_step})",
                                       min_value=1, value=_auto_step, step=1,
                                       key=f"pt_rx_step_{commodity}_{report}")

        # ── Threshold Rollex selector (determines Above / Below split) ────────
        _opt_cot    = "Latest COT"
        _opt_rx     = "Latest Rollex"
        _opt_custom = "Custom"
        with _c_radio:
            _rx_mode = st.radio(
                "Threshold Rollex",
                [_opt_cot, _opt_rx, _opt_custom],
                index=0, horizontal=True,
                key=f"pt_rx_mode_{commodity}_{report}",
                help="Rollex level used to split bins into Above / Below in the table below.",
            )
        _cot_px_str = f"{window_px:.2f}" if pd.notna(window_px) else "—"
        _rx_px_str  = f"{px_latest_rx:.2f}" if pd.notna(px_latest_rx) else "—"
        _cot_month  = f" · {window_active_lbl}" if window_active_lbl else ""
        _rx_month   = f" · {latest_rx_active_lbl}" if latest_rx_active_lbl else ""
        st.markdown(
            f"<p style='font-size:.72rem;color:#888;margin:-6px 0 8px 0;letter-spacing:.01em'>"
            f"Latest COT &nbsp;<b style='color:#444'>{_cot_px_str}</b>"
            f"<span style='color:#bbb'> &nbsp;{window_date}{_cot_month}&nbsp; </span>"
            f"<span style='color:#ccc;padding:0 10px'>|</span>"
            f"Latest Rollex &nbsp;<b style='color:#444'>{_rx_px_str}</b>"
            f"<span style='color:#bbb'> &nbsp;{date_latest_rx}{_rx_month}</span>"
            f"</p>",
            unsafe_allow_html=True,
        )
        if _rx_mode == _opt_cot:
            tbl_window_px, tbl_window_date = window_px, window_date
        elif _rx_mode == _opt_rx:
            tbl_window_px, tbl_window_date = px_latest_rx, date_latest_rx
        else:
            _c_custom, _c_pad2 = st.columns([0.2, 0.8])
            with _c_custom:
                tbl_window_px = st.number_input(
                    "Custom level",
                    value=float(window_px) if pd.notna(window_px) else 0.0,
                    step=float(rx_step),
                    format="%.1f",
                    key=f"pt_rx_custom_{commodity}_{report}",
                )
            tbl_window_date = "custom"

        if not tbl_df.empty and rx_step > 0:
            rx_floor = (tbl_df["Rollex"].min() // rx_step) * rx_step
            rx_ceil  = (tbl_df["Rollex"].max() // rx_step + 1) * rx_step
            bins     = np.arange(rx_floor, rx_ceil + rx_step, rx_step)
            bin_lbls = [f"{int(b)} – {int(b + rx_step)}" for b in bins[:-1]]

            tbl_df["RxBin"] = pd.cut(tbl_df["Rollex"], bins=bins, labels=bin_lbls, right=False)
            flow_cols = ["Long Add", "Long Liq", "Short Add", "Short Cover"]

            sorted_dates   = tbl_df["Date"].sort_values(ascending=False).reset_index(drop=True)
            week_label_map = {d: f"W-{i}" if i > 0 else "W0" for i, d in enumerate(sorted_dates)}
            tbl_df["_wlbl"] = tbl_df["Date"].map(week_label_map)

            agg = tbl_df.groupby("RxBin", observed=False)
            # Reverse bin_lbls (highest → lowest) for display; avoid alphabetical sort
            # which misaligns bins_desc when labels span a digit-length boundary (e.g. "9–10" vs "10–11")
            bin_lbls_desc = bin_lbls[::-1]
            grouped          = agg[flow_cols].sum().reindex(bin_lbls_desc)
            grouped["n"]     = agg["_wlbl"].count().reindex(bin_lbls_desc)
            grouped["Weeks"] = agg["_wlbl"].apply(
                lambda s: ",  ".join(s.sort_values().tolist())
            ).reindex(bin_lbls_desc)

            bins_desc  = bins[:-1][::-1]   # lower edges descending — aligned with bin_lbls_desc
            above_mask = bins_desc >= tbl_window_px if pd.notna(tbl_window_px) else np.array([False]*len(bins_desc))
            below_mask = (bins_desc + rx_step) <= tbl_window_px if pd.notna(tbl_window_px) else np.array([False]*len(bins_desc))

            grp_filled = grouped[flow_cols + ["n"]].fillna(0)
            above_row  = grp_filled[above_mask].sum()
            below_row  = grp_filled[below_mask].sum()
            above_row["Weeks"] = f"≥ {tbl_window_px:.0f}" if pd.notna(tbl_window_px) else ""
            below_row["Weeks"] = f"< {tbl_window_px:.0f}" if pd.notna(tbl_window_px) else ""

            total_row          = grp_filled.sum()
            total_row["Weeks"] = ""

            wpx_lbl = f"{tbl_window_px:.0f}" if pd.notna(tbl_window_px) else "—"
            summary_labels = [f"Above  ({wpx_lbl})", f"Below  ({wpx_lbl})", "TOTAL"]
            summary_df = pd.DataFrame([above_row, below_row, total_row], index=summary_labels)
            display_tbl = pd.concat([grouped, summary_df])
            display_tbl.index.name = "Rollex Range"

            def _style_bucket(df_):
                styles = pd.DataFrame("", index=df_.index, columns=df_.columns)
                for ri in df_.index:
                    is_summary = ri in summary_labels
                    for col in df_.columns:
                        v = df_.at[ri, col]
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(fv):
                            continue
                        bold = "font-weight:700;" if is_summary else "font-weight:600;"
                        if col in ("Long Add", "Short Cover"):
                            cc = "#1a6b1a" if fv > 0 else "#dc2626" if fv < 0 else ""
                        elif col in ("Long Liq", "Short Add"):
                            cc = "#dc2626" if fv != 0 else ""
                        else:
                            cc = ""
                        if cc:
                            styles.at[ri, col] = f"color:{cc};{bold}"
                return styles

            fmt = {c: "{:+.2f}" for c in flow_cols}
            fmt["n"] = "{:.0f}"
            styled = (display_tbl.style
                      .format(fmt, na_rep="—")
                      .apply(_style_bucket, axis=None)
                      .set_table_styles([
                          {"selector": "thead th",
                           "props": [("font-size", ".75rem"), ("color", "#444"),
                                     ("font-weight", "600"), ("text-align", "center")]},
                          {"selector": "td", "props": [("font-size", ".78rem"),
                                                       ("text-align", "right")]},
                          {"selector": "td:last-child",
                           "props": [("text-align", "left"), ("color", "#555"),
                                     ("font-size", ".72rem"), ("white-space", "nowrap"),
                                     ("font-family", "monospace")]},
                          {"selector": "th.row_heading",
                           "props": [("font-size", ".75rem"), ("text-align", "left"),
                                     ("color", "#555"), ("font-weight", "500")]},
                          # Summary section — dark top border on first summary row
                          {"selector": "tr:nth-last-child(3) td, tr:nth-last-child(3) th",
                           "props": [("border-top", "2.5px solid #1e293b !important"),
                                     ("background", "#f0f4f8"), ("font-weight", "700")]},
                          {"selector": "tr:nth-last-child(2) td, tr:nth-last-child(2) th",
                           "props": [("background", "#f0f4f8"), ("font-weight", "700")]},
                          # TOTAL row — strongest border + slightly darker bg
                          {"selector": "tr:last-child td, tr:last-child th",
                           "props": [("border-top", "2px solid #64748b !important"),
                                     ("border-bottom", "2px solid #1e293b !important"),
                                     ("background", "#e2e8f0"), ("font-weight", "700")]},
                      ]))
            # Must render as HTML — st.dataframe strips all pandas Styler CSS
            st.markdown(
                f'<div style="overflow-x:auto">'
                f'{styled.to_html(escape=False)}</div>',
                unsafe_allow_html=True,
            )
            _threshold_note = (
                f"threshold Rollex {wpx_lbl} ({tbl_window_date})"
                if tbl_window_date != "custom"
                else f"threshold Rollex {wpx_lbl} (custom)"
            )
            st.markdown(
                f"<p style='font-size:.68rem;color:#999;margin-top:4px'>"
                f"Values in k lots · {len(tbl_df)} COT observations in selected period · "
                f"step = {rx_step} · {_threshold_note}</p>",
                unsafe_allow_html=True,
            )


# Fragment wrappers — each tab only reruns itself when its own widgets change,
# preventing the full-page rerun that scrolls back to the top.
@st.fragment
def _tab_recap(d, report, color, commodity, is_options=False):
    render_recap(d, report, color, commodity, is_options)

@st.fragment
def _tab_recap_charts(d, report, color, commodity):
    render_recap_charts(d, report, color, commodity)

@st.fragment
def _tab_spec(d, report, color):
    render_spec(d, report, color)

@st.fragment
def _tab_commercial(d, report, color):
    render_commercial(d, report, color)

@st.fragment
def _tab_spreading(d, color, df_all_crops=None, commodity=""):
    render_spreading(d, color, df_all_crops, commodity)

@st.fragment
def _tab_old_new(d_crops, color, commodity=""):
    render_old_new(d_crops, color, commodity)

@st.fragment
def _tab_concentration(d, color):
    render_concentration(d, color)

@st.fragment
def _tab_analysis(d, report, color, commodity):
    render_analysis(d, report, color, commodity)

@st.fragment
def _tab_correlation(d, report, color):
    render_correlation(d, report, color)

@st.fragment
def _tab_comparison(commodity, start_date, end_date, color):
    render_comparison(commodity, start_date, end_date, color)

@st.fragment
def _tab_spec_var(commodity, df, report, color, start_date, end_date):
    render_spec_var(commodity, df, report, color, start_date, end_date)

@st.fragment
def _tab_pairs(start_date, end_date, commodity):
    render_pairs(start_date, end_date, commodity)

@st.fragment
def _tab_pain_trade(d, commodity, report, color, is_options):
    render_pain_trade(d, commodity, report, color, is_options)


def _na(msg):
    st.markdown(
        f"<div style='margin-top:24px;font-size:.83rem;color:#6b7280;"
        f"padding:12px 16px;background:#f9fafb;border:1px solid #e5e7eb;"
        f"border-radius:8px'>{msg}</div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION — DISTRIBUTION (ported from spec_distribution_app.py)
# Uses the full history of the sidebar commodity/report/version — the sidebar
# date range is deliberately ignored; each view has its own lookback control.
# ══════════════════════════════════════════════════════════════════════════════
DIST_LOOKBACKS = [1, 3, 5, 10]
DIST_CIT_CATS = {k: v for k, v in CIT_SPEC.items()} | {
    "Commercial": {"long": "Comm Long", "short": "Comm Short", "net": "Comm Net"},
}
DIST_DISAGG_CATS = {k: v for k, v in DISAGG_SPEC.items()} | {
    "MM + Non-Rep":          {"long": "MM+NonRep Long", "short": "MM+NonRep Short", "net": "MM+NonRep Net"},
    "Commercial (Producer)": {"long": "Producer Long",  "short": "Producer Short",  "net": "Comm Net"},
}
DIST_MATRIX_COMMS = [c for c in COMM_NAMES if c not in COMBINED_COMMS]

def _dist_prepare(d):
    """Add the MM + Non-Rep columns (not built by load_disagg) incl. their % of OI."""
    d = d.copy()
    if {"MM Long", "Non Rep Long"} <= set(d.columns):
        for side in ("Long", "Short"):
            d[f"MM+NonRep {side}"] = d[f"MM {side}"] + d[f"Non Rep {side}"]
        d["MM+NonRep Net"] = d["MM+NonRep Long"] - d["MM+NonRep Short"]
        if "Total OI" in d.columns:
            for c in ("MM+NonRep Long", "MM+NonRep Short", "MM+NonRep Net"):
                d[f"Pct OI {c}"] = (d[c] / d["Total OI"] * 100).round(2)
    return d

def _dist_zscore(series, years):
    if series.empty:
        return np.nan
    window = series[series.index >= series.index.max() - pd.DateOffset(years=years)]
    sd = window.std(ddof=0)
    if len(window) < 5 or pd.isna(sd) or sd == 0:
        return np.nan
    return float((series.iloc[-1] - window.mean()) / sd)

_ZBAR_CAP = 3.0   # bars saturate at ±3σ

_ZBAR_CSS = """<style>
.zbt{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  font-size:.8rem;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;background:#fff}
.zbt th{background:#f8fafc;color:#64748b;font-weight:600;font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;
  padding:9px 8px;border-bottom:1px solid #e5e7eb;text-align:center}
.zbt th:first-child{text-align:left;width:26%;padding-left:12px}
.zbt th:last-child,.zbt td:last-child{width:20%}
.zbt .zv{max-width:calc(100% - 12px)}
.zbt td{padding:7px 8px;border-bottom:1px solid #f1f5f9}
.zbt tr:last-child td{border-bottom:none}
.zbt tr:hover td{background:#fafbfc}
.zbt .zn{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#1f2937;font-weight:500;padding-left:12px}
.zbt .zd{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:middle}
.zbt .zt{position:relative;height:20px;background:#f5f6f8;border-radius:5px}
.zbt .zt::after{content:'';position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:#cbd5e1}
.zbt .zb{position:absolute;top:0;bottom:0}
.zbt .zv{position:absolute;top:50%;transform:translateY(-50%);font-size:.74rem;line-height:1;font-variant-numeric:tabular-nums;color:#374151;white-space:nowrap}
.zbt .zx{font-weight:700}
.zleg{font-size:.7rem;color:#94a3b8;margin-top:6px}
</style>"""

def _dist_zbar_cell(v, cap=_ZBAR_CAP, fmt="{:+.2f}", strong_at=2.0, bordered=False):
    """Diverging bar from the centre line; the value sits in the opposite half.
    `cap` sets what saturates the bar; `strong_at` sets the bold/dark threshold
    (both in the same units as `v`); `bordered` adds a left divider (for a
    column that isn't part of the z-score set, e.g. a price-change column)."""
    td = "<td style='border-left:1px solid #e5e7eb'>" if bordered else "<td>"
    if pd.isna(v):
        return f"{td}<div class='zt'><span class='zv' style='left:50%;transform:translate(-50%,-50%);color:#9ca3af'>—</span></div></td>"
    frac = min(abs(v), cap) / cap * 50          # % of track width, from centre
    pos, strong = v >= 0, abs(v) >= strong_at
    clr = ("#16a34a" if strong else "#86efac") if pos else ("#dc2626" if strong else "#fca5a5")
    bar = (f"left:50%;width:{frac:.1f}%;border-radius:0 5px 5px 0" if pos else
           f"right:50%;width:{frac:.1f}%;border-radius:5px 0 0 5px")
    lbl = "right:calc(50% + 6px)" if pos else "left:calc(50% + 6px)"
    txt = ("color:#15803d;font-weight:700" if pos else "color:#b91c1c;font-weight:700") if strong else ""
    return (f"{td}<div class='zt'><div class='zb' style='{bar};background:{clr}'></div>"
            f"<span class='zv' style='{lbl};{txt}'>{fmt.format(v)}</span></div></td>")

def _dist_zbar_table(rows, selected=None, price_col=None, price_label="Px Δ%"):
    """rows: {commodity_code: {years: z}} → HTML diverging-bar table; the sidebar
    commodity's row is highlighted. `price_col`, if given, is a
    {commodity_code: pct_change} dict appended as one extra bar column on the
    far right, scaled to its own (shared) range rather than the ±3σ z-score cap."""
    head = "<tr><th>Commodity</th>" + "".join(f"<th>{y}y</th>" for y in DIST_LOOKBACKS)
    if price_col is not None:
        head += (f"<th style='border-left:1px solid #e5e7eb' "
                 f"title='Latest weekly % price change, Tuesday COT date to Tuesday COT date'>{price_label}</th>")
    head += "</tr>"
    price_cap = max(3.0, np.nanmax(np.abs(list(price_col.values())))) if price_col and any(
        not pd.isna(v) for v in price_col.values()) else 3.0
    body = ""
    for code, zs in rows.items():
        name = COMM_NAMES[code].split(" : ")[1]
        sel = code == selected
        row_style = f" style='background:{COMM_COLORS[code]}14;box-shadow:inset 3px 0 0 {COMM_COLORS[code]}'" if sel else ""
        name_html = f"<b>{name}</b>" if sel else name
        row = (f"<tr{row_style}><td class='zn'><span class='zd' style='background:{COMM_COLORS[code]}'></span>{name_html}</td>"
               + "".join(_dist_zbar_cell(zs.get(y, np.nan)) for y in DIST_LOOKBACKS))
        if price_col is not None:
            row += _dist_zbar_cell(price_col.get(code, np.nan), cap=price_cap, fmt="{:+.1f}%",
                                   strong_at=price_cap * 2 / 3, bordered=True)
        body += row + "</tr>"
    return f"<table class='zbt'><thead>{head}</thead><tbody>{body}</tbody></table>"

def _dist_auto_bin(series_list, target_bins=60):
    """Round-ish bin width from the combined range of the given series."""
    vals = pd.concat([s for s in series_list if not s.empty]) if series_list else pd.Series(dtype=float)
    if vals.empty:
        return 1.0
    span = vals.max() - vals.min()
    raw_w = span / target_bins if span > 0 else 1.0
    mag = 10 ** np.floor(np.log10(raw_w))
    for m in (1, 2, 2.5, 5, 10):
        if raw_w <= m * mag:
            return round(m * mag, 6)
    return round(10 * mag, 6)

_DIST_FONT = dict(family="-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif", size=11)

@st.fragment
def render_distribution(full, commodity, report):
    if report == "Disagg" and version_key == "Opt":
        _na("Distribution is available for CIT, Disagg F&O and Disagg Futures — not Options only.")
        return
    cat_map = DIST_CIT_CATS if report == "CIT" else DIST_DISAGG_CATS
    c1, c2, c3, c4 = st.columns([1.8, 1.2, 2.6, 1.5])
    with c1:
        category = st.selectbox("Category", list(cat_map), key=f"dist_cat_{report}")
    with c2:
        unit = st.radio("Units", ["k lots", "% of OI"], horizontal=True, key="dist_unit")
    with c3:
        lb_choice = st.radio("History window", ["All", "1y", "3y", "5y", "10y"],
                             horizontal=True, key="dist_lookback")
    with c4:
        y_mode = st.radio("Y-axis", ["% of weeks", "Count"], horizontal=True, key="dist_ymode")

    cols = cat_map[category]
    d = _dist_prepare(full)
    if d.empty or any(c not in d.columns for c in (cols["net"], cols["long"], cols["short"])):
        st.warning(f"No data available for {commodity} / {report} / {category}.")
        return
    if lb_choice != "All":
        d = d[d["Date"] >= d["Date"].max() - pd.DateOffset(years=int(lb_choice[:-1]))]
    st.caption(f"Study period: **{d['Date'].min():%d %b %Y} → {d['Date'].max():%d %b %Y}** "
               f"({len(d)} weekly observations) · full history, independent of the sidebar date range")

    use_pct    = unit == "% of OI"
    unit_div   = 1.0 if use_pct else 1000.0
    hist_norm  = "percent" if y_mode == "% of weeks" else None
    y_title    = "% of weeks" if hist_norm else "Weeks (count)"

    panels = {}   # (row, col) -> (series, colour, title)
    for i, (name, base, clr) in enumerate([("Net", cols["net"], C_NET),
                                           ("Long", cols["long"], C_LONG),
                                           ("Short", cols["short"], C_SHORT)], start=1):
        col = f"Pct OI {base}" if use_pct else base
        if col not in d.columns:
            continue
        lvl = pd.to_numeric(d[col], errors="coerce").dropna() / unit_div
        panels[(1, i)] = (lvl, clr, f"{name} — Level ({unit})")
        panels[(2, i)] = (lvl.diff().dropna(), clr, f"{name} — Weekly Δ ({unit})")

    bins = {1: 0.5, 2: 0.5} if use_pct else {
        r: _dist_auto_bin([s for (rr, _), (s, _, _) in panels.items() if rr == r]) for r in (1, 2)}

    def _title(key):
        if key not in panels:
            return ""
        s, _, base_title = panels[key]
        if s.empty:
            return base_title
        return f"{base_title}   ·   latest ({d.loc[s.index[-1], 'Date']:%d %b %Y}) {s.iloc[-1]:,.1f}"

    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.22, horizontal_spacing=0.07,
                        subplot_titles=[_title((r, c)) for r in (1, 2) for c in (1, 2, 3)])
    for ann in fig.layout.annotations:
        ann.font.size = 12
    for (r, c), (s, clr, _) in panels.items():
        if s.empty:
            continue
        fig.add_trace(go.Histogram(x=s.values, xbins=dict(size=bins[r]), marker_color=clr,
                                   marker_line=dict(color="white", width=1), histnorm=hist_norm,
                                   opacity=0.85, showlegend=False), row=r, col=c)
        fig.add_vline(x=s.iloc[-1], line_dash="dash", line_color="#1a1a2e", line_width=2, row=r, col=c)
    fig.update_layout(height=700, template="plotly_white", bargap=0.04,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
                      margin=dict(l=30, r=30, t=70, b=40), font=_DIST_FONT)
    fig.update_yaxes(title_text=y_title, title_font_size=10, showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_xaxes(showgrid=False)
    _chart(fig, width='stretch')
    st.caption(f"Distribution of {category} Net/Long/Short — level and week-over-week change — over the "
               f"selected window. Dashed line marks the latest data point ({d['Date'].max():%d %b %Y}).")

    # Rollex weekly % change on the same COT dates
    st.divider()
    st.markdown("**Rollex Price Weekly % Change**")
    rx = load_rollex(commodity)
    if rx.empty:
        st.info("Rollex price data not available for this commodity.")
        return
    px_lvl = d[["Date"]].merge(rx[["Date", "rollex_px"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    chg = (px_lvl["rollex_px"].dropna().pct_change() * 100).dropna()
    if chg.empty:
        st.info("No overlapping weeks between COT dates and Rollex price data.")
        return
    fig_px = go.Figure(go.Histogram(x=chg.values, xbins=dict(size=_dist_auto_bin([chg])),
                                    marker_color="#f59e0b", marker_line=dict(color="white", width=1),
                                    histnorm=hist_norm, opacity=0.85, showlegend=False))
    fig_px.add_vline(x=chg.iloc[-1], line_color="#1a1a2e", line_width=2)
    fig_px.update_layout(
        title=dict(text=f"Weekly Price Change %   ·   latest ({px_lvl.loc[chg.index[-1], 'Date']:%d %b %Y}) "
                        f"{chg.iloc[-1]:+.2f}%", font=dict(size=13)),
        height=340, template="plotly_white", bargap=0.04, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafafa", margin=dict(l=30, r=30, t=50, b=40), font=_DIST_FONT)
    fig_px.update_yaxes(title_text=y_title, title_font_size=10, showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig_px.update_xaxes(title_text="Weekly Price Change %", showgrid=False)
    px_col, _ = st.columns(2)
    with px_col:
        _chart(fig_px, width='stretch')
    st.caption("Distribution of the Rollex (roll-adjusted) price's week-over-week % change over the same "
               "study window as the positioning histograms. Solid line marks the latest value.")

# Default category is different per basis — CIT's US-style combined leg vs
# Disagg's own combined leg — so each basis remembers its own selectbox state.
_MATRIX_DEFAULT_CAT = {"CIT": "Large Spec + Index + Non-Rep", "Disagg": "MM + Other + Non-Rep"}

@st.fragment
def render_zscore_matrix(commodity=None, report=None):
    st.caption(
        "Every single commodity side by side on **one common report basis** (chosen below), "
        "so commodities that don't share a CIT report (RC/LCC/LSU, metals) can still be "
        "compared — this is why it uses its own Report/Category controls instead of the "
        "sidebar's, which are set per-commodity and would leave those blank."
    )
    b1, b2 = st.columns([1, 2])
    with b1:
        basis_default = "CIT" if report == "CIT" else "Disagg"
        basis = st.radio("Report basis (matrix)", ["CIT", "Disagg"],
                         index=["CIT", "Disagg"].index(basis_default),
                         horizontal=True, key="dist_matrix_basis")
    if basis == "CIT":
        src, cat_map, comms = load_cit(), DIST_CIT_CATS, [c for c in DIST_MATRIX_COMMS if c in CIT_COMMS]
    else:
        src, cat_map, comms = _dist_prepare(load_disagg("Fut")), DIST_DISAGG_CATS, DIST_MATRIX_COMMS
    if "Crop" in src.columns:
        src = src[src["Crop"] == "All"]

    with b2:
        cats = list(cat_map)
        default_cat = _MATRIX_DEFAULT_CAT[basis]
        category = st.selectbox("Category (matrix)", cats, key=f"dist_matrix_cat_{basis}",
                                index=cats.index(default_cat) if default_cat in cats else 0)
    net_col = cat_map[category]["net"]
    if basis == "CIT":
        st.caption("KC / CC / SB / CT only — RC, LCC, LSU and the metals have no CIT (Index Traders) report.")
    st.caption(f"Latest COT date: **{src['Date'].max():%d %b %Y}**.")

    level_rows, chg_rows, price_rows = {}, {}, {}
    for cmm in comms:
        dc = src[src["Commodity"] == cmm].sort_values("Date")
        if dc.empty or net_col not in dc.columns:
            continue
        lvl = pd.to_numeric(dc.set_index("Date")[net_col], errors="coerce").dropna()
        level_rows[cmm] = {y: _dist_zscore(lvl, y) for y in DIST_LOOKBACKS}
        chg_rows[cmm]   = {y: _dist_zscore(lvl.diff().dropna(), y) for y in DIST_LOOKBACKS}
        # Tuesday-to-Tuesday price move: Rollex (roll-adjusted) price aligned to
        # this commodity's own COT report dates, latest week's % change.
        # No Rollex file (e.g. metals) -> _inject_rollex is a no-op, no "Px" added.
        px_df = _inject_rollex(dc[["Date"]], cmm)
        if "Px" in px_df.columns:
            px = pd.to_numeric(px_df.set_index("Date")["Px"], errors="coerce").dropna()
            chg_pct = px.pct_change().dropna()
            if not chg_pct.empty:
                price_rows[cmm] = float(chg_pct.iloc[-1]) * 100

    st.markdown(_ZBAR_CSS, unsafe_allow_html=True)
    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.markdown(f"**{category} Net — Z-score**")
        st.markdown(_dist_zbar_table(level_rows, commodity, price_col=price_rows), unsafe_allow_html=True)
    with m2:
        st.markdown(f"**{category} Weekly Change — Z-score**")
        st.markdown(_dist_zbar_table(chg_rows, commodity, price_col=price_rows), unsafe_allow_html=True)
    st.markdown("<div class='zleg'>Z-score bars run from the centre line (z = 0): green = above the window "
                "mean, red = below — scaled to ±3σ, darker/bold at |z| ≥ 2. The right-most Px Δ% column is "
                "each commodity's own most recent weekly price move (Tuesday COT date to Tuesday COT date), "
                "scaled to the largest move on screen.</div>", unsafe_allow_html=True)

def _view_distribution():
    full = raw[(raw["Commodity"] == commodity) & (raw["Crop"] == "All")] if "Crop" in raw.columns \
        else raw[raw["Commodity"] == commodity]
    render_distribution(full.sort_values("Date").reset_index(drop=True), commodity, report)


def _view_concentration():
    if report == "CIT":
        _na("Concentration data is only available in the Disaggregated report.")
    else:
        _tab_concentration(df, color)

def _view_spreading():
    if report == "Disagg" and not is_options:
        _tab_spreading(df, color, df_all_crops, commodity)
    else:
        _na("Spreading positions are only available in the Disaggregated report (Fut or F&O).")

def _view_old_new():
    if report == "CIT":
        _na("Old / New crop split is only available in the Disaggregated report.")
    elif df_all_crops is not None:
        _tab_old_new(df_all_crops, color, commodity)
    else:
        _na("Old / New crop split is not available for this commodity.")

def _view_comparison():
    if commodity in CIT_COMMS and not is_options:
        _tab_comparison(commodity, start_date, end_date, color)
    else:
        _na("CIT vs Disagg comparison is only available for KC, CC, SB, and CT with a non-Options report.")

VIEWS = {
    "Recap":              lambda: _tab_recap(df, report, color, commodity, is_options),
    "Recap (Charts)":     lambda: _tab_recap_charts(df, report, color, commodity),
    "Spec":               lambda: _tab_spec(df, report, color),
    "Commercial":         lambda: _tab_commercial(df, report, color),
    "Concentration":      _view_concentration,
    "Spreading":          _view_spreading,
    "Old / New":          _view_old_new,
    "Correlation":        lambda: _tab_correlation(df, report, color),
    "Spec Prediction":    lambda: _tab_analysis(df, report, color, commodity),
    "Specs in VaR":       lambda: _tab_spec_var(commodity, df, report, color, start_date, end_date),
    "CIT vs Disagg":      _view_comparison,
    "Pain Trade Monitor": lambda: _tab_pain_trade(df, commodity, report, color, is_options),
    "Spec Proximity":     lambda: render_spec_proximity(start_date, end_date, commodity),
    "Distribution":       _view_distribution,
    "Z-Score Matrix":     lambda: render_zscore_matrix(commodity, report),
}

# "buttons": segmented selector — only the chosen view is built on each rerun.
# "tabs":    original st.tabs — all 13 views are built on every rerun.
NAV_STYLE = "buttons"

VIEW_GROUPS = {
    "Positioning": ["Recap", "Recap (Charts)", "Spec", "Commercial", "Concentration",
                    "Spreading", "Old / New", "CIT vs Disagg"],
    "Analytics":   ["Correlation", "Spec Prediction", "Specs in VaR",
                    "Pain Trade Monitor", "Spec Proximity"],
    "Distribution": ["Distribution", "Z-Score Matrix"],
}

def _nav_css(accent):
    """Section row = pill switch, view row = underline tabs; both in the
    commodity accent. Scoped via st.container(key=...) → .st-key-<key>."""
    sec = ".st-key-nav_section"
    vw  = ".st-key-nav_view"
    return f"""<style>
  {sec} [data-testid="stButtonGroup"] {{ gap:0; }}
  {sec} [data-testid="stButtonGroup"] > div {{
    display:inline-flex; gap:4px; padding:4px; background:#f1f3f7;
    border:1px solid #e3e7ee; border-radius:999px;
  }}
  {sec} button[kind^="segmented_control"] {{
    border:none !important; border-radius:999px !important; margin:0 !important;
    padding:.35rem 1.25rem !important; min-height:0 !important;
    background:transparent !important; box-shadow:none !important;
    transition:background .15s ease, color .15s ease;
  }}
  {sec} button[kind^="segmented_control"] p {{
    font-size:.84rem !important; font-weight:600 !important; letter-spacing:.02em;
    color:#5b6472 !important;
  }}
  {sec} button[kind="segmented_control"]:hover {{ background:#e6e9f0 !important; }}
  {sec} button[kind="segmented_controlActive"] {{
    background:{accent} !important; box-shadow:0 1px 3px rgba(0,0,0,.18) !important;
  }}
  {sec} button[kind="segmented_controlActive"] p {{ color:#ffffff !important; }}

  {vw} {{ margin-top:-.35rem; border-bottom:1px solid #e3e7ee; gap:0; }}
  {vw} [data-testid="stButtonGroup"] > div {{ gap:2px; flex-wrap:wrap; }}
  {vw} button[kind^="segmented_control"] {{
    border:none !important; border-radius:6px 6px 0 0 !important; margin:0 0 -1px 0 !important;
    padding:.45rem .9rem !important; min-height:0 !important;
    background:transparent !important; box-shadow:none !important;
    border-bottom:2px solid transparent !important;
    transition:color .15s ease, border-color .15s ease, background .15s ease;
  }}
  {vw} button[kind^="segmented_control"] p {{
    font-size:.81rem !important; font-weight:500 !important; color:#6b7280 !important;
  }}
  {vw} button[kind="segmented_control"]:hover {{ background:#f5f6f9 !important; }}
  {vw} button[kind="segmented_control"]:hover p {{ color:#1f2937 !important; }}
  {vw} button[kind="segmented_controlActive"] {{
    border-bottom:2px solid {accent} !important;
  }}
  {vw} button[kind="segmented_controlActive"] p {{ color:{accent} !important; font-weight:600 !important; }}
</style>"""

if NAV_STYLE == "buttons":
    st.markdown(_nav_css(color), unsafe_allow_html=True)
    # Keyed widgets, so the selected group/view survive sidebar filter changes.
    with st.container(key="nav_section"):
        group = st.segmented_control("Section", list(VIEW_GROUPS), default="Positioning",
                                     key="main_group", label_visibility="collapsed") or "Positioning"
    group_views = VIEW_GROUPS[group]
    # Streamlit drops a widget's state while it isn't rendered, so the other
    # group's last view is remembered in a plain session key and fed back as default.
    last_key = f"_last_view_{group}"
    last = st.session_state.get(last_key, group_views[0])
    with st.container(key="nav_view"):
        view = st.segmented_control("View", group_views, default=last,
                                    key=f"main_view_{group}", label_visibility="collapsed") or last
    st.session_state[last_key] = view
    VIEWS[view]()
else:
    # Fixed tab count — Streamlit preserves the active tab when sidebar filters change.
    for _tab, _render in zip(st.tabs(list(VIEWS)), VIEWS.values()):
        with _tab:
            _render()

# Pairs tab hidden — re-enable by adding "Pairs" to st.tabs() and wiring:
# with tabs[11]:
#     if show_pairs: _tab_pairs(start_date, end_date, commodity)
#     else: _na("Pairs view is available for KC, RC, CC, LCC, SB, and LSU.")