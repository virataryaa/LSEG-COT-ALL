"""
spec_prediction_overview.py — Cross-Commodity Spec Change Prediction

Logic
-----
COT is released Tue-close data, published the following Friday, picked up
here the Monday after. On any day between the last COT release and the next
one, we already know the Tuesday-close price (via Rollex) even though the
new COT numbers aren't out yet. So:

  1. Fit  Δ(Spec Net)_week  =  beta * ΔPx%_week + alpha   on history
     (same regression as cot_app.py's "Spec Prediction" tab, per commodity).
  2. Take the price move from the LAST published COT's Tuesday close to the
     latest available Tuesday close (this week's, not yet in the COT data).
  3. Predicted Δ = beta * that price move % + alpha  →  Predicted Net = last
     published Spec Net + Predicted Δ.
  4. That prediction is logged (keyed by commodity + the COT date it targets).
     When the COT data eventually rolls forward to that target date, the row
     resolves: Actual = realized Δ, shown next to the Prediction that was
     made for it ahead of time. Until then it just says "Awaiting".

Spec definitions (mirrors cot_app.py's derived columns):
  NYC   (KC, CC, SB, CT)   = Spec + Non Rep + Index      → "Combined Spec Net"
  Europe (RC, LCC, LSU)    = Managed Money + Other + Non Rep → "MM+Other+NonRep Net"

Run: streamlit run spec_prediction_overview.py
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="SPEC CHANGE PREDICTION", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  :root { color-scheme: light !important; }
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background:#ffffff !important; color:#1a1a1a !important;
  }
  [data-testid="stHeader"] { background:transparent !important; }
  .block-container { padding-top:1rem !important; padding-bottom:1rem !important; max-width:1600px; }
  div[data-testid="stExpander"] summary { padding:2px 8px !important; min-height:0 !important; }
  div[data-testid="stExpander"] summary p { font-size:.74rem !important; }
  div[data-testid="stVerticalBlock"] { gap:.35rem !important; }

  /* Minimal underline tabs */
  div[data-testid="stTabs"] { margin-top:2px; }
  div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap:22px; border-bottom:1px solid #edeff3;
  }
  div[data-testid="stTabs"] button[data-baseweb="tab"] {
    padding:6px 2px !important; height:auto !important; background:transparent !important;
  }
  div[data-testid="stTabs"] button[data-baseweb="tab"] p {
    font-size:.78rem !important; font-weight:600 !important; letter-spacing:.02em;
    color:#9ca3af !important;
  }
  div[data-testid="stTabs"] button[aria-selected="true"] p { color:#111827 !important; }
  div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#111827 !important; height:2px !important; }
  div[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }

  div[data-testid="stSelectbox"] label p { font-size:.68rem !important; color:#9ca3af !important;
    font-weight:700 !important; letter-spacing:.04em !important; }
</style>""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────
DB_DIR      = Path(__file__).resolve().parent.parent / "Database"
CIT_FILE    = DB_DIR / "cot_cit.parquet"
FO_FILE     = DB_DIR / "cot_disagg_futopt.parquet"
ROLLEX_DIR  = DB_DIR / "Rollex"
ROLLYIELD_FILE = DB_DIR / "RollYield" / "roll_yield_data.parquet"
LOG_FILE    = DB_DIR / "spec_prediction_log.parquet"
# BRL: synced daily into this repo's own Database/ by Automator/run_daily.bat
# (Step 2b) so it travels with the repo to Streamlit Cloud; fall back to the
# source Roll Yield project's copy for local dev before the first sync runs.
FX_BRL_FILE = DB_DIR / "fx_brl.parquet"
_FX_BRL_FALLBACK = (Path(__file__).resolve().parent.parent.parent / "Roll Yield"
                     / "Database" / "fx_brl.parquet")
# Daily open interest: synced daily into this repo's own Database/Futures/ by
# Automator/run_daily.bat (Step 2c), same pattern as fx_brl above. Published
# with a ~1-trading-day lag by ICE, so it's ahead of the last COT Tuesday but
# usually a day or two behind "today" — that's real, not a bug.
FUTURES_DIR = DB_DIR / "Futures"
_FUTURES_FALLBACK_DIR = (Path(__file__).resolve().parent.parent.parent / "Futures" / "Database")
FUTURES_MAP = {"KC":"kc_futures.parquet","CC":"cc_futures.parquet","CT":"ct_futures.parquet",
               "SB":"sb_futures.parquet","RC":"rc_futures.parquet","LCC":"lcc_futures.parquet",
               "LSU":"lsu_futures.parquet"}

ROLLEX_MAP = {"KC":"rollex_KC.parquet","CC":"rollex_CC.parquet","CT":"rollex_CT.parquet",
              "SB":"rollex_SB.parquet","RC":"rollex_RC.parquet","LCC":"rollex_LCC.parquet",
              "LSU":"rollex_LSU.parquet"}
ROLLYIELD_MAP = {"KC":"KC","CC":"CC","SB":"SB","CT":"CT","RC":"RC","LCC":"LCC"}  # no LSU series

CIT_COMMS = {"KC","CC","SB","CT"}
COMMODITIES = ["KC","RC","CC","LCC","SB","LSU","CT"]
COMM_NAMES  = {"KC":"Arabica","RC":"Robusta","CC":"NYC Cocoa","LCC":"London Cocoa",
               "SB":"Sugar #11","LSU":"White Sugar","CT":"Cotton"}
SPEC_INCLUSION = {
    "KC":"Spec + Non Rep + Index",  "CC":"Spec + Non Rep + Index",
    "SB":"Spec + Non Rep + Index",  "CT":"Spec + Non Rep + Index",
    "RC":"Managed Money + Other + Non Rep", "LCC":"Managed Money + Other + Non Rep",
    "LSU":"Managed Money + Other + Non Rep",
}
COMM_COLORS = {"KC":"#1a56db","CC":"#d97706","SB":"#059669","CT":"#7c3aed",
               "RC":"#dc2626","LCC":"#0891b2","LSU":"#ea580c"}


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
def _derive_nets(df):
    pairs = [("Spec Long","Spec Short","Spec Net"), ("Index Long","Index Short","Index Net"),
             ("Non Rep Long","Non Rep Short","Non Rep Net"), ("MM Long","MM Short","MM Net"),
             ("Other Long","Other Short","Other Net"), ("Comm Long","Comm Short","Comm Net"),
             ("Producer Long","Producer Short","Producer Net")]
    for l, s, n in pairs:
        if l in df.columns and s in df.columns and n not in df.columns:
            df[n] = df[l] - df[s]
    return df

@st.cache_data(ttl=600)
def load_cit():
    df = pd.read_parquet(CIT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date","Commodity","Crop")]
    df[num] = df[num].astype(float)
    df = _derive_nets(df)
    for side in ("Long","Short"):
        df[f"Combined Spec {side}"] = df.get(f"Spec {side}",0) + df.get(f"Non Rep {side}",0) + df.get(f"Index {side}",0)
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    return df.sort_values(["Commodity","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_disagg():
    df = pd.read_parquet(FO_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date","Commodity","Crop")]
    df[num] = df[num].astype(float)
    df = _derive_nets(df)
    for side in ("Long","Short"):
        df[f"MM+Other+NonRep {side}"] = df.get(f"MM {side}",0) + df.get(f"Other {side}",0) + df.get(f"Non Rep {side}",0)
    df["MM+Other+NonRep Net"] = df["MM+Other+NonRep Long"] - df["MM+Other+NonRep Short"]
    return df.sort_values(["Commodity","Crop","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_rollex(commodity):
    path = ROLLEX_DIR / ROLLEX_MAP[commodity]
    if not path.exists():
        return pd.DataFrame(columns=["Date","rollex_px","active_label"])
    try:
        df = pd.read_parquet(path, columns=["rollex_px","active_label"])
    except Exception:
        df = pd.read_parquet(path, columns=["rollex_px"])
        df["active_label"] = np.nan
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.reset_index().sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=600)
def load_roll_yield():
    if not ROLLYIELD_FILE.exists():
        return pd.DataFrame(columns=["Date","Commodity","roll_yield_pct"])
    df = pd.read_parquet(ROLLYIELD_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["roll_yield_pct"] = df["Roll_Yield_1yr"] * 100
    return df[["Date","Commodity","roll_yield_pct"]].sort_values(["Commodity","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_brl():
    path = FX_BRL_FILE if FX_BRL_FILE.exists() else _FX_BRL_FALLBACK
    if not path.exists():
        return pd.DataFrame(columns=["Date","USDBRL"])
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=600)
def load_futures_oi(commodity):
    """Daily Total OI = sum of open_interest across all live contract months.
    ICE reports OI a day behind (yesterday's settle), so the latest date here
    trails the latest Rollex price date by ~1-2 sessions — drop unreported
    (NaN/0) days rather than showing a false zero."""
    fname = FUTURES_MAP.get(commodity)
    if fname is None:
        return pd.DataFrame(columns=["Date","Total_OI"])
    path = FUTURES_DIR / fname
    if not path.exists():
        path = _FUTURES_FALLBACK_DIR / fname
    if not path.exists():
        return pd.DataFrame(columns=["Date","Total_OI"])
    df = pd.read_parquet(path, columns=["Date","open_interest"])
    df["Date"] = pd.to_datetime(df["Date"])
    agg = df.groupby("Date")["open_interest"].sum(min_count=1).reset_index()
    agg = agg.rename(columns={"open_interest":"Total_OI"})
    agg = agg[agg["Total_OI"] > 0].sort_values("Date").reset_index(drop=True)
    return agg


def load_log():
    cols = ["Commodity","Target_Date","Base_Date","Base_Px","Base_Spec","Beta","Alpha",
            "R2","N","Predicted_Delta","Predicted_Net","Predicted_At"]
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_parquet(LOG_FILE)
        df["Target_Date"] = pd.to_datetime(df["Target_Date"])
        df["Base_Date"]   = pd.to_datetime(df["Base_Date"])
        return df
    except Exception:
        return pd.DataFrame(columns=cols)

def upsert_log(log_df, row):
    mask = ~((log_df["Commodity"] == row["Commodity"]) & (log_df["Target_Date"] == row["Target_Date"]))
    log_df = pd.concat([log_df[mask], pd.DataFrame([row])], ignore_index=True)
    try:
        log_df.to_parquet(LOG_FILE, index=False)
    except Exception:
        pass  # read-only environment — prediction just won't persist this run
    return log_df


# ══════════════════════════════════════════════════════════════════════════
# PER-COMMODITY COMPUTATION
# ══════════════════════════════════════════════════════════════════════════
def _fit_regression(spec_series, px_series):
    """Same fit as cot_app.py's Spec Prediction tab: Δspec vs weekly ΔPx%,
    with a 5-IQR clip to strip CFTC data-entry error weeks."""
    ds = spec_series.diff()
    px_chg = px_series.pct_change() * 100
    common = ~(ds.isna() | px_chg.isna())
    x, y = px_chg[common].values.astype(float), ds[common].values.astype(float)
    if len(y) >= 10:
        iqr = np.percentile(y, 75) - np.percentile(y, 25)
        med = np.median(y)
        clip = max(iqr * 5, 1)
        ok = np.abs(y - med) <= clip
        x, y = x[ok], y[ok]
    if len(x) < 10:
        return None
    beta, alpha = np.polyfit(x, y, 1)
    y_hat = beta * x + alpha
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(beta=float(beta), alpha=float(alpha), r2=float(r2), n=int(len(x)))


@st.cache_data(ttl=600)
def compute_commodity(commodity):
    is_cit = commodity in CIT_COMMS
    if is_cit:
        raw = load_cit()
        sub = raw[raw["Commodity"] == commodity].sort_values("Date").reset_index(drop=True)
        spec_col = "Combined Spec Net"
    else:
        raw = load_disagg()
        sub = raw[(raw["Commodity"] == commodity) & (raw["Crop"] == "All")].sort_values("Date").reset_index(drop=True)
        spec_col = "MM+Other+NonRep Net"
    if sub.empty:
        return None

    rx = load_rollex(commodity)
    if rx.empty:
        return None

    # Weekly Tuesday-close price at each COT date, via backward asof merge
    merged = pd.merge_asof(sub[["Date", spec_col, "Total OI"]].sort_values("Date"),
                            rx[["Date","rollex_px"]], on="Date", direction="backward")
    merged = merged.rename(columns={spec_col: "Spec Net"})

    fit = _fit_regression(merged["Spec Net"], merged["rollex_px"])
    if fit is None:
        return None

    last_cot_date = merged["Date"].iloc[-1]
    last_cot_spec = float(merged["Spec Net"].iloc[-1])
    last_cot_px   = float(merged["rollex_px"].iloc[-1])
    prev_cot_spec = float(merged["Spec Net"].iloc[-2]) if len(merged) >= 2 else np.nan

    latest_px    = float(rx["rollex_px"].iloc[-1])
    latest_date  = rx["Date"].iloc[-1]
    latest_label = rx["active_label"].iloc[-1] if "active_label" in rx.columns else None

    # Live OI change: last COT's Total OI vs the most recent actual daily
    # print (ICE reports OI ~1 session late), same nowcast logic as price —
    # not the stale week-over-week COT OI change.
    oi_daily = load_futures_oi(commodity)
    if not oi_daily.empty:
        last_oi = pd.merge_asof(pd.DataFrame({"Date":[last_cot_date]}), oi_daily,
                                 on="Date", direction="backward")["Total_OI"].iloc[0]
        latest_oi = float(oi_daily["Total_OI"].iloc[-1])
        latest_oi_date = oi_daily["Date"].iloc[-1]
        last_oi = float(last_oi) if not pd.isna(last_oi) else float(merged["Total OI"].iloc[-1])
    else:
        last_oi = float(merged["Total OI"].iloc[-1])
        latest_oi = last_oi
        latest_oi_date = last_cot_date

    px_move_pct   = (latest_px / last_cot_px - 1) * 100 if last_cot_px else np.nan
    px_move_abs   = latest_px - last_cot_px
    predicted_delta = fit["beta"] * px_move_pct + fit["alpha"]
    predicted_net    = last_cot_spec + predicted_delta

    return dict(
        commodity=commodity, spec_col=spec_col, is_cit=is_cit,
        last_cot_date=last_cot_date, last_cot_spec=last_cot_spec, last_cot_px=last_cot_px,
        prev_cot_spec=prev_cot_spec, last_oi=last_oi, latest_oi=latest_oi, latest_oi_date=latest_oi_date,
        latest_px=latest_px, latest_date=latest_date, latest_label=latest_label,
        px_move_pct=px_move_pct, px_move_abs=px_move_abs,
        beta=fit["beta"], alpha=fit["alpha"], r2=fit["r2"], n=fit["n"],
        predicted_delta=predicted_delta, predicted_net=predicted_net,
        hist=merged, sub=sub,
    )


def resolve_row(res, log_df):
    """Look up whether a prior ex-ante prediction targeted this commodity's
    current COT date; if so the row is 'Resolved' (Actual known). Otherwise
    log today's live prediction for the NEXT COT date and mark 'Awaiting'."""
    c = res["commodity"]
    prior = log_df[(log_df["Commodity"] == c) & (log_df["Target_Date"] == res["last_cot_date"])]
    if not prior.empty:
        p = prior.iloc[-1]
        actual_delta = res["last_cot_spec"] - res["prev_cot_spec"]
        return dict(status="Resolved", cot_date=res["last_cot_date"],
                    spec_net=res["last_cot_spec"],
                    prediction=float(p["Predicted_Delta"]), actual=actual_delta,
                    predicted_at=p["Predicted_At"]), log_df

    target_date = res["last_cot_date"] + pd.Timedelta(days=7)
    row = dict(Commodity=c, Target_Date=target_date, Base_Date=res["last_cot_date"],
               Base_Px=res["last_cot_px"], Base_Spec=res["last_cot_spec"],
               Beta=res["beta"], Alpha=res["alpha"], R2=res["r2"], N=res["n"],
               Predicted_Delta=res["predicted_delta"], Predicted_Net=res["predicted_net"],
               Predicted_At=pd.Timestamp.now())
    log_df = upsert_log(log_df, row)
    return dict(status="Awaiting", cot_date=res["last_cot_date"], spec_net=res["last_cot_spec"],
                prediction=res["predicted_delta"], actual=None,
                predicted_at=row["Predicted_At"]), log_df


# ══════════════════════════════════════════════════════════════════════════
# PLOT + TABLE HELPERS
# ══════════════════════════════════════════════════════════════════════════
_BASE = dict(
    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif", size=11, color="#1a1a1a"))

def _ax(x=False):
    b = dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
             zeroline=True, zerolinecolor="rgba(0,0,0,0.12)", zerolinewidth=1,
             showline=True, linecolor="rgba(0,0,0,0.08)", linewidth=1,
             tickfont=dict(size=10, color="#666"))
    if x:
        b.update(showgrid=False, tickangle=-35, nticks=16, hoverformat="%d %b %Y")
    return b

C_LONG, C_SHORT = "#16a34a", "#dc2626"

_th = ("padding:5px 10px;font-size:.6rem;font-weight:700;color:#9ca3af;letter-spacing:.03em;"
       "border-bottom:1px solid #edeff3;text-align:left;white-space:nowrap")
_td = ("padding:5px 10px;font-size:.75rem;font-weight:600;color:#1e293b;"
       "border-bottom:1px solid #f3f4f6;white-space:nowrap;line-height:1.3")

def _table(headers, rows_html):
    h = "".join(f"<th style='{_th}'>{x}</th>" for x in headers)
    return (f"<table style='border-collapse:collapse;width:100%;font-family:-apple-system,sans-serif'>"
            f"<tr>{h}</tr>{rows_html}</table>")


# ══════════════════════════════════════════════════════════════════════════
# LOAD ALL COMMODITIES ONCE
# ══════════════════════════════════════════════════════════════════════════
log_df = load_log()
rows = []
for c in COMMODITIES:
    res = compute_commodity(c)
    if res is None:
        continue
    resolved, log_df = resolve_row(res, log_df)
    rows.append((c, res, resolved))
res_by_c = {c: res for c, res, r in rows}

st.markdown(
    "<div style='font-size:1.05rem;font-weight:800;letter-spacing:-.01em;color:#111827'>Spec Change Prediction</div>"
    "<div style='font-size:.7rem;color:#9ca3af;margin:1px 0 4px'>"
    "NYC (KC/CC/SB/CT) = Spec + Non Rep + Index &nbsp;&middot;&nbsp; "
    "Europe (RC/LCC/LSU) = Managed Money + Other + Non Rep &nbsp;&middot;&nbsp; "
    "Prediction = &beta;&times;&Delta;Px% (last COT Tue &rarr; latest Tue close) + &alpha;</div>",
    unsafe_allow_html=True)

tab_overview, tab_regress, tab_position = st.tabs(
    ["Overview", "Regression & Correlation", "Positioning"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_overview:
    body = ""
    for c, res, r in rows:
        color = COMM_COLORS.get(c, "#374151")
        region = "NYC" if res["is_cit"] else "EU"
        region_bg = "#eff6ff" if res["is_cit"] else "#fdf4ff"
        region_fg = "#1d4ed8" if res["is_cit"] else "#a21caf"
        px_chg_pct = res["px_move_pct"]
        px_clr = C_LONG if px_chg_pct >= 0 else C_SHORT
        oi_chg = res["latest_oi"] - res["last_oi"] if not pd.isna(res["last_oi"]) else np.nan
        oi_chg_pct = (oi_chg / res["last_oi"] * 100) if not pd.isna(res["last_oi"]) and res["last_oi"] else np.nan
        oi_clr = C_LONG if (not pd.isna(oi_chg) and oi_chg >= 0) else C_SHORT
        oi_chg_k = oi_chg / 1000
        spec_net_k = r["spec_net"] / 1000
        pred_k = r["prediction"] / 1000
        pred_clr = C_LONG if pred_k >= 0 else C_SHORT
        if r["status"] == "Resolved":
            act_k = r["actual"] / 1000
            act_clr = C_LONG if act_k >= 0 else C_SHORT
            act_html = f"<span style='color:{act_clr}'>{act_k:+.1f}k</span>"
        else:
            act_html = "<i style='color:#c7cbd1;font-weight:400'>Awaiting</i>"

        body += (
            f"<tr>"
            f"<td style='{_td}'><span style='background:{region_bg};color:{region_fg};"
            f"border-radius:4px;padding:1px 6px;font-size:.62rem;font-weight:700'>{region}</span></td>"
            f"<td style='{_td};color:{color}'>{COMM_NAMES[c]} <span style='color:#c7cbd1;font-weight:400'>{c}</span></td>"
            f"<td style='{_td};color:#6b7280;font-weight:400'>{r['cot_date'].strftime('%d %b %y')}</td>"
            f"<td style='{_td}'>{spec_net_k:+.1f}k</td>"
            f"<td style='{_td};color:{pred_clr}'>{pred_k:+.1f}k</td>"
            f"<td style='{_td}'>{act_html}</td>"
            f"<td style='{_td};color:{px_clr}'>{px_chg_pct:+.1f}%</td>"
            f"<td style='{_td};color:#6b7280;font-weight:400'>{res['latest_label'] or '—'}</td>"
            f"<td style='{_td};color:{oi_clr}'>{oi_chg_k:+.1f}k <span style='color:#c7cbd1'>({oi_chg_pct:+.1f}%)</span></td>"
            f"<td style='{_td}'>{res['latest_px']:.2f}</td>"
            f"<td style='{_td};color:#6b7280;font-weight:400'>{res['latest_oi_date'].strftime('%d %b %y')}</td>"
            f"</tr>")
    st.markdown(_table(["Region","Commodity","COT Date","Spec Net","Prediction","Actual",
                         "Px Δ","Future","OI Δ","Px","OI Date"], body), unsafe_allow_html=True)
    st.markdown(
        f"<div style='margin-top:10px;font-size:.62rem;color:#c7cbd1'>"
        f"{len(log_df)} logged predictions &middot; rows flip Awaiting &rarr; Resolved once the next COT release lands</div>",
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — REGRESSION & CORRELATION
# ══════════════════════════════════════════════════════════════════════════
with tab_regress:
    sel = st.selectbox("Commodity", COMMODITIES, format_func=lambda x: f"{COMM_NAMES[x]} ({x})",
                        key="regress_commodity")
    res = res_by_c.get(sel)
    if res is None:
        st.info("No data.")
    else:
        color = COMM_COLORS[sel]
        hist = res["hist"]
        ds = hist["Spec Net"].diff()
        px_chg = hist["rollex_px"].pct_change() * 100
        common = ~(ds.isna() | px_chg.isna())
        x_hist = px_chg[common].values.astype(float)
        y_hist = ds[common].values.astype(float)
        dates_common = hist["Date"][common].reset_index(drop=True)
        beta, alpha, r2 = res["beta"], res["alpha"], res["r2"]
        x_line = np.linspace(x_hist.min(), x_hist.max(), 200)
        y_line = beta * x_line + alpha

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode="markers",
                marker=dict(color=color, size=6, opacity=.5, line=dict(width=.4, color="white")),
                hovertemplate="ΔPx%: %{x:.1f}%<br>ΔSpec: %{y:.1f} lots<extra></extra>", showlegend=False))
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                line=dict(color=color, width=2, dash="dash"), showlegend=False))
            fig.add_trace(go.Scatter(x=[x_hist[-1]], y=[y_hist[-1]], mode="markers", name="Latest",
                marker=dict(symbol="star", size=13, color="#f59e0b", line=dict(width=1, color="white")),
                hovertemplate=f"<b>{dates_common.iloc[-1].strftime('%d %b %y')}</b><extra></extra>"))
            fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                text=f"R² = {r2:.2f} · n = {len(x_hist)}", showarrow=False,
                font=dict(size=10, color="#6b7280"), xanchor="left", yanchor="top")
            fig.update_layout(**_BASE, height=340, showlegend=False,
                title=dict(text="ΔSpec Net vs ΔPx% · weekly", font=dict(size=11, color="#374151"), x=0),
                margin=dict(l=54, r=16, t=38, b=40),
                xaxis=dict(**_ax(), title_text="Px Δ% (weekly)", ticksuffix="%"),
                yaxis=dict(**_ax(), title_text="Δ Spec Net (lots)"))
            st.plotly_chart(fig, width='stretch')
        with c2:
            n_show = min(52, len(dates_common))
            bd = dates_common.iloc[-n_show:].dt.strftime("%d %b'%y")
            ba = y_hist[-n_show:]
            bp = beta * x_hist[-n_show:] + alpha
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=bd, y=ba, name="Actual", marker_color=color, opacity=.75))
            fig2.add_trace(go.Bar(x=bd, y=bp, name="Predicted", marker_color="#c7cbd1", opacity=.85))
            fig2.update_layout(**_BASE, height=340, barmode="group",
                title=dict(text="Actual vs Predicted Δ Spec · last 52w", font=dict(size=11, color="#374151"), x=0),
                margin=dict(l=54, r=16, t=38, b=40),
                xaxis=_ax(x=True),
                yaxis=dict(**_ax(), title_text="Δ (lots)"),
                legend=dict(orientation="h", y=1.16, x=1, xanchor="right", font=dict(size=9)))
            st.plotly_chart(fig2, width='stretch')

        st.markdown(
            f"<div style='font-size:.68rem;font-weight:700;color:#9ca3af;letter-spacing:.03em;"
            f"margin:6px 0 6px'>PAIRWISE COT CORRELATION — {sel} · weekly Δ</div>", unsafe_allow_html=True)

        sub = res["sub"]
        if res["is_cit"]:
            corr_cols = ["Spec Net","Comm Net","Non Rep Net","Index Net"]
        else:
            corr_cols = ["MM Net","Comm Net","Non Rep Net","Other Net"]
        pw = {lbl: sub[lbl].astype(float).diff().reset_index(drop=True)
              for lbl in corr_cols if lbl in sub.columns}
        pw["Px %Δ"] = hist["rollex_px"].pct_change().reset_index(drop=True) * 100
        pw_df = pd.DataFrame(pw).dropna(how="all")
        if len(pw_df) >= 4:
            corr = pw_df.corr()
            labels = list(corr.columns)
            z = corr.values
            fig3 = go.Figure(go.Heatmap(
                z=z, x=labels, y=labels,
                colorscale=[[0,"#dc2626"],[0.5,"#f9fafb"],[1,"#16a34a"]], zmid=0, zmin=-1, zmax=1,
                text=[[f"{v:+.2f}" for v in row] for row in z], texttemplate="%{text}",
                textfont=dict(size=10, color="#111"),
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b>: r=%{z:.2f}<extra></extra>",
                colorbar=dict(thickness=10, len=.7, tickfont=dict(size=9)), xgap=2, ygap=2))
            fig3.update_layout(**_BASE, height=270, margin=dict(l=90, r=20, t=10, b=10),
                xaxis=dict(side="top", tickfont=dict(size=9), showgrid=False, showline=False),
                yaxis=dict(autorange="reversed", tickfont=dict(size=9), showgrid=False, showline=False))
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Not enough overlapping history for a correlation matrix.")

    with st.expander("Diagnostics — R² across commodities · Roll Yield · USDBRL"):
        diag = pd.DataFrame([{
            "Commodity": c, "R²": f"{res['r2']:.2f}", "n obs": res["n"],
        } for c, res, r in rows]).set_index("Commodity")
        st.dataframe(diag, width='stretch', height=246)

        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.markdown("<div style='font-size:.68rem;font-weight:700;color:#9ca3af;letter-spacing:.03em;"
                        "margin:14px 0 6px'>ROLL YIELD — actual, week on week</div>", unsafe_allow_html=True)
            ry_all = load_roll_yield()
            ry_body = ""
            for c, res_c, r in rows:
                ry_code = ROLLYIELD_MAP.get(c)
                lt = old = np.nan
                if ry_code and not ry_all.empty:
                    s = ry_all[ry_all["Commodity"] == ry_code].sort_values("Date")
                    if not s.empty:
                        lt = pd.merge_asof(pd.DataFrame({"Date":[res_c["last_cot_date"]]}), s,
                                            on="Date", direction="backward")["roll_yield_pct"].iloc[0]
                        old_date = res_c["last_cot_date"] - pd.Timedelta(days=7)
                        old = pd.merge_asof(pd.DataFrame({"Date":[old_date]}), s,
                                             on="Date", direction="backward")["roll_yield_pct"].iloc[0]
                if pd.isna(lt) or pd.isna(old):
                    ry_body += (f"<tr><td style='{_td}'>{c}</td>"
                                f"<td style='{_td}' colspan='3'><i style='color:#c7cbd1;font-weight:400'>no series</i></td></tr>")
                    continue
                chg = lt - old
                clr = C_LONG if chg >= 0 else C_SHORT
                ry_body += (f"<tr><td style='{_td}'>{c}</td><td style='{_td}'>{lt:.1f}%</td>"
                            f"<td style='{_td}'>{old:.1f}%</td><td style='{_td};color:{clr}'>{chg:+.1f}%</td></tr>")
            st.markdown(_table(["Commodity","Lt COT","Old COT","Change"], ry_body), unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='font-size:.68rem;font-weight:700;color:#9ca3af;letter-spacing:.03em;"
                        "margin:14px 0 6px'>USDBRL MOVE</div>", unsafe_allow_html=True)
            fx = load_brl()
            if fx.empty or not rows:
                st.info("No BRL series found.")
            else:
                ref_last_cot = rows[0][1]["last_cot_date"]
                prev_cot = ref_last_cot - pd.Timedelta(days=7)
                latest_fx_date = fx["Date"].iloc[-1]
                latest_fx = float(fx["USDBRL"].iloc[-1])

                def _asof(fxdf, d):
                    m = pd.merge_asof(pd.DataFrame({"Date":[d]}), fxdf, on="Date", direction="backward")
                    return float(m["USDBRL"].iloc[0]) if not m["USDBRL"].isna().iloc[0] else np.nan

                px_last_cot = _asof(fx, ref_last_cot)
                px_prev_cot = _asof(fx, prev_cot)

                def _panel(title, new_lbl, new_val, old_lbl, old_val):
                    if pd.isna(new_val) or pd.isna(old_val) or old_val == 0:
                        return
                    mv = (new_val / old_val - 1) * 100
                    clr = C_LONG if mv >= 0 else C_SHORT
                    st.markdown(
                        f"<div style='border:1px solid #edeff3;border-radius:8px;padding:8px 12px;margin-bottom:8px'>"
                        f"<div style='font-size:.6rem;color:#9ca3af;font-weight:700;letter-spacing:.03em;margin-bottom:4px'>{title}</div>"
                        f"<table style='width:100%;font-size:.72rem'><tr>"
                        f"<td style='color:#9ca3af'>{new_lbl}</td><td style='color:#9ca3af'>{old_lbl}</td><td style='color:#9ca3af'>% Move</td></tr>"
                        f"<tr><td style='font-weight:700'>{new_val:.4f}</td><td style='font-weight:700'>{old_val:.4f}</td>"
                        f"<td style='font-weight:700;color:{clr}'>{mv:+.2f}%</td></tr></table></div>",
                        unsafe_allow_html=True)

                _panel("LATEST BRL MOVE WRT LAST COT", latest_fx_date.strftime('%d %b %y'), latest_fx,
                       ref_last_cot.strftime('%d %b %y'), px_last_cot)
                _panel("BRL MOVE IN PREVIOUS COT WINDOW", ref_last_cot.strftime('%d %b %y'), px_last_cot,
                       prev_cot.strftime('%d %b %y'), px_prev_cot)

        st.markdown(f"<div style='margin-top:10px;font-size:.6rem;color:#c7cbd1'>Log: {LOG_FILE.name}</div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — POSITIONING (Recap + Spec/Commercial, per commodity)
# ══════════════════════════════════════════════════════════════════════════
with tab_position:
    sel2 = st.selectbox("Commodity", COMMODITIES, format_func=lambda x: f"{COMM_NAMES[x]} ({x})",
                         key="position_commodity")
    res2 = res_by_c.get(sel2)
    if res2 is None:
        st.info("No data.")
    else:
        color = COMM_COLORS[sel2]
        sub = res2["sub"]
        is_cit = res2["is_cit"]
        latest, prev = sub.iloc[-1], (sub.iloc[-2] if len(sub) >= 2 else sub.iloc[-1])

        if is_cit:
            groups = [("Large Spec","Spec Long","Spec Short","Spec Net"),
                      ("Commercial","Comm Long","Comm Short","Comm Net"),
                      ("Non-Reportable","Non Rep Long","Non Rep Short","Non Rep Net"),
                      ("Index","Index Long","Index Short","Index Net")]
            spread_col, spread_lbl = "Spec Spread", "Spec Spread"
        else:
            groups = [("Managed Money","MM Long","MM Short","MM Net"),
                      ("Commercial (Prod.)","Producer Long","Producer Short","Producer Net"),
                      ("Non-Reportable","Non Rep Long","Non Rep Short","Non Rep Net"),
                      ("Other Reportable","Other Long","Other Short","Other Net")]
            spread_col, spread_lbl = "MM Spread", "MM Spread"

        cards = ""
        for name, lc, sc, nc in groups:
            if nc not in sub.columns:
                continue
            v = float(latest[nc])
            pv = float(prev[nc]) if nc in prev else np.nan
            chg = v - pv if not pd.isna(pv) else np.nan
            clr = C_LONG if v >= 0 else C_SHORT
            chg_html = (f"<span style='font-size:.62rem;color:{C_LONG if chg>=0 else C_SHORT}'>"
                        f"{chg/1000:+.1f}k w/w</span>") if not pd.isna(chg) else ""
            cards += (
                f"<div style='flex:1;min-width:150px;border:1px solid #edeff3;border-radius:8px;padding:8px 12px'>"
                f"<div style='font-size:.6rem;color:#9ca3af;font-weight:700;letter-spacing:.03em'>{name.upper()}</div>"
                f"<div style='font-size:1rem;font-weight:800;color:{clr}'>{v/1000:+.1f}k</div>{chg_html}</div>")
        if spread_col in sub.columns and pd.notna(latest.get(spread_col)):
            sv = float(latest[spread_col])
            cards += (
                f"<div style='flex:1;min-width:150px;border:1px solid #edeff3;border-radius:8px;padding:8px 12px'>"
                f"<div style='font-size:.6rem;color:#9ca3af;font-weight:700;letter-spacing:.03em'>{spread_lbl.upper()}</div>"
                f"<div style='font-size:1rem;font-weight:800;color:#374151'>{sv/1000:.1f}k</div></div>")
        st.markdown(f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px'>{cards}</div>",
                    unsafe_allow_html=True)

        # Every group's Net line on one chart — click a legend entry to
        # isolate it (Plotly default), so no separate Long/Short view is needed.
        fig = go.Figure()
        for name, lc, sc, nc in groups:
            if nc not in sub.columns:
                continue
            fig.add_trace(go.Scatter(x=sub["Date"], y=sub[nc]/1000, mode="lines",
                                      name=name, line=dict(width=1.8)))
        fig.update_layout(**_BASE, height=400,
            title=dict(text=f"{COMM_NAMES[sel2]} — Net Positioning", font=dict(size=11, color="#374151"), x=0),
            margin=dict(l=54, r=16, t=38, b=40),
            xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
            yaxis=dict(**_ax(), title_text="k lots"),
            legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9)))
        st.plotly_chart(fig, width='stretch')

