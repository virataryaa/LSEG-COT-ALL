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
  .block-container { padding-top:1.2rem !important; max-width:1700px; }
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
             ("Other Long","Other Short","Other Net")]
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
    last_oi       = float(merged["Total OI"].iloc[-1])
    prev_oi       = float(merged["Total OI"].iloc[-2]) if len(merged) >= 2 else np.nan

    latest_px    = float(rx["rollex_px"].iloc[-1])
    latest_date  = rx["Date"].iloc[-1]
    latest_label = rx["active_label"].iloc[-1] if "active_label" in rx.columns else None

    px_move_pct   = (latest_px / last_cot_px - 1) * 100 if last_cot_px else np.nan
    px_move_abs   = latest_px - last_cot_px
    predicted_delta = fit["beta"] * px_move_pct + fit["alpha"]
    predicted_net    = last_cot_spec + predicted_delta

    return dict(
        commodity=commodity, spec_col=spec_col,
        last_cot_date=last_cot_date, last_cot_spec=last_cot_spec, last_cot_px=last_cot_px,
        prev_cot_spec=prev_cot_spec, last_oi=last_oi, prev_oi=prev_oi,
        latest_px=latest_px, latest_date=latest_date, latest_label=latest_label,
        px_move_pct=px_move_pct, px_move_abs=px_move_abs,
        beta=fit["beta"], alpha=fit["alpha"], r2=fit["r2"], n=fit["n"],
        predicted_delta=predicted_delta, predicted_net=predicted_net,
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
# BUILD TABLE
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='background:#111827;color:#fff;padding:14px 20px;border-radius:8px 8px 0 0;"
    "font-size:1.05rem;font-weight:700;letter-spacing:.02em'>SPEC CHANGE PREDICTION &amp; ACTUAL</div>"
    "<div style='background:#f3f4f6;color:#4b5563;padding:8px 20px;font-size:.78rem;"
    "border-radius:0 0 8px 8px;margin-bottom:18px'>"
    "NYC (KC/CC/SB/CT) = Spec + Non Rep + Index &nbsp;&middot;&nbsp; "
    "Europe (RC/LCC/LSU) = Managed Money + Other + Non Rep &nbsp;&middot;&nbsp; "
    "Prediction = &beta;&times;&Delta;Px% (last COT Tue &rarr; latest Tue close) + &alpha;, fit on weekly history</div>",
    unsafe_allow_html=True)

log_df = load_log()
rows = []
for c in COMMODITIES:
    res = compute_commodity(c)
    if res is None:
        continue
    resolved, log_df = resolve_row(res, log_df)
    rows.append((c, res, resolved))

_th = ("padding:7px 12px;font-size:.62rem;font-weight:700;color:#94a3b8;letter-spacing:.05em;"
       "border:1px solid #e5e7eb;background:#f9fafb;text-align:left;white-space:nowrap")
_td = ("padding:8px 12px;font-size:.83rem;font-weight:600;color:#1e293b;"
       "border:1px solid #e5e7eb;white-space:nowrap")

html = "<table style='border-collapse:collapse;width:100%;font-family:-apple-system,sans-serif'><tr>"
for h in ["Spec Inclusion","Commodity","COT Date","Spec Net","Prediction","Actual",
          "Px Change","Future","OI Change (K)","OI Change %","Px","Latest Px Date"]:
    html += f"<th style='{_th}'>{h}</th>"
html += "</tr>"

for c, res, r in rows:
    color = COMM_COLORS.get(c, "#374151")
    px_chg_pct = res["px_move_pct"]
    px_clr = "#16a34a" if px_chg_pct >= 0 else "#dc2626"
    oi_chg = res["last_oi"] - res["prev_oi"] if not pd.isna(res["prev_oi"]) else np.nan
    oi_chg_pct = (oi_chg / res["prev_oi"] * 100) if not pd.isna(res["prev_oi"]) and res["prev_oi"] else np.nan
    oi_clr = "#16a34a" if (not pd.isna(oi_chg) and oi_chg >= 0) else "#dc2626"
    oi_chg_k = oi_chg / 1000
    spec_net_k = r["spec_net"] / 1000
    pred_k = r["prediction"] / 1000
    pred_clr = "#16a34a" if pred_k >= 0 else "#dc2626"
    if r["status"] == "Resolved":
        act_k = r["actual"] / 1000
        act_clr = "#16a34a" if act_k >= 0 else "#dc2626"
        act_html = f"<span style='color:{act_clr}'>{act_k:+.1f}k</span>"
    else:
        act_html = "<i style='color:#9ca3af;font-weight:400'>Awaiting</i>"

    html += (
        f"<tr>"
        f"<td style='{_td};color:#6b7280;font-weight:400'>{SPEC_INCLUSION[c]}</td>"
        f"<td style='{_td};color:{color}'>{COMM_NAMES[c]} ({c})</td>"
        f"<td style='{_td}'>{r['cot_date'].strftime('%d-%b-%y')}</td>"
        f"<td style='{_td}'>{spec_net_k:+.1f}k</td>"
        f"<td style='{_td};color:{pred_clr}'>{pred_k:+.1f}k</td>"
        f"<td style='{_td}'>{act_html}</td>"
        f"<td style='{_td};color:{px_clr}'>{px_chg_pct:+.1f}%</td>"
        f"<td style='{_td};color:#6b7280;font-weight:400'>{res['latest_label'] or '—'}</td>"
        f"<td style='{_td};color:{oi_clr}'>{oi_chg_k:+.1f}k</td>"
        f"<td style='{_td};color:{oi_clr}'>{oi_chg_pct:+.1f}%</td>"
        f"<td style='{_td}'>{res['latest_px']:.2f}</td>"
        f"<td style='{_td};color:#6b7280;font-weight:400'>{res['latest_date'].strftime('%d-%b-%y')}</td>"
        f"</tr>")
html += "</table>"
st.markdown(html, unsafe_allow_html=True)

with st.expander("Regression diagnostics (β, α, R², n obs)"):
    diag = pd.DataFrame([{
        "Commodity": c, "β (k lots / 1%)": f"{res['beta']:+.2f}", "α": f"{res['alpha']:+.2f}",
        "R²": f"{res['r2']:.2f}", "n obs": res["n"],
    } for c, res, r in rows]).set_index("Commodity")
    st.dataframe(diag, width='stretch')

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# ROLL YIELD  +  BRL PANELS
# ══════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown("<div style='font-size:.88rem;font-weight:700;color:#374151;margin-bottom:6px'>ROLL YIELD — actual, week on week</div>", unsafe_allow_html=True)
    ry_all = load_roll_yield()
    ry_rows = []
    for c, res, r in rows:
        ry_code = ROLLYIELD_MAP.get(c)
        if ry_code is None or ry_all.empty:
            ry_rows.append((c, np.nan, np.nan)); continue
        s = ry_all[ry_all["Commodity"] == ry_code].sort_values("Date")
        if s.empty:
            ry_rows.append((c, np.nan, np.nan)); continue
        lt = pd.merge_asof(pd.DataFrame({"Date":[res["last_cot_date"]]}), s, on="Date", direction="backward")["roll_yield_pct"].iloc[0]
        old_date = res["last_cot_date"] - pd.Timedelta(days=7)
        old = pd.merge_asof(pd.DataFrame({"Date":[old_date]}), s, on="Date", direction="backward")["roll_yield_pct"].iloc[0]
        ry_rows.append((c, lt, old))
    ry_html = "<table style='border-collapse:collapse;width:100%;font-family:-apple-system,sans-serif'><tr>"
    for h in ["Commodity","Lt COT","Old COT","Change"]:
        ry_html += f"<th style='{_th}'>{h}</th>"
    ry_html += "</tr>"
    for c, lt, old in ry_rows:
        if pd.isna(lt) or pd.isna(old):
            ry_html += f"<tr><td style='{_td}'>{c}</td><td style='{_td}' colspan='3'><i style='color:#9ca3af;font-weight:400'>no series</i></td></tr>"
            continue
        chg = lt - old
        clr = "#16a34a" if chg >= 0 else "#dc2626"
        ry_html += (f"<tr><td style='{_td}'>{c}</td><td style='{_td}'>{lt:.1f}%</td>"
                    f"<td style='{_td}'>{old:.1f}%</td><td style='{_td};color:{clr}'>{chg:+.1f}%</td></tr>")
    ry_html += "</table>"
    st.markdown(ry_html, unsafe_allow_html=True)

with col2:
    st.markdown("<div style='font-size:.88rem;font-weight:700;color:#374151;margin-bottom:6px'>USDBRL MOVE</div>", unsafe_allow_html=True)
    fx = load_brl()
    if fx.empty or not rows:
        st.info("No BRL series found.")
    else:
        # Use the most-recent commodity's COT calendar as the reference weeks
        ref_last_cot = rows[0][1]["last_cot_date"]
        prev_cot = ref_last_cot - pd.Timedelta(days=7)
        prev2_cot = ref_last_cot - pd.Timedelta(days=14)
        latest_fx_date = fx["Date"].iloc[-1]
        latest_fx = float(fx["USDBRL"].iloc[-1])

        def _asof(fx, d):
            m = pd.merge_asof(pd.DataFrame({"Date":[d]}), fx, on="Date", direction="backward")
            return float(m["USDBRL"].iloc[0]) if not m["USDBRL"].isna().iloc[0] else np.nan

        px_last_cot = _asof(fx, ref_last_cot)
        px_prev_cot = _asof(fx, prev_cot)
        px_prev2_cot = _asof(fx, prev2_cot)

        def _panel(title, new_lbl, new_val, old_lbl, old_val):
            if pd.isna(new_val) or pd.isna(old_val) or old_val == 0:
                return
            mv = (new_val / old_val - 1) * 100
            clr = "#16a34a" if mv >= 0 else "#dc2626"
            st.markdown(
                f"<div style='border:1px solid #e5e7eb;border-radius:8px;padding:10px 14px;margin-bottom:10px'>"
                f"<div style='font-size:.68rem;color:#9ca3af;font-weight:700;letter-spacing:.04em;margin-bottom:6px'>{title}</div>"
                f"<table style='width:100%;font-size:.8rem'><tr>"
                f"<td style='color:#6b7280'>{new_lbl}</td><td style='color:#6b7280'>{old_lbl}</td><td style='color:#6b7280'>% Move</td></tr>"
                f"<tr><td style='font-weight:700'>{new_val:.4f}</td><td style='font-weight:700'>{old_val:.4f}</td>"
                f"<td style='font-weight:700;color:{clr}'>{mv:+.2f}%</td></tr></table></div>",
                unsafe_allow_html=True)

        _panel("LATEST BRL MOVE WRT LAST COT",
               latest_fx_date.strftime('%d-%b-%y'), latest_fx,
               ref_last_cot.strftime('%d-%b-%y'), px_last_cot)
        _panel("BRL MOVE IN PREVIOUS COT WINDOW",
               ref_last_cot.strftime('%d-%b-%y'), px_last_cot,
               prev_cot.strftime('%d-%b-%y'), px_prev_cot)

st.markdown(
    "<div style='margin-top:18px;font-size:.72rem;color:#9ca3af'>"
    f"Log: {LOG_FILE.name} &middot; {len(log_df)} logged predictions &middot; "
    "refresh weekly once new COT data lands to see rows flip from Awaiting to Resolved.</div>",
    unsafe_allow_html=True)
