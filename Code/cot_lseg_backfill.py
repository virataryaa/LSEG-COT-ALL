"""
Hardmine — COT CFTC Full History Backfill (LSEG)
==================================================
LSEG-API replacement for ICEBREAKER/COT_ALL/Code/cot_backfill.py (icepython-based).

Produces the same three parquet files, with the same column names, as the ICE
pipeline, for the same 7 commodities (KC, CC, SB, CT, RC, LCC, LSU) in both
Futures+Options-combined and Futures-only flavours, plus the CIT/Supplemental
report for KC/CC/SB/CT. Also covers 3 COMEX metals (GC Gold, SI Silver, HG
Copper — added 2026-08-28), Disaggregated-only like RC/LCC/LSU: metals have
no CIT/Index Traders category on the CFTC (or on LSEG — verified live),
since CIT is the ag-only Commodity Index Supplemental report.

Known, confirmed-by-probe gaps vs. the ICE source (left as NaN, not estimated):
  - Per-category trader counts (Traders Comm/Spec/Index/Producer/Swap/MM/Other
    Long/Short/Spread). LSEG only publishes an aggregate "Total Reportable"
    trader count (TTLNG/TTSHT), which is used to fill "Traders Tot Rept Long/
    Short" only.
  - Concentration (Conc Gross/Net 4/8 Long/Short) — not published by LSEG for
    this report at all.
  - Old/New crop split — LSEG only carries this as a live snapshot field
    (GEN_VAL2/GEN_VAL3) with no historical archive, so it cannot be
    backfilled. Every row is written with Crop="All" only; no Old/Other rows
    are produced (unlike the ICE source, which has all three).

Usage:
    python cot_lseg_backfill.py            # full history from 2010-01-01
    python cot_lseg_backfill.py --start 2015-01-01
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd
pd.set_option("future.no_silent_downcasting", True)  # silences a harmless lseg.data internal FutureWarning

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "cot_lseg_backfill.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "Database"
CIT_FILE          = DB_DIR / "cot_cit.parquet"
DISAGG_FUT_FILE   = DB_DIR / "cot_disagg_fut.parquet"
DISAGG_FUTOPT_FILE = DB_DIR / "cot_disagg_futopt.parquet"

START_FULL = "2010-01-01"

# ══════════════════════════════════════════════════════════════════════════════
# COMMODITY / RIC CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# "cftc" commodities live under CFTC's own 6-digit code, with no "CFTC" token
# in the category RICs (only Total OI / Total Traders use the "CFTC" token).
# "lif"  commodities (ICE Futures Europe / LIFFE) use their ticker as the code,
# and DO carry the "LIF" token in every RIC, category fields included.
CIT_COMMODITIES = {
    "KC": {"code": "083731", "px": "KCc2"},
    "CC": {"code": "073732", "px": "CCc2"},
    "SB": {"code": "080732", "px": "SBc2"},
    "CT": {"code": "033661", "px": "CTc2"},
}

DISAGG_COMMODITIES = {
    "KC":  {"kind": "cftc", "code": "083731", "px": "KCc2"},
    "CC":  {"kind": "cftc", "code": "073732", "px": "CCc2"},
    "SB":  {"kind": "cftc", "code": "080732", "px": "SBc2"},
    "CT":  {"kind": "cftc", "code": "033661", "px": "CTc2"},
    "RC":  {"kind": "lif",  "code": "LRC",    "px": "LRCc2"},
    "LCC": {"kind": "lif",  "code": "LCC",    "px": "LCCc2"},
    "LSU": {"kind": "lif",  "code": "LSU",    "px": "LSUc2"},
    # Metals (COMEX) — verified live 2026-08-28 (OI + Producer/MM category
    # RICs + price all resolve). No CIT/Index Traders category exists for
    # these (also verified — 4<code>PLNG returns no data), which is
    # expected: CIT is CFTC's ag-only Commodity Index Supplemental report.
    # Same "no CIT" situation RC/LCC/LSU are already in above, so the
    # dashboard's existing CIT_COMMS gating already handles this cleanly.
    "GC":  {"kind": "cftc", "code": "088691", "px": "GCc2"},  # Gold
    "SI":  {"kind": "cftc", "code": "084691", "px": "SIc2"},  # Silver
    "HG":  {"kind": "cftc", "code": "085692", "px": "HGc2"},  # Copper
}

FIELD_BY_KIND = {"cftc": "COMM_LAST", "lif": "TRDPRC_1"}

# ── CIT category RICs (prefix "4") ──────────────────────────────────────────
CIT_CATS = {
    "Comm Long": "CLNG", "Comm Short": "CSHT",
    "Spec Long": "NLNG", "Spec Short": "NSHT", "Spec Spread": "NSPD",
    "Index Long": "PLNG", "Index Short": "PSHT",
    "Non Rep Long": "RLNG", "Non Rep Short": "RSHT",
    "Traders Tot Rept Long": "TTLNG", "Traders Tot Rept Short": "TTSHT",
}
CIT_POS_FOR_PCT = [
    "Comm Long", "Comm Short", "Spec Long", "Spec Short", "Spec Spread",
    "Index Long", "Index Short", "Non Rep Long", "Non Rep Short",
]
CIT_FINAL_COLS = (
    ["Commodity", "Date"]
    + ["Comm Long", "Comm Short", "Spec Long", "Spec Short", "Spec Spread",
       "Index Long", "Index Short", "Non Rep Long", "Non Rep Short", "Total OI"]
    + ["Traders Comm Long", "Traders Comm Short",
       "Traders Spec Long", "Traders Spec Short", "Traders Spec Spread",
       "Traders Index Long", "Traders Index Short",
       "Traders Tot Rept Long", "Traders Tot Rept Short"]
    + ["Pct OI " + c for c in CIT_POS_FOR_PCT]
    + ["Px"]
)

# ── Disaggregated category RICs (prefix "3"=COMB / "1"=FUT) ────────────────
DISAGG_CATS = {
    "Producer Long": "PLNG", "Producer Short": "PSHT",
    "Swap Long": "SLNG", "Swap Short": "SSHT", "Swap Spread": "SSPD",
    "MM Long": "MLNG", "MM Short": "MSHT", "MM Spread": "MSPD",
    "Other Long": "OLNG", "Other Short": "OSHT", "Other Spread": "OSPD",
    "Tot Rept Long": "TLNG", "Tot Rept Short": "TSHT",
    "Non Rep Long": "RLNG", "Non Rep Short": "RSHT",
    "Traders Tot Rept Long": "TTLNG", "Traders Tot Rept Short": "TTSHT",
}
DISAGG_POS_FOR_PCT = [
    "Producer Long", "Producer Short",
    "Swap Long", "Swap Short", "Swap Spread",
    "MM Long", "MM Short", "MM Spread",
    "Other Long", "Other Short", "Other Spread",
    "Tot Rept Long", "Tot Rept Short",
    "Non Rep Long", "Non Rep Short",
]
DISAGG_FINAL_COLS = (
    ["Commodity", "Crop", "Date", "Total OI"]
    + ["Producer Long", "Producer Short",
       "Swap Long", "Swap Short", "Swap Spread",
       "MM Long", "MM Short", "MM Spread",
       "Other Long", "Other Short", "Other Spread",
       "Tot Rept Long", "Tot Rept Short", "Non Rep Long", "Non Rep Short"]
    + ["Traders Total",
       "Traders Producer Long", "Traders Producer Short",
       "Traders Swap Long", "Traders Swap Short", "Traders Swap Spread",
       "Traders MM Long", "Traders MM Short", "Traders MM Spread",
       "Traders Other Long", "Traders Other Short", "Traders Other Spread",
       "Traders Tot Rept Long", "Traders Tot Rept Short"]
    + ["Conc Gross 4 Long", "Conc Gross 4 Short", "Conc Gross 8 Long", "Conc Gross 8 Short",
       "Conc Net 4 Long", "Conc Net 4 Short", "Conc Net 8 Long", "Conc Net 8 Short"]
    + ["Pct OI " + c for c in DISAGG_POS_FOR_PCT]
    + ["Px"]
)

# Columns LSEG cannot supply at all — always written as NaN, never estimated.
CIT_GAP_COLS = [
    "Traders Comm Long", "Traders Comm Short",
    "Traders Spec Long", "Traders Spec Short", "Traders Spec Spread",
    "Traders Index Long", "Traders Index Short",
]
DISAGG_GAP_COLS = [
    "Traders Producer Long", "Traders Producer Short",
    "Traders Swap Long", "Traders Swap Short", "Traders Swap Spread",
    "Traders MM Long", "Traders MM Short", "Traders MM Spread",
    "Traders Other Long", "Traders Other Short", "Traders Other Spread",
    "Conc Gross 4 Long", "Conc Gross 4 Short", "Conc Gross 8 Long", "Conc Gross 8 Short",
    "Conc Net 4 Long", "Conc Net 4 Short", "Conc Net 8 Long", "Conc Net 8 Short",
]


# ══════════════════════════════════════════════════════════════════════════════
# FETCH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _root(kind: str, code: str) -> str:
    return f"LIF{code}" if kind == "lif" else code


def _cftc_root(kind: str, code: str) -> str:
    """Root used only by Total OI / Total Traders (always carries CFTC/LIF token)."""
    return f"LIF{code}" if kind == "lif" else f"CFTC{code}"


def _single_ric_history(ld, ric: str, col: str, field: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch one RIC/field and return a single-column DataFrame named `col`,
    indexed by Date. get_history on a single-item universe does not always
    keep the RIC as the column label, so just take whatever column comes
    back rather than matching by name."""
    try:
        d = ld.get_history(universe=[ric], fields=[field], start=start, end=end,
                            interval="daily", count=10000)
    except Exception as e:
        log.warning("  MISSING: %s (%s) — %s", ric, col, str(e)[:120])
        return None
    if d is None or d.empty:
        log.warning("  EMPTY: %s (%s)", ric, col)
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d = d.iloc[:, [0]]
    d.columns = [col]
    return d


def _batch_history(ld, ric_to_col: dict, field: str, start: str, end: str) -> pd.DataFrame:
    """Fetch many RICs with one field. Uses a single multi-universe call when
    there's more than one RIC (columns come back keyed by RIC), and falls
    back to per-RIC fetches (matched positionally, not by name) when that's
    not possible or when RICs are missing from the batch response."""
    rics = list(ric_to_col.keys())

    if len(rics) == 1:
        ric = rics[0]
        d = _single_ric_history(ld, ric, ric_to_col[ric], field, start, end)
        return d if d is not None else pd.DataFrame()

    try:
        df = ld.get_history(universe=rics, fields=[field], start=start, end=end,
                             interval="daily", count=10000)
    except Exception as e:
        log.warning("  Batch fetch failed (%d RICs, field=%s): %s — falling back to per-RIC",
                    len(rics), field, e)
        df = None

    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        got = set(df.columns)
        missing = [r for r in rics if r not in got]
        if missing:
            log.warning("  %d/%d RICs missing from batch response, re-fetching individually: %s",
                        len(missing), len(rics), missing)
        else:
            return df.rename(columns=ric_to_col)
    else:
        missing = rics
        df = pd.DataFrame()

    df = df.rename(columns=ric_to_col) if not df.empty else pd.DataFrame()
    frames = [df] if not df.empty else []
    for ric in missing:
        d = _single_ric_history(ld, ric, ric_to_col[ric], field, start, end)
        if d is not None:
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def fetch_cit_commodity(ld, comm: str, cfg: dict, start: str, end: str) -> pd.DataFrame:
    code = cfg["code"]
    ric_to_col = {f"4{code}{suffix}": col for col, suffix in CIT_CATS.items()}
    log.info("  [CIT] %s — %d category RICs", comm, len(ric_to_col))
    df_cat = _batch_history(ld, ric_to_col, "COMM_LAST", start, end)

    oi_ric = f"3CFTC{code}OI"
    log.info("  [CIT] %s — Total OI (%s)", comm, oi_ric)
    df_oi = _batch_history(ld, {oi_ric: "Total OI"}, "COMM_LAST", start, end)

    log.info("  [CIT] %s — Px (%s)", comm, cfg["px"])
    df_px = _batch_history(ld, {cfg["px"]: "Px"}, "TRDPRC_1", start, end)

    df = df_cat.join(df_oi, how="outer").join(df_px, how="left")
    for col in CIT_GAP_COLS:
        df[col] = float("nan")
    for col in CIT_POS_FOR_PCT:
        if col in df.columns:
            df[f"Pct OI {col}"] = df[col] / df["Total OI"] * 100

    df.index.name = "Date"
    df = df.reset_index()
    df.insert(0, "Commodity", comm)
    df = df.dropna(subset=[c for c in CIT_POS_FOR_PCT if c in df.columns], how="all")
    for col in CIT_FINAL_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    log.info("  [CIT] %s -> %d rows", comm, len(df))
    return df[CIT_FINAL_COLS]


def fetch_disagg_commodity(ld, comm: str, cfg: dict, prefix: str, start: str, end: str) -> pd.DataFrame:
    kind, code = cfg["kind"], cfg["code"]
    field = FIELD_BY_KIND[kind]
    root = f"{prefix}{_root(kind, code)}"
    cftc_root = f"{prefix}{_cftc_root(kind, code)}"

    ric_to_col = {f"{root}{suffix}": col for col, suffix in DISAGG_CATS.items()}
    log.info("  [Disagg-%s] %s — %d category RICs (field=%s)",
             "COMB" if prefix == "3" else "FUT", comm, len(ric_to_col), field)
    df_cat = _batch_history(ld, ric_to_col, field, start, end)

    oi_ric = f"{cftc_root}OI"
    tt_ric = f"{cftc_root}TT"
    log.info("  [Disagg] %s — Total OI (%s) / Traders Total (%s)", comm, oi_ric, tt_ric)
    df_extra = _batch_history(ld, {oi_ric: "Total OI", tt_ric: "Traders Total"}, field, start, end)

    log.info("  [Disagg] %s — Px (%s)", comm, cfg["px"])
    df_px = _batch_history(ld, {cfg["px"]: "Px"}, "TRDPRC_1", start, end)

    df = df_cat.join(df_extra, how="outer").join(df_px, how="left")
    for col in DISAGG_GAP_COLS:
        df[col] = float("nan")
    for col in DISAGG_POS_FOR_PCT:
        if col in df.columns:
            df[f"Pct OI {col}"] = df[col] / df["Total OI"] * 100

    df.index.name = "Date"
    df = df.reset_index()
    df.insert(0, "Crop", "All")
    df.insert(0, "Commodity", comm)
    df = df.dropna(subset=[c for c in DISAGG_POS_FOR_PCT if c in df.columns], how="all")
    for col in DISAGG_FINAL_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    log.info("  [Disagg-%s] %s -> %d rows", "COMB" if prefix == "3" else "FUT", comm, len(df))
    return df[DISAGG_FINAL_COLS]


# ── Old/New(Other) crop split ────────────────────────────────────────────────
# Discovered live 2026-08-31: the standard Disagg category RICs (e.g.
# 3033661PLNG = Cotton COMB Producer Long) carry two companion fields beyond
# the usual COMM_LAST ("All" crop, what fetch_disagg_commodity() above pulls):
# MET_VAL = "Old" crop row, MET_VAL2 = "Other" crop row (CFTC's own report
# labels the non-old row "Other", not "New"). Verified byte-for-byte against
# the official CFTC report text (Cotton COMB, 2026-08-25 — All/Old/Other
# rows for Producer/Swap/MM/Other Reportables/Non-Rep Long+Short all matched
# exactly) and confirmed both fields support full get_history() backfill
# back to 2010-01-05 (869 daily rows) — this is a genuine LSEG historical
# archive, not a live-only snapshot (an earlier probe wrongly concluded no
# crop history existed at all, based on a different, live-only field
# (GEN_VAL2/3) that happens to hold the same current value but rejects any
# get_history() date range; MET_VAL/MET_VAL2 do not have that restriction).
#
# One real gap found and verified live, not assumed:
#   - Only works for "cftc"-kind commodities (KC/CC/SB/CT) — tested on a
#     "lif"-kind RIC (RC) and LSEG rejected the field outright. Old/New crop
#     is a US CFTC-specific concept; RC/LCC/LSU (London/ICE Europe) don't
#     carry it. No crop split fetched for those 3.
#
# Total OI has NO crop split via this mechanism either (tested MET_VAL,
# MET_VAL2, GEN_VAL2, GEN_VAL3 — all return null on the OI RIC), but it IS
# fully derivable from the category data we do have: OI = sum of every
# category's gross long side, where Swap/MM/Other Reportables' "Long"
# figure excludes their Spread leg (reported as a separate number), so
# Spread has to be added back in per category. See the derivation and its
# verification in fetch_disagg_commodity_crop() below.
CROP_FIELD = {"Old": "MET_VAL", "Other": "MET_VAL2"}

def fetch_disagg_commodity_crop(ld, comm: str, cfg: dict, prefix: str, crop: str,
                                start: str, end: str) -> pd.DataFrame:
    kind, code = cfg["kind"], cfg["code"]
    if kind != "cftc":
        return pd.DataFrame()  # crop split not available for lif-kind (RC/LCC/LSU) — verified live

    root = f"{prefix}{code}"
    field = CROP_FIELD[crop]
    ric_to_col = {f"{root}{suffix}": col for col, suffix in DISAGG_CATS.items()}
    log.info("  [Disagg-%s-%s] %s — %d category RICs (field=%s)",
             "COMB" if prefix == "3" else "FUT", crop, comm, len(ric_to_col), field)
    df = _batch_history(ld, ric_to_col, field, start, end)
    if df.empty:
        return pd.DataFrame()

    for col in DISAGG_GAP_COLS:
        df[col] = float("nan")
    # Total OI itself has no crop split on LSEG (verified live — see comment
    # above), but it's fully DERIVABLE from category data we do have: OI =
    # sum of each category's gross long side, where Swap/MM/Other's "Long"
    # figure excludes their Spread leg (reported separately), so Spread has
    # to be added back in. Verified against the real "All"-crop Total OI
    # (2026-08-31): matches to within +/-1 contract on the All row itself,
    # and Old-derived + Other-derived reproduces the real All OI to within
    # +/-3 contracts on SB/GC (869/869 weeks) — pure rounding, not an
    # approximation error.
    oi_parts = ["Producer Long", "Swap Long", "Swap Spread", "MM Long", "MM Spread",
                "Other Long", "Other Spread", "Non Rep Long"]
    df["Total OI"] = sum(df[c] for c in oi_parts if c in df.columns)
    df["Traders Total"] = float("nan")  # trader COUNTS aren't derivable from position sums
    for col in DISAGG_POS_FOR_PCT:
        if col in df.columns:
            df[f"Pct OI {col}"] = df[col] / df["Total OI"] * 100

    df.index.name = "Date"
    df = df.reset_index()
    df.insert(0, "Crop", crop)
    df.insert(0, "Commodity", comm)
    df = df.dropna(subset=[c for c in DISAGG_POS_FOR_PCT if c in df.columns], how="all")
    for col in DISAGG_FINAL_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    log.info("  [Disagg-%s-%s] %s -> %d rows", "COMB" if prefix == "3" else "FUT", crop, comm, len(df))
    return df[DISAGG_FINAL_COLS]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="COT CFTC Full History Backfill (LSEG)")
    parser.add_argument("--start", default=START_FULL)
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("COT LSEG Backfill  |  %s  |  start=%s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), args.start)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    import lseg.data as ld
    ld.open_session()
    log.info("LSEG session opened.")

    try:
        # ── CIT ──────────────────────────────────────────────────────────────
        log.info("--- CIT ---")
        cit_frames = []
        for comm, cfg in CIT_COMMODITIES.items():
            try:
                cit_frames.append(fetch_cit_commodity(ld, comm, cfg, args.start, today))
            except Exception as e:
                log.error("  ERROR fetching CIT %s: %s", comm, e)
        cit_df = pd.concat(cit_frames, ignore_index=True).sort_values(["Commodity", "Date"]).reset_index(drop=True)
        cit_df.to_parquet(CIT_FILE, engine="pyarrow", index=False)
        log.info("CIT saved -> %s | %d rows", CIT_FILE, len(cit_df))

        # ── Disaggregated: Futures+Options combined ─────────────────────────
        log.info("--- Disaggregated (Futures+Options combined) ---")
        futopt_frames = []
        for comm, cfg in DISAGG_COMMODITIES.items():
            try:
                futopt_frames.append(fetch_disagg_commodity(ld, comm, cfg, "3", args.start, today))
            except Exception as e:
                log.error("  ERROR fetching Disagg-COMB %s: %s", comm, e)
            for crop in ("Old", "Other"):
                try:
                    futopt_frames.append(fetch_disagg_commodity_crop(ld, comm, cfg, "3", crop, args.start, today))
                except Exception as e:
                    log.error("  ERROR fetching Disagg-COMB-%s %s: %s", crop, comm, e)
        futopt_frames = [f for f in futopt_frames if f is not None and not f.empty]
        futopt_df = pd.concat(futopt_frames, ignore_index=True).sort_values(["Commodity", "Date"]).reset_index(drop=True)
        futopt_df.to_parquet(DISAGG_FUTOPT_FILE, engine="pyarrow", index=False)
        log.info("Disagg FutOpt saved -> %s | %d rows", DISAGG_FUTOPT_FILE, len(futopt_df))

        # ── Disaggregated: Futures-only ──────────────────────────────────────
        log.info("--- Disaggregated (Futures-only) ---")
        fut_frames = []
        for comm, cfg in DISAGG_COMMODITIES.items():
            try:
                fut_frames.append(fetch_disagg_commodity(ld, comm, cfg, "1", args.start, today))
            except Exception as e:
                log.error("  ERROR fetching Disagg-FUT %s: %s", comm, e)
            for crop in ("Old", "Other"):
                try:
                    fut_frames.append(fetch_disagg_commodity_crop(ld, comm, cfg, "1", crop, args.start, today))
                except Exception as e:
                    log.error("  ERROR fetching Disagg-FUT-%s %s: %s", crop, comm, e)
        fut_frames = [f for f in fut_frames if f is not None and not f.empty]
        fut_df = pd.concat(fut_frames, ignore_index=True).sort_values(["Commodity", "Date"]).reset_index(drop=True)
        fut_df.to_parquet(DISAGG_FUT_FILE, engine="pyarrow", index=False)
        log.info("Disagg Fut saved -> %s | %d rows", DISAGG_FUT_FILE, len(fut_df))

    finally:
        ld.close_session()
        log.info("LSEG session closed.")

    log.info("=" * 70)


if __name__ == "__main__":
    main()
