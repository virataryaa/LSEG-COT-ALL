"""
notify.py — COT_ALL (LSEG) Automator email summary
Called by run.bat (weekly) or run_daily.bat (daily Rollex/RollYield sync).
Usage: python notify.py <status> <git_status> [mode]
  status     : ok | error
  git_status : pushed | skipped | failed
  mode       : weekly (default) | daily

Adapted from ICEBREAKER/COT_ALL/Automator/notify.py — same logic, repointed
at the LSEG folder.
"""

import sys
import datetime
import pandas as pd
from pathlib import Path

TO_EMAIL      = "virat.arya@etgworld.com"
DB_DIR        = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Database")
AUTOMATOR_DIR = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Automator")
ROLLEX_DIR    = DB_DIR / "Rollex"
ROLLYIELD_DIR = DB_DIR / "RollYield"

CIT_FILE       = DB_DIR / "cot_cit.parquet"
DISAGG_FO_FILE = DB_DIR / "cot_disagg_futopt.parquet"
DISAGG_F_FILE  = DB_DIR / "cot_disagg_fut.parquet"

status     = sys.argv[1] if len(sys.argv) > 1 else "ok"
git_status = sys.argv[2] if len(sys.argv) > 2 else "unknown"
mode       = sys.argv[3] if len(sys.argv) > 3 else "weekly"
run_dt     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
today      = datetime.date.today().strftime("%Y-%m-%d")


def staleness_warnings(path: Path, label: str, group_by=None) -> list[str]:
    if not path.exists():
        return []
    group_by = group_by or ["Commodity"]
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    max_date  = df["Date"].max()
    threshold = max_date - pd.Timedelta(days=14)
    warnings  = []
    for keys, grp in df.groupby(group_by):
        latest = grp["Date"].max()
        if latest < threshold:
            key_str = " / ".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
            warnings.append(f"  STALE [{label}] {key_str:<25}  latest {latest.date()}  (max {max_date.date()})")
    return warnings


def parquet_summary(path: Path, label: str, group_by=None) -> str:
    if not path.exists():
        return f"  {label}: FILE NOT FOUND\n"
    group_by = group_by or ["Commodity"]
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    lines = [f"  {label}:"]
    for keys, grp in df.groupby(group_by):
        key_str = "  ".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
        lines.append(f"    {key_str:<25}  {len(grp):>5} rows   {grp['Date'].min().date()} -> {grp['Date'].max().date()}")
    return "\n".join(lines) + "\n"


def _max_date(df: pd.DataFrame):
    """Find the max date in a parquet whether Date is a column or the index."""
    if "Date" in df.columns:
        return pd.to_datetime(df["Date"]).max().date()
    if df.index.name == "Date":
        return pd.to_datetime(df.index).max().date()
    return "n/a"


def rollex_sync_summary() -> str:
    """List Rollex + RollYield parquets with file mtime and data max-date."""
    files = sorted(ROLLEX_DIR.glob("rollex_*.parquet")) + sorted(ROLLYIELD_DIR.glob("*.parquet"))
    if not files:
        return "  (no synced files found)\n"
    lines = []
    for p in files:
        mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size_kb = p.stat().st_size / 1024
        try:
            max_date = _max_date(pd.read_parquet(p))
        except Exception:
            max_date = "n/a"
        rel = p.relative_to(DB_DIR).as_posix()
        lines.append(f"    {rel:<32}  file mtime {mtime}   data->{max_date}   {size_kb:>6.0f} KB")
    return "\n".join(lines) + "\n"


def send_outlook_email(subject: str, body: str):
    try:
        import win32com.client
        outlook      = win32com.client.Dispatch("Outlook.Application")
        mail         = outlook.CreateItem(0)
        mail.To      = TO_EMAIL
        mail.Subject = subject
        mail.Body    = body
        mail.Send()
        print(f"  Email sent -> {TO_EMAIL}")
    except Exception as e:
        print(f"  Email failed: {e}")


ok = status == "ok"

git_line = {
    "pushed":  "GitHub  : Pushed successfully",
    "skipped": "GitHub  : No changes — push skipped",
    "failed":  "GitHub  : PUSH FAILED",
}.get(git_status, f"GitHub  : {git_status}")

if mode == "daily":
    # ── DAILY ROLLEX/ROLLYIELD SYNC EMAIL ──────────────────────────────────────
    tag = "[ERROR]" if not ok else ("[WARN-PUSH]" if git_status == "failed" else "[DAILY]")
    subject = f"{tag} COT_ALL (LSEG) Rollex/RollYield Daily Sync — {today}"

    body = f"""LSEG COT_ALL (LSEG) — DAILY ROLLEX/ROLLYIELD SYNC
Run time : {run_dt}
Status   : {"OK" if ok else "ERROR — check run_daily_log.txt"}
{git_line}

{"=" * 55}
SYNCED FILES (Rollex + Roll Yield)
{"=" * 55}
{rollex_sync_summary()}
{"=" * 55}
Note: This is the DAILY sync (xcopy only). COT parquets are NOT touched here —
those refresh weekly via run.bat on Fridays after CFTC release.

Log: C:\\Users\\virat.arya\\ETG\\SoftsDatabase - Documents\\Database\\Hardmine\\LSEG\\COT_ALL\\Automator\\run_daily_log.txt
"""

else:
    # ── WEEKLY COT UPDATE EMAIL ────────────────────────────────────────────────
    stale_warnings = (
        staleness_warnings(CIT_FILE,       "CIT",        group_by=["Commodity"]) +
        staleness_warnings(DISAGG_FO_FILE, "DISAGG F&O", group_by=["Commodity", "Crop"]) +
        staleness_warnings(DISAGG_F_FILE,  "DISAGG Fut", group_by=["Commodity", "Crop"])
    )
    has_warnings = bool(stale_warnings)

    tag = "[ERROR]" if not ok else ("[WARNING]" if has_warnings else "[OK]")
    subject = f"{tag} LSEG-COT_ALL (LSEG) Weekly — {today}"

    warnings_block = ""
    if stale_warnings:
        lines = ["=" * 55, "WARNINGS", "=" * 55, "  Stale data detected:"] + stale_warnings
        warnings_block = "\n".join(lines) + "\n\n"

    body = f"""LSEG COT_ALL (LSEG) — WEEKLY COT UPDATE
Run time : {run_dt}
Status   : {"OK" if ok else "ERROR — ingest failed, check run_log.txt"}
{git_line}

{warnings_block}{"=" * 55}
DATABASE SUMMARY
{"=" * 55}
{parquet_summary(CIT_FILE,       "CIT       (KC / CC / SB / CT)")}
{parquet_summary(DISAGG_FO_FILE, "DISAGG F&O (LSEG: All-crop only)", group_by=["Commodity", "Crop"])}
{parquet_summary(DISAGG_F_FILE,  "DISAGG Fut (LSEG: All-crop only)", group_by=["Commodity", "Crop"])}
{"=" * 55}
Note: this is the LSEG interim-migration pipeline. Known gaps vs. the ICE
source: no Old/New crop split (LSEG has no historical series for it), no
per-category trader counts, no concentration (Conc Gross/Net 4/8). Rollex
and Roll Yield ARE synced — Rollex + Roll Yield weekly here, and daily via
the separate run_daily.bat.

Log: C:\\Users\\virat.arya\\ETG\\SoftsDatabase - Documents\\Database\\Hardmine\\LSEG\\COT_ALL\\Automator\\run_log.txt
"""

print(body)
send_outlook_email(subject, body)
