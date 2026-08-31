"""
Hardmine — COT CFTC Weekly Ingest (LSEG)
==========================================
Incremental companion to cot_lseg_backfill.py — meant to run weekly (Friday,
after the CFTC release) via Automator/run.bat. Re-fetches only the current
report-year for each commodity (CFTC revises prior weeks within a year), then
merges + dedups into the existing parquet files rather than re-pulling all of
history every run.

Usage:
    python cot_lseg_ingest.py            # incremental (default)
    python cot_lseg_ingest.py --full     # delegates to cot_lseg_backfill.py's
                                          # full-history path instead
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import cot_lseg_backfill as backfill  # reuse RIC maps + fetch logic, no duplication

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "cot_lseg_ingest.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def incremental_start(path: Path, date_col: str = "Date") -> str:
    """Re-fetch from the start of the latest year on file, to pick up CFTC revisions."""
    existing = pd.read_parquet(path, columns=[date_col])
    latest = pd.to_datetime(existing[date_col]).max()
    return f"{latest.year}-01-01"


def merge_and_dedup(old: pd.DataFrame, new: pd.DataFrame, keys: list) -> pd.DataFrame:
    merged = pd.concat([old, new], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=keys, keep="last")
    merged = merged.sort_values(keys).reset_index(drop=True)
    log.info("  Dedup: %d -> %d rows (-%d)", before, len(merged), before - len(merged))
    return merged


def run_cit(ld, today: str):
    log.info("--- CIT ---")
    if not backfill.CIT_FILE.exists():
        start = backfill.START_FULL
        log.info("No existing CIT file — full history from %s", start)
    else:
        start = incremental_start(backfill.CIT_FILE)
        log.info("Incremental CIT from %s", start)

    frames = []
    for comm, cfg in backfill.CIT_COMMODITIES.items():
        try:
            frames.append(backfill.fetch_cit_commodity(ld, comm, cfg, start, today))
        except Exception as e:
            log.error("  ERROR fetching CIT %s: %s", comm, e)
    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if backfill.CIT_FILE.exists() and not new_df.empty:
        old_df = pd.read_parquet(backfill.CIT_FILE)
        final = merge_and_dedup(old_df, new_df, ["Commodity", "Date"])
    elif not new_df.empty:
        final = new_df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
    else:
        log.warning("  No new CIT data fetched — leaving existing file untouched.")
        return

    final.to_parquet(backfill.CIT_FILE, engine="pyarrow", index=False)
    log.info("CIT saved -> %s | %d rows", backfill.CIT_FILE, len(final))


def run_disagg(ld, prefix: str, path: Path, today: str):
    label = "COMB" if prefix == "3" else "FUT"
    log.info("--- Disaggregated (%s) ---", label)
    if not path.exists():
        start = backfill.START_FULL
        log.info("No existing Disagg-%s file — full history from %s", label, start)
    else:
        start = incremental_start(path)
        log.info("Incremental Disagg-%s from %s", label, start)

    frames = []
    for comm, cfg in backfill.DISAGG_COMMODITIES.items():
        try:
            frames.append(backfill.fetch_disagg_commodity(ld, comm, cfg, prefix, start, today))
        except Exception as e:
            log.error("  ERROR fetching Disagg-%s %s: %s", label, comm, e)
        for crop in ("Old", "Other"):
            try:
                frames.append(backfill.fetch_disagg_commodity_crop(ld, comm, cfg, prefix, crop, start, today))
            except Exception as e:
                log.error("  ERROR fetching Disagg-%s-%s %s: %s", label, crop, comm, e)
    frames = [f for f in frames if f is not None and not f.empty]
    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if path.exists() and not new_df.empty:
        old_df = pd.read_parquet(path)
        final = merge_and_dedup(old_df, new_df, ["Commodity", "Crop", "Date"])
    elif not new_df.empty:
        final = new_df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
    else:
        log.warning("  No new Disagg-%s data fetched — leaving existing file untouched.", label)
        return

    final.to_parquet(path, engine="pyarrow", index=False)
    log.info("Disagg-%s saved -> %s | %d rows", label, path, len(final))


def main():
    parser = argparse.ArgumentParser(description="COT CFTC Weekly Ingest (LSEG)")
    parser.add_argument("--full", action="store_true",
                         help="Full history rebuild (delegates to cot_lseg_backfill.py logic)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("COT LSEG Ingest  |  %s  |  mode=%s",
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "FULL" if args.full else "INCREMENTAL")

    import lseg.data as ld
    ld.open_session()
    log.info("LSEG session opened.")

    today = datetime.date.today().isoformat()
    try:
        if args.full:
            start = backfill.START_FULL
            cit_frames = [backfill.fetch_cit_commodity(ld, c, cfg, start, today)
                          for c, cfg in backfill.CIT_COMMODITIES.items()]
            pd.concat(cit_frames, ignore_index=True).sort_values(["Commodity", "Date"]) \
                .reset_index(drop=True).to_parquet(backfill.CIT_FILE, engine="pyarrow", index=False)

            for prefix, path in (("3", backfill.DISAGG_FUTOPT_FILE), ("1", backfill.DISAGG_FUT_FILE)):
                frames = [backfill.fetch_disagg_commodity(ld, c, cfg, prefix, start, today)
                          for c, cfg in backfill.DISAGG_COMMODITIES.items()]
                for c, cfg in backfill.DISAGG_COMMODITIES.items():
                    for crop in ("Old", "Other"):
                        frames.append(backfill.fetch_disagg_commodity_crop(ld, c, cfg, prefix, crop, start, today))
                frames = [f for f in frames if f is not None and not f.empty]
                pd.concat(frames, ignore_index=True).sort_values(["Commodity", "Date"]) \
                    .reset_index(drop=True).to_parquet(path, engine="pyarrow", index=False)
        else:
            run_cit(ld, today)
            run_disagg(ld, "3", backfill.DISAGG_FUTOPT_FILE, today)
            run_disagg(ld, "1", backfill.DISAGG_FUT_FILE, today)
    finally:
        ld.close_session()
        log.info("LSEG session closed.")

    log.info("=" * 70)


if __name__ == "__main__":
    main()
