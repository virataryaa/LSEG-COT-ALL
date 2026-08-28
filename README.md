# COT_ALL — Interim Migration (LSEG)

Interim replacement for `ICEBREAKER/COT_ALL`, rebuilt against the **LSEG Data
API** (`lseg.data`) instead of ICE Connect (`icepython`), for the period while
ICE API access is unavailable.

## What's here

- **`Code/`** — `cot_lseg_backfill.py` (full-history builder) and
  `cot_lseg_ingest.py` (weekly incremental, imports the backfill script's RIC
  maps and fetch logic — no duplication).
- **`Database/`** — `cot_cit.parquet`, `cot_disagg_futopt.parquet`,
  `cot_disagg_fut.parquet`, plus `Database/Rollex/` (synced from the sibling
  `Interim_Migration/Rollex` project). Covers 7 softs (KC/CC/SB/CT/RC/LCC/
  LSU) + 3 COMEX metals (GC/SI/HG, added 2026-08-28). Metals are
  Disaggregated-only (`cot_disagg_*.parquet`) — no CIT rows, since CIT is
  CFTC's ag-only Commodity Index Supplemental report (same situation
  RC/LCC/LSU are already in; the dashboard's `CIT_COMMS` gating already
  handles it).
- **`Dashboard/`** — `cot_app.py`, copied verbatim from the ICE source (it has
  no ICE dependency at all — pure parquet consumer). Deploy this file's
  directory to Streamlit Cloud; `requirements.txt` sits alongside it.
- **`Studies/`** — six research/backtest scripts, also copied verbatim
  (parquet-only, no API dependency).
- **`Automator/`** — `run.bat` (weekly LSEG ingest + Rollex sync + git push +
  email), `notify.py`, `probe_lseg.py` (connectivity smoke test).

## Known gaps vs. the ICE source

Confirmed by direct probing of LSEG's API — not assumptions:

- **Old/New crop split** — not available as a time series on LSEG at all
  (only a live, non-historical snapshot field exists). Every Disagg row is
  written with `Crop="All"` only; no `Old`/`Other` rows.
- **Per-category trader counts** (Traders Producer/Swap/MM/Other Long/Short)
  — not published by LSEG. Only the aggregate "Total Reportable" trader count
  is available and is mapped into `Traders Tot Rept Long/Short`.
- **Concentration** (`Conc Gross/Net 4/8 Long/Short`) — not published by
  LSEG for this report.
- A small number of historical weeks (2012, 2013, part of 2015, part of 2018)
  show a data-revision discrepancy vs. the ICE archive for KC/CC/SB/CT
  specifically — traced to LSEG holding a different (likely later-revised)
  CFTC print than ICE's archived snapshot for those weeks. Not a mapping bug.

## Running it

```bash
python Code/cot_lseg_backfill.py --start 2010-01-01   # full rebuild
python Code/cot_lseg_ingest.py                          # weekly incremental
streamlit run Dashboard/cot_app.py
```

Requires an authenticated LSEG Workspace/Eikon session on the host running
the ingest scripts.
