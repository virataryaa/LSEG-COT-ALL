@echo off
:: DAILY - Rollex + Roll Yield sync only (no COT ingest, no LSEG session needed)
:: Schedule: daily
:: Adapted from ICEBREAKER/COT_ALL/Automator/run_daily.bat. COT parquets only
:: refresh weekly (via run.bat, Fridays after CFTC release) - this script
:: just keeps COT_ALL's copies of Rollex/Roll Yield fresh day to day, since
:: those two source projects update daily on their own schedules.
set PYTHONIOENCODING=utf-8
set PYTHON=C:\Users\virat.arya\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe
set LOG=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Automator\run_daily_log.txt
set REPO=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL
set GIT_STATUS=skipped

:: Prevent Git Credential Manager from showing an interactive dialog in unattended runs.
:: If credentials are cached it pushes silently; if not, it fails immediately instead of hanging.
set GCM_INTERACTIVE=never
set GIT_TERMINAL_PROMPT=0
echo. >> "%LOG%"
echo ============================= >> "%LOG%"
echo Daily run started: %date% %time% >> "%LOG%"
echo ============================= >> "%LOG%"

:: Step 1 - Sync Rollex parquets from the Rollex (LSEG) master database
echo [1] Syncing Rollex parquets... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Rollex\Database\rollex_*.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Database\Rollex\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Rollex sync had issues >> "%LOG%"

:: Step 2 - Sync Roll Yield parquet from the Roll Yield (LSEG) master database
echo [2] Syncing Roll Yield parquet... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Roll Yield\Database\roll_yield_data.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Database\RollYield\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Roll Yield sync had issues >> "%LOG%"

:: Step 2b - Sync USDBRL parquet (feeds the Spec Prediction overview's BRL panel)
echo [2b] Syncing USDBRL parquet... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Roll Yield\Database\fx_brl.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Database\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: USDBRL sync had issues >> "%LOG%"

:: Step 2c - Sync daily Futures OI parquets (feeds the Spec Prediction overview's
:: live OI Change columns; source project updates these daily on its own schedule)
echo [2c] Syncing Futures OI parquets... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Futures\Database\*_futures.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Database\Futures\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Futures OI sync had issues >> "%LOG%"

:: Step 3 - Push updated parquets to GitHub
:: (spec_prediction_log.parquet is included so Awaiting/Resolved state on the
:: Spec Prediction overview survives Streamlit Cloud redeploys, which wipe
:: the filesystem; it's written locally by spec_prediction_overview.py on
:: every dashboard run, so this just picks up whatever's there.)
echo [3] Pushing to GitHub... >> "%LOG%"
cd /d "%REPO%"
git add Database\Rollex\rollex_*.parquet Database\RollYield\roll_yield_data.parquet Database\fx_brl.parquet Database\Futures\*_futures.parquet Database\spec_prediction_log.parquet >> "%LOG%" 2>&1
git diff --cached --quiet
if %ERRORLEVEL% NEQ 0 (
    git -c core.askpass= commit -m "Daily Rollex/RollYield/BRL/Futures sync: %date%" >> "%LOG%" 2>&1
    git -c core.askpass= pull --rebase --autostash >> "%LOG%" 2>&1
    git -c core.askpass= push >> "%LOG%" 2>&1
    if not errorlevel 1 (
        set GIT_STATUS=pushed
        echo Git push done. >> "%LOG%"
    ) else (
        set GIT_STATUS=failed
        echo ERROR: git push failed >> "%LOG%"
    )
) else (
    echo No changes to commit. >> "%LOG%"
    set GIT_STATUS=skipped
)

:: Step 4 - Email notification (daily mode - Rollex/RollYield-focused body)
echo [4] Sending notification... >> "%LOG%"
"%PYTHON%" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Automator\notify.py" ok %GIT_STATUS% daily >> "%LOG%" 2>&1

echo Daily run finished: %date% %time% >> "%LOG%"
