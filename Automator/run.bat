@echo off
:: WEEKLY - COT ingest (LSEG) + Rollex sync + Roll Yield sync + git push + email
:: Schedule: Friday, after CFTC release
:: Adapted from ICEBREAKER/COT_ALL/Automator/run.bat. Rollex and Roll Yield
:: are both now built (see the sibling LSEG/Rollex and
:: LSEG/Roll Yield projects) and synced in below via xcopy,
:: same as the ICE original's run_daily.bat does for both.
set PYTHONIOENCODING=utf-8
set PYTHON=C:\Users\virat.arya\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe
set LOG=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Automator\run_log.txt
set REPO=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL
set INGEST_STATUS=ok
set GIT_STATUS=skipped

set GCM_INTERACTIVE=never
set GIT_TERMINAL_PROMPT=0
echo. >> "%LOG%"
echo ============================= >> "%LOG%"
echo Run started: %date% %time% >> "%LOG%"
echo ============================= >> "%LOG%"

:: Step 1 - Incremental COT ingest (LSEG, all 7 commodities)
echo [1] Running cot_lseg_ingest.py... >> "%LOG%"
"%PYTHON%" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Code\cot_lseg_ingest.py" >> "%LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cot_lseg_ingest.py failed >> "%LOG%"
    set INGEST_STATUS=error
    goto notify
)

:: Step 1b - Sync Rollex parquets from the Rollex (LSEG) master database
echo [1b] Syncing Rollex parquets... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Rollex\Database\rollex_*.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Database\Rollex\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Rollex sync had issues >> "%LOG%"

:: Step 1c - Sync Roll Yield parquet from the Roll Yield (LSEG) master database
echo [1c] Syncing Roll Yield parquet... >> "%LOG%"
xcopy /Y "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Roll Yield\Database\roll_yield_data.parquet" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Database\RollYield\" >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 echo WARNING: Roll Yield sync had issues >> "%LOG%"

:: Step 2 - Push updated parquets to GitHub
echo [2] Pushing to GitHub... >> "%LOG%"
cd /d "%REPO%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cd to LSEG COT_ALL repo failed >> "%LOG%"
    set GIT_STATUS=failed
    goto notify
)
git add Database\cot_cit.parquet Database\cot_disagg_futopt.parquet Database\cot_disagg_fut.parquet Database\Rollex\rollex_*.parquet Database\RollYield\roll_yield_data.parquet >> "%LOG%" 2>&1
git diff --cached --quiet >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    git -c core.askpass= commit -m "Auto update: COT_ALL (LSEG) %date%" >> "%LOG%" 2>&1
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

:notify
echo [3] Sending email notification... >> "%LOG%"
"%PYTHON%" "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\COT_ALL\Automator\notify.py" %INGEST_STATUS% %GIT_STATUS% >> "%LOG%" 2>&1

echo Run finished: %date% %time% >> "%LOG%"
