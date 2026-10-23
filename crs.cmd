@echo off
setlocal
rem ---------------------------------------------------------------
rem  CRS - local AI agent launcher
rem  Usage:  crs                 start an interactive session
rem           crs "do this task"  run a single task and exit
rem           crs --demo build-file
rem ---------------------------------------------------------------
cd /d "%~dp0"
if not exist ".\venv\Scripts\python.exe" (
    echo CRS: virtual environment not found at .\venv
    echo      create it with:  python -m venv venv
    echo      then install:    .\venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)
".\venv\Scripts\python.exe" -u run.py %*
endlocal
