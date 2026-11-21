@echo off
rem SO-101 simulator launcher for Windows. Double-click it, or run it from a terminal
rem with extra options, e.g.  run_app.bat --randomize full --dataset my_demos
rem The first run creates .venv here and installs everything (a few minutes, ~2 GB).
rem Set SO101_SIM_COMMAND to run another command, e.g. "check" or "generate".
setlocal
if not defined SO101_SIM_COMMAND set "SO101_SIM_COMMAND=app"
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto run

echo Setting up the Python environment (first run only)...
set "PYCMD="
where py >nul 2>nul && py -3.12 -c "import sys" >nul 2>nul && set "PYCMD=py -3.12"
if not defined PYCMD (
  where python >nul 2>nul && python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 12) else 1)" >nul 2>nul && set "PYCMD=python"
)
if not defined PYCMD (
  echo.
  echo Python 3.12 was not found. Install it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH"^), then run this file again.
  if not defined CI pause
  exit /b 1
)
%PYCMD% -m venv .venv || goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto fail
".venv\Scripts\python.exe" -m pip install -e ".[record]" || goto fail

:run
".venv\Scripts\python.exe" -m so101_sim %SO101_SIM_COMMAND% %*
if errorlevel 1 (
  if not defined CI pause
  exit /b 1
)
exit /b 0

:fail
echo.
echo Setup failed. Delete the .venv folder and try again; the messages above say why.
if not defined CI pause
exit /b 1
