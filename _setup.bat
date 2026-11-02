@echo off
rem Finds a Python that has everything installed; on a new computer it sets one up once in .venv
set PY=
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
  goto :eof
)
python --version >nul 2>&1
if errorlevel 1 goto nopython
python -c "import streamlit, cv2, pandas, requests, PIL, numpy" >nul 2>&1
if not errorlevel 1 (
  set "PY=python"
  goto :eof
)
echo First run on this computer: installing what the app needs. This takes 1 to 3 minutes...
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
set "PY=.venv\Scripts\python.exe"
echo Setup finished.
goto :eof
:nopython
echo.
echo Python is not installed. Install Python 3.10 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer, then double-click run_app.bat again.
echo.
pause
exit /b 1
:failed
echo.
echo Setup failed - check the internet connection, then double-click run_app.bat again.
if exist .venv rmdir /s /q .venv
echo.
pause
exit /b 1
