@echo off
cd /d "%~dp0"
title Vial Inspector - batch test
call _setup.bat
if "%PY%"=="" exit /b 1
if not exist nvidia_key.txt (
  echo First run: paste your NVIDIA API key, then press Enter. It is saved only in this folder.
  set /p KEY=NVIDIA key: 
  call echo %%KEY%%> nvidia_key.txt
)
"%PY%" batch_eval.py %*
echo.
echo Finished. This window closes in 20 seconds.
timeout /t 20
