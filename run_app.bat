@echo off
cd /d "%~dp0"
title Vial Inspector - dashboard
call _setup.bat
if "%PY%"=="" exit /b 1
if exist nvidia_key.txt set /p NVIDIA_API_KEY=<nvidia_key.txt
if exist "%USERPROFILE%\.streamlit\credentials.toml" goto credok
mkdir "%USERPROFILE%\.streamlit" 2>nul
>"%USERPROFILE%\.streamlit\credentials.toml" echo [general]
>>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
:credok
echo Starting the website - it opens in your browser at http://localhost:8501
"%PY%" -m streamlit run app.py
pause
