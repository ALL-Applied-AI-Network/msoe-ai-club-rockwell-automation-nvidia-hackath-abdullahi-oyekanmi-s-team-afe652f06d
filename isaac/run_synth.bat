@echo off
cd /d "%~dp0"
title Generating synthetic images in Isaac Sim - leave this window open
set OMNI_KIT_ACCEPT_EULA=YES
if not exist C:\isaac_env\Scripts\python.exe (echo Isaac Sim is not installed yet - run setup_isaac.bat first. & pause & exit /b 1)
if not exist logs mkdir logs
echo Rendering synthetic CapSealCam images from the digital twin (starts a NEW set).
echo Leave this window OPEN until it says Finished - it restarts Isaac Sim by itself after a crash.
echo If a white "Assertion failed" box pops up, click Ignore.
taskkill /f /im py-spy.exe >nul 2>&1
taskkill /f /im crashpad_handler.exe >nul 2>&1
C:\isaac_env\Scripts\python.exe gen_synth.py --fresh %* > logs\console_0.txt 2>&1
call "%~dp0resume_synth.bat" %*
