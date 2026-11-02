@echo off
cd /d "%~dp0"
title Reading the digital twin
set OMNI_KIT_ACCEPT_EULA=YES
if not exist C:\isaac_env\Scripts\python.exe (echo Isaac Sim is not installed yet - run setup_isaac.bat first. & pause & exit /b 1)
echo Opening the digital twin in Isaac Sim (first start can take a few minutes)...
C:\isaac_env\Scripts\python.exe twin_inventory.py > inventory_console.txt 2>&1
echo Finished. Tell Claude "inventory done". Closing in 20 seconds.
timeout /t 20
