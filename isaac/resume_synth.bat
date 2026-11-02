@echo off
cd /d "%~dp0"
title Continuing synthetic images in Isaac Sim - leave this window open
set OMNI_KIT_ACCEPT_EULA=YES
if not exist logs mkdir logs
echo Continuing from the last saved image. Leave this window OPEN until it says Finished.
echo (Isaac Sim may crash now and then inside the NVIDIA driver - this window restarts it by itself.)
set tries=0
:loop
findstr /c:"RENDER_DONE" synth_status.txt >nul && goto fin
set /a tries+=1
if %tries% geq 80 goto fin
taskkill /f /im py-spy.exe >nul 2>&1
taskkill /f /im crashpad_handler.exe >nul 2>&1
echo Starting Isaac Sim (run %tries%)...
C:\isaac_env\Scripts\python.exe gen_synth.py %* > logs\console_%tries%.txt 2>&1
goto loop
:fin
taskkill /f /im py-spy.exe >nul 2>&1
taskkill /f /im crashpad_handler.exe >nul 2>&1
echo Applying the real camera lens and packing the images...
C:\isaac_env\Scripts\python.exe finalize_synth.py > logs\finalize.txt 2>&1
findstr /c:"DONE" synth_status.txt
echo.
echo Finished. Tell Claude "synth done". Closing in 30 seconds.
timeout /t 30
