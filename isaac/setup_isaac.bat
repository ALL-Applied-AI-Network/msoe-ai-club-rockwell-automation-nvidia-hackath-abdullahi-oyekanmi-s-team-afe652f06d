@echo off
setlocal
cd /d "%~dp0"
title Isaac Sim setup for Vial Inspector
set "LOG=%~dp0setup_log.txt"
set "PIPLOG=%~dp0pip_log.txt"
echo ==== Isaac Sim setup %date% %time% > "%LOG%"
echo.
echo  This installs NVIDIA Isaac Sim 6.0.1 into its own environment (C:\isaac_env).
echo  Your normal Python is not touched. Download is large (about 20-30 GB) and can take 30-60 minutes.
echo.
echo  Isaac Sim requires accepting the NVIDIA Omniverse License Agreement:
echo    (the full text is on NVIDIA's Isaac Sim / Omniverse website)
set /p EULA= Type Y to accept the license and continue, anything else to stop: 
if /i not "%EULA%"=="Y" (echo Stopped - license not accepted. & echo EULA_DECLINED>>"%LOG%" & pause & exit /b 1)
setx OMNI_KIT_ACCEPT_EULA YES >nul
set OMNI_KIT_ACCEPT_EULA=YES
echo EULA_ACCEPTED_BY_USER>>"%LOG%"
echo.
echo [1/6] Checking GPU, driver, memory...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> "%LOG%" 2>&1
systeminfo | findstr /C:"OS Name" /C:"Total Physical Memory" >> "%LOG%" 2>&1
reg query HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled >> "%LOG%" 2>&1
echo [2/6] Windows long paths (needed by Isaac Sim) and Python 3.12...
reg query HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled | find "0x1" >nul
if errorlevel 1 (
  echo     Turning on Windows long paths - click YES on the admin prompt...
  powershell -NoProfile -Command "Start-Process reg -ArgumentList 'add','HKLM\SYSTEM\CurrentControlSet\Control\FileSystem','/v','LongPathsEnabled','/t','REG_DWORD','/d','1','/f' -Verb RunAs -Wait"
)
reg query HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled >> "%LOG%" 2>&1
set "P312=C:\Python312\python.exe"
if not exist "%P312%" (
  echo     Downloading the official Python 3.12 installer from python.org...
  powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile \"$env:TEMP\python-3.12.10-amd64.exe\""
  echo     Installing Python 3.12 into C:\Python312 - this takes a minute...
  "%TEMP%\python-3.12.10-amd64.exe" /quiet InstallAllUsers=0 TargetDir=C:\Python312 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0
)
if not exist "%P312%" (echo PYTHON312_MISSING>>"%LOG%" & echo Could not install Python 3.12 - tell Claude. & pause & exit /b 1)
"%P312%" --version >> "%LOG%" 2>&1
echo [3/6] Creating C:\isaac_env ...
if not exist C:\isaac_env\Scripts\python.exe "%P312%" -m venv C:\isaac_env >> "%LOG%" 2>&1
set "PY=C:\isaac_env\Scripts\python.exe"
if not exist "%PY%" (echo VENV_MISSING>>"%LOG%" & echo Could not create C:\isaac_env - tell Claude. & pause & exit /b 1)
"%PY%" -m pip install --upgrade pip --log "%PIPLOG%"
echo [4/6] PyTorch for CUDA 12.8 (large)...
"%PY%" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --log "%PIPLOG%"
echo [5/6] Isaac Sim 6.0.1 (very large - leave this window open)...
"%PY%" -m pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --log "%PIPLOG%"
"%PY%" -m pip install opencv-python-headless pillow --log "%PIPLOG%"
echo [6/6] Unpacking the digital twin to C:\csi_twin ...
if not exist C:\csi_twin\csi_digital_twin_v1\README.md powershell -NoProfile -Command "Expand-Archive -Force -LiteralPath (Join-Path $env:USERPROFILE 'Downloads\Copy of csi_digital_twin.zip') -DestinationPath 'C:\csi_twin'" >> "%LOG%" 2>&1
if exist C:\csi_twin\csi_digital_twin_v1\README.md (echo TWIN_OK>>"%LOG%") else (echo TWIN_MISSING>>"%LOG%")
"%PY%" -c "import torch; ok=torch.cuda.is_available(); print('TORCH_OK', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else '-')" >> "%LOG%" 2>&1
"%PY%" -c "import isaacsim, sys; print('ISAACSIM_OK', sys.version.split()[0])" >> "%LOG%" 2>&1
echo SETUP_FINISHED %date% %time%>>"%LOG%"
echo.
echo Done. Tell Claude "setup done". This window closes in 60 seconds.
timeout /t 60
