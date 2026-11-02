@echo off
cd /d "%~dp0"
if exist nvidia_key.txt del nvidia_key.txt
echo Old key removed.
call run_test.bat %*
