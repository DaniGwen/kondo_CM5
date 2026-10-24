@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_training_ui.ps1"
if errorlevel 1 pause
