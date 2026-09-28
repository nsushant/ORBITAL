@echo off
cd /d "%~dp0.."
python -m SLA.mvp.server
if errorlevel 1 pause
