@echo off
title Cangqiong Attendance - PUBLIC mode
cd /d "%~dp0"

echo ============================================================
echo   Cangqiong Attendance  -  PUBLIC mode
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Environment not found.
    echo Please run  start.bat  once first to set it up, then run this again.
    echo.
    pause
    exit /b 1
)

echo [1/2] Starting the local service on port 8000 ...
start "Cangqiong Backend" /min ".venv\Scripts\python.exe" run.py
timeout /t 4 /nobreak >nul

echo [2/2] Opening a public HTTPS tunnel ...
echo.
echo ------------------------------------------------------------
echo   Your PUBLIC address will appear a few lines below.
echo   Look for a line containing  https://
echo   Open THAT address on your phone.
echo.
echo   - Phones can sign in (camera works, because it is HTTPS).
echo   - Phones CANNOT add or change face data. That is by design,
echo     and this computer is the only place that can do it.
echo.
echo   Keep this window OPEN. Closing it kills the public link.
echo   Press Ctrl+C to stop the tunnel.
echo ------------------------------------------------------------
echo.

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -R 80:localhost:8000 serveo.net

echo.
echo Tunnel closed. The public link is no longer available.
pause
