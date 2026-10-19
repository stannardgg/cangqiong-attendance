@echo off
title Allow phone access - Cangqiong Attendance (port 8000)

net session >nul 2>&1
if errorlevel 1 (
    echo Requesting administrator rights...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo.
echo Opening TCP port 8000 for devices on the same local network...
echo.

netsh advfirewall firewall delete rule name="Cangqiong Attendance 8000" >nul 2>&1
netsh advfirewall firewall add rule name="Cangqiong Attendance 8000" dir=in action=allow protocol=TCP localport=8000 remoteip=LocalSubnet

echo Done. Now open one of these addresses in your PHONE browser
echo (the phone must be on the SAME WiFi as this computer):
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do echo     http://%%b:8000
)
echo.
echo If the phone still cannot connect, the campus/office WiFi may have
echo "client isolation" turned on. In that case ask WorkBuddy to set up
echo a tunnel instead.
echo.
pause
