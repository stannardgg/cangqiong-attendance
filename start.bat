@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [Cangqiong Attendance] First run: initializing environment, please wait...
    where python >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [ERROR] Python 3.11 or newer not found.
        echo Please install Python from https://www.python.org/downloads/
        echo and check "Add Python to PATH" during installation, then run this again.
        echo.
        pause
        exit /b 1
    )
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

".venv\Scripts\python.exe" run.py
pause
