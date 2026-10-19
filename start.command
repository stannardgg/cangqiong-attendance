#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
    echo "[Cangqiong Attendance] First run: initializing environment, please wait..."
    if ! command -v python3 >/dev/null 2>&1; then
        echo ""
        echo "[ERROR] Python 3.11 or newer not found."
        echo "Please install Python from https://www.python.org/downloads/ then run this again."
        echo ""
        exit 1
    fi
    python3 -m venv .venv
    ".venv/bin/python" -m pip install --upgrade pip -q
    ".venv/bin/python" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

".venv/bin/python" run.py
