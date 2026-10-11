#!/bin/bash
# Double-click on macOS (or run ./run.command) to start the app.
cd "$(dirname "$0")"
[ -d venv ] || { python3 -m venv venv && source venv/bin/activate && python -m pip install -r requirements.txt; }
source venv/bin/activate
python app.py
