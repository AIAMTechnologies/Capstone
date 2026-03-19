#!/bin/bash
cd "$(dirname "$0")/backend"
source /Users/ammaralam/Documents/lap_portal_v3_nov6/.venv/bin/activate
exec python -m uvicorn main:app --reload --port "${PORT:-8000}"
