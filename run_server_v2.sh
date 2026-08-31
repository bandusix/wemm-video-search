#!/bin/bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn server_v2:app --host 127.0.0.1 --port 8766
