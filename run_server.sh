#!/bin/bash
# torchcodec 需要在进程启动时就能找到 Homebrew ffmpeg 的动态库
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8765
