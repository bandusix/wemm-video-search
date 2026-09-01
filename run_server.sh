#!/bin/bash
# 统一版 v3：本地文件 + 在线 m3u8 + 判重 + 代理，端口 8765
# torchcodec 需要在进程启动时就能找到 Homebrew ffmpeg 的动态库
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn server_v3:app --host 127.0.0.1 --port 8765
