#!/bin/bash
# 并行抽帧分支的验证实例：端口 8767 + 独立数据目录，不干扰 8766
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
export WEMM_DATA=webapp_data_v3
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn server_v2:app --host 127.0.0.1 --port 8767
