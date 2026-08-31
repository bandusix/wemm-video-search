#!/bin/bash
# 一键对外分享：确保本地服务在跑，起 quick tunnel，打印公网地址
cd "$(dirname "$0")"

if ! curl -s -o /dev/null --max-time 2 http://127.0.0.1:8765/api/status; then
  echo "启动本地服务…"
  nohup ./run_server.sh > server.log 2>&1 &
  until curl -s -o /dev/null --max-time 2 http://127.0.0.1:8765/api/status; do sleep 2; done
fi
echo "本地服务 OK (http://localhost:8765)"

pkill -f "cloudflared tunnel" 2>/dev/null
rm -f cloudflared.log
nohup cloudflared tunnel --protocol http2 --edge-ip-version 4 --url http://localhost:8765 > cloudflared.log 2>&1 &

echo "申请公网地址…"
for i in $(seq 1 30); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" cloudflared.log | grep -v "^https://api" | head -1)
  [ -n "$URL" ] && break
  sleep 2
done

if [ -n "$URL" ]; then
  echo ""
  echo "✅ 公网地址: $URL"
  echo "   (停止分享: pkill -f 'cloudflared tunnel')"
else
  echo "❌ 隧道建立失败，看 cloudflared.log"
fi
