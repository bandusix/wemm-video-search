#!/bin/bash
# 用法：先 `hf auth login`（贴你的 HF token），再 `./publish_hf.sh <你的HF用户名>`
set -e
[ -z "$1" ] && echo "用法: ./publish_hf.sh <你的HF用户名>" && exit 1
SPACE="$1/wemm-video-search"
cd "$(dirname "$0")"

echo "创建 Space（Docker SDK）..."
hf repo create "$SPACE" --repo-type space --space_sdk docker -y 2>/dev/null || echo "（已存在，跳过创建）"

STAGE=$(mktemp -d)
cp Dockerfile server_v2.py LICENSE "$STAGE/"
cp -r static "$STAGE/"
cp HF_SPACE_README.md "$STAGE/README.md"   # HF Space 需要带 YAML 配置头的 README

echo "上传文件到 Space..."
hf upload "$SPACE" "$STAGE" . --repo-type space
rm -rf "$STAGE"
echo "✅ 已发布: https://huggingface.co/spaces/$SPACE"
echo "提示：默认免费 CPU，索引很慢；在 Space Settings 升级 GPU 硬件后才实用。"
