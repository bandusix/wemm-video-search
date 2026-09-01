#!/bin/bash
# 把项目作为「模型仓库」镜像到 HuggingFace（免费；Docker Space 需 PRO，故不用 Space）。
# 用法：先 `.venv/bin/hf auth login`（贴 HF token），再 `./publish_hf.sh <你的HF用户名>`
set -e
[ -z "$1" ] && echo "用法: ./publish_hf.sh <你的HF用户名>" && exit 1
REPO="$1/wemm-video-search"
cd "$(dirname "$0")"
HF="./.venv/bin/hf"   # 用项目 venv 里的 hf，免全局 PATH 问题

"$HF" auth whoami >/dev/null 2>&1 || { echo "请先登录: $HF auth login"; exit 1; }

echo "创建 model 仓库..."
"$HF" repo create "$REPO" --repo-type model -y 2>/dev/null || echo "（已存在，跳过创建）"

STAGE=$(mktemp -d)
git archive HEAD | tar -x -C "$STAGE"          # 只导出 git 跟踪的源码，不含数据/venv
printf -- '---\nlicense: mit\ntags:\n- video-search\n- multimodal-embedding\n---\n\n' \
  | cat - "$STAGE/README.md" > "$STAGE/.rdme" && mv "$STAGE/.rdme" "$STAGE/README.md"

echo "上传到 HF..."
"$HF" upload "$REPO" "$STAGE" . --repo-type model
rm -rf "$STAGE"
echo "✅ 已发布: https://huggingface.co/$REPO"
echo "提示：这是代码镜像仓库；要在线可运行的 demo 需 HF PRO（Docker Space），或本地/自有 GPU 运行。"
