# wemm-video-search

用 [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) 多模态嵌入模型做的**视频"时刻"语义检索** —— 加载一个视频（本地文件或在线 m3u8），输入一句文字描述，定位到最匹配的时间段并直接跳播。

> "找到你脑海中的那一刻。"

中英日韩西葡等多语言查询均可直接使用（模型多语言，实测同一语义不同语言定位到同一片段）。

## 能力一览

- **文字 → 视频时刻检索**：视频按窗口切片编码成 2048 维向量，文字 query 编码后余弦相似度排序，毫秒级定位时间段
- **两种输入**：本地视频文件上传，或**在线 m3u8 / HLS 流**（不下载全片，流式抽帧索引）
- **在线流不下载全片**：自动选最低码率变体、只解关键帧、并行拉分片，一部 2.5h 电影约几分钟索引完
- **重复视频判重**：清单指纹（挡 token 轮换，零下载）+ 抽样帧感知指纹（挡同片不同清晰度），避免重复索引
- **防盗链 / 需要 Referer 的流**：可填来源页地址，服务端注入 Referer/Origin 拉流；内置 HLS 代理让浏览器也能播放
- **本地 Web 界面**：视频库切换、时间轴相似度柱状图、结果列表点击跳播

## 目录结构

| 文件 | 说明 |
|---|---|
| `server.py` + `static/index.html` | **v1**：本地视频文件检索（端口 8765） |
| `server_v2.py` + `static/m3u8.html` | **v2**：m3u8 在线流式索引 + 判重 + 代理（端口 8766） |
| `WeMM_video_search_colab.ipynb` | Colab notebook：官方示例 + 视频检索 demo（免 GPU 本地环境时用） |
| `verify_local.py` | 命令行验证脚本 |

## 快速开始

需要 Python 3.12、ffmpeg、以及一块 GPU（NVIDIA CUDA / Apple Silicon MPS，或 CPU 也能小规模跑）。

```bash
# 1. 建环境（推荐 uv）
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
  "transformers==5.2.0" "qwen-vl-utils==0.0.14" "sentence-transformers>=5.7.0" \
  "accelerate>=1.1.0" av torchcodec fastapi uvicorn python-multipart pillow

# 2. 启动 v2（m3u8 流式索引）
./run_server_v2.sh          # → http://localhost:8766
# 或 v1（本地文件检索）
./run_server.sh             # → http://localhost:8765
```

> **macOS 注意**：torchcodec 需要找到 Homebrew ffmpeg 动态库，启动脚本已设 `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`。首次运行会下载约 6GB 模型权重。

打开页面后：粘贴一个 m3u8 链接或上传视频 → 等索引完成 → 输入文字描述检索。

## 硬件要求

- **最低**：单卡 8GB 显存（纯文本/短视频，batch=1）
- **推荐**：12–16GB 显存（T4 / 4070 等），图文视频混合无压力
- **Apple Silicon**：MPS 直接可跑（`device="mps"` + float16）；本仓库在 M5 Max 上开发验证
- **云平台**：Google Colab 免费 T4 即可跑通

## 工作原理

```
视频 ──切窗(默认8s)──► 每窗抽关键帧 ──► WeMM 编码 2048维向量 ──► 存索引
文字 query ──► WeMM 编码 ──► 与所有窗向量余弦相似度 ──► Top-K 时间段 ──► 跳播
```

在线 m3u8 的关键优化（PoC 实测）：

- **不下载全片**：解析 master playlist 选最低码率变体，只对它抽帧
- **只解关键帧**：`-skip_frame nokey`，解码量降一个数量级
- **并行拉分片**：公网 CDN 瓶颈在每分片响应延迟，16 并发下载比 ffmpeg 单连接顺序读快约 **3.6x**（实测一部 39min 电影抽帧 300s → 83s）
- **播放用原始流**：命中后 hls.js 直接 seek 原 m3u8，不产生转码副本

判重三层（成本从低到高）：业务 `content_id` → 清单指纹（分片时长序列 hash，零下载）→ 抽样帧感知指纹（dHash，抗清晰度/码率差异，实测同片相似度 0.99、异片 0.50）。

## 规模化方向

- 向量入 FAISS / Milvus / pgvector，PQ 压缩后单机可扛千万级窗口
- 索引服务与对象存储同区域内网部署，读取免流量费
- 部署走 vLLM / SGLang 提升编码吞吐

## 致谢与许可

- 模型：[tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B)（腾讯）
- 本项目基于 [MIT License](LICENSE) 开源
- 请仅对你有权处理的视频内容做索引；本工具不含、也不鼓励任何绕过内容保护的用途
