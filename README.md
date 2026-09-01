# wemm-video-search

**Text-to-video moment search** powered by the [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) multimodal embedding model — load a video (local file or online m3u8/HLS), type a description, and jump straight to the matching time range.

> *"Find that moment in your head."*

Queries work in many languages (Chinese, English, Japanese, Korean, Spanish, Portuguese, …) — the model is multilingual, and the same meaning in different languages lands on the same clip in practice.

English · [中文说明 ↓](#中文说明)

## Features

- **Text → video-moment retrieval**: the video is sliced into windows, each encoded to a 2048-dim vector; a text query is encoded and ranked by cosine similarity, locating the time range in milliseconds.
- **Two input types**: local video files, or **online m3u8 / HLS streams** (indexed by streaming frames, without downloading the whole file).
- **No full download for online streams**: auto-picks the lowest-bitrate variant, decodes keyframes only, and fetches segments in parallel — a 2.5-hour movie indexes in under 6 minutes.
- **No blind spots**: when a stream's keyframe interval exceeds the window length, the windows in between would be unsearchable; those gaps are filled by decoding one extra frame while the segment is still local, so coverage is 100% and every second of the timeline is reachable.
- **Duplicate detection**: manifest fingerprint (defeats token rotation, zero download) + sampled-frame perceptual hash (catches the same title at a different resolution), avoiding redundant indexing.
- **Referer-gated streams**: supply the origin page URL and the server injects `Referer`/`Origin`; a built-in HLS proxy lets the browser play them too.
- **Local web UI**: video library switcher, similarity timeline bar chart, click-to-seek result list.

## Layout

| File | Purpose |
|---|---|
| `server_v3.py` + `static/index.html` | The unified server — local files, online m3u8, dedup, HLS proxy (port 8765) |
| `WeMM_video_search_colab.ipynb` | Colab notebook: official example + search demo (when you have no local GPU) |
| `verify_local.py` | Command-line verification script |
| `Dockerfile` | Container deployment |

## Quick start

Requires Python 3.12, ffmpeg, and a GPU (NVIDIA CUDA / Apple Silicon MPS; CPU also works for small runs).

```bash
# 1. Create the environment (uv recommended)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
  "transformers==5.2.0" "qwen-vl-utils==0.0.14" "sentence-transformers>=5.7.0" \
  "accelerate>=1.1.0" av torchcodec fastapi uvicorn python-multipart pillow

# 2. Start the server
./run_server.sh             # → http://localhost:8765
```

> **macOS note**: torchcodec needs to find the Homebrew ffmpeg dylibs, so the launch scripts set `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. The first run downloads ~6 GB of model weights.

Open the page, paste an m3u8 link or upload a video, wait for indexing, then search by text.

## Hardware

- **Minimum**: single 8 GB GPU (text/short video, batch=1)
- **Recommended**: 12–16 GB (T4 / 4070, etc.) — mixed image/video with ease
- **Apple Silicon**: runs on MPS out of the box (`device="mps"` + float16); developed and verified on an M5 Max
- **Cloud**: Google Colab's free T4 is enough

## How it works

```
video ──window (8s default)──► keyframes per window ──► WeMM encode (2048-dim) ──► index
text query ──► WeMM encode ──► cosine sim vs all windows ──► top-K ranges ──► seek
```

Key optimizations for online m3u8 (measured in the PoC):

- **No full download**: parse the master playlist, pick the lowest-bitrate variant, index only that.
- **Keyframes only**: `-skip_frame nokey`, decoded locally per segment.
- **Parallel segment fetch**: on public CDNs the bottleneck is per-segment latency, not bandwidth (measured: pure download 34 s vs 0.18 s of CPU). 16 concurrent downloads beat ffmpeg's single-connection sequential read by ~**3.6x**.
- **Fingerprints reuse the downloaded segments**: the dedup fingerprint needs 16 frames at fixed timestamps. Seeking them over the network cost **101.7 s**; taking them from segments already on disk costs **~0.75 s** — a 135x difference for an identical result.
- **Gap filling for 100% coverage**: a window with no keyframe is unsearchable. Those windows get one extra locally-decoded frame (+0.01 s per segment), lifting coverage from 77% to 100%.

Measured on a 2.4-hour film (Apple M5 Max, MPS):

| Stage | Before | After |
|---|---|---|
| Extraction | 146 s | 147 s (now includes fingerprints) |
| Dedup fingerprint | ~100 s | folded into extraction |
| Clip synthesis | ~210 s (serial) | ~10 s (parallel) |
| Vector encoding | 125 s | 171 s (+30% windows from gap filling) |
| **Total** | **581 s** | **336 s** |
| **Coverage** | **77%** | **100%** |

Two optimizations were tried and rejected by measurement: larger encode batches are *slower* on MPS (batch 4 is optimal), and feeding images instead of video clips is 7x faster but retrieves noticeably worse.
- **Playback from the original stream**: on a hit, hls.js seeks the original m3u8 — no transcoded copies.

Three-layer dedup (cheapest first): business `content_id` → manifest fingerprint (hash of segment-duration sequence, zero download) → sampled-frame perceptual hash (dHash, robust to resolution/bitrate; measured 0.99 same-title vs 0.50 different-title).

## Scaling

- Store vectors in FAISS / Milvus / pgvector; PQ compression lets a single box hold tens of millions of windows.
- Co-locate the indexing service with object storage in the same region — no egress cost.
- Serve encoding through vLLM / SGLang for higher throughput.

## Credits & license

- Model: [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) (Tencent)
- Released under the [MIT License](LICENSE)
- Please only index content you have the right to process; this tool neither includes nor encourages any means of bypassing content protection.

---

## 中文说明

用 [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) 多模态嵌入模型做的**视频"时刻"语义检索** —— 加载一个视频（本地文件或在线 m3u8），输入一句文字描述，定位到最匹配的时间段并直接跳播。中英日韩西葡等多语言查询均可直接使用。

### 能力一览

- **文字 → 视频时刻检索**：视频按窗口切片编码成 2048 维向量，文字 query 编码后余弦相似度排序，毫秒级定位时间段。
- **两种输入**：本地视频文件，或**在线 m3u8 / HLS 流**（不下载全片，流式抽帧索引）。
- **在线流不下载全片**：自动选最低码率变体、只解关键帧、并行拉分片，一部 2.5h 电影 6 分钟内索引完。
- **没有检索盲区**：当流的关键帧间隔大于窗口长度时，中间那些窗口本来搜不到；趁分片还在本地补解一帧填上，覆盖率 100%，时间轴上每一秒都可检索。
- **重复视频判重**：清单指纹（挡 token 轮换，零下载）+ 抽样帧感知指纹（挡同片不同清晰度），避免重复索引。
- **防盗链 / 需要 Referer 的流**：可填来源页地址，服务端注入 Referer/Origin；内置 HLS 代理让浏览器也能播放。
- **本地 Web 界面**：视频库切换、时间轴相似度柱状图、结果列表点击跳播。

### 目录结构

| 文件 | 说明 |
|---|---|
| `server_v3.py` + `static/index.html` | 统一服务端——本地文件、在线 m3u8、判重、HLS 代理（端口 8765） |
| `WeMM_video_search_colab.ipynb` | Colab notebook：官方示例 + 视频检索 demo |
| `verify_local.py` | 命令行验证脚本 |
| `Dockerfile` | 容器部署 |

### 快速开始

需要 Python 3.12、ffmpeg、以及一块 GPU（NVIDIA CUDA / Apple Silicon MPS，或 CPU 也能小规模跑）。

```bash
# 1. 建环境（推荐 uv）
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
  "transformers==5.2.0" "qwen-vl-utils==0.0.14" "sentence-transformers>=5.7.0" \
  "accelerate>=1.1.0" av torchcodec fastapi uvicorn python-multipart pillow

# 2. 启动服务
./run_server.sh             # → http://localhost:8765
```

> **macOS 注意**：torchcodec 需要找到 Homebrew ffmpeg 动态库，启动脚本已设 `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`。首次运行会下载约 6GB 模型权重。

### 硬件要求

- **最低**：单卡 8GB 显存（纯文本/短视频，batch=1）
- **推荐**：12–16GB 显存（T4 / 4070 等）
- **Apple Silicon**：MPS 直接可跑（`device="mps"` + float16）；本仓库在 M5 Max 上开发验证
- **云平台**：Google Colab 免费 T4 即可跑通

### 工作原理

在线 m3u8 的关键优化（均为实测）：

- **不下载全片**：解析主清单选最低码率变体，只索引它
- **只解关键帧**：`-skip_frame nokey`，逐分片本地解码
- **并行拉分片**：公网 CDN 的瓶颈是每分片延迟而非带宽（实测纯下载 34s、CPU 仅 0.18s），16 并发比 ffmpeg 单连接顺序读快约 **3.6x**
- **判重指纹复用已下载分片**：指纹需要固定时刻的 16 帧，跨公网 seek 要 **101.7s**，从本地已有分片取只要 **~0.75s**，结果相同但差 135 倍
- **补解非关键帧达成 100% 覆盖**：没有关键帧的窗口检索不到，补解一帧填上（每分片 +0.01s），覆盖率 77% → 100%
- **播放走原始流**：hls.js 直接 seek 原 m3u8，不产生转码副本

一部 2.4 小时影片实测（Apple M5 Max，MPS）：

| 阶段 | 优化前 | 优化后 |
|---|---|---|
| 抽帧 | 146s | 147s（已含判重指纹） |
| 判重指纹 | ~100s | 折入抽帧 |
| 合成片段 | ~210s（串行） | ~10s（并行） |
| 编码向量 | 125s | 171s（补帧后窗口 +30%） |
| **总计** | **581s** | **336s** |
| **覆盖率** | **77%** | **100%** |

另有两项优化被实测否决：MPS 上加大编码 batch 反而更慢（batch=4 最优）；用图片替代视频 clip 快 7 倍但检索质量明显下降。

判重三层（成本从低到高）：业务 `content_id` → 清单指纹（分片时长序列 hash，零下载）→ 抽样帧感知指纹（dHash，实测同片 0.99、异片 0.50）。

### 许可

模型 [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B)（腾讯）；本项目基于 [MIT License](LICENSE) 开源。请仅对你有权处理的视频内容做索引；本工具不含、也不鼓励任何绕过内容保护的用途。
