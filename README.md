# WeMM-mvp — 视频文字搜索

用 [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) 做的 MVP：加载一个视频，输入文字描述，定位到最匹配的时间段并播放。

## 用法

1. 打开 [Google Colab](https://colab.research.google.com/)，`文件 → 上传笔记本`，选择 `WeMM_video_search_colab.ipynb`
2. `代码执行程序 → 更改运行时类型 → T4 GPU`
3. 从上到下依次运行。默认用 Big Buck Bunny 前 3 分钟做演示
4. 换自己的视频：左侧文件面板上传，改第 3 节的 `VIDEO_PATH`，重跑第 3 节之后的 cell

## 本地 Web 可视化（已验证）

```bash
~/DIY/WeMM-mvp/run_server.sh
```

然后打开 http://localhost:8765 ：视频库点击切换视频、上传新视频自动切片编码索引、文字检索定位时间段（时间轴柱状图 + 结果列表，点击跳播）。索引数据缓存在 `webapp_data/`，重启秒开。

## 本地跑（Mac，已验证）

模型和依赖已装好（`.venv`，Python 3.12；权重在 `~/.cache/huggingface`）：

```bash
cd ~/DIY/WeMM-mvp && DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python verify_local.py
```

M5 Max 实测：加载 33s，片段编码 1.1s/段，23 段总耗时 115s（含官方示例）。
`DYLD_FALLBACK_LIBRARY_PATH` 是给 torchcodec 找 Homebrew ffmpeg 动态库用的，不能省。

## 原理

视频按 8 秒切片 → 每片编码成 2048 维向量（只算一次）→ 文字 query 编码后余弦相似度排序 → 得到时间段。
