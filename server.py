"""WeMM 视频时刻检索 · 本地 Web 服务

启动: ./run_server.sh  (需要 DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib 给 torchcodec 找 ffmpeg)
"""
import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent
DATA = BASE / "webapp_data"
VIDEOS = DATA / "videos"
VIDEOS.mkdir(parents=True, exist_ok=True)

SEGMENT_SECONDS = 8

app = FastAPI(title="WeMM Video Moment Search")

# ---------- 模型（后台线程加载，编码用锁串行） ----------
_model = None
_model_lock = threading.Lock()
_model_status = {"state": "loading", "device": None, "error": None}


def _load_model():
    global _model
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        _model = SentenceTransformer(
            "tencent/WeMM-Embedding-2B",
            trust_remote_code=True,
            device=device,
            model_kwargs={"dtype": dtype},
        )
        _model_status.update(state="ready", device=device)
        _requeue_orphans()
        _seed_demo_videos()
    except Exception as e:  # noqa: BLE001
        _model_status.update(state="error", error=str(e))


threading.Thread(target=_load_model, daemon=True).start()


# ---------- 视频索引 ----------
def _meta_path(vid):
    return VIDEOS / vid / "meta.json"


def _read_meta(vid):
    p = _meta_path(vid)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _write_meta(vid, meta):
    _meta_path(vid).write_text(json.dumps(meta, ensure_ascii=False))


def _ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _index_video(vid, src_path, name):
    vdir = VIDEOS / vid
    clips_dir = vdir / "clips"
    thumbs_dir = vdir / "thumbs"
    clips_dir.mkdir(exist_ok=True)
    thumbs_dir.mkdir(exist_ok=True)
    meta = {"id": vid, "name": name, "status": "processing",
            "stage": "转码切片", "progress": 0, "seg": SEGMENT_SECONDS}
    try:
        duration = _ffprobe_duration(src_path)
        meta["duration"] = duration
        _write_meta(vid, meta)

        # 转码为浏览器友好的 360p mp4，用 -progress 实时回报进度（长视频要转很久）
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_path),
             "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "28", "-c:a", "aac", "-movflags", "+faststart",
             "-progress", "pipe:1", "-nostats", str(vdir / "source.mp4")],
            stdout=subprocess.PIPE, text=True,
        )
        for line in proc.stdout:
            if line.startswith("out_time_ms="):
                try:
                    done = int(line.split("=")[1]) / 1e6
                    meta.update(stage="转码", progress=round(done / duration * 100))
                    _write_meta(vid, meta)
                except ValueError:
                    pass
        if proc.wait() != 0:
            raise RuntimeError("转码失败")
        # 每段独立截取：不依赖 segment muxer（源时间戳有偏移时它可能漏切），边界精确可控
        n_seg = max(1, -(-int(duration * 1000) // (SEGMENT_SECONDS * 1000)))
        clips, clip_files = [], []
        for i in range(n_seg):
            start = i * SEGMENT_SECONDS
            end = min(start + SEGMENT_SECONDS, duration)
            if end - start < 0.5:
                break
            cf = clips_dir / f"clip_{i:04d}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", str(start), "-t", str(SEGMENT_SECONDS),
                 "-i", str(vdir / "source.mp4"),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                 "-c:a", "aac", str(cf)],
                check=True,
            )
            clips.append({"start": start, "end": round(end, 2)})
            clip_files.append(cf)
            if i % 10 == 0:
                meta.update(stage="切片", progress=round((i + 1) / n_seg * 100))
                _write_meta(vid, meta)

        meta.update(stage="生成缩略图", clips=clips)
        _write_meta(vid, meta)
        for i, cf in enumerate(clip_files):
            mid = (clips[i]["end"] - clips[i]["start"]) / 2
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{mid:.2f}",
                 "-i", str(cf), "-frames:v", "1", "-vf", "scale=-2:120",
                 str(thumbs_dir / f"{i:04d}.jpg")],
                check=True,
            )

        # 等模型就绪后编码
        while _model_status["state"] == "loading":
            meta.update(stage="等待模型加载")
            _write_meta(vid, meta)
            time.sleep(2)
        if _model_status["state"] != "ready":
            raise RuntimeError(f"模型加载失败: {_model_status['error']}")

        embs = []
        t0 = time.time()
        CHUNK = 4  # 128GB 统一内存下小批量编码，长视频提速明显
        for i in range(0, len(clip_files), CHUNK):
            batch = clip_files[i:i + CHUNK]
            with _model_lock:
                es = _model.encode_document(
                    [{"video": str(cf), "text": "Represent this video."} for cf in batch],
                    batch_size=CHUNK)
            embs.extend(np.asarray(e, dtype=np.float32) for e in es)
            done = min(i + CHUNK, len(clip_files))
            meta.update(stage="编码向量", progress=round(done / len(clip_files) * 100))
            _write_meta(vid, meta)
        np.save(vdir / "embeds.npy", np.stack(embs))

        meta.update(status="ready", stage="完成", progress=100,
                    encode_seconds=round(time.time() - t0, 1))
        _write_meta(vid, meta)
    except Exception as e:  # noqa: BLE001
        meta.update(status="error", error=str(e))
        _write_meta(vid, meta)


def _start_index(src_path, name):
    vid = uuid.uuid4().hex[:8]
    vdir = VIDEOS / vid
    vdir.mkdir()
    dst = vdir / ("upload" + Path(name).suffix)
    shutil.copy(src_path, dst)
    _write_meta(vid, {"id": vid, "name": name, "status": "processing",
                      "stage": "排队中", "progress": 0, "seg": SEGMENT_SECONDS})
    threading.Thread(target=_index_video, args=(vid, dst, name), daemon=True).start()
    return vid


def _requeue_orphans():
    """服务重启时接续上次没做完的索引（进程一死索引线程就没了，不重排会永远卡在 processing）。"""
    for d in VIDEOS.iterdir():
        if not d.is_dir():
            continue
        m = _read_meta(d.name)
        if m and m.get("status") == "processing":
            srcs = list(d.glob("upload.*"))
            if srcs:
                threading.Thread(target=_index_video,
                                 args=(m["id"], srcs[0], m["name"]), daemon=True).start()
            else:
                m.update(status="error", error="索引中断且源文件缺失，请重新上传")
                _write_meta(d.name, m)


def _seed_demo_videos():
    """首次启动时预置两个演示视频。"""
    if any(VIDEOS.iterdir()):
        return
    seeds = [
        (BASE / "demo_source.mp4", "Big Buck Bunny (前3分钟)"),
        (Path.home() / ".claude/uploads/df660e94-d987-4350-b69d-06c34c73bc41/1f898eea-ssstwitter.com_1788132109792.mp4",
         "参考 Demo 录屏"),
    ]
    for path, name in seeds:
        if path.exists():
            _start_index(path, name)


# ---------- API ----------
@app.get("/api/status")
def status():
    return _model_status


@app.get("/api/videos")
def list_videos():
    out = []
    for d in sorted(VIDEOS.iterdir()):
        m = _read_meta(d.name)
        if m:
            out.append(m)
    return out


@app.get("/api/videos/{vid}")
def get_video(vid: str):
    m = _read_meta(vid)
    if not m:
        raise HTTPException(404)
    return m


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    tmp = DATA / f"tmp_{uuid.uuid4().hex}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    vid = _start_index(tmp, file.filename or "上传视频")
    tmp.unlink()
    return {"id": vid}


class SearchReq(BaseModel):
    video_id: str
    query: str
    top_k: int = 5


@app.post("/api/search")
def search(req: SearchReq):
    if _model_status["state"] != "ready":
        raise HTTPException(503, "模型尚未就绪")
    m = _read_meta(req.video_id)
    if not m or m.get("status") != "ready":
        raise HTTPException(400, "视频尚未索引完成")
    embs = np.load(VIDEOS / req.video_id / "embeds.npy")
    t0 = time.time()
    with _model_lock:
        q = np.asarray(_model.encode_query([req.query])[0], dtype=np.float32)
    scores = embs @ q / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-8)
    order = np.argsort(-scores)[: req.top_k]
    return {
        "query": req.query,
        "encode_ms": round((time.time() - t0) * 1000),
        "scores": [round(float(s), 4) for s in scores],
        "results": [
            {"index": int(i), "start": m["clips"][int(i)]["start"],
             "end": m["clips"][int(i)]["end"], "score": round(float(scores[i]), 4)}
            for i in order
        ],
    }


app.mount("/media", StaticFiles(directory=VIDEOS), name="media")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
