"""WeMM 视频时刻检索 v2 · m3u8 流式索引 PoC （端口 8766，与 v1 并存）

验证点：在线 m3u8 不下载全片——选最低码率变体流、按窗口流式拉取、只解关键帧，
每窗口抽 ≤8 帧关键帧合成微型 clip 送入模型编码。
"""
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent
DATA = BASE / "webapp_data_v2"
VIDEOS = DATA / "videos"
VIDEOS.mkdir(parents=True, exist_ok=True)

SEGMENT_SECONDS = 8
MAX_FRAMES = 8
PHASH_K = 16  # 判重指纹采样点数：存储与预检必须一致，都按 (i+0.5)/K 的时间分数取，保证逐位对齐
# 公网 m3u8 分片并行下载并发数。实测 cinejoy(公网CDN,每分片~2.3s延迟)下载吞吐随并发扩展：
#   8并发 13.6x / 16并发 25.6x / 24并发 32.6x / 32并发 41x 实时（单趟顺序读仅 7.8x）。
# 16 是甜点区（3.3x 提速、对第三方 CDN 温和不易触发限流）；自有 OSS/内网可上调到 24~32。
# 解码开销可忽略（8并发全流水线 22.9s vs 纯下载 22.1s），瓶颈纯在下载。
HLS_FETCH_WORKERS = 16
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")


def _origin_of(referer):
    """从 Referer 推导 Origin（scheme://host）。"""
    if not referer:
        return None
    p = urllib.parse.urlparse(referer)
    return f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else None


def _http_headers(referer=None):
    """urllib 用的请求头字典：UA + 可选 Referer/Origin（应对防盗链）。"""
    h = {"User-Agent": UA}
    if referer:
        h["Referer"] = referer
        origin = _origin_of(referer)
        if origin:
            h["Origin"] = origin
    return h


def _ffmpeg_header_args(referer=None):
    """ffmpeg 用的头参数：-user_agent + -headers（CRLF 分隔）。"""
    args = ["-user_agent", UA]
    if referer:
        lines = [f"Referer: {referer}"]
        origin = _origin_of(referer)
        if origin:
            lines.append(f"Origin: {origin}")
        args += ["-headers", "\r\n".join(lines) + "\r\n"]
    return args

app = FastAPI(title="WeMM m3u8 PoC")

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
        _cleanup_orphans()
        _backfill_manifest_fp()
        _seed_demo()
    except Exception as e:  # noqa: BLE001
        _model_status.update(state="error", error=str(e))


threading.Thread(target=_load_model, daemon=True).start()


def _meta_path(vid):
    return VIDEOS / vid / "meta.json"


def _read_meta(vid):
    p = _meta_path(vid)
    return json.loads(p.read_text()) if p.exists() else None


def _write_meta(vid, meta):
    _meta_path(vid).write_text(json.dumps(meta, ensure_ascii=False))


# ---------- m3u8 解析 ----------
def _fetch_text(url, referer=None):
    req = urllib.request.Request(url, headers=_http_headers(referer))
    last = None
    for _attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001 —— 公网抖动重试
            last = e
            time.sleep(2)
    raise last


def _pick_lowest_variant(master_url, referer=None):
    """master playlist 里选带宽最低的变体；若本身是媒体清单则原样返回。
    返回 (index_url, bandwidth, media_text)"""
    text = _fetch_text(master_url, referer)
    lines = [ln.strip() for ln in text.splitlines()]
    variants = []
    for i, ln in enumerate(lines):
        if ln.startswith("#EXT-X-STREAM-INF"):
            m = re.search(r"BANDWIDTH=(\d+)", ln)
            bw = int(m.group(1)) if m else 0
            for j in range(i + 1, len(lines)):
                if lines[j] and not lines[j].startswith("#"):
                    variants.append((bw, urllib.parse.urljoin(master_url, lines[j])))
                    break
    if variants:
        variants.sort()
        bw, url = variants[0]
        return url, bw, _fetch_text(url, referer)
    return master_url, None, text


def _duration_from_playlist(text):
    return sum(float(m) for m in re.findall(r"#EXTINF:([\d.]+)", text))


# ---------- 判重指纹 ----------
def _manifest_fp(media_text, duration):
    """清单指纹：分片时长序列 + 总时长 + 分片数。同一压制换 token 后完全一致。"""
    segs = re.findall(r"#EXTINF:([\d.]+)", media_text)
    seq = ",".join(f"{float(s):.2f}" for s in segs)
    raw = f"{round(duration)}|{len(segs)}|{seq}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _dhash_hex(path, size=8):
    """dHash：缩到 (size+1)×size 灰度，比较相邻像素得 64bit。抗分辨率/码率。"""
    from PIL import Image
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return f"{bits:016x}"


def _hamming_hex(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _dhash_or_none(p):
    try:
        return _dhash_hex(p)
    except Exception:  # noqa: BLE001
        return None


def _phash_of_frames(frame_paths, k=PHASH_K):
    """预检用：frame_paths 是定长 k 列表（失败位置为 None），逐位算 dHash 并保位。"""
    return [_dhash_or_none(p) if p else None for p in frame_paths]


def _phash_similarity(fa, fb):
    """两组感知指纹按位置对齐比（都是定长 k、按 (i+0.5)/k 时间分数采样）。
    只统计两边都有帧的位置，避免个别采样失败导致错位。"""
    n = min(len(fa), len(fb))
    same, cnt = 0, 0
    for i in range(n):
        if fa[i] and fb[i]:
            same += 64 - _hamming_hex(fa[i], fb[i])
            cnt += 1
    return round(same / (64 * cnt), 4) if cnt else 0.0


def _sample_frames_quick(index_url, duration, out_dir, k=PHASH_K, referer=None):
    """在 (i+0.5)/k 时间分数处精确 seek 采帧。返回定长 k 列表，失败位置为 None（保位）。
    索引期算存储指纹、预检期算候选指纹都用它 → 两侧采样同一时刻，同片相似度可达 0.95+。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = _ffmpeg_header_args(referer)
    paths = []
    for i in range(k):
        t = duration * (i + 0.5) / k
        p = out_dir / f"s{i:02d}.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", *hdr,
                 "-allowed_extensions", "ALL", "-extension_picky", "0",
                 "-ss", str(t), "-i", index_url, "-frames:v", "1",
                 "-vf", "scale=-2:120", str(p)],
                check=False, timeout=45)
        except subprocess.TimeoutExpired:
            paths.append(None)  # 保留位置，避免错位
            continue
        paths.append(p if p.exists() else None)
    return paths


# 存储侧与预检侧都按 (i+0.5)/K 精确时间 seek 采帧后，同影片不同码率≈0.90+、不同影片≈0.50。
# 阈值 0.82：稳稳判出同片、又远离随机水平。
PHASH_DUP_THRESHOLD = 0.82


def _best_perceptual(duration, phash, exclude_id=None):
    """返回时长同桶内感知指纹相似度最高的候选 (meta, sim)。用于判重和透明展示。"""
    best_m, best_sim = None, 0.0
    if not phash:
        return None, 0.0
    for d in VIDEOS.iterdir():
        m = _read_meta(d.name) if d.is_dir() else None
        if not m or m.get("status") != "ready" or m["id"] == exclude_id:
            continue
        if abs((m.get("duration") or 0) - duration) > 3:  # 时长粗桶
            continue
        sim = _phash_similarity(phash, m.get("phash") or [])
        if sim > best_sim:
            best_m, best_sim = m, sim
    return best_m, best_sim


def _find_duplicate(manifest_fp, duration, phash, exclude_id=None):
    """在已索引库里找重复：先清单指纹精确匹配，再时长分桶 + 感知指纹比对。"""
    for d in VIDEOS.iterdir():
        m = _read_meta(d.name) if d.is_dir() else None
        if not m or m.get("status") != "ready" or m["id"] == exclude_id:
            continue
        if manifest_fp and m.get("manifest_fp") == manifest_fp:
            return {"id": m["id"], "name": m["name"], "kind": "manifest",
                    "confidence": 1.0, "reason": "清单指纹完全一致（同一压制，仅换了 token/URL）"}
    m, sim = _best_perceptual(duration, phash, exclude_id)
    if m and sim >= PHASH_DUP_THRESHOLD:
        return {"id": m["id"], "name": m["name"], "kind": "perceptual",
                "confidence": sim,
                "reason": f"画面感知指纹匹配 {round(sim * 100)}%（同一内容，可能不同清晰度/压制）"}
    return None


# ---------- 索引 ----------
def _extract_all_keyframes(index_url, out_dir, duration, progress_cb=None, referer=None):
    """单趟拉取最低码率流，一次性抽出全部关键帧（showinfo 打印每帧 pts_time）。
    一次连接顺序读，比每窗口单独 seek 快一个数量级，且流量只拉一遍。
    progress_cb(pct): ffmpeg -progress 实时回报当前处理到的时间点，用于更新进度。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info", *_ffmpeg_header_args(referer),
           # 有些站点把 .ts/.mp4 分片伪装成 .html 等扩展名，放开 ffmpeg 的 HLS 扩展名白名单
           "-allowed_extensions", "ALL", "-extension_picky", "0",
           "-i", index_url,
           "-vf", r"select='eq(pict_type\,I)',scale=-2:360,showinfo",
           "-fps_mode", "vfr", "-q:v", "5",
           "-progress", "pipe:1", "-nostats", str(out_dir / "kf_%05d.jpg")]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err_buf = []
    threading.Thread(target=lambda: err_buf.extend(proc.stderr), daemon=True).start()
    deadline = time.time() + int(duration) + 900
    last_pct = -1
    for line in proc.stdout:  # 读 -progress 流：out_time_us 反映已处理到的时间戳
        if line.startswith("out_time_us=") and progress_cb:
            try:
                pct = min(99, round(int(line.split("=")[1]) / 1e6 / duration * 100))
                if pct != last_pct:
                    last_pct = pct
                    progress_cb(pct)
            except ValueError:
                pass
        if time.time() > deadline:
            proc.kill()
            break
    proc.wait()
    times = [float(m) for m in re.findall(r"pts_time:([\d.]+)", "".join(err_buf))]
    frames = sorted(out_dir.glob("kf_*.jpg"))
    pairs = list(zip(times, frames))  # showinfo 行与输出帧同序一一对应
    by_win = {}
    for t, fp in pairs:
        by_win.setdefault(int(t // SEGMENT_SECONDS), []).append((t, fp))
    # 每个窗口最多保留 MAX_FRAMES 帧（均匀下采样）
    out = {}
    for w, items in by_win.items():
        items.sort()
        if len(items) > MAX_FRAMES:
            step = len(items) / MAX_FRAMES
            items = [items[int(k * step)] for k in range(MAX_FRAMES)]
        out[w] = [fp for _, fp in items]
    return out, pairs  # pairs=(time, path) 全量，供感知指纹按时间取样


def _download_bytes(url, referer, dest, timeout=45):
    """下载单个分片到文件，失败重试。仅用于公网 m3u8 并行抽帧。"""
    req = urllib.request.Request(url, headers=_http_headers(referer))
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                dest.write_bytes(r.read())
            return dest.stat().st_size > 0
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def _parse_segments(media_url, media_text):
    """从媒体清单解析 (idx, 绝对URL, 起始秒) 列表 + 可选 EXT-X-MAP init 段 URL。"""
    init_url = None
    mm = re.search(r'#EXT-X-MAP:URI="([^"]+)"', media_text)
    if mm:
        init_url = urllib.parse.urljoin(media_url, mm.group(1))
    segs, t, dur = [], 0.0, None
    for ln in media_text.splitlines():
        s = ln.strip()
        if s.startswith("#EXTINF:"):
            dm = re.match(r"#EXTINF:([\d.]+)", s)
            dur = float(dm.group(1)) if dm else 0.0
        elif s and not s.startswith("#"):
            segs.append((len(segs), urllib.parse.urljoin(media_url, s), t))
            t += dur or 0.0
            dur = None
    return segs, init_url


def _extract_all_keyframes_parallel(media_url, media_text, out_dir, duration,
                                    progress_cb=None, referer=None):
    """公网 m3u8 专用：并行下载分片、各自本地抽关键帧。瓶颈是 CDN 每分片延迟，
    并发拉取比 ffmpeg 单连接顺序读快 ~5x（实测）。解码用 -skip_frame nokey（本地、极廉价）。
    返回 (by_win, timed)，与单趟版签名一致。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = out_dir / "seg"
    seg_dir.mkdir(exist_ok=True)
    segs, init_url = _parse_segments(media_url, media_text)
    if not segs:
        return {}, []
    init_path = None
    if init_url:  # fMP4：init 段需拼在每个分片前才能解码
        init_path = out_dir / "init.bin"
        _download_bytes(init_url, referer, init_path)

    done = [0]
    lock = threading.Lock()
    results = {}

    def _work(item):
        idx, url, start = item
        raw = seg_dir / f"{idx:05d}.bin"
        if not _download_bytes(url, referer, raw):
            return
        dec_in = raw
        if init_path:
            merged = seg_dir / f"{idx:05d}.m4s"
            merged.write_bytes(init_path.read_bytes() + raw.read_bytes())
            dec_in = merged
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-skip_frame", "nokey",
             "-i", str(dec_in), "-vf", "scale=-2:360", "-fps_mode", "vfr",
             "-q:v", "5", "-frames:v", str(MAX_FRAMES), str(seg_dir / f"{idx:05d}_%02d.jpg")],
            check=False, timeout=60)
        frames = sorted(seg_dir.glob(f"{idx:05d}_*.jpg"))
        raw.unlink(missing_ok=True)
        if dec_in != raw:
            dec_in.unlink(missing_ok=True)
        if frames:
            results[idx] = (start, frames)
        with lock:
            done[0] += 1
            if progress_cb and done[0] % 5 == 0:
                progress_cb(min(99, round(done[0] / len(segs) * 100)))

    with ThreadPoolExecutor(max_workers=HLS_FETCH_WORKERS) as ex:
        list(ex.map(_work, segs))

    by_win, timed = {}, []
    for idx in sorted(results):
        start, frames = results[idx]
        by_win.setdefault(int(start // SEGMENT_SECONDS), []).extend(frames)
        timed.extend((start, fp) for fp in frames)
    return by_win, timed


def _frames_to_clip(frames, clip_path):
    """把任意一组关键帧图片（文件名可能不连续）合成微型 clip。用 concat demuxer 显式列出。"""
    lst = clip_path.with_suffix(".txt")
    lst.write_text("".join(f"file '{f.resolve()}'\nduration 0.5\n" for f in frames)
                   + f"file '{frames[-1].resolve()}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-vf", "fps=2,scale=-2:360", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(clip_path)],
        check=True, timeout=60)
    lst.unlink(missing_ok=True)


def _index_hls(vid, master_url, name, referer=None):
    vdir = VIDEOS / vid
    clips_dir, thumbs_dir = vdir / "clips", vdir / "thumbs"
    clips_dir.mkdir(exist_ok=True)
    thumbs_dir.mkdir(exist_ok=True)
    meta = {"id": vid, "name": name, "status": "processing", "stage": "解析清单",
            "progress": 0, "seg": SEGMENT_SECONDS, "source_type": "hls",
            "playback_url": master_url, "referer": referer}
    _write_meta(vid, meta)
    try:
        t_all = time.time()
        index_url, bw, media_text = _pick_lowest_variant(master_url, referer)
        duration = _duration_from_playlist(media_text)
        if duration <= 0:
            raise RuntimeError("无法从 m3u8 清单解析时长（可能是直播流，PoC 仅支持点播）")
        if not bw:  # 媒体清单无 BANDWIDTH，用逐分片 #EXT-X-BITRATE(kbps) 估平均码率
            brs = [int(m) for m in re.findall(r"#EXT-X-BITRATE:(\d+)", media_text)]
            if brs:
                bw = int(sum(brs) / len(brs) * 1000)
        est_mb = round(bw * duration / 8 / 1e6, 1) if bw else None
        meta.update(duration=duration, bandwidth=bw, est_mb=est_mb, index_url=index_url,
                    manifest_fp=_manifest_fp(media_text, duration))
        _write_meta(vid, meta)

        n = max(1, int(duration // SEGMENT_SECONDS) + (1 if duration % SEGMENT_SECONDS > 0.5 else 0))
        meta.update(stage="单趟流式抽帧", progress=0)
        _write_meta(vid, meta)
        t_ext = time.time()

        def _ext_progress(pct):
            meta.update(stage="单趟流式抽帧", progress=pct)
            _write_meta(vid, meta)

        # 公网 m3u8：并行下载分片抽帧（瓶颈在 CDN 每分片延迟，并发比单连接顺序读快 ~5x）
        by_win, _ = _extract_all_keyframes_parallel(index_url, media_text, vdir / "frames",
                                                    duration, _ext_progress, referer)
        if not by_win:  # 并行失败则回退到经过验证的单趟顺序读
            by_win, _ = _extract_all_keyframes(index_url, vdir / "frames", duration,
                                               _ext_progress, referer)
        extract_seconds = round(time.time() - t_ext, 1)
        if not by_win:
            raise RuntimeError("未能从流中抽出任何帧，检查链接是否可访问 / 是否有防盗链")

        # 感知指纹：按 (i+0.5)/K 精确时间 seek 采帧（与预检完全同一时刻，同片相似度 0.95+）
        meta.update(stage="计算判重指纹")
        _write_meta(vid, meta)
        fp_frames = _sample_frames_quick(index_url, duration, vdir / "fp", referer=referer)
        meta["phash"] = _phash_of_frames(fp_frames)
        _write_meta(vid, meta)
        shutil.rmtree(vdir / "fp", ignore_errors=True)

        clips = []
        for w in range(n):
            frames = by_win.get(w)
            if not frames:
                continue
            start = w * SEGMENT_SECONDS
            cf = clips_dir / f"clip_{len(clips):04d}.mp4"
            _frames_to_clip(frames, cf)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(frames[0]),
                            "-vf", "scale=-2:120", str(thumbs_dir / f"{len(clips):04d}.jpg")],
                           check=False, timeout=30)
            clips.append({"start": start, "end": round(min(start + SEGMENT_SECONDS, duration), 2),
                          "frames": len(frames)})
            if w % 10 == 0:
                meta.update(stage="合成片段", progress=round((w + 1) / n * 100))
                _write_meta(vid, meta)

        while _model_status["state"] == "loading":
            meta.update(stage="等待模型加载")
            _write_meta(vid, meta)
            time.sleep(2)
        if _model_status["state"] != "ready":
            raise RuntimeError(f"模型加载失败: {_model_status['error']}")

        clip_files = sorted(clips_dir.glob("clip_*.mp4"))
        embs = []
        t_enc = time.time()
        CHUNK = 4
        for i in range(0, len(clip_files), CHUNK):
            batch = clip_files[i:i + CHUNK]
            with _model_lock:
                es = _model.encode_document(
                    [{"video": str(cf), "text": "Represent this video."} for cf in batch],
                    batch_size=CHUNK)
            embs.extend(np.asarray(e, dtype=np.float32) for e in es)
            meta.update(stage="编码向量",
                        progress=round(min(i + CHUNK, len(clip_files)) / len(clip_files) * 100))
            _write_meta(vid, meta)
        np.save(vdir / "embeds.npy", np.stack(embs))
        shutil.rmtree(vdir / "frames", ignore_errors=True)

        total = round(time.time() - t_all, 1)
        meta.update(status="ready", stage="完成", progress=100, clips=clips,
                    skipped_windows=n - len(clips),
                    extract_seconds=extract_seconds,
                    encode_seconds=round(time.time() - t_enc, 1),
                    total_seconds=total,
                    realtime_factor=round(duration / total, 1))
        _write_meta(vid, meta)
    except Exception as e:  # noqa: BLE001
        meta.update(status="error", error=str(e))
        _write_meta(vid, meta)


def _index_upload(vid, src_path, name):
    """本地上传沿用 v1 思路：转码 360p + 精确切片。"""
    vdir = VIDEOS / vid
    clips_dir, thumbs_dir = vdir / "clips", vdir / "thumbs"
    clips_dir.mkdir(exist_ok=True)
    thumbs_dir.mkdir(exist_ok=True)
    meta = {"id": vid, "name": name, "status": "processing", "stage": "转码",
            "progress": 0, "seg": SEGMENT_SECONDS, "source_type": "upload"}
    try:
        t_all = time.time()
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(src_path)],
            capture_output=True, text=True, check=True).stdout.strip()
        duration = float(out)
        meta["duration"] = duration
        _write_meta(vid, meta)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_path),
             "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "28", "-c:a", "aac", "-movflags", "+faststart",
             str(vdir / "source.mp4")], check=True)
        n = max(1, -(-int(duration * 1000) // (SEGMENT_SECONDS * 1000)))
        clips, clip_files = [], []
        for i in range(n):
            start = i * SEGMENT_SECONDS
            end = min(start + SEGMENT_SECONDS, duration)
            if end - start < 0.5:
                break
            cf = clips_dir / f"clip_{i:04d}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
                 "-t", str(SEGMENT_SECONDS), "-i", str(vdir / "source.mp4"),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an", str(cf)],
                check=True)
            mid = (end - start) / 2
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{mid:.2f}", "-i", str(cf),
                 "-frames:v", "1", "-vf", "scale=-2:120",
                 str(thumbs_dir / f"{i:04d}.jpg")], check=False)
            clips.append({"start": start, "end": round(end, 2)})
            clip_files.append(cf)
            if i % 10 == 0:
                meta.update(stage="切片", progress=round((i + 1) / n * 100))
                _write_meta(vid, meta)

        while _model_status["state"] == "loading":
            time.sleep(2)
        if _model_status["state"] != "ready":
            raise RuntimeError(f"模型加载失败: {_model_status['error']}")
        embs = []
        t_enc = time.time()
        CHUNK = 4
        for i in range(0, len(clip_files), CHUNK):
            batch = clip_files[i:i + CHUNK]
            with _model_lock:
                es = _model.encode_document(
                    [{"video": str(cf), "text": "Represent this video."} for cf in batch],
                    batch_size=CHUNK)
            embs.extend(np.asarray(e, dtype=np.float32) for e in es)
            meta.update(stage="编码向量",
                        progress=round(min(i + CHUNK, len(clip_files)) / len(clip_files) * 100))
            _write_meta(vid, meta)
        np.save(vdir / "embeds.npy", np.stack(embs))
        total = round(time.time() - t_all, 1)
        meta.update(status="ready", stage="完成", progress=100, clips=clips,
                    encode_seconds=round(time.time() - t_enc, 1),
                    total_seconds=total, realtime_factor=round(duration / total, 1))
        _write_meta(vid, meta)
    except Exception as e:  # noqa: BLE001
        meta.update(status="error", error=str(e))
        _write_meta(vid, meta)


def _start_hls(url, name, referer=None):
    vid = uuid.uuid4().hex[:8]
    (VIDEOS / vid).mkdir()
    _write_meta(vid, {"id": vid, "name": name, "status": "processing",
                      "stage": "排队中", "progress": 0, "source_type": "hls",
                      "playback_url": url, "referer": referer, "seg": SEGMENT_SECONDS})
    threading.Thread(target=_index_hls, args=(vid, url, name, referer), daemon=True).start()
    return vid


def _cleanup_orphans():
    """服务重启会杀死索引线程，把残留的 processing 标记为 error，避免永久卡住。"""
    for d in VIDEOS.iterdir():
        m = _read_meta(d.name) if d.is_dir() else None
        if m and m.get("status") == "processing":
            m.update(status="error", error="服务重启中断，请重新添加")
            _write_meta(d.name, m)


def _backfill_manifest_fp():
    """给判重功能上线前索引的老 HLS 视频回填清单指纹（只需重取清单，零媒体下载）。"""
    for d in VIDEOS.iterdir():
        m = _read_meta(d.name) if d.is_dir() else None
        if (m and m.get("status") == "ready" and m.get("source_type") == "hls"
                and not m.get("manifest_fp") and m.get("playback_url")):
            try:
                _, _bw, txt = _pick_lowest_variant(m["playback_url"], m.get("referer"))
                dur = _duration_from_playlist(txt) or m.get("duration", 0)
                m["manifest_fp"] = _manifest_fp(txt, dur)
                _write_meta(d.name, m)
            except Exception:  # noqa: BLE001 —— 老链接可能已失效，跳过
                pass


def _seed_demo():
    if any(VIDEOS.iterdir()):
        return
    _start_hls("https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
               "Big Buck Bunny (在线 HLS 流)")


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


class HlsReq(BaseModel):
    url: str
    name: str = ""
    referer: str = ""  # 防盗链站点需要：填来源页地址，服务端据此带上 Referer/Origin


@app.post("/api/add_m3u8")
def add_m3u8(req: HlsReq):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "请填入 http(s) 开头的链接（m3u8 清单或其主清单接口）")
    name = req.name.strip() or url.rsplit("/", 1)[-1]
    return {"id": _start_hls(url, name, req.referer.strip() or None)}


@app.post("/api/check_duplicate")
def check_duplicate(req: HlsReq):
    """索引前判重：① 清单指纹（免费，挡 token 轮换）② 采样帧感知指纹（挡同片不同版本）。"""
    url = req.url.strip()
    referer = req.referer.strip() or None
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "请填入 http(s) 开头的链接（m3u8 清单或其主清单接口）")
    t0 = time.time()
    try:
        index_url, _bw, media_text = _pick_lowest_variant(url, referer)
        duration = _duration_from_playlist(media_text)
        if duration <= 0:
            raise RuntimeError("无法解析时长")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"无法读取清单：{e}")

    mfp = _manifest_fp(media_text, duration)
    # 第一层：清单指纹精确匹配（0 媒体下载）
    hit = _find_duplicate(mfp, duration, None)
    if hit:
        return {"duplicate": True, "checked_ms": round((time.time() - t0) * 1000),
                "manifest_fp": mfp, "duration": round(duration), "phash_sampled": 0, **hit}

    # 第二层：仅当存在同时长候选时，才采样算感知指纹（省则省）
    same_dur = any((m := _read_meta(d.name)) and m.get("status") == "ready"
                   and m.get("phash") and abs((m.get("duration") or 0) - duration) <= 3
                   for d in VIDEOS.iterdir() if d.is_dir())
    phash = []
    if same_dur:
        tmp = DATA / f"probe_{uuid.uuid4().hex}"
        try:
            frames = _sample_frames_quick(index_url, duration, tmp, referer=referer)
            phash = _phash_of_frames(frames)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        hit = _find_duplicate(None, duration, phash)
        best_m, best_sim = _best_perceptual(duration, phash)
        if hit:
            return {"duplicate": True, "checked_ms": round((time.time() - t0) * 1000),
                    "manifest_fp": mfp, "duration": round(duration),
                    "phash_sampled": sum(1 for h in phash if h), **hit}
        return {"duplicate": False, "checked_ms": round((time.time() - t0) * 1000),
                "manifest_fp": mfp, "duration": round(duration), "phash_sampled": sum(1 for h in phash if h),
                "best_perceptual": round(best_sim, 4) if best_m else None,
                "best_perceptual_name": best_m["name"] if best_m else None,
                "threshold": PHASH_DUP_THRESHOLD}

    return {"duplicate": False, "checked_ms": round((time.time() - t0) * 1000),
            "manifest_fp": mfp, "duration": round(duration), "phash_sampled": sum(1 for h in phash if h)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    vid = uuid.uuid4().hex[:8]
    vdir = VIDEOS / vid
    vdir.mkdir()
    dst = vdir / ("upload" + Path(file.filename or "v.mp4").suffix)
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    _write_meta(vid, {"id": vid, "name": file.filename or "上传视频",
                      "status": "processing", "stage": "排队中", "progress": 0,
                      "source_type": "upload", "seg": SEGMENT_SECONDS})
    threading.Thread(target=_index_upload, args=(vid, dst, file.filename or "上传视频"),
                     daemon=True).start()
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


# ---------- HLS 代理：浏览器无法自设 Referer/跨域，改由服务端带头拉流后回吐 ----------
def _proxy_url(upstream, vid):
    return f"/api/proxy?u={urllib.parse.quote(upstream, safe='')}&vid={vid}"


def _rewrite_playlist(text, base_url, vid):
    """把清单里的分片/子清单地址（相对或绝对）改写成走本代理，并处理 URI="..." 属性。"""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append(line)
            continue
        if s.startswith("#"):
            mm = re.search(r'URI="([^"]+)"', s)
            if mm:
                abs_u = urllib.parse.urljoin(base_url, mm.group(1))
                s = s.replace(mm.group(1), _proxy_url(abs_u, vid))
            out.append(s)
        else:
            abs_u = urllib.parse.urljoin(base_url, s)
            out.append(_proxy_url(abs_u, vid))
    return "\n".join(out)


@app.get("/api/proxy")
def proxy(u: str, request: Request, vid: str = ""):
    referer = None
    if vid:
        m = _read_meta(vid)
        referer = m.get("referer") if m else None
    headers = _http_headers(referer)
    if request.headers.get("range"):  # 转发 Range，支持 seek/分段
        headers["Range"] = request.headers["range"]
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(u, headers=headers), timeout=30)
    except urllib.error.HTTPError as e:
        return Response(status_code=e.code, content=e.read())
    except Exception as e:  # noqa: BLE001
        return Response(status_code=502, content=str(e).encode())
    data = resp.read()
    ctype = resp.headers.get("Content-Type", "application/octet-stream")
    if "mpegurl" in ctype.lower() or data[:7] == b"#EXTM3U":
        return Response(content=_rewrite_playlist(data.decode("utf-8", "replace"), u, vid),
                        media_type="application/vnd.apple.mpegurl")
    passthru = {h: resp.headers[h] for h in ("Content-Range", "Accept-Ranges")
                if resp.headers.get(h)}
    return Response(content=data, media_type=ctype, headers=passthru,
                    status_code=resp.status)


app.mount("/media", StaticFiles(directory=VIDEOS), name="media")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "m3u8.html")
