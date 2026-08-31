"""本地（Mac MPS）验证 WeMM-Embedding-2B 完整流程：
1. 官方示例：query -> 文本/图片/视频 相似度
2. 视频时刻检索：23 个 8 秒片段编码 + 文字搜索定位时间段
"""
import time

import torch
from sentence_transformers import SentenceTransformer

t0 = time.time()
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
print(f"device={device}, dtype={dtype}")

model = SentenceTransformer(
    "tencent/WeMM-Embedding-2B",
    trust_remote_code=True,
    device=device,
    model_kwargs={"dtype": dtype},
)
print(f"[1] 模型加载完成 ({time.time() - t0:.1f}s)")

# ---------- 官方示例 ----------
queries = [
    "Which Llama 4 model variants are available?",
    "How is mapo tofu prepared?",
]
documents = [
    "Mapo tofu is a Sichuan dish of soft tofu simmered in a spicy, numbing sauce.",
    {
        "image": "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/llama4_hgf.png",
        "text": "Represent this image.",
    },
    {
        "video": "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/mapo_tofu.mp4",
        "text": "Represent this video.",
    },
]

t1 = time.time()
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
similarities = model.similarity(query_embeddings, document_embeddings)
print(f"[2] 官方示例完成 ({time.time() - t1:.1f}s)")
print("相似度矩阵（行=query，列=[文本, 图片, 视频]）:")
print(similarities)

# ---------- 视频时刻检索 ----------
import glob

SEGMENT_SECONDS = 8
clip_paths = sorted(glob.glob("clips/clip_*.mp4"))
clips = [
    {"path": p, "start": i * SEGMENT_SECONDS, "end": (i + 1) * SEGMENT_SECONDS}
    for i, p in enumerate(clip_paths)
]
print(f"[3] 开始编码 {len(clips)} 个视频片段…")

t2 = time.time()
clip_docs = [{"video": c["path"], "text": "Represent this video."} for c in clips]
clip_embeddings = model.encode_document(clip_docs, batch_size=1, show_progress_bar=True)
per_clip = (time.time() - t2) / len(clips)
print(f"[4] 片段编码完成 ({time.time() - t2:.1f}s, 平均 {per_clip:.1f}s/段)")
print("向量矩阵形状:", clip_embeddings.shape)


def fmt(sec):
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def search(query, top_k=3):
    q = model.encode_query([query])
    scores = model.similarity(q, clip_embeddings)[0]
    ranked = scores.argsort(descending=True)[:top_k]
    print(f'\nQuery: "{query}"')
    for rank, idx in enumerate(ranked, 1):
        c = clips[int(idx)]
        print(f"  #{rank}  {fmt(c['start'])} - {fmt(c['end'])}  score={scores[idx]:.4f}")


search("a giant rabbit comes out of its burrow and stretches")
search("a butterfly flying near flowers")
search("three small animals looking down from a tree")
search("兔子从洞里钻出来伸懒腰")

print(f"\n全部完成，总耗时 {time.time() - t0:.1f}s")
