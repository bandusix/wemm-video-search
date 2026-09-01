# HuggingFace Space (Docker SDK) / 通用容器部署
# 注意：WeMM-Embedding-2B 需要 GPU 才实用；HF 免费 CPU Space 仅够跑通检索，索引会很慢。
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision \
    "transformers==5.2.0" "qwen-vl-utils==0.0.14" "sentence-transformers>=5.7.0" \
    "accelerate>=1.1.0" av torchcodec fastapi uvicorn python-multipart pillow

COPY server_v2.py .
COPY static/ ./static/

ENV WEMM_DATA=webapp_data_v2
EXPOSE 7860
CMD ["uvicorn", "server_v2:app", "--host", "0.0.0.0", "--port", "7860"]
