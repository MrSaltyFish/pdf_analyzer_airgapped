# Base image — using lightweight CUDA-capable if available, fallback to slim
# For GPU: nvidia/cuda:12.1.1-base-ubuntu22.04
# For CPU only: python:3.10-slim
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# -----------------------------
# Environment + Work Directory
# -----------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    MODEL_DIR=/models/mistral \
    MODEL_NAME=mistral-7b-v0.1.Q4_0 \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# -----------------------------
# Install Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps separately for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------
# ✅ Copy Pre-downloaded Model (if exists)
# ---------------------------------------
# Avoid re-downloading large weights from HF
RUN mkdir -p /models

# --------------------------------------------
# 🧩 Fallback: Auto-download model if not found
# --------------------------------------------
RUN if [ ! -d "${MODEL_DIR}" ]; then \
    echo "⚠️ Mistral model not found locally, downloading from HF..."; \
    python -c "\
    from transformers import AutoModelForCausalLM, AutoTokenizer; \
    m = 'mistralai/Mistral-7B-Instruct-v0.2'; \
    AutoTokenizer.from_pretrained(m, cache_dir='/models/mistral'); \
    AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto', device_map='auto', cache_dir='/models/mistral')"; \
    else \
    echo '✅ Using locally cached Mistral model'; \
    fi

# -----------------------------
# Copy Application Code
# -----------------------------
COPY . .

EXPOSE 8002

# -----------------------------
# Entry Point
# -----------------------------
CMD ["python", "run.py"]
