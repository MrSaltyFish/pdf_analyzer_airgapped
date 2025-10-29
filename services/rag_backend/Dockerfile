FROM python:3.10-slim

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    SENTENCE_TRANSFORMERS_HOME=/models

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# ✅ Copy local model if available
# -------------------------------
# This prevents redownloading during build
# Make sure the folder exists in `../shared/models/...`
ARG MODEL_NAME=all-MiniLM-L12-v2
COPY ../../shared/models/${MODEL_NAME} /models/${MODEL_NAME}

# Fallback: if the folder doesn’t exist (fresh clone),
# download the model automatically.
RUN if [ ! -d "/models/${MODEL_NAME}" ]; then \
    echo "⚠️ Model not found locally, downloading..."; \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}').save('/models/${MODEL_NAME}')"; \
    else \
    echo "✅ Using pre-downloaded model from shared folder"; \
    fi

# Copy application code
COPY . .

EXPOSE 8001

CMD ["python", "run.py"]
