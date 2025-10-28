FROM python:3.10-slim
# TODO: create a new Dockerfile for the FastAPI thing


# Set working directory
WORKDIR /app

# Install dependencies first to cache layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model using a temp Python script during build
RUN python -c "\
    from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L12-v2').save('/models/all-MiniLM-L12-v2')"

# Copy your app code
COPY . .

# Set offline mode and force local cache usage
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

CMD ["python", "main.py"]
