FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FINRAG_PROJECT_ROOT=/app \
    HF_HOME=/app/.hf_cache

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# CPU-only torch keeps the image several gigabytes smaller than the default wheel.
RUN pip install --upgrade pip \
    && pip install "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu \
    && pip install .
COPY configs ./configs
COPY scripts ./scripts
COPY data/manifest.json ./data/manifest.json
COPY data/held_out ./data/held_out
COPY artifacts ./artifacts
COPY app.py ./app.py

# Mount the downloaded FinQA splits at /app/data/raw and (optionally) a model cache
# at /app/.hf_cache: docker run -v $PWD/data/raw:/app/data/raw -p 8501:8501 finrag
VOLUME ["/app/data/raw", "/app/.hf_cache", "/app/artifacts"]

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
