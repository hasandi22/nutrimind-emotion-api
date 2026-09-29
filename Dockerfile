FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image so containers don't re-download ~438MB on every
# start. Cache lands under HF_HOME (kept by the later COPY, which only merges).
ENV HF_HOME=/app/hf-cache
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Hasandi/nutrimind-emotion-model')"

COPY . .

# Model is already cached in the image, so run fully offline at runtime.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 5000

# Single worker: each worker would load its own ~1.2GB copy of the model, and
# only one fits in 2GB RAM. Threads give light concurrency with no extra copies.
# --preload loads the model in the master before forking.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "--preload", "app:app"]
