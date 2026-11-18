FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# CPU-only PyTorch from the official PyTorch index keeps the image small.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fixed-weight TorchScript models and sample images are shipped in the repo:
#   app/model.pt          normal 10-class classifier
#   app/model_anomaly.pt  demo model that raises on overexposed frames
#   samples/              incl. sample_overexposed.png (the "poison" frame)
COPY app ./app
COPY samples ./samples

# Runtime configuration (all optional, defaults shown):
#   MAX_CAPACITY=16        max images queued or in-flight
#   BATCH_SIZE=4           max images per batch
#   BATCH_WAIT_MS=50       idle model waits at most this long to fill a batch
#   RETRY_ATTEMPTS=1       bounded per-frame retries after a batch failure
#   INFERENCE_DELAY_MS=0   artificial delay to demo slow inference
#   MODEL_PATH             set to /srv/app/model_anomaly.pt for the anomaly demo
ENV MAX_CAPACITY=16 \
    BATCH_SIZE=4 \
    BATCH_WAIT_MS=50 \
    RETRY_ATTEMPTS=1 \
    INFERENCE_DELAY_MS=0

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
