FROM python:3.11-slim

WORKDIR /app

# Install system libraries required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and calibrated model bundle
COPY ARTSS/src ./ARTSS/src
COPY ARTSS/models/dr_efficientnet_b2_224_best.pth ./ARTSS/models/
COPY ARTSS/models/dr_efficientnet_b2_224_results.json ./ARTSS/models/
COPY ARTSS/models/preprocess_config.json ./ARTSS/models/

EXPOSE 8000

CMD ["uvicorn", "ARTSS.src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]