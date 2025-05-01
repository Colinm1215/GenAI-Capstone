FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy code
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_trf

# Set Transformer Cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# Expose Flask port
ENV PORT=7860
CMD ["python", "app.py"]