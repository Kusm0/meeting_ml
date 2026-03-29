FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create directories for models, metrics and checkpoints
RUN mkdir -p models/tfidf/decision models/tfidf/topic_type models/tfidf/da \
    models/bert/decision models/bert/topic_type models/bert/da \
    metrics/tfidf/decision metrics/tfidf/topic_type metrics/tfidf/da \
    metrics/bert/decision metrics/bert/topic_type metrics/bert/da \
    checkpoints/bert/decision checkpoints/bert/topic_type checkpoints/bert/da \
    data datasets

# Expose ports
EXPOSE 8000 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: run both API and Gradio
CMD ["python", "-m", "src.web.gradio_app"]

