# Dockerfile for Hugging Face Spaces deployment
# BNM Policy RAG Agent with Gradio frontend

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for HF Spaces
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gradio>=4.0.0

# Copy project files
COPY src/ ./src/
COPY app.py .
COPY chroma_db/ ./chroma_db/
COPY data/ ./data/

# Create a non-root user for security (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user

# Expose the Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1

# Run the Gradio app
CMD ["python", "app.py"]
