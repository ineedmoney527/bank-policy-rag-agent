# Dockerfile for AWS EC2 deployment
# BNM Policy RAG Agent with Gradio frontend + ChromaDB (using OpenRouter API)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# ChromaDB connection (can be overridden in docker-compose)
ENV CHROMA_HOST=chromadb
ENV CHROMA_PORT=8000
# Use OpenRouter API for LLM (no local Ollama needed)
ENV USE_OPENROUTER=true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
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

# Copy BNM policy documents for ingestion
COPY data/bnm/ ./data/bnm/

# Create directories for ChromaDB
RUN mkdir -p /app/chroma_db

# Create entrypoint script that runs ingestion if chroma_db is empty
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Check if ChromaDB has been ingested (check for parent_store.pkl)\n\
if [ ! -f /app/chroma_db/parent_store.pkl ]; then\n\
    echo "ChromaDB is empty. Running ingestion..."\n\
    python src/ingestion.py\n\
    echo "Ingestion complete!"\n\
else\n\
    echo "ChromaDB already has data. Skipping ingestion."\n\
fi\n\
\n\
# Start the app\n\
exec python app.py\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Create a non-root user for security
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Expose the Gradio port
EXPOSE 7860

# Health check using curl (more reliable)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run via entrypoint (checks and runs ingestion if needed)
CMD ["/app/entrypoint.sh"]
