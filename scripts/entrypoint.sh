#!/bin/bash
# Entrypoint script for BNM Policy RAG Agent
# Checks if data exists and runs ingestion if needed

set -e

echo "=== BNM RAG Agent Startup ==="
echo "CHROMA_HOST: ${CHROMA_HOST:-not set}"
echo "CHROMA_PORT: ${CHROMA_PORT:-8000}"

# Function to check ChromaDB document count
check_chromadb() {
    python3 -c "
import os
import sys

try:
    import chromadb
    from chromadb.config import Settings
    
    host = os.environ.get('CHROMA_HOST', '')
    port = int(os.environ.get('CHROMA_PORT', '8000'))
    
    if host:
        # HTTP client mode
        client = chromadb.HttpClient(host=host, port=port)
    else:
        # Local mode
        client = chromadb.PersistentClient(path='/app/chroma_db')
    
    # Try to get the collection
    try:
        collection = client.get_collection('bnm_docs')
        count = collection.count()
        print(f'ChromaDB document count: {count}')
        sys.exit(0 if count > 0 else 1)
    except Exception as e:
        print(f'Collection not found or empty: {e}')
        sys.exit(1)
        
except Exception as e:
    print(f'Error connecting to ChromaDB: {e}')
    sys.exit(1)
"
}

# Wait for ChromaDB to be ready (with retry)
echo ""
echo "Waiting for ChromaDB to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Use Python to check ChromaDB heartbeat (more portable than curl/wget)
    if python3 -c "import urllib.request; urllib.request.urlopen('http://${CHROMA_HOST:-localhost}:${CHROMA_PORT:-8000}/api/v1/heartbeat', timeout=5)" 2>/dev/null; then
        echo "ChromaDB is ready!"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Waiting for ChromaDB... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "WARNING: ChromaDB did not become ready in time, continuing anyway..."
fi

# Check if ingestion is needed
echo ""
echo "Checking if data exists..."

if check_chromadb; then
    echo "✓ ChromaDB already has data. Skipping ingestion."
else
    echo "ChromaDB is empty. Running ingestion..."
    echo ""
    
    # Run ingestion
    python -m src.ingestion
    
    echo ""
    echo "✓ Ingestion complete!"
fi

# Check parent store
if [ -f /app/chroma_db/parent_store.pkl ]; then
    echo "✓ Parent store exists."
else
    echo "⚠ Warning: Parent store not found at /app/chroma_db/parent_store.pkl"
fi

echo ""
echo "=== Starting Gradio App ==="
exec python app.py
