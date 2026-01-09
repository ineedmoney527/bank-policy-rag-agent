"""
FastAPI Server for Compliance-Aware Banking Agent.

Endpoints:
- POST /query: Submit a query and get the agent's response
- POST /ingest: Trigger document ingestion
- GET /health: Health check endpoint
"""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Server configuration from environment or defaults
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
from src.graph import run_agent
from src.ingest import run_ingestion
from src.retriever import get_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    include_sources: bool = Field(default=True, description="Include source documents in response")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    response: str
    status: str
    iterations: int
    sources: Optional[list] = None


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    force: bool = Field(default=False, description="Force re-ingestion even if data exists")


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    success: bool
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    retriever_ready: bool
    details: dict


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes the retriever on startup.
    """
    logger.info("Starting Compliance-Aware Banking Agent...")
    
    # Initialize retriever
    retriever = get_retriever()
    try:
        retriever.initialize()
        logger.info("Retriever initialized successfully")
    except Exception as e:
        logger.warning(f"Retriever initialization failed (run /ingest first): {e}")
    
    yield
    
    logger.info("Shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Compliance-Aware Banking Agent",
    description="A RAG-based banking agent for Malaysian market (BNM regulations)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Submit a query to the compliance agent.
    
    The agent will:
    1. Retrieve relevant documents using hybrid search
    2. Generate a response using Ollama/Mistral
    3. Validate the response for compliance
    4. Return the verified answer or a fallback message
    """
    try:
        logger.info(f"Received query: {request.query[:50]}...")
        
        # Run the agent
        result = run_agent(request.query)
        
        # Extract sources if requested
        sources = None
        if request.include_sources and result.get("retrieved_docs"):
            sources = [
                {
                    "filename": doc["metadata"].get("filename"),
                    "category": doc["metadata"].get("category"),
                    "score": doc.get("score")
                }
                for doc in result["retrieved_docs"]
            ]
        
        return QueryResponse(
            response=result["final_response"],
            status=result["compliance_status"],
            iterations=result["iteration_count"],
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Trigger document ingestion.
    
    This will:
    1. Parse PDFs from /data/bnm and /data/product
    2. Chunk and tag documents with metadata
    3. Store in ChromaDB and build BM25 index
    """
    try:
        logger.info("Starting document ingestion...")
        
        run_ingestion()
        
        # Reinitialize retriever
        retriever = get_retriever()
        retriever._initialized = False
        retriever.initialize()
        
        return IngestResponse(
            success=True,
            message="Documents ingested successfully"
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return IngestResponse(
            success=False,
            message=f"Ingestion failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the agent and its components.
    """
    retriever = get_retriever()
    
    details = {
        "ollama_model": "mistral",
        "embedding_model": "all-MiniLM-L6-v2",
        "reranker_model": "ms-marco-MiniLM-L-6-v2"
    }
    
    return HealthResponse(
        status="healthy",
        retriever_ready=retriever._initialized,
        details=details
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Compliance-Aware Banking Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the server."""
    uvicorn.run(
        "src.server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True
    )


if __name__ == "__main__":
    main()
