"""
BNM Policy Document Retriever using LangChain Built-in Components

This module implements a production-ready retrieval pipeline:
1. EnsembleRetriever (Vector + BM25 with weighted fusion)
2. ContextualCompressionRetriever (Cross-encoder reranking)
3. Child-to-parent expansion
4. Multi-query expansion with BNM terminology
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Ensemble and Compression moved to langchain_classic in this environment
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# =============================================================================
# Configuration
# =============================================================================

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Check if using OpenRouter (cloud deployment without local Ollama)
USE_OPENROUTER = os.environ.get("USE_OPENROUTER", "false").lower() == "true"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Ollama configuration (for local development)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Embedding model (HuggingFace - works everywhere, no external API needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM for query expansion 
# - Uses OpenRouter if USE_OPENROUTER=true
# - Falls back to Ollama for local development
QUERY_EXPANSION_MODEL_OPENROUTER = "xiaomi/mimo-v2-flash:free"
QUERY_EXPANSION_MODEL_OLLAMA = "qwen2.5:3b-instruct"

# ChromaDB settings
CHROMA_COLLECTION_NAME = "bnm_docs"
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")

# Retrieval settings
ENSEMBLE_VECTOR_WEIGHT = 0.7
ENSEMBLE_BM25_WEIGHT = 0.3
INITIAL_K = 20  # Documents to retrieve before reranking
RERANK_TOP_N = 5   # Documents to return after reranking

# Cross-encoder model for reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


# =============================================================================
# Global State (Lazy Initialization)
# =============================================================================

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None
_ensemble_retriever: Optional[EnsembleRetriever] = None
_compression_retriever: Optional[ContextualCompressionRetriever] = None
_query_expansion_llm: Optional[Any] = None  # Can be ChatOllama or ChatOpenAI
_all_docs_for_bm25: Optional[List[Document]] = None


# =============================================================================
# Initialization Functions
# =============================================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or initialize HuggingFace embeddings (works without external API)."""
    global _embeddings
    if _embeddings is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        print(f"Embeddings loaded successfully")
    return _embeddings


def get_vectorstore() -> Chroma:
    """Get or initialize Chroma vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings()
        )
        print(f"Connected to Chroma: {CHROMA_COLLECTION_NAME}")
    return _vectorstore


def get_all_docs_for_bm25() -> List[Document]:
    """Load all documents from vectorstore for BM25."""
    global _all_docs_for_bm25
    if _all_docs_for_bm25 is None:
        vectorstore = get_vectorstore()
        # Get all documents from Chroma
        results = vectorstore.get(include=["documents", "metadatas"])
        _all_docs_for_bm25 = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        print(f"Loaded {len(_all_docs_for_bm25)} documents for BM25")
    return _all_docs_for_bm25


def get_ensemble_retriever() -> EnsembleRetriever:
    """Get or initialize EnsembleRetriever (Vector + BM25)."""
    global _ensemble_retriever
    if _ensemble_retriever is None:
        # Vector retriever from Chroma
        vectorstore = get_vectorstore()
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": INITIAL_K})
        
        # BM25 retriever
        all_docs = get_all_docs_for_bm25()
        if not all_docs:
            print("WARNING: No documents found in database. Using only vector retriever.")
            # Dummy ensemble with only vector retriever
            # Or just return vector_retriever (but the return type is EnsembleRetriever)
            # We can create a dummy list for BM25 to avoid crashing
            all_docs = [Document(page_content="Empty Database Placeholder", metadata={})]
        
        bm25_retriever = BM25Retriever.from_documents(all_docs, k=INITIAL_K)
        
        # Ensemble with weighted fusion
        _ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[ENSEMBLE_VECTOR_WEIGHT, ENSEMBLE_BM25_WEIGHT]
        )
        print(f"Initialized EnsembleRetriever (vector={ENSEMBLE_VECTOR_WEIGHT}, bm25={ENSEMBLE_BM25_WEIGHT})")
    return _ensemble_retriever


def get_compression_retriever() -> ContextualCompressionRetriever:
    """Get or initialize ContextualCompressionRetriever with cross-encoder reranking."""
    global _compression_retriever
    if _compression_retriever is None:
        # Cross-encoder reranker
        cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=RERANK_TOP_N)
        
        # Wrap ensemble retriever with compression
        _compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=get_ensemble_retriever()
        )
        print(f"Initialized CrossEncoderReranker: {CROSS_ENCODER_MODEL}")
    return _compression_retriever


def get_query_expansion_llm():
    """Get or initialize LLM for query expansion (OpenRouter or Ollama)."""
    global _query_expansion_llm
    if _query_expansion_llm is None:
        if USE_OPENROUTER and OPENROUTER_API_KEY:
            # Use OpenRouter API (cloud deployment)
            from langchain_openai import ChatOpenAI
            print(f"Using OpenRouter for query expansion: {QUERY_EXPANSION_MODEL_OPENROUTER}")
            _query_expansion_llm = ChatOpenAI(
                model=QUERY_EXPANSION_MODEL_OPENROUTER,
                temperature=0.3,
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY
            )
        else:
            # Use local Ollama (development)
            try:
                from langchain_ollama import ChatOllama
                print(f"Using Ollama for query expansion: {QUERY_EXPANSION_MODEL_OLLAMA}")
                _query_expansion_llm = ChatOllama(
                    model=QUERY_EXPANSION_MODEL_OLLAMA,
                    base_url=OLLAMA_HOST,
                    temperature=0.3
                )
            except Exception as e:
                print(f"Ollama not available ({e}), query expansion disabled")
                _query_expansion_llm = None
    return _query_expansion_llm


# =============================================================================
# Query Expansion
# =============================================================================

def expand_query_to_bnm_speak(query: str) -> List[str]:
    """
    Multi-Query Expansion: Translate user query into BNM terminology.
    
    Generates alternative phrasings using official BNM regulatory language.
    """
    llm = get_query_expansion_llm()
    
    # If LLM not available, return original query
    if llm is None:
        print("  Query expansion skipped (no LLM available)")
        return [query]
    
    expansion_prompt = f"""You are an expert on Bank Negara Malaysia (BNM) regulatory terminology.

Translate this user question into official BNM regulatory language. Generate 2-3 alternative phrasings using:
- Official BNM terms (e.g., "financial institution" instead of "bank")
- Regulatory terminology (e.g., "prudential requirements" instead of "rules")
- Common BNM policy phrases (e.g., "shall ensure", "minimum threshold")

User Query: {query}

Return JSON with query variations:
{{"queries": ["variation 1", "variation 2", "variation 3"]}}"""

    try:
        response = llm.invoke(expansion_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"  Query expansion failed (model unavailable): {e}")
        return [query]  # Fallback to original query
    
    try:
        # Clean and parse JSON
        content = re.sub(r'```\w*\s*', '', content).replace('```', '')
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            content = content[start:end+1]
        
        result = json.loads(content)
        variations = result.get("queries", [])
        return [query] + variations[:3]
    except:
        return [query]


# =============================================================================
# Main Retrieval Function
# =============================================================================

def get_relevant_documents(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    num_results: int = 6,
    use_query_expansion: bool = False
) -> List[Document]:
    """
    Retrieve relevant BNM policy documents.
    
    Pipeline:
    1. (Optional) Multi-query expansion with BNM terminology
    2. EnsembleRetriever (Vector + BM25 with RRF fusion)
    3. CrossEncoderReranker for precision
    
    Args:
        query: User search query
        filters: Optional metadata filters (applied post-retrieval)
        num_results: Number of documents to return
        use_query_expansion: If True, expand query with BNM terminology
    
    Returns:
        List of relevant Document objects
    """
    # Step 1: Optional query expansion
    if use_query_expansion:
        print("Expanding query to BNM terminology...")
        query_variations = expand_query_to_bnm_speak(query)
        print(f"  Generated {len(query_variations)} query variations")
    else:
        query_variations = [query]
    
    # Step 2: Retrieve with compression (ensemble + reranking)
    retriever = get_compression_retriever()
    
    all_docs = []
    seen_content = set()
    
    for q in query_variations:
        print(f"Retrieving for: {q[:50]}...")
        docs = retriever.invoke(q)
        
        # Deduplicate by content hash
        for doc in docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
    
    print(f"  Found {len(all_docs)} unique documents")
    
    # Step 3: Apply metadata filters (post-retrieval)
    # Supports both single-value and list-based metadata
    if filters:
        filtered_docs = []
        for doc in all_docs:
            matches = True
            for key, value in filters.items():
                print(f"Applying filter: {key} = {value}")
                meta_value = doc.metadata.get(key, "")
                
                # Handle JSON-serialized lists
                if isinstance(meta_value, str) and meta_value.startswith("["):
                    try:
                        import json
                        meta_value = json.loads(meta_value)
                    except:
                        pass
                
                # Check if filter value is in list or equals string
                if isinstance(meta_value, list):
                    if value not in meta_value:
                        matches = False
                        break
                else:
                    if str(meta_value) != str(value):
                        matches = False
                        break
            
            if matches:
                filtered_docs.append(doc)
        
        # FALLBACK: If topic filter removed ALL docs, drop topic filter and use semantic results
        if len(filtered_docs) == 0 and len(all_docs) > 0 and "topic" in filters:
            print(f"  Note: Topic filter '{filters['topic']}' removed all {len(all_docs)} docs, using unfiltered results")
            # Keep other filters (like is_islamic) but drop topic
            other_filters = {k: v for k, v in filters.items() if k != "topic"}
            if other_filters:
                # Re-filter with remaining filters only
                for doc in all_docs:
                    matches = True
                    for key, value in other_filters.items():
                        meta_value = doc.metadata.get(key, "")
                        if str(meta_value) != str(value):
                            matches = False
                            break
                    if matches:
                        filtered_docs.append(doc)
            else:
                # No other filters, use all docs
                filtered_docs = all_docs
        
        all_docs = filtered_docs
        print(f"  After filtering: {len(all_docs)} documents")
    
    return all_docs[:num_results]


# =============================================================================
# Convenience Functions
# =============================================================================

def search_standards(query: str, num_results: int = 6) -> List[Document]:
    """Search only mandatory Standard (S) requirements."""
    return get_relevant_documents(query, filters={"regulatory_type": "Standard"}, num_results=num_results)


def search_guidance(query: str, num_results: int = 6) -> List[Document]:
    """Search only Guidance (G) best practices."""
    return get_relevant_documents(query, filters={"regulatory_type": "Guidance"}, num_results=num_results)


def search_islamic(query: str, num_results: int = 6) -> List[Document]:
    """Search only Islamic finance / Shariah-related documents."""
    return get_relevant_documents(query, filters={"is_islamic": True}, num_results=num_results)


def search_with_expansion(query: str, num_results: int = 6) -> List[Document]:
    """Search with multi-query expansion using BNM terminology."""
    return get_relevant_documents(query, use_query_expansion=True, num_results=num_results)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the minimum monthly repayment requirements for credit cards?"
    
    print(f"\nQuery: {query}\n")
    print("=" * 60)
    
    docs = get_relevant_documents(query)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source_pdf', 'N/A')}")
        print(f"Product: {doc.metadata.get('product_type', 'N/A')}")
        print(f"Domain: {doc.metadata.get('regulatory_domain', 'N/A')}")
        print(f"Content Preview: {doc.page_content[:500]}...")
