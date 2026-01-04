"""
Self-Correcting RAG Workflow using LangGraph

This module implements a sophisticated RAG pipeline with:
1. Query decomposition for complex questions
2. Metadata filter extraction
3. Retrieval (Fetches Child Chunks first for speed)
4. Document grading (Grades small chunks)
5. Expansion (Swaps relevant chunks for full Parent Docs)
6. Generation with long-context reordering
7. Hallucination checking
"""

import hashlib
import pickle
import os
import re
from pathlib import Path
from typing import TypedDict, List, Annotated, Literal, Optional, Dict, Union
from operator import add
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field, field_validator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers import LongContextReorder
from langgraph.graph import StateGraph, END

# Import retriever function
from retriever import get_relevant_documents

# =============================================================================
# Global Cache for Parent Store (loaded once, not per-request)
# =============================================================================
# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

_parent_store_cache: Optional[Dict[str, Document]] = None
_parent_store_mtime: Optional[float] = None  # Track file modification time
PARENT_STORE_PATH = PROJECT_ROOT / "chroma_db" / "parent_store.pkl"

def get_parent_store() -> Dict[str, Document]:
    """Get cached parent store, reloading if file was modified."""
    global _parent_store_cache, _parent_store_mtime
    
    if Path(PARENT_STORE_PATH).exists():
        current_mtime = Path(PARENT_STORE_PATH).stat().st_mtime
        
        # Reload if file was modified since last load
        if _parent_store_cache is None or _parent_store_mtime != current_mtime:
            with open(PARENT_STORE_PATH, "rb") as f:
                _parent_store_cache = pickle.load(f)
            _parent_store_mtime = current_mtime
            print(f"Loaded {len(_parent_store_cache)} parents from disk")
    else:
        _parent_store_cache = {}
    
    return _parent_store_cache


# =============================================================================
# Configuration
# =============================================================================
# OpenRouter Setup
# -----------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

MODEL_NAME = "xiaomi/mimo-v2-flash:free"
# LLM_MODEL = "mistral:7b-instruct"  # Using Qwen 2.5 7B (available locally)
MAX_RETRY_COUNT = 3
MAX_REVISION_COUNT = 2


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class QueryRefinement(BaseModel):
    """Refined query with changes made."""
    refined_query: str = Field(description="The refined query")
    changes_made: List[str] = Field(description="List of changes made to the query")

class QueryDecomposition(BaseModel):
    """Query decomposition result."""
    needs_decomposition: bool = Field(description="Whether the query needs decomposition")
    sub_questions: Optional[List[str]] = Field(default_factory=list, description="List of sub-questions")
    
    @field_validator("sub_questions", mode="before")
    def convert_none_to_empty_list(cls, v):
        """Convert None to empty list to handle LLM returning null."""
        if v is None:
            return []
        return v

class MetadataFilters(BaseModel):
    """Metadata filters extracted from query."""
    topic: Optional[str] = Field(None, description="Document topic like 'Debit Card', 'Credit Card', 'E-Money'")
    is_islamic: Optional[bool] = Field(None, description="Filter for Islamic finance")
    regulatory_type: Optional[str] = Field(None, description="Standard or Guidance")
    part_id: Optional[str] = Field(None, description="PART identifier")
    section_id: Optional[str] = Field(None, description="SECTION identifier")

class GradingResult(BaseModel):
    """Batch grading result."""
    verdicts: List[str] = Field(description="List of RELEVANT or IRRELEVANT verdicts")

class HallucinationCheck(BaseModel):
    """Hallucination check result."""
    status: Literal["Pass", "Fail"] = Field(description="Whether answer is grounded")
    critique: Optional[Union[str, List[str]]] = Field(default="", description="Specific unsupported claim or empty if Pass")

    @field_validator("critique")
    def convert_critique_to_string(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return "; ".join(str(i) for i in v)
        return v

# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict):
    """State for the self-correcting RAG workflow."""
    original_query: str                      # User's initial input
    current_query: str                       # Query being used for retrieval
    sub_questions: List[str]  # Result of decomposition
    documents: List[Document]                # Retrieved & graded context
    generation: str                          # LLM's answer
    filters: dict                            # Metadata filters
    retry_count: int                         # Track retrieval attempts
    revision_count: int                      # Track hallucination fix loops
    grade_status: str                        # "Relevant" or "Irrelevant"
    hallucination_status: str                # "Pass" or "Fail"
    critique: str                            # Error message


# =============================================================================
# Initialize LLM
# =============================================================================

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


# =============================================================================
# Node 0: Refine Query (Clarify & Add Context)
# =============================================================================

def refine_query(state: AgentState) -> AgentState:
    """
    Refine the user query to avoid vagueness:
    - Clarify ambiguous terms
    - Add domain-specific context (BNM terminology)
    - Expand abbreviations
    - Make implicit context explicit
    """
    print("\n[Node: refine_query]")
    original_query = state["original_query"]
    
    refine_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Bank Negara Malaysia (BNM) regulatory research assistant.

Your task is to MINIMALLY refine a user query for document retrieval.

STRICT RULES:
- Expand abbreviations in the query **only if you are certain of the correct full form** in context. For example, "KPI" → "Key Performance Indicator", "CPI" → "Consumer Price Index".
- Preserve the exact original meaning of the query. Do not substitute synonyms.
- Do not introduce new acronyms, abbreviations, or capitalized terms.
- Do not rewrite the query structure.
- Only fix obvious typos, grammatical mistakes, or add the context "BNM" if missing and necessary for clarity.

If the query is already clear and specific: return it unchanged.

Note:
'takaful' → 'takaual' is incorrect, takaful is a valid word

OUTPUT FORMAT:
- refined_query: the corrected query (unchanged if already clear)
- changes_made: a list of changes made (empty if none)
""")
,
    ("user", "{query}")
])


    
    llm_with_structure = llm.with_structured_output(QueryRefinement)
    chain = refine_prompt | llm_with_structure
    result = chain.invoke({"query": original_query})
    
    if result.refined_query and result.refined_query != original_query:
        state["current_query"] = result.refined_query
        print(f"  Original: {original_query}")
        print(f"  Refined:  {result.refined_query}")
        if result.changes_made:
            print(f"  Changes:  {', '.join(result.changes_made)}")
    else:
        state["current_query"] = original_query
        print("  Query is already clear, no refinement needed")
    
    return state


# =============================================================================
# Node 1: Query Decomposition
# =============================================================================

def decompose_query(state: AgentState) -> AgentState:
    print("\n[Node: decompose_query]")
    original_query = state["original_query"]
    # Use the refined query from Node 0
    query_to_decompose = state.get("current_query", state["original_query"])
    
    decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are analyzing BNM policy queries to determine if they must be broken into separate retrievals.

CRITICAL: Decompose if the query contains MULTIPLE DISTINCT INFORMATION REQUESTS that may reside in different document sections.

## WHEN TO DECOMPOSE:

1. **Multiple numeric values**: Query asks for "minimum AND maximum" (two retrievals)
2. **Multiple entities**: Query asks about "banks AND insurers" (two different regulated entities)
3. **Multiple unrelated requirements**: Query asks about "licensing requirements AND reporting requirements" (likely in different sections)
4. **Multiple distinct aspects of same topic**: Query asks about BOTH:
   - Conditions/criteria/requirements to do X, AND
   - Obligations/duties/responsibilities when doing X
   - These are distinct pieces of information even if about the same topic

## WHEN NOT TO DECOMPOSE:

- Simple queries about one topic: "What are the requirements for X?"
- Queries about a single process: "What is the process for customer due diligence?"
- Simple list queries: "List the capital adequacy requirements"
- Any query that asks for ONE type of information about ONE topic

## EXAMPLES:

DO NOT DECOMPOSE:
- "What are the requirements for electronic money issuers?"
- "What is the process for customer due diligence?"
- "List the capital adequacy requirements"
- "What conditions must be met for Simplified CDD?" (single aspect)

DECOMPOSE:
- "What are the capital requirements for banks and insurers?" 
  → ["What are the capital requirements for banks?", "What are the capital requirements for insurers?"]
- "What are the minimum and maximum loan amounts?"
  → ["What is the minimum loan amount?", "What is the maximum loan amount?"]
- "What are the conditions for Simplified CDD, and what obligations remain when it is applied?"
  → ["What are the conditions that must be met to apply Simplified CDD?", "What obligations remain even when Simplified CDD is applied?"]
- "What triggers Enhanced CDD, and what additional measures are required?"
  → ["What triggers Enhanced Customer Due Diligence?", "What additional measures are required under Enhanced CDD?"]

Return JSON with needs_decomposition and sub_questions.
"""),
    ("user", "{query}")
])

    
    llm_with_structure = llm.with_structured_output(QueryDecomposition)
    chain = decompose_prompt | llm_with_structure
    result = chain.invoke({"query": query_to_decompose})
    
    if result.needs_decomposition:
        # Ensure sub_questions are strings
        sub_questions = [str(sq) for sq in result.sub_questions]
        print(f"  Decomposed into {len(sub_questions)} sub-questions")
        for i, sq in enumerate(sub_questions, 1):
            print(f"    {i}. {sq}")
        state["sub_questions"] = sub_questions
    else:
        print("  Query is simple, no decomposition needed")
        state["sub_questions"] = []
    
    return state


# =============================================================================
# Node 2: Extract Metadata Filters
# =============================================================================

def extract_filters(state: AgentState) -> AgentState:
    print("\n[Node: extract_filters]")
    original_query = state["original_query"]
    
    filter_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a metadata filter extractor for Bank Negara Malaysia (BNM) policy documents.

Your job is to extract STRICT metadata filters only when the user's intent is explicit and unambiguous.

IMPORTANT PRINCIPLE:
Filters must reflect explicit document scoping, NOT inferred intent.

## Available Filters
- topic: Document topic (see Topic Rules below)
- regulatory_type: "Standard" or "Guidance" ONLY
- is_islamic: true or false ONLY
- part_id
- section_id

## Topic Rules (CRITICAL for product filtering)

Set topic based on EXPLICIT product mentions in the query:
w2çç
| Query mentions | Set topic to |
|---------------|--------------|
| "debit card", "debit card-i" | "Debit Card" |
| "credit card", "credit card-i" | "Credit Card" |
| "e-money", "electronic money", "e-wallet" | "E-Money" |
| "takaful" (non-medical) | "Takaful" |
| "RMiT", "technology risk" | "rmit Technology Risk" |

IMPORTANT: Only set topic if the product type is EXPLICITLY mentioned. Never guess.

## Regulatory Type Rules (CRITICAL)

1. Set regulatory_type = "Standard" ONLY IF:
   - The query explicitly asks for legally binding obligations
   - AND contains NO guidance-style language

   Examples:
   - "What is the standard requirement for capital adequacy?"
   - "mandatory requirement"
   - "must comply with BNM"
   - "BNM requires issuers to"

2. Set regulatory_type = "Guidance" ONLY IF:
   - "What is the BNM guidance on ......"
   - The query explicitly asks for recommendations or best practices
   - AND contains NO mandatory language

   Examples:
   - "best practice"
   - "recommended approach"
   - "BNM guidance on"

3. If BOTH mandatory and guidance signals appear:
   - DO NOT set regulatory_type (leave null)

4. If intent is unclear or exploratory:
   - DO NOT set regulatory_type

## Islamic Rule
- Set is_islamic = true ONLY if Islamic terms, like "Islamic", "takaful", "Shariah", "credit card-i", "debit card-i" are explicitly mentioned

## Absolute Rule
- Never guess.
- Never infer.
- When in doubt, leave null.
"""),
        ("user", "{query}")
    ])
    
    # Use structured output for reliable parsing
    llm_with_structure = llm.with_structured_output(MetadataFilters)
    chain = filter_prompt | llm_with_structure
    result = chain.invoke({"query": original_query})
    
    # Convert Pydantic model to dict, excluding None values
    filters = result.model_dump(exclude_none=True)
    
    # VALIDATION: Remove invalid filter values that the LLM might hallucinate
    valid_regulatory_types = {"Standard", "Guidance"}
    if "regulatory_type" in filters:
        if filters["regulatory_type"] not in valid_regulatory_types:
            print(f"  Removing invalid regulatory_type: {filters['regulatory_type']}")
            del filters["regulatory_type"]
        # FIX: Prevent aggressive Standard/Guidance inference (only keep if explicit keywords present)
        elif filters["regulatory_type"] == "Standard":
            # Only keep Standard if query has explicit mandatory-style language
            standard_keywords = ["mandatory", "must comply", "legally binding", "bnm requires", "shall", "compulsory"]
            is_explicit = any(kw in original_query.lower() for kw in standard_keywords)
            if not is_explicit:
                print(f"  Removing hallucinated regulatory_type=Standard (no explicit mandatory keyword)")
                del filters["regulatory_type"]
        elif filters["regulatory_type"] == "Guidance":
            # Only keep Guidance if query explicitly asks for recommendations
            guidance_keywords = ["best practice", "recommended", "guidance on", "advisable", "suggestion"]
            is_explicit = any(kw in original_query.lower() for kw in guidance_keywords)
            if not is_explicit:
                print(f"  Removing hallucinated regulatory_type=Guidance (no explicit guidance keyword)")
                del filters["regulatory_type"]

    
    if "is_islamic" in filters:
        if not isinstance(filters["is_islamic"], bool):
            print(f"  Removing invalid is_islamic: {filters['is_islamic']}")
            del filters["is_islamic"]
        # FIX: Prevent aggressive False inference (treat as None unless "conventional" is explicitly asked)
        elif filters["is_islamic"] is False:
             # Basic check for explicit non-islamic intent
            non_islamic_keywords = ["conventional", "non-islamic", "standard", "commercial"]
            is_explicit = any(kw in original_query.lower() for kw in non_islamic_keywords)
            if not is_explicit:
                print("  Removing hallucinated is_islamic=False (no explicit keyword)")
                del filters["is_islamic"]
    
    # VALIDATION: Validate and normalize topic filter
    if "topic" in filters:
        # FIX: Check if query is about cross-cutting regulatory concepts
        # These concepts appear across multiple policy documents (e.g., CDD in AML/CFT policy)
        # and should NOT be filtered by product-specific topics
        cross_cutting_concepts = [
            "cdd", "customer due diligence", "simplified cdd", "enhanced cdd",
            "aml", "anti-money laundering", "cft", "counter financing of terrorism",
            "kyc", "know your customer", "know-your-customer",
            "customer identification", "beneficial owner",
            "screening", "sanction", "pep", "politically exposed person",
            "transaction monitoring", "suspicious transaction"
        ]
        query_lower = original_query.lower()
        is_cross_cutting = any(concept in query_lower for concept in cross_cutting_concepts)
        
        if is_cross_cutting:
            print(f"  Removing topic filter '{filters['topic']}' (query about cross-cutting regulatory concept)")
            del filters["topic"]
        else:
            # Normalize topic to match metadata format
            topic_value = filters["topic"]
            # Map common LLM outputs to actual metadata values
            topic_map = {
                # Short forms → Full metadata values
                "debit card": "Debit Card Policy",
                "debit card policy": "Debit Card Policy",
                "credit card": "Credit Card Policy",
                "credit card policy": "Credit Card Policy",
                "e-money": "Electronic Money Policy",
                "e-money policy": "Electronic Money Policy",
                "electronic money": "Electronic Money Policy",
                "electronic money policy": "Electronic Money Policy",
                "rmit": "Rmit Technology Risk Policy",
                "rmit policy": "Rmit Technology Risk Policy",
                "rmit technology risk": "Rmit Technology Risk Policy",
                "rmit technology risk policy": "Rmit Technology Risk Policy",
                "technology risk": "Rmit Technology Risk Policy",
                "technology risk policy": "Rmit Technology Risk Policy",
                # MHIT - Medical Health Insurance Takaful
                "mhit": "Medical Health Insurance Takaful Policy",
                "mhit policy": "Medical Health Insurance Takaful Policy",
                "medical health insurance": "Medical Health Insurance Takaful Policy",
                "medical health insurance takaful": "Medical Health Insurance Takaful Policy",
                "medical health insurance takaful policy": "Medical Health Insurance Takaful Policy",
                "medical insurance": "Medical Health Insurance Takaful Policy",
                "health insurance": "Medical Health Insurance Takaful Policy",
                "health takaful": "Medical Health Insurance Takaful Policy",
                # Life Insurance / Family Takaful
                "life insurance": "Life Insurance Family Takaful Framework",
                "life insurance family takaful": "Life Insurance Family Takaful Framework",
                "life insurance family takaful framework": "Life Insurance Family Takaful Framework",
                "family takaful": "Life Insurance Family Takaful Framework",
            }
            normalized = topic_map.get(topic_value.lower().strip(), topic_value)
            filters["topic"] = normalized
    
    # The retriever expects "filters" as a dict
    state["filters"] = filters
    print(f"  Extracted filters: {filters}")
    return state


# =============================================================================
# Node 3: Retrieve Documents (MODIFIED: Returns Children)
# =============================================================================

def retrieve(state: AgentState) -> AgentState:
    """
    Retrieve CHILD CHUNKS using hybrid search. 
    We get more results (10-15) because grading small chunks is fast.
    """
    print("\n[Node: retrieve]")
    
    current_query = state["current_query"]
    sub_questions = state.get("sub_questions", [])
    filters = state.get("filters", {})
    retry_count = state["retry_count"]
    use_expansion = retry_count > 0  # Use query expansion on retries
    
    if use_expansion:
        print(f"  Retry attempt {retry_count} - enabling query expansion")
    
    all_docs = []
    
    if sub_questions:
        print(f"  Retrieving for {len(sub_questions)} sub-questions...")
        for sub_q in sub_questions:
            docs = get_relevant_documents(
                sub_q,
                filters=filters,
                num_results=5, 
                use_query_expansion=use_expansion
            )
            print(f"  DEBUG: Input filters = {filters}")
            for doc in docs[:3]:
                print(f"  DEBUG: Doc metadata = {doc.metadata}")
            all_docs.extend(docs)
    else:
        print(f"  Retrieving for: {current_query[:60]}...")
        all_docs = get_relevant_documents(
            current_query,
            filters=filters,
            num_results=10,
            use_query_expansion=use_expansion
        )
    
    # Deduplicate by content hash
    seen_hashes = set()
    unique_docs = []
    for doc in all_docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    print(f"  Retrieved {len(unique_docs)} unique child chunks")
    state["documents"] = unique_docs
    return state


# =============================================================================
# Node 4: Grade Documents (With Parent Expansion)
# =============================================================================

def grade_documents(state: AgentState) -> AgentState:
    """
    Grade retrieved clause chunks for relevance using BATCH grading.
    Then expand to parent SECTION documents for fuller context.
    """
    print("\n[Node: grade_documents]")
    
    documents = state["documents"]
    current_query = state["current_query"]
    
    if not documents:
        state["grade_status"] = "Irrelevant"
        return state
    
    # FIX: Batch grading instead of N+1 individual LLM calls
    # Send all documents at once for grading
    batch_grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a STRICT relevance grader for Bank Negara Malaysia (BNM) policy documents.

You will be given:
- ONE query asking for SPECIFIC information
- MULTIPLE document excerpts

For EACH excerpt, decide: Does this excerpt DIRECTLY ANSWER the specific question?

RELEVANT means: The excerpt contains the EXACT answer to what was asked.
IRRELEVANT means: The excerpt does NOT answer the question, even if from the same policy.

EXAMPLES:
Query: "What is the minimum age requirement for debit card?"
- Excerpt about minimum age = RELEVANT
- Excerpt about transaction alerts = IRRELEVANT (wrong topic)
- Excerpt about debit card disclosure = IRRELEVANT (wrong topic)

Query: "What is the interest rate cap for credit cards?"
- Excerpt about interest rate limits = RELEVANT
- Excerpt about minimum repayment = IRRELEVANT (different topic)

STRICT RULES:
- ONLY mark RELEVANT if document contains the SPECIFIC ANSWER requested
- From same policy document but different topic = IRRELEVANT
- Mentions keywords but doesn't answer = IRRELEVANT
- General information without specific answer = IRRELEVANT

DO NOT assume relevance. Be VERY strict.
If the excerpt does not stand on its own as an answer, mark it IRRELEVANT.

OUTPUT FORMAT:
- verdicts: list of "RELEVANT" or "IRRELEVANT", one per document, in order
"""),
    ("user", "Query:\n{query}\n\nDocuments:\n{documents}")
])

    
    # Format documents for batch grading (limit to first 800 chars each)
    doc_texts = []
    for i, doc in enumerate(documents):
        doc_texts.append(f"[Doc {i+1}]: {doc.page_content[:800]}")
    docs_formatted = "\n\n---\n\n".join(doc_texts)
    
    print(f"  Batch grading {len(documents)} chunks...")
    
    llm_with_structure = llm.with_structured_output(GradingResult)
    chain = batch_grade_prompt | llm_with_structure
    
    try:
        result = chain.invoke({
            "query": current_query,
            "documents": docs_formatted[:5000]  # Cap total length
        })
        
        relevant_chunks = []
        for i, doc in enumerate(documents):
            if i < len(result.verdicts) and "RELEVANT" in result.verdicts[i].upper():
                relevant_chunks.append(doc)
    except Exception as e:
        print(f"  Batch grading failed ({e}), falling back to accepting all docs")
        relevant_chunks = documents
    
    print(f"  {len(relevant_chunks)}/{len(documents)} chunks passed grading")
    
    # PARENT EXPANSION: Use cached parent store (no per-request loading)
    # Smart expansion: check if parent actually contains the child's content
    if relevant_chunks:
        parent_dict = get_parent_store()
        
        if parent_dict:
            final_docs = []
            seen_parent_ids = set()
            
            for chunk in relevant_chunks:
                parent_id = chunk.metadata.get("parent_id")
                clause_id = chunk.metadata.get("clause_id", "")
                
                # Always keep the child chunk - it has the specific answer with correct clause ID
                final_docs.append(chunk)
                
                # Also add the parent for surrounding context (if not already added)
                if parent_id and parent_id in parent_dict and parent_id not in seen_parent_ids:
                    parent_doc = parent_dict[parent_id]
                    # Only add parent if it's not too long (to avoid context overflow)
                    if len(parent_doc.page_content) < 25000:
                        final_docs.append(parent_doc)
                        seen_parent_ids.add(parent_id)
            
            if final_docs:
                parent_count = sum(1 for d in final_docs if d.metadata.get("is_parent"))
                child_count = len(final_docs) - parent_count
                print(f"  Expanded to {parent_count} parents + {child_count} kept children")
                state["documents"] = final_docs
            else:
                print("  No documents after expansion, using clause chunks")
                state["documents"] = relevant_chunks
        else:
            print("  Parent store empty, using clause chunks")
            state["documents"] = relevant_chunks
        
        state["grade_status"] = "Relevant"
        
        # APPENDIX HOPPING: Check if documents reference appendices and fetch them
        if parent_dict and state["documents"]:
            appendix_refs = set()
            # Pattern to find appendix references like "Appendix 10", "in Appendix 5"
            appendix_pattern = re.compile(r'(?:in |see |specified in |refer to )?Appendix\s+(\d+)', re.IGNORECASE)
            
            for doc in state["documents"]:
                matches = appendix_pattern.findall(doc.page_content)
                appendix_refs.update(matches)
            
            if appendix_refs:
                # Find matching appendices in parent store
                appendix_docs = []
                for k, v in parent_dict.items():
                    section_id = str(v.metadata.get('section_id', ''))
                    if v.metadata.get('is_appendix'):
                        # Check if this appendix number is referenced
                        for ref_num in appendix_refs:
                            if f"Appendix {ref_num}:" in section_id:
                                appendix_docs.append(v)
                                break
                
                if appendix_docs:
                    print(f"  → Hopped to {len(appendix_docs)} referenced appendices: {list(appendix_refs)}")
                    state["documents"].extend(appendix_docs)
    else:
        state["documents"] = []
        state["grade_status"] = "Irrelevant"
    
    return state

# =============================================================================
# Node 5: Rewrite Query
# =============================================================================

def rewrite_query(state: AgentState) -> AgentState:
    print("\n[Node: rewrite_query]")
    original_query = state["original_query"]
    retry_count = state["retry_count"]
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are assisting with retrying a failed retrieval against Bank Negara Malaysia (BNM)
policy documents.

Your task is to rewrite the query ONLY to improve document retrieval coverage,
without changing the original meaning.

STRICT RULES:
- Preserve the original intent and semantic meaning exactly.
- Do NOT reinterpret values as requirements or obligations.
- Do NOT change the type of information requested.
- Do NOT introduce assumptions or answers.
- Do NOT narrow the scope.

YOU MAY:
- Add explicit references to "Bank Negara Malaysia" or "BNM policy"
- Expand abbreviations if unambiguous
- Replace vague pronouns with explicit terms
- Add commonly used BNM regulatory phrasing if neutral

If no safe improvement is possible:
- Return the original query unchanged.

Return ONLY the rewritten query.
"""),
    ("user", "Original query:\n{query}\n\nRewritten query:")
])

    
    chain = rewrite_prompt | llm
    response = chain.invoke({"query": original_query})
    rewritten = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    print(f"  Rewritten: {rewritten}")
    state["current_query"] = rewritten
    state["retry_count"] = retry_count + 1
    
    # FIX: Clear sub_questions to prevent "zombie sub-question" bug
    # This ensures the rewritten query is used directly, not old sub-questions
    state["sub_questions"] = []
    
    return state

# =============================================================================
# Node 6: Generate Answer
# =============================================================================

def generate(state: AgentState) -> AgentState:
    print("\n[Node: generate]")
    documents = state["documents"]
    original_query = state["original_query"]
    
    # Reorder docs
    reorderer = LongContextReorder()
    reordered_docs = reorderer.transform_documents(documents)
    
    # Build context
    context_parts = []
    for i, doc in enumerate(reordered_docs, 1):
        # Use document_title if available, otherwise fall back to source_pdf
        doc_title = doc.metadata.get("document_title") or doc.metadata.get("source_pdf", "Unknown")
        clause = doc.metadata.get("clause_id", "")
        section = doc.metadata.get("section_id", "")
        
        # Build citation: prefer clause (S 13.1), fallback to section (SECTION 13)
        if clause:
            cite_ref = f"{doc_title}, {clause}"
        elif section:
            # Shorten "SECTION 13: Minimum monthly repayment" to "SECTION 13"
            section_short = section.split(":")[0] if ":" in section else section
            cite_ref = f"{doc_title}, {section_short}"
        else:
            cite_ref = doc_title
        
        context_parts.append(f"CITATION_SOURCE: {cite_ref}\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Bank Negara Malaysia (BNM) regulatory assistant.

Answer the question using ONLY the provided policy excerpts.
Do not introduce requirements, rights, obligations, etc. that are not stated in the excerpts.

Before answering, determine whether the excerpts contain the specific information requested:
- If the excerpts do not directly address the question, respond that the policy does not specify the requested information.

When forming the answer:
- Summarise only the requirements that directly answer the question.
- Do not answer any additional information that is not directly related to the question, even though available in the clause.
- Do not expand the answer by referring to related controls, governance measures, or general security practices unless they are explicitly part of the same requirement.
- Avoid generalisation or inference beyond what is expressly stated.


If a requirement is not explicitly listed under the clause(s) above,
it MUST NOT be included in the answer.

If excerpts do NOT answer the query completely, respond EXACTLY answer that implies you don't know, eg: The credit card policy does not specify minimum age or income requirements for principal cardholders."
Do not change 'should' to 'must', and vice versa.

IMPORTANT -- STRICT CITATION RULES:
If the answer IS found in the retrieved excerpts:

1. EVERY factual statement (even in point form) MUST have an immediate precise citation to the relevant clause or part.
2. Place the citation at the end of the sentence or bullet that contains the fact.
3. The citation MUST:
   - Follow the format: `(Policy Document Name, Clause ID)`
   - Use the EXACT Policy Name and Clause ID provided in the CITATION_SOURCE line
   - NOT include the Section Name/Title (e.g. "SECTION 14B: CDD")
   - Be wrapped in parentheses
   - Use a comma to separate the Policy Name and Clause ID

4. Valid examples:
   - "(Claims Settlement Practices, S 12.2(a))"
   - "(Anti-Money Laundering Policy, S 14A.9.6)"
   - "(RMiT Policy, G 12.1)"

5. INVALID examples (DO NOT DO THIS):
   - Missing citations
   - Citations grouped at the end of a paragraph
   - Vague references like "(the policy)" or "(Appendix)"
   - A reference list at the end of the answer
   - Inventing subsections that don't exist in the source (e.g., adding "(a)", "(b)" when source has a paragraph)
   - Changing S to G or G to S in citations

6. If a sentence contains multiple facts, use semicolon (;) to separate citations.

"""),
    ("user", """Question: {query}

Policy excerpts:
{context}

Answer:
""")
])

    
    chain = generation_prompt | llm
    response = chain.invoke({"query": original_query, "context": context})
    state["generation"] = response.content if hasattr(response, 'content') else str(response)
    print("  Generated answer.")
    
    return state


# =============================================================================
# Node 7: Hallucination Check
# =============================================================================

def hallucination_check(state: AgentState) -> AgentState:
    print("\n[Node: hallucination_check]")
    generation = state["generation"]
    documents = state["documents"]
    
    # Handle empty generation or NO_DATA_FOUND cases
    if not generation or "NO_DATA_FOUND" in generation or "Not found" in generation:
        print("  Status: Pass (no data to verify)")
        state["hallucination_status"] = "Pass"
        state["critique"] = "No factual claims to verify"
        return state
    
    # Extract ALL factual claims from the generation
    doc_texts = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    check_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are verifying if an answer is factually grounded in source documents.

TASK: Check if key facts in the answer have basis in the source documents.
Critique must point out what is factually incorrect/hallucinated/contradicted.

OUTPUT: {{"status": "Pass"}} or {{"status": "Fail", "critique": "..."}}

CRITICAL: Be LENIENT. Only flag actual factual errors, not formatting or phrasing differences.

ALWAYS PASS if:
- Numbers and limits mentioned in answer appear somewhere in the documents (even if phrased differently)
- The answer summarizes or interprets policy content reasonably
- Language like "must"/"should"/"encouraged" may differ - this is OK
- Answer covers only some aspects of a topic - this is OK
- Citation format differs (e.g., "S 14D.12.2" vs "Section 14D.12.2", or adding subsections like "(a)" when the source lists items as a paragraph)
- Phrasing is equivalent (e.g., "between RM3,000 and RM4,999" vs "below RM5,000" - these mean the same thing)
- The answer correctly interprets ranges, limits, or conditions even if the exact wording differs
- The answer extracts information from a paragraph and presents it as a list or structured format
- The answer cites a Standard (S) clause and does NOT include optional details from a related Guidance (G) clause
  (Example: S 14C.16.17 says "make video calls", G 14C.16.18 says "may consider unannounced calls" - the answer is CORRECT to say "video calls" without "unannounced" because the S clause does not require it)
- The answer focuses on ONE aspect of a requirement when the source mentions multiple (e.g., source says "customers or counterparties" but answer only discusses "customers" because that's what was asked)
- The answer omits additional details from the same clause that were not asked about (e.g., "maximum tolerable downtime of 120 minutes per incident" can be omitted if only asked about cumulative downtime)

ONLY FAIL if:
- The answer states something that DIRECTLY contradicts what the documents say (e.g., says 8 hours when source says 4 hours)
- The answer makes claims about requirements/obligations that are not mentioned in the documents at all
- The answer attributes content from a Guidance (G) clause as if it were a Standard (S) requirement
"""),
    ("user", """Answer:
{generation}

Source documents:
{context}

JSON result:""")
])

    
    try:
        # First try structured output
        llm_with_structure = llm.with_structured_output(HallucinationCheck)
        chain = check_prompt | llm_with_structure
        
        result = chain.invoke({
            "generation": generation, 
            "context": doc_texts[:100000]  # 262K context model
        })
        state["hallucination_status"] = result.status
        state["critique"] = result.critique if isinstance(result.critique, str) else "; ".join(result.critique)
        print(f"  Status: {state['hallucination_status']}")
        if state["hallucination_status"] == "Fail":
            print(f"  Critique: {state['critique']}")
    except Exception as e:
        # Fallback: try to parse raw response
        print(f"  Structured output failed: {e}")
        try:
            # Try raw LLM call and parse manually
            raw_chain = check_prompt | llm
            raw_response = raw_chain.invoke({
                "generation": generation,
                "context": doc_texts[:80000]
            })
            raw_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            # Simple heuristic: check if "Pass" is in the response
            if '"status": "Pass"' in raw_text or '"Pass"' in raw_text:
                state["hallucination_status"] = "Pass"
                state["critique"] = "All claims verified"
                print("  Status: Pass (fallback)")
            else:
                state["hallucination_status"] = "Fail"
                state["critique"] = "Could not verify claims"
                print("  Status: Fail (fallback)")
        except Exception as fallback_error:
            print(f"  Fallback also failed: {fallback_error}")
            # Default to Pass to avoid blocking good answers due to LLM issues
            state["hallucination_status"] = "Pass"
            state["critique"] = "Verification skipped due to LLM error"
    
    return state
# =============================================================================
# Node 8: Revise Generation
# =============================================================================

def revise_generation(state: AgentState) -> AgentState:
    print("\n[Node: revise_generation]")
    
    revision_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are revising a Bank Negara Malaysia (BNM) regulatory answer
that failed factual verification.

Your task is to FIX the identified issues WITHOUT adding any new information.

REVISION RULES:
- Remove or correct ONLY the claims identified in the critique.
- Use ONLY facts explicitly present in the source documents.
- Do NOT add new claims, explanations, or context.
- Do NOT rephrase unaffected sentences.
- Every remaining factual statement MUST be cited.

If, after removal, the documents no longer contain an answerpy:
- State that the specific information is not specified in the policy documents provided.

OUTPUT:
- Revised answer only
- No commentary or explanation
"""),
    ("user", """Original answer:
{generation}

Issues to fix:
{critique}

Source documents:
{context}

Revised answer:""")
])

    
    documents = state["documents"]
    context = "\n\n".join([doc.page_content[:4000] for doc in documents[:5]])
    
    chain = revision_prompt | llm
    response = chain.invoke({
        "generation": state["generation"],
        "critique": state["critique"],
        "context": context
    })
    
    state["generation"] = response.content if hasattr(response, 'content') else str(response)
    state["revision_count"] += 1
    print(f"  Revised (attempt {state['revision_count']})")
    
    return state


# =============================================================================
# Routing
# =============================================================================

def should_retrieve_or_end(state: AgentState) -> str:
    if len(state["documents"]) == 0:
        if state["retry_count"] < MAX_RETRY_COUNT:
            return "rewrite"
        else:
            state["generation"] = "No relevant documents found."
            return "end"
    return "generate"

def should_revise_or_end(state: AgentState) -> str:
    if state["hallucination_status"] == "Fail" and state["revision_count"] < MAX_REVISION_COUNT:
        return "revise"
    return "end"

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("extract_filters", extract_filters)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination_check", hallucination_check)
    workflow.add_node("revise_generation", revise_generation)
    
    # Start with query refinement
    workflow.set_entry_point("refine_query")
    workflow.add_edge("refine_query", "decompose_query")
    workflow.add_edge("decompose_query", "extract_filters")
    workflow.add_edge("extract_filters", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges("grade_documents", should_retrieve_or_end, 
                                   {"rewrite": "rewrite_query", "generate": "generate", "end": END})
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", should_revise_or_end, 
                                   {"revise": "revise_generation", "end": END})
    workflow.add_edge("revise_generation", "hallucination_check")
    
    return workflow.compile()

app = create_graph()

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the minimum age and income requirement for a principal cardholder of a debit card-i?"
    initial_state = {
        "original_query": query, 
        "current_query": query, 
        "retry_count": 0, 
        "revision_count": 0,
        "sub_questions": [],
        "documents": [],
        "generation": "",
        "filters": {},
        "grade_status": "",
        "hallucination_status": "",
        "critique": ""
    }
    for s in app.stream(initial_state):
        pass
    print("\nFINAL ANSWER:\n", list(s.values())[0].get("generation", "Error"))