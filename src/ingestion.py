"""
BNM Policy Document Ingestion Pipeline
Hierarchical Parent-Child Chunking

Structure: Document → PART → SECTION → CLAUSE
- Parents (SECTION level): Stored in pickle for context expansion
- Children (CLAUSE level): Stored in ChromaDB for search
"""

import hashlib
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import pymupdf4llm
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_COLLECTION_NAME = "bnm_docs"
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
PARENT_STORE_PATH = str(PROJECT_ROOT / "chroma_db" / "parent_store.pkl")

# Islamic finance keywords
ISLAMIC_KEYWORDS = ["islamic", "shariah", "takaful", "sukuk", "-i", "murabahah"]

# =============================================================================
# Regex Patterns for BNM Document Structure
# =============================================================================

# PART: Matches various formats:
#   **PART A  OVERVIEW**  |  **PART A** **OVERVIEW**  |  # **Part A** **Title**
#   ## **PART C: TITLE**  |  ### **PART A: OVERVIEW**
PART_PATTERN = re.compile(
    r'(?:#{1,4}\s*)?\*\*PART\s+([A-Z]):?\*?\*?\s+\*?\*?([A-Z][A-Za-z\s,\-/()]+)',
    re.MULTILINE | re.IGNORECASE
)

# SECTION: Matches various formats:
#   **13** **Title**  |  **13.** **Title**  |  **13 Title**
#   **1.** **Objective**  |  #### **1. Introduction**
#   **14A** **CDD: Banking**  (alphanumeric section like 14A, 14B, 14C)
#   **10 Application of Risk-Based Approach** (all in one bold block)
#   **1.** Objective ...  (period inside bold, title not bold - ALTERNATE FORMAT)
#   **10    Technology Operations** (RMIT format - number + spaces + title all in one bold)
SECTION_PATTERN = re.compile(
    r'^(?:#{1,4}\s*)?\*\*(\d+[A-Z]?)\.?\s+([A-Z][A-Za-z\s,\-/:()…]+?)\*\*(?:\s*\.{3,}.*)?$|'  # All-in-one bold: **10 Title**
    r'^(?:#{1,4}\s*)?\*\*(\d+[A-Z]?)\.?\*\*\s*\*?\*?([A-Z][A-Za-z\s,\-/:()…]+?)(?:\*\*)?(?:\s*\.{3,}.*)?$',  # Separate: **10** **Title**
    re.MULTILINE
)



# CLAUSE: **S** 13.1, **G** 13.1, **S** 14A.10.2 (handles alphanumeric like 14A.10.2)
# Note: Uses DOTALL so . matches newlines, captures until next clause or section
CLAUSE_PATTERN = re.compile(
    r'^\*\*([SG])\*\*\s+(\d+[A-Z]?\.\d+(?:\.\d+)?)\s+(.+?)(?=\n\*\*[SG]\*\*\s+\d+[A-Z]?\.\d+|\n\*\*\d+[A-Z]?\s+\*\*|\n\*\*PART|\Z)', 
    re.MULTILINE | re.DOTALL
)



# Fallback: Simple numbered clause like "13.1 Text...", "S 13.1 Text...", "14A.10.2 Text..."
SIMPLE_CLAUSE_PATTERN = re.compile(
    r'^(?:[SG]\s+)?(\d+[A-Z]?\.\d+(?:\.\d+)?)\s+(.+?)(?=\n(?:[SG]\s+)?\d+[A-Z]?\.\d+\s|\n\*\*|\Z)', 
    re.MULTILINE | re.DOTALL
)


# ToC line pattern
TOC_PATTERN = re.compile(r'\.{5,}\s*\d+\s*$')

# APPENDIX: Matches "Appendix 1 Title", "Appendix 1: Title", "**Appendix 1** Title"
# APPENDIX: Matches "**Appendix 1** Title", "**APPENDIX 4a** Title"
# Also matches "**Appendix 2 Control Measures on SSTs**" (title inside bold)
# Requires bolding to avoid false positives in text/ToC
APPENDIX_PATTERN = re.compile(
    r'^(?:#{1,4}\s*)?\*\*Appendix\s+([A-Z0-9]+)\s+([A-Z][A-Za-z\s,\-/:()]+?)\*\*\s*$|'  # Title inside bold: **Appendix 2 Title**
    r'^(?:#{1,4}\s*)?\*\*Appendix\s+([A-Z0-9]+)\*\*[:\s]+(.+?)$',  # Title outside bold: **Appendix 2** Title
    re.MULTILINE | re.IGNORECASE
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HierarchyNode:
    """Node in the document hierarchy tree."""
    level: str  # "PART", "SECTION", "CLAUSE"
    id: str     # "PART A", "SECTION 13", "S 13.1"
    title: str
    content: str = ""
    children: List["HierarchyNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_full_text(self) -> str:
        """Get text of this node and all children."""
        parts = [f"{self.id}: {self.title}"]
        if self.content:
            parts.append(self.content)
        for child in self.children:
            parts.append(child.get_full_text())
        return "\n\n".join(parts)


# =============================================================================
# Metadata Extraction
# =============================================================================

def extract_metadata_from_filename(filename: str) -> dict:
    """Extract metadata from BNM filename pattern."""
    name = filename.replace(".pdf", "").replace("bnm_", "")
    
    # Extract year
    year_match = re.search(r'(\d{4})$', name)
    year = year_match.group(1) if year_match else "Unknown"
    
    # Topic
    topic = re.sub(r'_?\d{4}$', '', name).replace("_", " ").title()
    
    # Islamic detection
    is_islamic = any(kw in filename.lower() for kw in ISLAMIC_KEYWORDS)
    
    return {
        "source_pdf": filename,
        "topic": topic,
        "year": year,
        "is_islamic": is_islamic
    }


def extract_document_title(first_page_markdown: str) -> str:
    """
    Extract the official document title from the first page markdown.
    
    BNM PDFs have titles in the format: # **Title**
    Some titles span multiple lines for long document names.
    
    Returns:
        The extracted title, or "Unknown" if not found.
    """
    # Pattern 1: Single-line title like "# **Credit Card and Credit Card-i**"
    single_line = re.search(r'^#\s*\*\*(.+?)\*\*', first_page_markdown, re.MULTILINE)
    if single_line:
        title = single_line.group(1).strip()
        # Check if there are continuation lines (multi-line titles)
        # e.g., "# **Anti-Money Laundering,** **Countering Financing...**"
        remaining = first_page_markdown[single_line.end():]
        continuation = re.match(r'\s*\*\*(.+?)\*\*', remaining)
        if continuation:
            # Multi-line title - combine them
            title = title.rstrip(',') + " " + continuation.group(1).strip()
        return title
    
    # Pattern 2: Fallback - any bold text at the start
    fallback = re.search(r'\*\*([A-Z][A-Za-z\s,\-/()]+)\*\*', first_page_markdown[:500])
    if fallback:
        return fallback.group(1).strip()
    
    return "Unknown"


def clean_toc(markdown: str) -> str:
    """Remove ToC entries from markdown."""
    lines = markdown.split('\n')
    return '\n'.join(line for line in lines if not TOC_PATTERN.search(line))


def clean_false_strikethrough(markdown: str) -> str:
    """
    Remove false strikethrough markers (~~text~~) from pymupdf4llm output.
    
    pymupdf4llm sometimes incorrectly parses normal text as strikethrough,
    especially around line breaks. This causes the hallucination checker to
    misinterpret valid policy text as deleted content.
    """
    # Remove ~~ markers while preserving the text inside
    return re.sub(r'~~([^~]+)~~', r'\1', markdown)


# =============================================================================
# Hierarchy Parsing
# =============================================================================

def parse_hierarchy(markdown: str, base_meta: dict) -> List[HierarchyNode]:
    """Parse markdown into hierarchical structure."""
    
    # Clean ToC first, then remove false strikethrough markers
    content = clean_toc(markdown)
    content = clean_false_strikethrough(content)
    
    # Find all PARTs
    parts = []
    part_matches = list(PART_PATTERN.finditer(content))
    
    for i, match in enumerate(part_matches):
        part_id = f"PART {match.group(1)}"
        part_title = match.group(2).strip().rstrip('*')
        
        # Get content until next PART
        start = match.end()
        end = part_matches[i + 1].start() if i + 1 < len(part_matches) else len(content)
        part_content = content[start:end]
        
        part_node = HierarchyNode(
            level="PART",
            id=part_id,
            title=part_title,
            metadata={**base_meta, "part_id": part_id}
        )
        
        # Find SECTIONs within this PART
        section_matches = list(SECTION_PATTERN.finditer(part_content))
        
        for j, sec_match in enumerate(section_matches):
            # Handle alternation: first alternative uses groups 1,2; second uses groups 3,4
            section_num = sec_match.group(1) or sec_match.group(3)
            section_title = (sec_match.group(2) or sec_match.group(4) or '').strip().rstrip('*')
            section_id = f"SECTION {section_num}: {section_title}"
            
            # Get content until next SECTION
            sec_start = sec_match.end()
            sec_end = section_matches[j + 1].start() if j + 1 < len(section_matches) else len(part_content)
            section_content = part_content[sec_start:sec_end]
            
            section_node = HierarchyNode(
                level="SECTION",
                id=section_id,
                title=section_title,
                content=section_content[:500],  # Preview only
                metadata={
                    **base_meta,
                    "part_id": part_id,
                    "section_id": section_id,
                    "section_num": section_num
                }
            )
            
            # Find CLAUSEs within this SECTION
            clause_matches = list(CLAUSE_PATTERN.finditer(section_content))
            
            if clause_matches:
                for clause_match in clause_matches:
                    reg_type = "Standard" if clause_match.group(1) == "S" else "Guidance"
                    clause_num = clause_match.group(2)
                    clause_text = clause_match.group(3).strip()
                    clause_id = f"{clause_match.group(1)} {clause_num}"
                    
                    clause_node = HierarchyNode(
                        level="CLAUSE",
                        id=clause_id,
                        title=clause_text[:100],
                        content=clause_text,
                        metadata={
                            **base_meta,
                            "part_id": part_id,
                            "section_id": section_id,
                            "clause_id": clause_id,
                            "regulatory_type": reg_type
                        }
                    )
                    section_node.children.append(clause_node)
            else:
                # Fallback: try simple numbered clauses
                simple_matches = list(SIMPLE_CLAUSE_PATTERN.finditer(section_content))
                for simple_match in simple_matches:
                    clause_num = simple_match.group(1)
                    clause_text = simple_match.group(2).strip()
                    clause_id = clause_num
                    
                    clause_node = HierarchyNode(
                        level="CLAUSE",
                        id=clause_id,
                        title=clause_text[:100],
                        content=clause_text,
                        metadata={
                            **base_meta,
                            "part_id": part_id,
                            "section_id": section_id,
                            "clause_id": clause_id,
                            "regulatory_type": "General"
                        }
                    )
                    section_node.children.append(clause_node)
            
            part_node.children.append(section_node)
        
        parts.append(part_node)
    
    # Parse APPENDIX sections (standalone, not inside PARTs)
    appendix_matches = list(APPENDIX_PATTERN.finditer(content))
    
    for i, app_match in enumerate(appendix_matches):
        # Handle alternation: first alternative uses groups 1,2 (title inside bold); second uses groups 3,4 (title outside bold)
        app_num = app_match.group(1) or app_match.group(3)
        app_title = (app_match.group(2) or app_match.group(4) or '').strip().rstrip('*')
        app_id = f"Appendix {app_num}: {app_title}"
        
        # Get content until next appendix or end
        start = app_match.end()
        end = appendix_matches[i + 1].start() if i + 1 < len(appendix_matches) else len(content)
        app_content = content[start:end].strip()
        
        # Create appendix as a pseudo-PART with the content as a single SECTION
        appendix_part = HierarchyNode(
            level="PART",
            id=f"APPENDIX {app_num}",
            title=app_title,
            metadata={**base_meta, "part_id": f"APPENDIX {app_num}"}
        )
        
        appendix_section = HierarchyNode(
            level="SECTION",
            id=app_id,
            title=app_title,
            metadata={
                **base_meta,
                "part_id": f"APPENDIX {app_num}",
                "section_id": app_id,
                "section_num": app_num,
                "is_appendix": True
            }
        )
        
        # Add full content as a single clause
        appendix_clause = HierarchyNode(
            level="CLAUSE",
            id=f"Appendix {app_num}",
            title=app_title,
            content=app_content,
            metadata={
                **base_meta,
                "part_id": f"APPENDIX {app_num}",
                "section_id": app_id,
                "clause_id": f"Appendix {app_num}",
                "is_appendix": True
            }
        )
        appendix_section.children.append(appendix_clause)
        appendix_part.children.append(appendix_section)
        parts.append(appendix_part)
    
    return parts


# =============================================================================
# Document Creation
# =============================================================================

def create_documents(hierarchy: List[HierarchyNode]) -> tuple[List[Document], List[Document]]:
    """
    Create parent and child documents from hierarchy.
    
    Parents: SECTION-level (for context expansion)
    Children: CLAUSE-level (for search)
    """
    parents = []
    children = []
    
    for part in hierarchy:
        for section in part.children:
            # Create parent document (SECTION level)
            parent_id = hashlib.md5(
                f"{section.metadata['source_pdf']}_{section.id}".encode()
            ).hexdigest()
            
            # Full section text = section content + all clauses
            section_text = section.get_full_text()
            
            parent_doc = Document(
                page_content=section_text[:50000],  # Increased for 262K context models
                metadata={
                    "id": parent_id,
                    "is_parent": True,
                    **section.metadata
                }
            )
            parents.append(parent_doc)
            
            # Create child documents (CLAUSE level)
            # Chunk long clauses to avoid embedding context length errors
            MAX_CLAUSE_LEN = 1500  # Conservative limit (~375 tokens)
            OVERLAP = 200
            
            for clause in section.children:
                clause_text = f"{clause.id}: {clause.content}"
                
                if len(clause_text) <= MAX_CLAUSE_LEN:
                    child_doc = Document(
                        page_content=clause_text,
                        metadata={
                            "parent_id": parent_id,
                            "is_parent": False,
                            **clause.metadata
                        }
                    )
                    children.append(child_doc)
                else:
                    # Split long clause into overlapping chunks
                    chunk_start = 0
                    chunk_idx = 0
                    while chunk_start < len(clause_text):
                        chunk_end = min(chunk_start + MAX_CLAUSE_LEN, len(clause_text))
                        chunk_content = clause_text[chunk_start:chunk_end]
                        
                        child_doc = Document(
                            page_content=chunk_content,
                            metadata={
                                "parent_id": parent_id,
                                "is_parent": False,
                                "chunk_index": chunk_idx,
                                **clause.metadata
                            }
                        )
                        children.append(child_doc)
                        # Move forward with overlap
                        chunk_start = chunk_end - OVERLAP if chunk_end < len(clause_text) else chunk_end
                        chunk_idx += 1
    
    return parents, children


# =============================================================================
# PDF Processing
# =============================================================================

def process_pdf(pdf_path: Path) -> tuple[List[Document], List[Document]]:
    """Process a single PDF into parent/child documents."""
    filename = pdf_path.name
    
    # Convert to markdown with page chunks to access first page
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    
    # Extract document title from first page
    first_page = page_chunks[0]['text'] if page_chunks else ""
    document_title = extract_document_title(first_page)
    
    # Combine all pages for hierarchy parsing
    markdown = "\n\n".join(chunk['text'] for chunk in page_chunks)
    
    # Extract base metadata
    base_meta = extract_metadata_from_filename(filename)
    base_meta["document_title"] = document_title
    
    # Check Islamic keywords in content
    if not base_meta["is_islamic"]:
        base_meta["is_islamic"] = any(kw in markdown[:5000].lower() for kw in ISLAMIC_KEYWORDS)
    
    # Parse hierarchy
    hierarchy = parse_hierarchy(markdown, base_meta)
    
    # Create documents
    return create_documents(hierarchy)


# =============================================================================
# Main Ingestion
# =============================================================================

def ingest_all(pdf_dir: str):
    """Main ingestion pipeline."""
    pdf_path = Path(pdf_dir)
    files = list(pdf_path.glob("*.pdf"))
    
    if not files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(files)} PDF files")
    print("Initializing embeddings...")
    
    embeddings = None
    try:
        embeddings = OllamaEmbeddings(model=PRIMARY_EMBEDDING_MODEL)
        embeddings.embed_query("test")
        print(f"Using embedding model: {PRIMARY_EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Ollama failed ({e}), using HuggingFace sentence-transformers")
        # Fallback to HuggingFace embeddings (works without Ollama)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    all_parents = []
    all_children = []
    
    for pdf in tqdm(files, desc="Processing PDFs"):
        try:
            parents, children = process_pdf(pdf)
            all_parents.extend(parents)
            all_children.extend(children)
            print(f"  {pdf.name}: {len(parents)} sections, {len(children)} clauses")
        except Exception as e:
            print(f"  ERROR {pdf.name}: {e}")
    
    # Save parents to pickle
    print(f"\nSaving {len(all_parents)} parent sections to {PARENT_STORE_PATH}...")
    Path(PARENT_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    parent_dict = {doc.metadata["id"]: doc for doc in all_parents}
    with open(PARENT_STORE_PATH, "wb") as f:
        pickle.dump(parent_dict, f)
    
    # Add children to ChromaDB
    if all_children:
        print(f"Adding {len(all_children)} child clauses to ChromaDB...")
        vectorstore.add_documents(all_children)
    
    print(f"\n✓ Ingestion complete!")
    print(f"  Parents: {len(all_parents)} → {PARENT_STORE_PATH}")
    print(f"  Children: {len(all_children)} → {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    pdf_directory = str(PROJECT_ROOT / "data" / "bnm")
    ingest_all(pdf_directory)
