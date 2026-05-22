"""
Chunker module for the DIEM Agentic RAG project.

Reads raw_documents.json produced by pdf_filter.py and splits each document
into parent and child chunks using a two-level RecursiveCharacterTextSplitter
strategy.

Architecture:
    - Child chunks (CHILD_CHUNK_SIZE chars) are embedded in ChromaDB.
      They contain only the raw child text — no header or breadcrumb
      injection. This allows the bi-encoder to match on the actual content
      without interference from URL-derived metadata.
    - Parent chunks (PARENT_CHUNK_SIZE chars) are stored in a separate
      JSON file and loaded into an InMemoryStore at runtime by retriever.py.
      They provide richer context to the LLM than child chunks.

ID scheme (all deterministic and stable across re-crawls):
    doc_id       = sha256(source_url)
    parent_id    = sha256(source_url + str(parent_position))
    chunk_id     = sha256(child_text + source_url + str(child_position))
    content_hash = sha256(" ".join(text.split()))

Metadata (domain, section_type, header) is derived deterministically from
the source URL via metadata_extractor.py and stored on both parent and child
chunks. It is available for filtering and display but is NOT injected into
the embedded child text.

Outputs:
    CHUNKED_DOCUMENTS_PATH — child chunks, consumed by indexer.py
    CHUNKED_PARENTS_PATH   — parent chunks, loaded into InMemoryStore

Usage:
    python -m ingestion.chunker
"""

import hashlib
import json
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingestion.metadata_extractor import extract_metadata


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _make_doc_id(source_url: str) -> str:
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()


def _make_parent_id(source_url: str, position: int) -> str:
    return hashlib.sha256(
        f"{source_url}::{position}".encode("utf-8")
    ).hexdigest()


def _make_chunk_id(child_text: str, source_url: str, position: int) -> str:
    return hashlib.sha256(
        f"{child_text}::{source_url}::{position}".encode("utf-8")
    ).hexdigest()


def _make_content_hash(text: str) -> str:
    """
    Returns a normalised SHA-256 hash of the document text.
    Whitespace is collapsed before hashing to avoid spurious hash changes
    caused by minor formatting differences between crawl runs.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _chunk_document(
    doc: Dict,
    parent_splitter: RecursiveCharacterTextSplitter,
    child_splitter: RecursiveCharacterTextSplitter,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits a single raw document into parent and child chunks.

    Parent chunks (PARENT_CHUNK_SIZE chars) are stored in the docstore
    and returned to the LLM as context.

    Child chunks (CHILD_CHUNK_SIZE chars) are embedded in ChromaDB.
    The child_text contains only the raw text — no header or breadcrumb
    is prepended, so the bi-encoder embeds the actual content directly.

    Args:
        doc:              A raw document dict from raw_documents.json.
        parent_splitter:  Splitter for parent chunks.
        child_splitter:   Splitter for child chunks.

    Returns:
        parents:  List of parent chunk dicts for chunked_parents.json.
        children: List of child chunk dicts for chunked_documents.json.
    """
    source = doc["source"]
    text = doc["text"]
    doc_type = doc.get("doc_type", "html")
    content_hash = _make_content_hash(text)
    doc_id = _make_doc_id(source)

    # Extract deterministic metadata from the source URL.
    meta = extract_metadata(source)
    domain = meta["domain"]
    section_type = meta["section_type"]

    # Split into parent chunks.
    parent_texts = parent_splitter.split_text(text)

    parents: List[Dict] = []
    children: List[Dict] = []
    child_global_idx = 0

    for p_idx, parent_text in enumerate(parent_texts):
        if not parent_text.strip():
            continue
        if len(parent_text.strip()) < config.CHUNK_MIN_CHARS:
                continue

        parent_id = _make_parent_id(source, p_idx)

        parent = {
            "parent_id":    parent_id,
            "doc_id":       doc_id,
            "content_hash": content_hash,
            "source":       source,
            "doc_type":     doc_type,
            "domain":       domain,
            "section_type": section_type,
            "parent_text":  parent_text,
        }
        parents.append(parent)

        # Split parent into child chunks.
        child_texts = child_splitter.split_text(parent_text)

        for raw_child_text in child_texts:
            if not raw_child_text.strip():
                continue
           
            if len(raw_child_text.strip()) < config.CHUNK_MIN_CHARS:
                continue

            chunk_id = _make_chunk_id(
                raw_child_text, source, child_global_idx
            )
            child_global_idx += 1

            child = {
                "chunk_id":     chunk_id,
                "parent_id":    parent_id,
                "doc_id":       doc_id,
                "source":       source,
                "doc_type":     doc_type,
                "domain":       domain,
                "section_type": section_type,
                "child_text":   raw_child_text,
            }
            children.append(child)

    return parents, children


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Executes the full chunking pipeline:
        1. Load raw documents from RAW_DOCUMENTS_PATH.
        2. Build parent and child text splitters from config parameters.
        3. For each document, extract metadata and split into chunks.
        4. Write chunked_documents.json (consumed by indexer.py).
        5. Write chunked_parents.json (loaded into InMemoryStore at runtime).
    """
    print("=== DIEM Agentic RAG: Chunker Started ===\n")

    # ------------------------------------------------------------------
    # Load raw documents
    # ------------------------------------------------------------------
    print(f"[Chunker] Loading documents from {config.RAW_DOCUMENTS_PATH} ...")
    with open(config.RAW_DOCUMENTS_PATH, "r", encoding="utf-8") as fh:
        documents = json.load(fh)
    print(f"[Chunker] {len(documents)} documents loaded.\n")

    # ------------------------------------------------------------------
    # Build splitters
    # ------------------------------------------------------------------
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    all_parents: List[Dict] = []
    all_children: List[Dict] = []

    skipped = 0
    for i, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        if not text.strip():
            skipped += 1
            continue

        parents, children = _chunk_document(doc, parent_splitter, child_splitter)
        all_parents.extend(parents)
        all_children.extend(children)

        if i % 500 == 0:
            print(
                f"[Chunker] Processed {i}/{len(documents)} documents "
                f"({len(all_parents)} parents, {len(all_children)} children so far)."
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n[Chunker] Chunking complete.")
    print(f"[Chunker]   Documents processed : {len(documents) - skipped}")
    print(f"[Chunker]   Documents skipped   : {skipped} (empty text)")
    print(f"[Chunker]   Parent chunks       : {len(all_parents)}")
    print(f"[Chunker]   Child chunks        : {len(all_children)}")
    if all_parents:
        print(
            f"[Chunker]   Avg children/parent : "
            f"{len(all_children) / len(all_parents):.1f}"
        )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    with open(config.CHUNKED_DOCUMENTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(all_children, fh, ensure_ascii=False, indent=2)
    print(f"\n[Chunker] Children saved to {config.CHUNKED_DOCUMENTS_PATH}")

    with open(config.CHUNKED_PARENTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(all_parents, fh, ensure_ascii=False, indent=2)
    print(f"[Chunker] Parents saved to {config.CHUNKED_PARENTS_PATH}")

    print("\n=== Chunker Completed ===")


if __name__ == "__main__":
    run()