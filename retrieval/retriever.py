"""
Retriever module for the DIEM Agentic RAG project.

Implements a two-stage retrieval pipeline:

    Stage 1 — Bi-encoder similarity search + manual parent lookup:
        Searches over child chunk embeddings in ChromaDB, then looks up
        the corresponding parent documents from the InMemoryStore docstore
        using the parent_id metadata field stored on each child.
        Deduplicates by parent_id — only one parent per unique parent_id
        is returned, keeping the child with the highest similarity score.

    Stage 2 — Cross-encoder reranking (Cohere rerank-4-pro):
        Reranks the parent documents retrieved in Stage 1 by relevance
        to the query, returning the top RERANK_TOP_N results.

Parent-child architecture:
    Child chunks are indexed in ChromaDB and used for similarity search.
    Their embeddings carry the enriched text (header + child_text) for
    better bi-encoder matching.
    Parent chunks are stored in an InMemoryStore loaded from
    chunked_parents.json at initialization. The InMemoryStore is rebuilt
    from this file on every system startup.

    The link between child and parent is the parent_id metadata field
    stored on each child document in ChromaDB. After similarity search,
    each child's parent_id is used to look up the full parent document
    in the docstore.

Deduplication:
    Multiple children may point to the same parent_id (due to child
    overlap). After the similarity search, results are deduplicated by
    parent_id — only the first occurrence (highest similarity score)
    is kept for each unique parent_id.

Usage:
    from retrieval.retriever import initialize, retrieve

    initialize()
    docs = retrieve("Quali sono gli orari di ricevimento del professor Capuano?")
"""

import json
import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

import config
from retrieval.reranker import initialize as initialize_reranker
from retrieval.reranker import rerank

logger = logging.getLogger("DIEM_Retriever")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_embed_model: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[Chroma] = None
_docstore: Optional[InMemoryStore] = None
_initialized: bool = False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize() -> None:
    """
    Loads all retrieval components.

    Steps:
        1. Load the HuggingFace embedding model.
        2. Open the ChromaDB vector store (child chunk embeddings).
        3. Load chunked_parents.json into an InMemoryStore (docstore).
        4. Initialize the Cohere cross-encoder reranker.

    Must be called once before any call to retrieve().
    Subsequent calls are no-ops if all components are already loaded.
    """
    global _embed_model, _vector_store, _docstore, _initialized

    if _initialized:
        return

    # --- Embedding model ---
    print(f"[Retriever] Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    _embed_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        cache_folder=config.MODELS_DIR,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"batch_size": config.EMBEDDING_BATCH_SIZE},
    )
    print("[Retriever] Embedding model ready.")

    # --- ChromaDB vector store ---
    print(f"[Retriever] Connecting to ChromaDB at {config.CHROMA_DB_DIR} ...")
    _vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=_embed_model,
        persist_directory=config.CHROMA_DB_DIR,
    )
    doc_count = _vector_store._collection.count()
    print(
        f"[Retriever] Collection '{config.COLLECTION_NAME}' ready "
        f"({doc_count} documents indexed)."
    )

    # --- Docstore ---
    print(f"[Retriever] Loading parent docstore from {config.CHUNKED_PARENTS_PATH} ...")
    _docstore = _build_docstore(config.CHUNKED_PARENTS_PATH)

    # --- Reranker ---
    initialize_reranker()
    print("[Retriever] Retrieval pipeline fully initialized.\n")

    _initialized = True


def _build_docstore(path: str) -> InMemoryStore:
    """
    Loads parent chunks from chunked_parents.json into an InMemoryStore.

    Each parent is stored as a LangChain Document with page_content set
    to parent_text and all other fields stored as metadata.
    The store key is parent_id.

    Args:
        path: Path to chunked_parents.json produced by chunker.py.

    Returns:
        A populated InMemoryStore keyed by parent_id.
    """
    with open(path, "r", encoding="utf-8") as fh:
        parents = json.load(fh)

    store = InMemoryStore()
    entries = []
    for p in parents:
        doc = Document(
            page_content=p["parent_text"],
            metadata={
                "source":       p["source"],
                "doc_type":     p["doc_type"],
                "doc_id":       p["doc_id"],
                "domain":       p["domain"],
                "section_type": p["section_type"],
            },
        )
        entries.append((p["parent_id"], doc))

    store.mset(entries)
    print(f"[Retriever] Docstore loaded: {len(entries)} parent documents.")
    return store


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str) -> List[Document]:
    """
    Retrieves and reranks documents for a query.

    Pipeline:
        1. Similarity search on child embeddings in ChromaDB (top_k).
        2. For each child, look up the parent document in the docstore
           using the parent_id metadata field.
        3. Deduplicate by parent_id — keep only the first occurrence
           (highest similarity score) for each unique parent_id.
        4. Rerank the deduplicated parent documents with Cohere.
        5. Return the top RERANK_TOP_N results.

    Args:
        query: The user query string.

    Returns:
        A list of up to RERANK_TOP_N ranked LangChain Documents.

    Raises:
        RuntimeError: if initialize() has not been called.
    """
    if not _initialized:
        raise RuntimeError(
            "Retriever has not been initialized. "
            "Call retriever.initialize() before using retrieve()."
        )

    logger.info(f"[Retriever] Query: '{query}'")

    # Stage 1 — similarity search on child embeddings.
    logger.info(
        f"[Retriever] Stage 1 — Bi-encoder search "
        f"(top_k={config.SIMILARITY_TOP_K}) + parent lookup ..."
    )
    child_results = _vector_store.similarity_search(
        query, k=config.SIMILARITY_TOP_K
    )

    if not child_results:
        logger.info("[Retriever] No documents retrieved.")
        return []

    # Look up parent documents from docstore, deduplicate by parent_id.
    seen_parent_ids = set()
    parent_docs: List[Document] = []

    for child in child_results:
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue

        # Fetch parent from docstore.
        results = _docstore.mget([parent_id])
        parent = results[0] if results else None

        if parent is None:
            logger.warning(
                f"[Retriever] parent_id '{parent_id}' not found in docstore."
            )
            continue

        seen_parent_ids.add(parent_id)
        parent_docs.append(parent)

    if not parent_docs:
        logger.info("[Retriever] No parent documents found in docstore.")
        return []

    logger.info(
        f"[Retriever] Retrieved {len(parent_docs)} unique parent document(s)."
    )
    for i, doc in enumerate(parent_docs, 1):
        source = doc.metadata.get("source", "unknown")
        logger.info(f"[Retriever]   #{i:>2} {source} ")

    # Stage 2 — cross-encoder reranking.
    logger.info(
        f"[Retriever] Stage 2 — Cross-encoder reranking "
        f"(top_n={config.RERANK_TOP_N}) ..."
    )
    ranked_docs = rerank(query=query, docs=parent_docs)

    logger.info(f"[Retriever] Final ranked results:")
    for i, doc in enumerate(ranked_docs, 1):
        source = doc.metadata.get("source", "unknown")
        rerank_score = doc.metadata.get("relevance_score", 0.0)
        logger.info(f"[Retriever]   #{i:>2} rerank={rerank_score:+.4f} | {source}")

    return ranked_docs