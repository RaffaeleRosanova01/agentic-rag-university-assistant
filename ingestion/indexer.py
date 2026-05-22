"""
Indexer module for the DIEM Agentic RAG project.

Reads child chunks from chunked_documents.json and parent chunks from
chunked_parents.json, both produced by chunker.py. Computes embeddings
for each child chunk and upserts them into a persistent ChromaDB collection.

Architecture:
    - Child chunks are indexed in ChromaDB with their embeddings.
      Each child carries a parent_id and doc_id as metadata, but NOT
      the parent_text — that is stored separately in the docstore.
    - Parent chunks are not embedded. They are serialised in
      chunked_parents.json and loaded into an InMemoryStore at runtime
      by retriever.py, implementing the ParentDocumentRetriever pattern.

Incremental update strategy:
    Change detection is performed at document level using doc_id and
    content_hash, both derived from parent chunks (the authoritative
    source). The logic for each document in the current crawl is:

        - doc_id NOT in ChromaDB         → new document, add all its chunks
        - doc_id in ChromaDB, hash equal → unchanged, skip entirely
        - doc_id in ChromaDB, hash diff  → changed, delete all old chunks
                                           and re-index the new version

    Documents present in ChromaDB but absent from the current crawl
    (i.e. pages removed from the site) are deleted from ChromaDB.

Usage:
    python -m ingestion.indexer
"""

import json
from typing import Dict, List, Set

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

import config


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_children() -> List[Dict]:
    """
    Loads the list of child chunks from chunked_documents.json.

    Returns:
        A list of child chunk dicts, each containing at minimum:
        chunk_id, parent_id, doc_id, source, doc_type, domain,
        section_type, header, child_text.

    Raises:
        FileNotFoundError: if chunked_documents.json does not exist.
    """
    print(f"[Indexer] Loading child chunks from {config.CHUNKED_DOCUMENTS_PATH} ...")
    with open(config.CHUNKED_DOCUMENTS_PATH, "r", encoding="utf-8") as fh:
        chunks: List[Dict] = json.load(fh)
    print(f"[Indexer] {len(chunks)} child chunks loaded.")
    return chunks


def _load_parents() -> List[Dict]:
    """
    Loads the list of parent chunks from chunked_parents.json.

    Parent chunks are used here exclusively for change detection:
    they carry the doc_id and content_hash that the incremental update
    logic needs. They are NOT indexed in ChromaDB.

    Returns:
        A list of parent chunk dicts, each containing at minimum:
        parent_id, doc_id, content_hash, source.

    Raises:
        FileNotFoundError: if chunked_parents.json does not exist.
    """
    print(f"[Indexer] Loading parent chunks from {config.CHUNKED_PARENTS_PATH} ...")
    with open(config.CHUNKED_PARENTS_PATH, "r", encoding="utf-8") as fh:
        parents: List[Dict] = json.load(fh)
    print(f"[Indexer] {len(parents)} parent chunks loaded.")
    return parents


# ---------------------------------------------------------------------------
# Model and collection
# ---------------------------------------------------------------------------

def _build_embedding_model() -> HuggingFaceEmbeddings:
    """
    Instantiates the sentence-transformer embedding model.

    The model is downloaded on first run and cached in config.MODELS_DIR.
    Subsequent runs load directly from the local cache without network access.
    """
    print(f"[Indexer] Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        cache_folder=config.MODELS_DIR,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"batch_size": config.EMBEDDING_BATCH_SIZE},
    )
    print("[Indexer] Embedding model ready.")
    return model


def _build_chroma_collection() -> chromadb.Collection:
    """
    Opens (or creates) the persistent ChromaDB collection defined in config.

    Returns:
        The ChromaDB collection object ready for upsert operations.
    """
    print(f"[Indexer] Connecting to ChromaDB at {config.CHROMA_DB_DIR} ...")
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(
        f"[Indexer] Collection '{config.COLLECTION_NAME}' ready "
        f"({collection.count()} documents currently stored)."
    )
    return collection


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _build_metadata(chunk: Dict) -> Dict:
    """
    Constructs the ChromaDB metadata dict for a single child chunk.

    ChromaDB requires all metadata values to be scalar types
    (str, int, or float). parent_text is intentionally excluded —
    it is stored in the InMemoryStore docstore, not in ChromaDB.

    The parent_id stored here is the key used by ParentDocumentRetriever
    to look up the corresponding parent document in the docstore.
    """
    return {
        "source":       chunk["source"],
        "doc_type":     chunk["doc_type"],
        "doc_id":       chunk["doc_id"],
        "parent_id":    chunk["parent_id"],
        "domain":       chunk["domain"],
        "section_type": chunk["section_type"],
    }


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def _build_doc_hash_map(parents: List[Dict]) -> Dict[str, str]:
    """
    Builds a mapping from doc_id to content_hash from the parent chunks.

    Since multiple parent chunks may share the same doc_id (one document
    can produce multiple parent chunks), we take the first content_hash
    encountered — they are identical for all parents of the same document.

    Args:
        parents: List of parent chunk dicts from chunked_parents.json.

    Returns:
        A dict mapping doc_id → content_hash.
    """
    doc_hash_map: Dict[str, str] = {}
    for parent in parents:
        doc_id = parent["doc_id"]
        if doc_id not in doc_hash_map:
            doc_hash_map[doc_id] = parent.get("content_hash", "")
    return doc_hash_map


def _perform_incremental_update(
    collection: chromadb.Collection,
    children: List[Dict],
    doc_hash_map: Dict[str, str],
) -> List[Dict]:
    """
    Performs incremental update of the ChromaDB collection and returns
    only the child chunks that need to be (re-)embedded and upserted.

    Change detection is performed at document level (doc_id):
        - doc_id not in ChromaDB       → new, include all its children
        - doc_id in ChromaDB, same hash → unchanged, skip all its children
        - doc_id in ChromaDB, diff hash → changed, delete old chunks,
                                          include new children for upsert

    Additionally removes documents that are in ChromaDB but no longer
    present in the current crawl (deleted pages).

    Args:
        collection:   The ChromaDB collection.
        children:     All child chunks from chunked_documents.json.
        doc_hash_map: Mapping doc_id → content_hash from parent chunks.

    Returns:
        The subset of children that must be upserted.
    """
    current_doc_ids: Set[str] = set(doc_hash_map.keys())

    # Retrieve all existing doc_ids and their stored content_hash from ChromaDB.
    # We query in batches to avoid loading all metadata at once.
    print("[Indexer] Scanning existing ChromaDB entries for incremental update ...")

    existing_doc_hashes: Dict[str, str] = {}   # doc_id → stored content_hash
    existing_chunk_ids_by_doc: Dict[str, List[str]] = {}  # doc_id → [chunk_ids]

    offset = 0
    batch_size = 1000
    while True:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        if not result["ids"]:
            break
        for chunk_id, meta in zip(result["ids"], result["metadatas"]):
            doc_id = meta.get("doc_id", "")
            if doc_id:
                existing_doc_hashes.setdefault(doc_id, meta.get("content_hash", ""))
                existing_chunk_ids_by_doc.setdefault(doc_id, []).append(chunk_id)
        offset += batch_size
        if len(result["ids"]) < batch_size:
            break

    existing_doc_ids: Set[str] = set(existing_doc_hashes.keys())

    # ------------------------------------------------------------------
    # Classify documents
    # ------------------------------------------------------------------
    new_doc_ids = current_doc_ids - existing_doc_ids
    removed_doc_ids = existing_doc_ids - current_doc_ids
    changed_doc_ids = {
        doc_id for doc_id in current_doc_ids & existing_doc_ids
        if doc_hash_map.get(doc_id, "") != existing_doc_hashes.get(doc_id, "")
    }
    unchanged_doc_ids = (current_doc_ids & existing_doc_ids) - changed_doc_ids

    print(
        f"[Indexer]   New documents      : {len(new_doc_ids)}\n"
        f"[Indexer]   Changed documents  : {len(changed_doc_ids)}\n"
        f"[Indexer]   Unchanged documents: {len(unchanged_doc_ids)}\n"
        f"[Indexer]   Removed documents  : {len(removed_doc_ids)}"
    )

    # ------------------------------------------------------------------
    # Delete changed and removed documents from ChromaDB
    # ------------------------------------------------------------------
    docs_to_delete = changed_doc_ids | removed_doc_ids
    if docs_to_delete:
        chunk_ids_to_delete: List[str] = []
        for doc_id in docs_to_delete:
            chunk_ids_to_delete.extend(existing_chunk_ids_by_doc.get(doc_id, []))
        if chunk_ids_to_delete:
            collection.delete(ids=chunk_ids_to_delete)
            print(
                f"[Indexer] Deleted {len(chunk_ids_to_delete)} stale chunk(s) "
                f"from {len(docs_to_delete)} document(s)."
            )

    # ------------------------------------------------------------------
    # Return only children that need to be upserted
    # ------------------------------------------------------------------
    docs_to_upsert = new_doc_ids | changed_doc_ids
    children_to_upsert = [
        c for c in children if c["doc_id"] in docs_to_upsert
    ]
    print(
        f"[Indexer] Children to upsert: {len(children_to_upsert)} "
        f"({len(unchanged_doc_ids * 0)} skipped)."  # cosmetic
    )
    return children_to_upsert


# ---------------------------------------------------------------------------
# Batch upsert
# ---------------------------------------------------------------------------

def _upsert_batches(
    collection: chromadb.Collection,
    chunks: List[Dict],
    embeddings: List[List[float]],
) -> None:
    """
    Upserts all child chunks and their precomputed embeddings into ChromaDB
    in fixed-size batches to avoid excessive memory usage.

    Args:
        collection: The target ChromaDB collection.
        chunks:     Child chunk dicts to upsert.
        embeddings: Precomputed embedding vectors in the same order as chunks.
    """
    batch_size = 500
    total = len(chunks)
    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)

        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end]

        collection.upsert(
            ids=[c["chunk_id"] for c in batch_chunks],
            documents=[c["child_text"] for c in batch_chunks],
            embeddings=batch_embeddings,
            metadatas=[_build_metadata(c) for c in batch_chunks],
        )

        print(
            f"[Indexer] Upserted batch {batch_idx + 1}/{num_batches} "
            f"({end}/{total} chunks)."
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Executes the full indexing pipeline:
        1. Load child chunks from CHUNKED_DOCUMENTS_PATH.
        2. Load parent chunks from CHUNKED_PARENTS_PATH (for change detection).
        3. Load the embedding model.
        4. Perform incremental update: detect new, changed, and removed docs.
        5. Compute embeddings only for chunks that need to be upserted.
        6. Upsert new/changed chunks into ChromaDB.
    """
    print("=== DIEM Agentic RAG: Indexer Started ===\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    children = _load_children()
    parents = _load_parents()

    if not children:
        print("[Indexer] No child chunks to index. Exiting.")
        return

    # Build doc_id → content_hash map from parents (authoritative source).
    doc_hash_map = _build_doc_hash_map(parents)
    print(f"[Indexer] {len(doc_hash_map)} unique documents in current crawl.\n")

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------
    embed_model = _build_embedding_model()

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------
    collection = _build_chroma_collection()

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------
    print()
    children_to_upsert = _perform_incremental_update(
        collection, children, doc_hash_map
    )

    if not children_to_upsert:
        print("[Indexer] All documents are up to date. Nothing to upsert.")
        print(f"\n[Indexer] Total docs in ChromaDB: {collection.count()}")
        print("\n=== Indexer Completed ===")
        return

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    print(
        f"\n[Indexer] Computing embeddings for {len(children_to_upsert)} "
        f"child chunks (batch size: {config.EMBEDDING_BATCH_SIZE}) ..."
    )
    child_texts = [c["child_text"] for c in children_to_upsert]
    embeddings: List[List[float]] = embed_model.embed_documents(child_texts)
    print(f"[Indexer] Embeddings computed. Vector dimension: {len(embeddings[0])}.")

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    print(f"\n[Indexer] Starting upsert ...")
    _upsert_batches(collection, children_to_upsert, embeddings)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    final_count = collection.count()
    print(f"\n[Indexer] Upsert complete.")
    print(f"[Indexer]   Children upserted    : {len(children_to_upsert)}")
    print(f"[Indexer]   Total docs in ChromaDB: {final_count}")
    print(f"\n=== Indexer Completed ===")


if __name__ == "__main__":
    run()