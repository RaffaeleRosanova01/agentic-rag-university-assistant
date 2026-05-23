"""
Indexer module for the DIEM Agentic RAG project.

Reads child chunks from chunked_documents.json and parent chunks from
chunked_parents.json, both produced by chunker.py. Computes both dense
and sparse embeddings using BGE-M3 (FlagEmbedding) and stores them in
two separate ChromaDB collections.

Architecture:
    - Dense collection (COLLECTION_NAME):
        Stores dense semantic vectors produced by BGE-M3.
        Used for semantic similarity search.

    - Sparse collection (COLLECTION_NAME + '_sparse'):
        Stores sparse lexical vectors (learned sparse, not BM25) produced
        by BGE-M3. Used for exact keyword matching on proper nouns,
        course codes, and faculty names.

    Both collections share identical metadata and chunk IDs, enabling
    Reciprocal Rank Fusion (RRF) at query time in retriever.py.

    Parent chunks are not embedded. They are serialised in
    chunked_parents.json and loaded into an InMemoryStore at runtime
    by retriever.py.

Incremental update strategy:
    Change detection is performed at document level using doc_id and
    content_hash derived from parent chunks:
        - doc_id NOT in ChromaDB         → new document, add all its chunks
        - doc_id in ChromaDB, hash equal → unchanged, skip entirely
        - doc_id in ChromaDB, hash diff  → changed, delete old chunks
                                           and re-index the new version
    Documents removed from the site are deleted from both collections.

Usage:
    python -m ingestion.indexer
"""

import json
from typing import Dict, List, Set, Tuple

import chromadb
import numpy as np

import config


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_children() -> List[Dict]:
    print(f"[Indexer] Loading child chunks from {config.CHUNKED_DOCUMENTS_PATH} ...")
    with open(config.CHUNKED_DOCUMENTS_PATH, "r", encoding="utf-8") as fh:
        chunks: List[Dict] = json.load(fh)
    print(f"[Indexer] {len(chunks)} child chunks loaded.")
    return chunks


def _load_parents() -> List[Dict]:
    print(f"[Indexer] Loading parent chunks from {config.CHUNKED_PARENTS_PATH} ...")
    with open(config.CHUNKED_PARENTS_PATH, "r", encoding="utf-8") as fh:
        parents: List[Dict] = json.load(fh)
    print(f"[Indexer] {len(parents)} parent chunks loaded.")
    return parents


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

def _build_embedding_model():
    """
    Loads BGE-M3 via FlagEmbedding for simultaneous dense and sparse
    embedding generation.

    Returns:
        BGEM3FlagModel instance ready for encoding.
    """
    from FlagEmbedding import BGEM3FlagModel

    print(f"[Indexer] Loading BGE-M3 model via FlagEmbedding ...")
    model = BGEM3FlagModel(
        config.EMBEDDING_MODEL_NAME,
        use_fp16=True,
        device=config.EMBEDDING_DEVICE,
    )
    print("[Indexer] BGE-M3 model ready.")
    return model


def _encode_batch(
    model,
    texts: List[str],
    batch_size: int,
) -> Tuple[List[List[float]], List[Dict[int, float]]]:
    """
    Encodes a list of texts using BGE-M3, producing both dense and sparse
    vectors in batches to avoid out-of-memory errors.

    Args:
        model:      BGEM3FlagModel instance.
        texts:      List of text strings to encode.
        batch_size: Number of texts per encoding batch.

    Returns:
        dense_vecs:   List of dense float vectors (one per text).
        sparse_vecs:  List of sparse dicts {token_id: weight} (one per text).
    """
    all_dense: List[List[float]] = []
    all_sparse: List[Dict[int, float]] = []

    total = len(texts)
    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = texts[start:end]

        output = model.encode(
            batch,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_batch = output["dense_vecs"]
        sparse_batch = output["lexical_weights"]

        # Convert numpy arrays to plain Python lists for ChromaDB.
        for vec in dense_batch:
            all_dense.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))

        # Convert sparse weights: ensure keys are int and values are float.
        for weights in sparse_batch:
            all_sparse.append({int(k): float(v) for k, v in weights.items()})

        print(
            f"[Indexer] Encoded batch {batch_idx + 1}/{num_batches} "
            f"({end}/{total} chunks)."
        )

    return all_dense, all_sparse


# ---------------------------------------------------------------------------
# ChromaDB collections
# ---------------------------------------------------------------------------

def _build_chroma_collections() -> Tuple[chromadb.Collection, chromadb.Collection]:
    """
    Opens (or creates) the two persistent ChromaDB collections:
        - Dense collection:  HNSW cosine index for dense vectors.
        - Sparse collection: stores sparse vectors as metadata for manual
                             dot-product scoring at query time.

    Returns:
        (dense_collection, sparse_collection)
    """
    print(f"[Indexer] Connecting to ChromaDB at {config.CHROMA_DB_DIR} ...")
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)

    dense_collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    sparse_collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME + "_sparse",
        metadata={"hnsw:space": "cosine"},
    )

    print(
        f"[Indexer] Dense collection '{config.COLLECTION_NAME}' ready "
        f"({dense_collection.count()} documents stored)."
    )
    print(
        f"[Indexer] Sparse collection '{config.COLLECTION_NAME}_sparse' ready "
        f"({sparse_collection.count()} documents stored)."
    )
    return dense_collection, sparse_collection


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _build_metadata(chunk: Dict) -> Dict:
    """
    Constructs the ChromaDB metadata dict for a single child chunk.
    All values must be scalar (str, int, float).
    parent_text is intentionally excluded — stored in the docstore.
    """
    return {
        "source":       chunk["source"],
        "doc_type":     chunk["doc_type"],
        "doc_id":       chunk["doc_id"],
        "parent_id":    chunk["parent_id"],
        "domain":       chunk["domain"],
        "section_type": chunk["section_type"],
        "content_hash": chunk.get("content_hash", "")
    }


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def _build_doc_hash_map(parents: List[Dict]) -> Dict[str, str]:
    doc_hash_map: Dict[str, str] = {}
    for parent in parents:
        doc_id = parent["doc_id"]
        if doc_id not in doc_hash_map:
            doc_hash_map[doc_id] = parent.get("content_hash", "")
    return doc_hash_map


def _perform_incremental_update(
    dense_collection: chromadb.Collection,
    sparse_collection: chromadb.Collection,
    children: List[Dict],
    doc_hash_map: Dict[str, str],
) -> List[Dict]:
    """
    Performs incremental update on both collections simultaneously.
    Change detection is based on the dense collection (authoritative source).
    Both collections are always kept in sync.
    """
    current_doc_ids: Set[str] = set(doc_hash_map.keys())

    print("[Indexer] Scanning existing ChromaDB entries for incremental update ...")

    existing_doc_hashes: Dict[str, str] = {}
    existing_chunk_ids_by_doc: Dict[str, List[str]] = {}

    offset = 0
    batch_size = 1000
    while True:
        result = dense_collection.get(
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

    docs_to_delete = changed_doc_ids | removed_doc_ids
    if docs_to_delete:
        chunk_ids_to_delete: List[str] = []
        for doc_id in docs_to_delete:
            chunk_ids_to_delete.extend(existing_chunk_ids_by_doc.get(doc_id, []))
        if chunk_ids_to_delete:
            dense_collection.delete(ids=chunk_ids_to_delete)
            sparse_collection.delete(ids=chunk_ids_to_delete)
            print(
                f"[Indexer] Deleted {len(chunk_ids_to_delete)} stale chunk(s) "
                f"from {len(docs_to_delete)} document(s) in both collections."
            )

    docs_to_upsert = new_doc_ids | changed_doc_ids
    children_to_upsert = [c for c in children if c["doc_id"] in docs_to_upsert]
    print(
        f"[Indexer] Children to upsert: {len(children_to_upsert)} "
        f"({len(unchanged_doc_ids)} skipped)."
    )
    return children_to_upsert


# ---------------------------------------------------------------------------
# Batch upsert
# ---------------------------------------------------------------------------

def _upsert_batches(
    dense_collection: chromadb.Collection,
    sparse_collection: chromadb.Collection,
    chunks: List[Dict],
    dense_vecs: List[List[float]],
    sparse_vecs: List[Dict[int, float]],
) -> None:
    """
    Upserts child chunks into both collections in batches.

    Dense collection: stores dense embeddings for HNSW cosine search.
    Sparse collection: stores sparse vectors as ChromaDB embeddings
                       (converted to fixed-size representation) plus
                       the raw sparse dict serialized in metadata for
                       exact dot-product scoring at query time.
    """
    batch_size = 500
    total = len(chunks)
    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)

        batch_chunks = chunks[start:end]
        batch_dense = dense_vecs[start:end]
        batch_sparse = sparse_vecs[start:end]
        batch_ids = [c["chunk_id"] for c in batch_chunks]
        batch_docs = [c["child_text"] for c in batch_chunks]
        batch_meta = [_build_metadata(c) for c in batch_chunks]

        # Upsert dense vectors.
        dense_collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_dense,
            metadatas=batch_meta,
        )

        # Upsert sparse vectors.
        # Store the sparse dict serialized as JSON in metadata for retrieval.
        import json as _json
        sparse_meta = []
        for base_meta, sparse_dict in zip(batch_meta, batch_sparse):
            m = dict(base_meta)
            m["sparse_json"] = _json.dumps(sparse_dict)
            sparse_meta.append(m)

        # Use dense vectors as placeholder embeddings in sparse collection
        # (required by ChromaDB — actual scoring uses sparse_json from metadata).
        sparse_collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_dense,
            metadatas=sparse_meta,
        )

        print(
            f"[Indexer] Upserted batch {batch_idx + 1}/{num_batches} "
            f"({end}/{total} chunks) into both collections."
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Executes the full indexing pipeline:
        1. Load child and parent chunks.
        2. Load BGE-M3 via FlagEmbedding.
        3. Open (or create) dense and sparse ChromaDB collections.
        4. Detect new/changed/removed documents incrementally.
        5. Encode new/changed children with BGE-M3 (dense + sparse).
        6. Upsert into both collections.
    """
    print("=== DIEM Agentic RAG: Indexer Started ===\n")

    children = _load_children()
    parents = _load_parents()

    if not children:
        print("[Indexer] No child chunks to index. Exiting.")
        return

    doc_hash_map = _build_doc_hash_map(parents)
    print(f"[Indexer] {len(doc_hash_map)} unique documents in current crawl.\n")

    embed_model = _build_embedding_model()
    dense_collection, sparse_collection = _build_chroma_collections()

    print()
    children_to_upsert = _perform_incremental_update(
        dense_collection, sparse_collection, children, doc_hash_map
    )

    if not children_to_upsert:
        print("[Indexer] All documents are up to date. Nothing to upsert.")
        print(f"\n[Indexer] Total docs in dense ChromaDB: {dense_collection.count()}")
        print(f"[Indexer] Total docs in sparse ChromaDB: {sparse_collection.count()}")
        print("\n=== Indexer Completed ===")
        return

    print(
        f"\n[Indexer] Computing BGE-M3 embeddings (dense + sparse) for "
        f"{len(children_to_upsert)} child chunks "
        f"(batch size: {config.EMBEDDING_BATCH_SIZE}) ..."
    )
    child_texts = [c["child_text"] for c in children_to_upsert]
    dense_vecs, sparse_vecs = _encode_batch(
        embed_model, child_texts, config.EMBEDDING_BATCH_SIZE
    )
    print(
        f"[Indexer] Embeddings computed. "
        f"Dense dim: {len(dense_vecs[0])}. "
        f"Sparse avg tokens: {sum(len(s) for s in sparse_vecs) / len(sparse_vecs):.1f}."
    )

    print(f"\n[Indexer] Starting upsert into both collections ...")
    _upsert_batches(
        dense_collection, sparse_collection,
        children_to_upsert, dense_vecs, sparse_vecs
    )

    print(f"\n[Indexer] Upsert complete.")
    print(f"[Indexer]   Children upserted            : {len(children_to_upsert)}")
    print(f"[Indexer]   Total docs in dense collection : {dense_collection.count()}")
    print(f"[Indexer]   Total docs in sparse collection: {sparse_collection.count()}")
    print(f"\n=== Indexer Completed ===")


if __name__ == "__main__":
    run()