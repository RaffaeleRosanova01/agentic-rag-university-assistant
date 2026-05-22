"""
Reranker module for the DIEM Agentic RAG project.

Supports two reranking backends, selected via config.RERANKER_PROVIDER:

    "local"       — cross-encoder inference via sentence_transformers.CrossEncoder,
                    running on the local machine (CPU or GPU).

    "openrouter"  — REST API call to the OpenRouter rerank endpoint, using a
                    hosted model such as cohere/rerank-4-pro. No local model is
                    loaded; each call consumes API credits.

Both backends expose the same public interface (initialize, rerank) and produce
the same output: a list of LangChain Document objects with relevance_score in
metadata, sorted by descending score, trimmed to RERANK_TOP_N.

Usage:
    from retrieval.reranker import initialize, rerank

    initialize()
    ranked_docs = rerank(query="...", docs=[...])
"""

import requests
from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import config

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_cross_encoder: Optional[CrossEncoder] = None


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize() -> None:
    """
    Prepares the reranking backend selected by config.RERANKER_PROVIDER.

    For the "local" provider, loads the cross-encoder model into memory.
    Subsequent calls are no-ops if the model is already loaded.

    For the "openrouter" provider, no local model needs to be loaded.
    This function verifies that OPENROUTER_API_KEY is set and logs a
    confirmation message.

    Raises:
        ValueError: if config.RERANKER_PROVIDER is not "local" or "openrouter".
    """
    global _cross_encoder

    if config.RERANKER_PROVIDER == "local":
        if _cross_encoder is not None:
            return
        print(f"[Reranker] Loading cross-encoder model: {config.RERANKER_MODEL_NAME} ...")
        _cross_encoder = CrossEncoder(
            model_name_or_path=config.RERANKER_MODEL_NAME,
            model_kwargs={"cache_dir": config.MODELS_DIR},
        )
        print(f"[Reranker] Cross-encoder model ready (local).")

    elif config.RERANKER_PROVIDER == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to the .env file before using the openrouter reranker."
            )
        print(
            f"[Reranker] OpenRouter reranker ready: "
            f"{config.RERANKER_MODEL_NAME_OPENROUTER}"
        )

    else:
        raise ValueError(
            f"Unknown RERANKER_PROVIDER: '{config.RERANKER_PROVIDER}'. "
            "Valid values are 'local' and 'openrouter'."
        )


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _rerank_local(query: str, docs: List[Document]) -> List[Document]:
    """
    Scores candidate documents using the local cross-encoder model.

    Args:
        query: The user query string.
        docs:  Candidate Document objects from the bi-encoder stage.

    Returns:
        Documents sorted by descending relevance_score, trimmed to RERANK_TOP_N.
    """
    pairs = [(query, doc.page_content) for doc in docs]
    scores = _cross_encoder.predict(pairs)

    for doc, score in zip(docs, scores):
        doc.metadata["relevance_score"] = float(score)

    ranked_docs = sorted(docs, key=lambda d: d.metadata["relevance_score"], reverse=True)
    return ranked_docs[:config.RERANK_TOP_N]


def _rerank_openrouter(query: str, docs: List[Document]) -> List[Document]:
    """
    Scores candidate documents via the OpenRouter rerank REST endpoint.

    Sends a single POST request with the query and all candidate texts.
    The API returns results ordered by relevance with their original index,
    which is used to map scores back to the corresponding Document objects.

    Args:
        query: The user query string.
        docs:  Candidate Document objects from the bi-encoder stage.

    Returns:
        Documents sorted by descending relevance_score, trimmed to RERANK_TOP_N.

    Raises:
        RuntimeError: if the OpenRouter API returns a non-200 status code.
    """
    payload = {
        "model": config.RERANKER_MODEL_NAME_OPENROUTER,
        "query": query,
        "documents": [doc.page_content for doc in docs],
        "top_n": config.RERANK_TOP_N,
    }
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/rerank",
        json=payload,
        headers=headers,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter rerank API returned status {response.status_code}: "
            f"{response.text}"
        )

    results = response.json().get("results", [])

    # Map relevance scores back to the original Document objects by index.
    for result in results:
        original_index = result["index"]
        docs[original_index].metadata["relevance_score"] = float(
            result["relevance_score"]
        )

    # Assign a default score to any document not returned by the API.
    for doc in docs:
        if "relevance_score" not in doc.metadata:
            doc.metadata["relevance_score"] = float("-inf")

    ranked_docs = sorted(docs, key=lambda d: d.metadata["relevance_score"], reverse=True)
    return ranked_docs[:config.RERANK_TOP_N]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Re-scores the candidate documents using the configured reranking backend
    and returns the top RERANK_TOP_N documents sorted by descending relevance.

    The relevance score is added to each document's metadata under the key
    'relevance_score' for downstream logging and traceability.

    Args:
        query: The user query string.
        docs:  Candidate LangChain Document objects from the bi-encoder stage.

    Returns:
        A list of up to RERANK_TOP_N Documents sorted by descending
        relevance_score.

    Raises:
        RuntimeError: if initialize() has not been called before rerank(),
                      or if the OpenRouter API call fails.
    """
    if not docs:
        return []

    if config.RERANKER_PROVIDER == "local":
        if _cross_encoder is None:
            raise RuntimeError(
                "Reranker has not been initialized. "
                "Call initialize() before rerank()."
            )
        top_docs = _rerank_local(query, docs)

    elif config.RERANKER_PROVIDER == "openrouter":
        top_docs = _rerank_openrouter(query, docs)

    else:
        raise ValueError(
            f"Unknown RERANKER_PROVIDER: '{config.RERANKER_PROVIDER}'."
        )

    print(f"[Reranker] Scored {len(docs)} chunks. Top {len(top_docs)} retained.")
    for i, doc in enumerate(top_docs, 1):
        source = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("relevance_score", 0.0)
        print(f"[Reranker]   #{i:>2} score={score:+.4f} | {source}")

    return top_docs