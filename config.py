"""
Configuration module for the DIEM Agentic RAG project.

Centralizes all environment variables, path definitions, model identifiers,
crawling parameters, chunking settings, and retrieval constants.
All other modules must import their settings exclusively from this file.
"""

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project Root and Environment
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Only required when LLM_PROVIDER is "openai" or "google".
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not HF_TOKEN and not OPENROUTER_API_KEY:
    import warnings
    warnings.warn(
        "Neither HF_TOKEN nor OPENROUTER_API_KEY is set in the .env file. "
        "Set at least one before running the agent."
    )

# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

# Supported providers: "openrouter", "ollama", "openai", "google", "huggingface".
# The "google" provider requires the optional package langchain-google-genai.
LLM_PROVIDER = "openrouter"

# Ollama base URL (only used when LLM_PROVIDER = "ollama").
OLLAMA_BASE_URL = "http://localhost:11434"

# OpenRouter base URL.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Language Models
# ---------------------------------------------------------------------------

# Primary model — used by the agent for tool calling and answer generation.
# Should be a capable model with reliable tool calling support.
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# Guardrail model — used by input and output guardrail nodes.
# A smaller, faster model is sufficient for classification tasks.
GUARDRAIL_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

# Judge model used by evaluate.py to grade answers. Using a different model
# from the agent reduces self-evaluation bias.
EVAL_JUDGE_MODEL_NAME = "openai/gpt-oss-120b"

# Maximum number of tokens the LLM may generate in a single call.
LLM_MAX_NEW_TOKENS = 4096

# Sampling temperature. Lower values produce more deterministic, factual responses.
LLM_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

# Backend for cross-encoder reranking: "local" or "openrouter".
RERANKER_PROVIDER = "openrouter"

# Local cross-encoder model (used when RERANKER_PROVIDER = "local").
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# OpenRouter reranker model (used when RERANKER_PROVIDER = "openrouter").
RERANKER_MODEL_NAME_OPENROUTER = "cohere/rerank-4-pro"

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

# Primary embedding model — superior on multilingual benchmarks (570M params).
# Changing this model requires rebuilding the ChromaDB index.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Device for embedding inference. Auto-selects CUDA if available.
import torch as _torch
EMBEDDING_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Number of child chunks embedded in a single forward pass.
# 128 recommended on GPU, 64 on CPU.
EMBEDDING_BATCH_SIZE = 128 if EMBEDDING_DEVICE == "cuda" else 64

# ---------------------------------------------------------------------------
# Directory Layout
# ---------------------------------------------------------------------------

# Local cache for HuggingFace model downloads.
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Persistent ChromaDB vector store.
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Intermediate JSON produced by crawler.py, consumed by chunker.py.
RAW_DOCUMENTS_PATH = os.path.join(BASE_DIR, "data", "raw_documents.json")

# Intermediate JSON produced by chunker.py, consumed by indexer.py.
CHUNKED_DOCUMENTS_PATH = os.path.join(BASE_DIR, "data", "chunked_documents.json")

# JSON produced by chunker.py, loaded into InMemoryStore at runtime by retriever.py.
CHUNKED_PARENTS_PATH = os.path.join(BASE_DIR, "data", "chunked_parents.json")

# Directory where PDF files are downloaded during crawling.
PDF_DOWNLOAD_DIR = os.path.join(BASE_DIR, "data", "pdfs")

# Audit log listing every URL successfully processed by the crawler.
CRAWL_AUDIT_PATH = os.path.join(BASE_DIR, "data", "crawl_audit.txt")

# Create all required directories at import time.
for _dir in (MODELS_DIR, CHROMA_DB_DIR, PDF_DOWNLOAD_DIR, os.path.join(BASE_DIR, "data")):
    os.makedirs(_dir, exist_ok=True)

# Force HuggingFace libraries to use the project-local cache.
os.environ["HF_HOME"] = MODELS_DIR

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

COLLECTION_NAME = "diem_knowledge_base"

# ---------------------------------------------------------------------------
# Crawling Parameters
# ---------------------------------------------------------------------------

# Seed URLs used to build domain-specific whitelists.
DIEM_FACULTY_SEED_URL = "https://www.diem.unisa.it/dipartimento/personale"
DIEM_COURSES_SEED_URL = "https://www.diem.unisa.it/didattica/offerta-formativa"

# Entry points and maximum BFS depths for each domain.
CRAWL_TARGETS = {
    "diem": {
        "seed": "https://www.diem.unisa.it/",
        "depth": 4,
        "allowed_domain": "www.diem.unisa.it",
    },
    "docenti": {
        # Individual faculty URLs are generated dynamically by the whitelist builder.
        # Depth 1 captures all sub-pages of each faculty profile.
        "seed": None,
        "depth": 1,
        "allowed_domain": "docenti.unisa.it",
    },
    "corsi": {
        # Individual course URLs are generated dynamically by the whitelist builder.
        # Depth 3 captures syllabus, timetable, and exam pages nested inside each course.
        "seed": None,
        "depth": 3,
        "allowed_domain": "corsi.unisa.it",
    },
    "easycourse": {
        "seed": "https://easycourse.unisa.it/EasyCourse/Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2025-2026/index.html",
        "depth": 2,
        "allowed_domain": "easycourse.unisa.it",
    },
}

# Domains from which content may be harvested.
ALLOWED_DOMAINS = frozenset([
    "www.diem.unisa.it",
    "docenti.unisa.it",
    "corsi.unisa.it",
])

# URL substrings that identify pages with no informational value.
URL_BLACKLIST = frozenset([
    "unisa-rescue-page", "/rescue/", "password-recovery",
    "idp/profile", "idp.unisa.it", "/sitemap", "?sitemap",
    "/login", "/logout", "/search", ".xls", ".xlsx",
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    "/home/news", "/news/", "/news?", "/home/eventi", "/didattica/orari", 
    "/comunicazioni-docenti", "?avvisi=",
])

# Minimum characters a cleaned page must contain to be retained.
MIN_PAGE_LENGTH = 200

# HTTP request timeout in seconds.
REQUEST_TIMEOUT = 15

# Delay between consecutive HTTP requests to avoid overloading servers.
CRAWL_DELAY = 0.3

# ---------------------------------------------------------------------------
# Chunking Parameters
# ---------------------------------------------------------------------------

# Child chunks — used for vector indexing (small, precise embeddings).
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

# Parent chunks — returned to the LLM as context (richer content).
PARENT_CHUNK_SIZE = 1200
PARENT_CHUNK_OVERLAP = 100

CHUNK_MIN_CHARS = 100

# ---------------------------------------------------------------------------
# Retrieval Parameters
# ---------------------------------------------------------------------------

# Number of candidate chunks retrieved by the bi-encoder.
SIMILARITY_TOP_K = 50

# Number of chunks retained after cross-encoder reranking.
RERANK_TOP_N = 15

# ---------------------------------------------------------------------------
# Agentic Loop Parameters
# ---------------------------------------------------------------------------

# Maximum number of tool call iterations per query before forcing a final answer.
MAX_AGENT_ITERATIONS = 5

# Maximum number of output guardrail retries before returning the fallback message.
MAX_GUARDRAIL_RETRIES = 1

# ---------------------------------------------------------------------------
# Output Guardrail Parameters
# ---------------------------------------------------------------------------
# Enable the output guardrail node.
ENABLE_OUTPUT_GUARDRAIL = True