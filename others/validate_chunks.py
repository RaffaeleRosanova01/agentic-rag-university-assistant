"""
Chunk validator for the DIEM Agentic RAG project.

Reads chunked_documents.json (child chunks) and chunked_parents.json
and performs quality checks:

    1. Size distribution  — how many chunks fall in each size bucket,
                            to detect excessive small or large chunks.
    2. Duplicate IDs      — checks chunk_id and parent_id uniqueness.
    3. Navigation noise   — scans child_text for known navigation patterns
                            that should not appear in embedded content.
    4. Empty / short      — chunks below minimum useful length.
    5. Header coverage    — how many children have a non-trivial section
                            breadcrumb vs just the URL header.
    6. Parent coverage    — checks that every parent_id in children
                            exists in chunked_parents.json.

Usage:
    python validate_chunks.py
"""

import json
import os
import re
import statistics
from collections import Counter, defaultdict

CHUNKED_DOCUMENTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "chunked_documents.json"
)
CHUNKED_PARENTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "chunked_parents.json"
)

# Navigation noise patterns — text that should not appear in clean chunks.
_NOISE_PATTERNS = [
    re.compile(r"skip to main content", re.IGNORECASE),
    re.compile(r"toggle navigation", re.IGNORECASE),
    re.compile(r"cerca persona", re.IGNORECASE),
    re.compile(r"^search$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"archivio mail", re.IGNORECASE),
    re.compile(r"helpdesk", re.IGNORECASE),
    re.compile(r"wi-fi@unisa", re.IGNORECASE),
    re.compile(r"tutti i servizi", re.IGNORECASE),
    re.compile(r"futuro studente", re.IGNORECASE),
    re.compile(r"studente iscritto", re.IGNORECASE),
    re.compile(r"personale tecnico amministrativo", re.IGNORECASE),
    re.compile(r"webmail studenti", re.IGNORECASE),
    re.compile(r"sitemap", re.IGNORECASE),
]

BUCKETS = [
    ("  < 100",   lambda x: x < 100),
    ("100-200",   lambda x: 100 <= x < 200),
    ("200-400",   lambda x: 200 <= x < 400),
    ("400-600",   lambda x: 400 <= x < 600),
    ("600-800",   lambda x: 600 <= x < 800),
    ("800-1200",  lambda x: 800 <= x < 1200),
    ("> 1200",    lambda x: x >= 1200),
]


def _check_sizes(chunks, label):
    lengths = [len(c["child_text"]) for c in chunks]
    if not lengths:
        return
    n = len(lengths)
    print(f"\n--- {label} size distribution ({n} chunks) ---")
    for name, fn in BUCKETS:
        count = sum(1 for l in lengths if fn(l))
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"  {name}: {count:>6} ({pct:>5.1f}%)  {bar}")
    print(f"\n  Min    : {min(lengths)} chars")
    print(f"  Max    : {max(lengths)} chars")
    print(f"  Mean   : {int(statistics.mean(lengths))} chars")
    print(f"  Median : {int(statistics.median(lengths))} chars")


def _check_duplicate_ids(chunks):
    print(f"\n--- Duplicate ID check ---")
    chunk_ids = [c["chunk_id"] for c in chunks]
    parent_ids = [c["parent_id"] for c in chunks]

    dup_chunk = [cid for cid, cnt in Counter(chunk_ids).items() if cnt > 1]
    dup_parent_in_children = [pid for pid, cnt in Counter(parent_ids).items() if cnt > 1]

    if dup_chunk:
        print(f"  WARNING: {len(dup_chunk)} duplicate chunk_id(s) found!")
        for cid in dup_chunk[:5]:
            print(f"    - {cid}")
    else:
        print(f"  chunk_id: all {len(chunk_ids)} are unique. OK")

    # Multiple children per parent is expected — just report the distribution.
    children_per_parent = Counter(parent_ids)
    avg = statistics.mean(children_per_parent.values())
    max_children = max(children_per_parent.values())
    singletons = sum(1 for v in children_per_parent.values() if v == 1)
    print(f"\n  Children per parent:")
    print(f"    Unique parents     : {len(children_per_parent)}")
    print(f"    Avg children/parent: {avg:.1f}")
    print(f"    Max children/parent: {max_children}")
    print(f"    Parents with 1 child (short sections): {singletons} "
          f"({singletons / len(children_per_parent) * 100:.1f}%)")


def _check_noise(chunks):
    print(f"\n--- Navigation noise check ---")
    noisy = []
    for c in chunks:
        text = c["child_text"]
        for pattern in _NOISE_PATTERNS:
            if pattern.search(text):
                noisy.append((c["source"], pattern.pattern, text[:200]))
                break

    if noisy:
        print(f"  WARNING: {len(noisy)} chunk(s) contain navigation noise.")
        print(f"  First 10 examples:")
        for source, pattern, preview in noisy[:10]:
            print(f"\n    Source  : {source}")
            print(f"    Pattern : {pattern}")
            print(f"    Preview : {preview[:100]!r}")
    else:
        print(f"  No navigation noise detected. OK")


def _check_short_chunks(chunks, min_chars=100):
    print(f"\n--- Short chunk check (< {min_chars} chars) ---")
    short = [c for c in chunks if len(c["child_text"]) < min_chars]
    if short:
        print(f"  WARNING: {len(short)} chunk(s) shorter than {min_chars} chars.")
        print(f"  First 5 examples:")
        for c in short[:5]:
            print(f"\n    Source  : {c['source']}")
            print(f"    Length  : {len(c['child_text'])} chars")
            print(f"    Content : {c['child_text']!r}")
    else:
        print(f"  No chunks shorter than {min_chars} chars. OK")


def _check_header_coverage(chunks):
    print(f"\n--- Section breadcrumb coverage ---")
    with_section = sum(1 for c in chunks if "\nSection:" in c["child_text"])
    n = len(chunks)
    pct = with_section / n * 100 if n else 0
    print(f"  Chunks with Section breadcrumb : {with_section} ({pct:.1f}%)")
    print(f"  Chunks with URL header only    : {n - with_section} ({100 - pct:.1f}%)")


def _check_parent_coverage(chunks, parents):
    print(f"\n--- Parent coverage check ---")
    parent_ids_in_store = {p["parent_id"] for p in parents}
    child_parent_ids = {c["parent_id"] for c in chunks}

    missing = child_parent_ids - parent_ids_in_store
    orphaned = parent_ids_in_store - child_parent_ids

    if missing:
        print(f"  WARNING: {len(missing)} parent_id(s) referenced by children "
              f"but NOT in chunked_parents.json!")
    else:
        print(f"  All {len(child_parent_ids)} parent_ids in children "
              f"exist in chunked_parents.json. OK")

    if orphaned:
        print(f"  INFO: {len(orphaned)} parent(s) in chunked_parents.json "
              f"have no children (short documents passed as single parent).")


def _check_domain_distribution(chunks):
    print(f"\n--- Domain distribution ---")
    domain_counts = Counter(c.get("domain", "unknown") for c in chunks)
    total = len(chunks)
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {domain:<15}: {count:>6} ({pct:>5.1f}%)  {bar}")


def main():
    print(f"Loading children from {CHUNKED_DOCUMENTS_PATH} ...")
    with open(CHUNKED_DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Total child chunks: {len(chunks)}")

    print(f"\nLoading parents from {CHUNKED_PARENTS_PATH} ...")
    with open(CHUNKED_PARENTS_PATH, "r", encoding="utf-8") as f:
        parents = json.load(f)
    print(f"Total parent chunks: {len(parents)}")

    html_chunks = [c for c in chunks if c.get("doc_type") == "html"]
    pdf_chunks  = [c for c in chunks if c.get("doc_type") == "pdf"]

    print(f"\n{'=' * 60}")
    print(f"  CHUNK VALIDATION REPORT")
    print(f"{'=' * 60}")

    _check_sizes(html_chunks, "HTML chunks")
    _check_sizes(pdf_chunks,  "PDF chunks")
    _check_sizes(chunks,      "ALL chunks")

    print(f"\n{'=' * 60}")
    _check_duplicate_ids(chunks)

    print(f"\n{'=' * 60}")
    _check_noise(chunks)

    print(f"\n{'=' * 60}")
    _check_short_chunks(chunks, min_chars=100)

    print(f"\n{'=' * 60}")
    _check_header_coverage(chunks)

    print(f"\n{'=' * 60}")
    _check_parent_coverage(chunks, parents)

    print(f"\n{'=' * 60}")
    _check_domain_distribution(chunks)

    print(f"\n{'=' * 60}")
    print(f"  Validation complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()