"""
Document length analyzer for the DIEM Agentic RAG project.

Reads raw_documents.json and prints statistical analysis of document
lengths, separated by doc_type (html vs pdf). Useful for choosing
optimal parent chunk sizes before indexing.

Usage:
    python analyze_docs.py
"""

import json
import os
import statistics
from collections import Counter

RAW_DOCUMENTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "raw_documents.json"
)

PERCENTILES = [10, 25, 50, 75, 90, 95, 99]


def analyze(docs, label):
    lengths = [len(d["text"]) for d in docs]
    if not lengths:
        print(f"\n[{label}] No documents found.")
        return

    lengths.sort()
    n = len(lengths)

    print(f"\n{'=' * 60}")
    print(f"  {label}  ({n} documents)")
    print(f"{'=' * 60}")
    print(f"  Min    : {min(lengths):>8} chars")
    print(f"  Max    : {max(lengths):>8} chars")
    print(f"  Mean   : {int(statistics.mean(lengths)):>8} chars")
    print(f"  Median : {int(statistics.median(lengths)):>8} chars")
    print(f"  StdDev : {int(statistics.stdev(lengths)):>8} chars" if n > 1 else "")

    print(f"\n  Percentiles:")
    for p in PERCENTILES:
        idx = min(int(n * p / 100), n - 1)
        print(f"    P{p:>2} : {lengths[idx]:>8} chars")

    # Bucket distribution
    buckets = [
        (" < 200",   lambda x: x < 200),
        (" 200-500",  lambda x: 200 <= x < 500),
        (" 500-1000", lambda x: 500 <= x < 1000),
        ("1000-2000", lambda x: 1000 <= x < 2000),
        ("2000-4000", lambda x: 2000 <= x < 4000),
        ("4000-8000", lambda x: 4000 <= x < 8000),
        (" > 8000",   lambda x: x >= 8000),
    ]
    print(f"\n  Distribution:")
    for name, fn in buckets:
        count = sum(1 for l in lengths if fn(l))
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {name}: {count:>5} ({pct:>5.1f}%)  {bar}")

    # Recommendation
    p50 = lengths[int(n * 0.50)]
    p75 = lengths[int(n * 0.75)]
    p90 = lengths[int(n * 0.90)]
    print(f"\n  Chunk size suggestions:")
    print(f"    Small  parent (~P50) : {_round(p50)} chars")
    print(f"    Medium parent (~P75) : {_round(p75)} chars")
    print(f"    Large  parent (~P90) : {_round(p90)} chars")
    print(f"    (child = parent / 3 is a reasonable starting point)")


def _round(n):
    """Round to nearest 100."""
    return int(round(n / 100) * 100)


def main():
    print(f"Loading documents from {RAW_DOCUMENTS_PATH} ...")
    with open(RAW_DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"Total documents: {len(docs)}")

    html_docs = [d for d in docs if d.get("doc_type") == "html"]
    pdf_docs  = [d for d in docs if d.get("doc_type") == "pdf"]

    analyze(html_docs, "HTML DOCUMENTS")
    analyze(pdf_docs,  "PDF DOCUMENTS")

    # Combined
    analyze(docs, "ALL DOCUMENTS")


if __name__ == "__main__":
    main()