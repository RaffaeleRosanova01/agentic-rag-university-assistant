"""
PDF filter module for the DIEM Agentic RAG project.

Reads raw_documents.json produced by crawler.py, removes unwanted PDF
categories (Pass 0), deduplicates remaining PDFs (Passes 1-4), and
optionally deletes excluded files from the PDF download directory.

Passes:
    Pass 0 — Exclude unwanted PDF categories (noise reduction):
        Removes AlmaLaurea statistics, historical course statistics,
        old student guides, administrative decree documents, and other
        categories with no informational value for Q&A.

    Pass 1 — Year-based deduplication:
        For groups of PDFs that differ only in the year component of
        their URL (e.g. regolamenti-cds/2023/ vs 2024/), keeps only
        the most recent version.

    Pass 2 — Content hash deduplication:
        Removes PDFs with identical text content regardless of URL.

    Pass 3 — Filename deduplication across CMS directory IDs:
        For the same filename stored under different numeric directory
        IDs, keeps the copy with the highest (most recent) directory ID.

    Pass 4 — Cross-domain deduplication:
        Removes PDFs linked from both corsi.unisa.it and diem.unisa.it
        under identical /rescue/ paths.

Outputs:
    raw_documents.json      — filtered version, consumed by chunker.py.
    raw_documents_full.json — unmodified backup of the original.

Configuration:
    DELETE_EXCLUDED_PDFS_FROM_DISK (bool) — if True, deletes excluded
    PDF files from config.PDF_DOWNLOAD_DIR after filtering. Set to False
    first to verify which files would be removed before committing.

Usage:
    python -m ingestion.pdf_filter
"""

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import config

# ---------------------------------------------------------------------------
# Configuration flag
# ---------------------------------------------------------------------------

# Set to True only after verifying the exclusion list is correct.
# When False, excluded PDFs are reported but NOT deleted from disk.
DELETE_EXCLUDED_PDFS_FROM_DISK = True

# ---------------------------------------------------------------------------
# Pass 0 — Exclusion patterns
# ---------------------------------------------------------------------------

_PDF_EXCLUDE_PATTERNS = [
    # AlmaLaurea occupational statistics — not useful for Q&A
    (re.compile(r"/__almalaurea/"),
     "AlmaLaurea statistics"),
    # Historical course statistics — not useful for Q&A
    (re.compile(r"/__statistiche-corsi/"),
     "Historical course statistics"),
    # Old student guides (2005-2012)
    (re.compile(r"guida_(?:dello_studente|ingegneria_informatica)_\d{4}[-_]\d{4}\.pdf$", re.IGNORECASE),
     "Old student guides"),
    # Old exam traces
    (re.compile(r"tracce\d{2}-\d{2}\.pdf$", re.IGNORECASE),
     "Old exam traces"),
    # Old committee minutes (before 2020)
    (re.compile(r"verbale-comitato-indirizzo-(?:13|14|17|2017|2018|2019)-", re.IGNORECASE),
     "Old committee minutes"),
    # Administrative decree documents
    (re.compile(r"decreto-(?:approvazione-atti|nomina(?:-com)?|revoca|scorrimento-graduatoria|indizione)", re.IGNORECASE),
     "Administrative decrees"),
    # Approval acts
    (re.compile(r"(?:apporvazione|approvazione)-atti", re.IGNORECASE),
     "Approval acts"),
    # Commission nomination documents
    (re.compile(r"nomina-commissione-(?:esaminatrice|lavori|com)", re.IGNORECASE),
     "Commission nominations"),
    # Selection calendars and final rankings
    (re.compile(r"calendario-colloqui-tutor", re.IGNORECASE),
     "Selection calendars"),
    (re.compile(r"graduatoria-finale", re.IGNORECASE),
     "Final rankings"),
    # Errata corrige
    (re.compile(r"errata-corrige", re.IGNORECASE),
     "Errata corrige"),
    # Files with double .pdf extension (corrupted)
    (re.compile(r"\.pdf\.pdf$", re.IGNORECASE),
     "Double extension files"),
    # Image PDFs (not text content)
    (re.compile(r"(?:image-urbanmis|disegniprovvisori)", re.IGNORECASE),
     "Image PDFs"),
    # PhD student election documents
    (re.compile(r"(?:elettorato-attivo|modello-di-candidatura-dottorandi|manifesto-delle-candidature)", re.IGNORECASE),
     "PhD election documents"),
    # Conclusion letters and generic nominations without content
    (re.compile(r"(?:conclusione-iter|repnomina-commissione|decreto-nomina-com-lav-aut)", re.IGNORECASE),
     "Generic administrative letters"),
    (re.compile(r"/uploads/rescue/\d+/\d+/curriculum.*\.pdf$", re.IGNORECASE),
     "Faculty CV PDFs (covered by HTML profile)"),
    (re.compile(r"/uploads/rescue/\d+/\d+/cv[-_].*\.pdf$", re.IGNORECASE),
     "Faculty CV PDFs (covered by HTML profile)"),
    (re.compile(r"/uploads/rescue/496/", re.IGNORECASE),
     "Erasmus agreements (scanned, poor OCR quality)"),
    (re.compile(r"ettorato-attivo|elettorato-attivo|elettorato-passivo", re.IGNORECASE),
     "Electoral lists (matricola numbers only, no informational value)"),
]


def _get_exclusion_reason(url: str) -> Optional[str]:
    """
    Returns the exclusion reason if the PDF URL matches any exclusion
    pattern, or None if the PDF should be retained.

    Args:
        url: The source URL of the PDF document.

    Returns:
        A human-readable reason string if excluded, None otherwise.
    """
    for pattern, reason in _PDF_EXCLUDE_PATTERNS:
        if pattern.search(url):
            return reason
    return None


# ---------------------------------------------------------------------------
# Year Extraction (Pass 1)
# ---------------------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"(?<=/)(199\d|20\d{2})(?=[/_\-]|$)")


def _extract_year_and_key(url: str) -> Tuple[Optional[int], str]:
    """
    Extracts the year from a PDF source URL and returns a normalized group
    key obtained by replacing the year segment with a fixed placeholder.
    """
    match = _YEAR_PATTERN.search(url)
    if not match:
        return None, url
    year = int(match.group(0))
    group_key = _YEAR_PATTERN.sub("__YEAR__", url)
    return year, group_key


# ---------------------------------------------------------------------------
# Filter Logic
# ---------------------------------------------------------------------------

def filter_pdf_documents(documents: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Filters PDF documents through five passes, returning only high-quality,
    deduplicated documents alongside a detailed report.

    Args:
        documents: The full list of raw documents from crawler.py.

    Returns:
        A tuple (filtered_documents, report).
    """
    html_docs: List[Dict] = []
    pdf_docs: List[Dict] = []

    for doc in documents:
        if doc.get("doc_type") == "pdf":
            pdf_docs.append(doc)
        else:
            html_docs.append(doc)

    report: Dict = {
        "html_documents": len(html_docs),
        "pdf_documents_before": len(pdf_docs),
    }

    # ------------------------------------------------------------------
    # Pass 0 — Exclude unwanted PDF categories
    # ------------------------------------------------------------------
    excluded_docs: List[Dict] = []
    retained_after_exclusion: List[Dict] = []
    exclusion_counts: Dict[str, int] = defaultdict(int)

    for doc in pdf_docs:
        reason = _get_exclusion_reason(doc["source"])
        if reason:
            excluded_docs.append(doc)
            exclusion_counts[reason] += 1
        else:
            retained_after_exclusion.append(doc)

    report["excluded_unwanted"] = len(excluded_docs)
    report["exclusion_by_category"] = dict(exclusion_counts)
    report["excluded_sources"] = [d["source"] for d in excluded_docs]
    pdf_docs = retained_after_exclusion

    # ------------------------------------------------------------------
    # Pass 1 — Year-based deduplication
    # ------------------------------------------------------------------
    groups: Dict[Tuple, Dict] = {}
    group_sources: Dict[str, Set[str]] = defaultdict(set)

    for doc in pdf_docs:
        url = doc["source"]
        year, group_key = _extract_year_and_key(url)
        group_sources[group_key].add(url)
        slot = group_key

        if slot not in groups:
            groups[slot] = {"best_year": year, "best_doc": doc}
        else:
            existing_year = groups[slot]["best_year"]
            if year is not None and (existing_year is None or year > existing_year):
                groups[slot]["best_year"] = year
                groups[slot]["best_doc"] = doc

    retained_pdfs = [g["best_doc"] for g in groups.values()]
    report["duplicates_removed_year"] = len(pdf_docs) - len(retained_pdfs)
    report["groups_with_duplicates"] = [
        {"key": key, "sources": sorted(sources)}
        for key, sources in group_sources.items()
        if len(sources) > 1
    ]

    # ------------------------------------------------------------------
    # Pass 2 — Content hash deduplication
    # ------------------------------------------------------------------
    seen_hashes: Dict[str, Dict] = {}
    for doc in retained_pdfs:
        text_hash = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes[text_hash] = doc
        else:
            if doc["source"] < seen_hashes[text_hash]["source"]:
                seen_hashes[text_hash] = doc

    deduped_pdfs = list(seen_hashes.values())
    report["duplicates_removed_content"] = len(retained_pdfs) - len(deduped_pdfs)

    # ------------------------------------------------------------------
    # Pass 3 — Filename deduplication across CMS directory IDs
    # ------------------------------------------------------------------
    filename_groups: Dict[str, Dict] = {}
    for doc in deduped_pdfs:
        url = doc["source"]
        filename = url.split("/")[-1].split("?")[0].lower()

        if filename not in filename_groups:
            filename_groups[filename] = doc
        else:
            def _dir_id(u: str) -> int:
                parts = u.split("/")
                for part in reversed(parts[:-1]):
                    if part.isdigit():
                        return int(part)
                return 0
            if _dir_id(url) > _dir_id(filename_groups[filename]["source"]):
                filename_groups[filename] = doc

    filename_deduped = list(filename_groups.values())
    report["duplicates_removed_filename"] = len(deduped_pdfs) - len(filename_deduped)

    # ------------------------------------------------------------------
    # Pass 4 — Cross-domain deduplication
    # ------------------------------------------------------------------
    _RESCUE_PATTERN = re.compile(r"/rescue/(.+)")
    path_groups: Dict[str, Dict] = {}
    no_rescue: List[Dict] = []

    for doc in filename_deduped:
        url = doc["source"]
        m = _RESCUE_PATTERN.search(url)
        if not m:
            no_rescue.append(doc)
            continue
        rescue_path = m.group(1).lower()
        if rescue_path not in path_groups:
            path_groups[rescue_path] = doc
        else:
            if "corsi.unisa.it" in url:
                path_groups[rescue_path] = doc

    cross_domain_deduped = list(path_groups.values()) + no_rescue
    report["duplicates_removed_domain"] = len(filename_deduped) - len(cross_domain_deduped)

    report["pdf_documents_after"] = len(cross_domain_deduped)
    return html_docs + cross_domain_deduped, report, excluded_docs


# ---------------------------------------------------------------------------
# Disk Cleanup
# ---------------------------------------------------------------------------

def _delete_excluded_pdfs(excluded_docs: List[Dict]) -> int:
    """
    Deletes PDF files from the download directory that correspond to
    excluded documents.

    Matches files by comparing the filename (last URL segment) against
    files in config.PDF_DOWNLOAD_DIR. The crawler prefixes each file
    with a sequential index (e.g. '0001_filename.pdf'), so we match
    by checking if the URL filename is contained in the disk filename.

    Args:
        excluded_docs: List of excluded document dicts with 'source' URLs.

    Returns:
        Number of files deleted.
    """
    pdf_dir = Path(config.PDF_DOWNLOAD_DIR)
    if not pdf_dir.exists():
        return 0

    # Build set of excluded filenames (lowercase, no query string).
    excluded_filenames: Set[str] = set()
    for doc in excluded_docs:
        fname = doc["source"].rstrip("/").split("/")[-1].split("?")[0].lower()
        excluded_filenames.add(fname)

    deleted = 0
    for pdf_file in pdf_dir.glob("*.pdf"):
        # Disk filename format: '{index:04d}_{original_name}'
        # Strip the leading index to get the original name.
        disk_name = pdf_file.name.lower()
        # Remove leading numeric prefix if present.
        clean_name = re.sub(r"^\d+_", "", disk_name)
        if clean_name in excluded_filenames:
            pdf_file.unlink()
            deleted += 1

    return deleted


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Executes the PDF filtering pipeline:
        1. Load raw_documents.json.
        2. Back up to raw_documents_full.json.
        3. Apply Pass 0 (exclusion) + Passes 1-4 (deduplication).
        4. Write filtered result back to raw_documents.json.
        5. Update the crawl audit log.
        6. Optionally delete excluded PDFs from disk.
        7. Print a detailed report.
    """
    print("=== DIEM Agentic RAG: PDF Filter Started ===\n")

    raw_path = Path(config.RAW_DOCUMENTS_PATH)
    full_backup_path = raw_path.parent / "raw_documents_full.json"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"[Filter] Loading raw documents from {raw_path} ...")
    with open(raw_path, "r", encoding="utf-8") as fh:
        documents: List[Dict] = json.load(fh)
    print(f"[Filter] {len(documents)} documents loaded.")

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------
    shutil.copy2(raw_path, full_backup_path)
    print(f"[Filter] Full backup saved to {full_backup_path}")

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------
    print(f"\n[Filter] Applying filters ...")
    filtered_documents, report, excluded_docs = filter_pdf_documents(documents)

    # ------------------------------------------------------------------
    # Pass 0 detail report
    # ------------------------------------------------------------------
    print(f"\n[Filter] Pass 0 — Excluded PDFs by category:")
    if report["exclusion_by_category"]:
        for category, count in sorted(report["exclusion_by_category"].items()):
            print(f"[Filter]   {count:>4}  {category}")
        print(f"\n[Filter] Excluded sources:")
        for url in sorted(report["excluded_sources"]):
            print(f"[Filter]   - {url}")
    else:
        print(f"[Filter]   No PDFs excluded.")

    # ------------------------------------------------------------------
    # Pass 1 detail report
    # ------------------------------------------------------------------
    if report["groups_with_duplicates"]:
        print(f"\n[Filter] Pass 1 — Year duplicates discarded:")
        for group in report["groups_with_duplicates"]:
            urls = group["sources"]
            years = [int(m.group(0)) if (m := _YEAR_PATTERN.search(u)) else None for u in urls]
            best_year = max((y for y in years if y is not None), default=None)
            kept_url = next(
                u for u, y in zip(urls, years)
                if y == best_year or best_year is None
            )
            for d in [u for u in urls if u != kept_url]:
                print(f"[Filter]   Discarded : {d}")

    # ------------------------------------------------------------------
    # Write filtered output
    # ------------------------------------------------------------------
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(filtered_documents, fh, ensure_ascii=False, indent=2)
    print(f"\n[Filter] Filtered documents saved to {raw_path}")

    # ------------------------------------------------------------------
    # Update audit log
    # ------------------------------------------------------------------
    audit_path = Path(config.CRAWL_AUDIT_PATH)
    if audit_path.exists():
        urls = sorted({doc["source"] for doc in filtered_documents})
        with open(audit_path, "w", encoding="utf-8") as fh:
            fh.write("=== DIEM Agentic RAG: Crawl Audit Log (post-filter) ===\n")
            fh.write(f"Total documents: {len(filtered_documents)}\n")
            fh.write(f"Unique sources:  {len(urls)}\n\n")
            for url in urls:
                fh.write(f"{url}\n")
        print(f"[Filter] Audit log updated: {len(urls)} unique sources.")

    # ------------------------------------------------------------------
    # Optional disk cleanup
    # ------------------------------------------------------------------
    if DELETE_EXCLUDED_PDFS_FROM_DISK:
        deleted = _delete_excluded_pdfs(excluded_docs)
        print(f"[Filter] Deleted {deleted} excluded PDF file(s) from disk.")
    else:
        print(
            f"\n[Filter] Disk cleanup SKIPPED (DELETE_EXCLUDED_PDFS_FROM_DISK=False).\n"
            f"[Filter] Review the excluded sources above, then set "
            f"DELETE_EXCLUDED_PDFS_FROM_DISK=True to delete them."
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n[Filter] Summary:")
    print(f"[Filter]   HTML documents (unchanged)       : {report['html_documents']}")
    print(f"[Filter]   PDF documents before filter      : {report['pdf_documents_before']}")
    print(f"[Filter]   Excluded (unwanted categories)   : {report['excluded_unwanted']}")
    print(f"[Filter]   Duplicates removed (year)        : {report['duplicates_removed_year']}")
    print(f"[Filter]   Duplicates removed (content)     : {report['duplicates_removed_content']}")
    print(f"[Filter]   Duplicates removed (filename)    : {report['duplicates_removed_filename']}")
    print(f"[Filter]   Duplicates removed (domain)      : {report['duplicates_removed_domain']}")
    print(f"[Filter]   PDF documents after filter       : {report['pdf_documents_after']}")
    print(f"[Filter]   Total documents after filter     : {len(filtered_documents)}")
    print("\n=== PDF Filter Completed ===")


if __name__ == "__main__":
    run()