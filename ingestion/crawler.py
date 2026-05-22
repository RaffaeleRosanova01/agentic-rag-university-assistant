"""
Crawler module for the DIEM Agentic RAG project.

Implements a BFS-based web crawler that harvests textual content from four
authorized domains:
    - www.diem.unisa.it      (main department site)
    - docenti.unisa.it       (faculty profiles)
    - corsi.unisa.it         (course pages)
    - easycourse.unisa.it    (lesson timetables)

Intercepts PDF links during HTML crawling and downloads them separately.
Produces a serialized JSON file of raw documents consumed by chunker.py.

Key design decisions:
    - Trafilatura with output_format='markdown' for content extraction.
      Preserves table structure (| col | col |) and heading hierarchy.
      No BeautifulSoup fallback — pages where Trafilatura finds no content
      are discarded as navigation shells with no informational value.
    - Use of pymupdf4llm. Each PDF is treated as a single
      Markdown document instead of page-by-page plain text, preserving
      headings, tables, and structure.
    - content_hash (SHA-256 of normalized text) stored alongside each
      document to support incremental indexing updates.
    - Query parameters preserved by default (only tracking/session params
      stripped), so content-identifying params like ?id=, ?dettaglio= are
      retained and their pages are crawled correctly.
    - EasyCourse timetable pages (static HTML) crawled in Phase 4.
      The seed URL must be updated in config.py each academic year.

Usage:
    python -m ingestion.crawler
"""

import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.parse
from collections import deque
from typing import Dict, List, Set, Tuple

import requests
import trafilatura
import urllib3
from bs4 import BeautifulSoup

import config

# ---------------------------------------------------------------------------
# HTTP Session
# ---------------------------------------------------------------------------

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _build_session() -> requests.Session:
    """
    Builds a persistent HTTP session with SSL verification disabled.

    SSL verification is disabled because the UniSA web infrastructure
    serves certificates that are occasionally invalid or self-signed.
    The scope of this bypass is intentionally limited to this session
    object and does not affect any other component of the system.
    """
    session = requests.Session()
    session.verify = False
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; DIEM-RAG-Crawler/1.0; "
            "+https://www.diem.unisa.it/)"
        )
    })
    return session


SESSION = _build_session()


# ---------------------------------------------------------------------------
# URL Utilities
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """
    Normalizes a URL by applying the following transformations in order:

        1. Force HTTPS scheme to eliminate http/https duplicates.
        2. Strip fragment identifiers (#...).
        3. Strip trailing slashes.
        4. For docenti.unisa.it: remove the redundant /home suffix from
           profile root URLs (e.g. /003495/home -> /003495) so that the
           base profile URL and its /home variant are treated as one.
        5. For www.diem.unisa.it: remove the redundant /home suffix to
           eliminate the duplicate between the site root and /home.
        6. Strip only known tracking and session query parameters, preserving
           all content-identifying parameters (e.g. ?id=, ?dettaglio=) that
           point to unique pages on UniSA CMS sites.
    """
    # 1. Force HTTPS.
    if url.startswith("http://"):
        url = "https://" + url[7:]

    # 2. Strip fragment.
    url = url.split("#")[0]

    # 3. Strip trailing slash.
    url = url.rstrip("/")

    # 4. Normalize docenti.unisa.it profile URLs.
    if "docenti.unisa.it/" in url and url.endswith("/home"):
        url = url[: -len("/home")]

    # 5. Normalize www.diem.unisa.it root URL.
    if url == "https://www.diem.unisa.it/home":
        url = "https://www.diem.unisa.it"

    # 6. Handle query parameters.
    # Strip only parameters that do not identify content — tracking pixels,
    # session tokens, language switches, and pagination params generate
    # duplicates and must be removed.
    #
    # Exceptions:
    #   - /home/bandi: anno, struttura and modulo are content-identifying —
    #     they filter by year and department. Without them the page shows
    #     bandi from all departments and all years.
    #   - /ricerca/pubblicazioni: anno is content-identifying — without it
    #     the page shows only the most recent year (server-side default).
    if "?" not in url:
        return url

    _STRIP_PARAMS = {
        "lang", "language", "lng", "locale", "session", "sessid", "token",
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "page", "paged", "p", "print", "format", "output", "anno", "struttura", 
        "modulo", "bando", "idConcorso", "categoria", "start", "end", "from", "to",
        "progetto", "tip", "sitemap", "stato",
    }

    # For bandi pages, preserve anno + struttura + modulo.
    if "/home/bandi" in url:
        _STRIP_PARAMS -= {"anno", "struttura", "modulo"}

    # For pubblicazioni pages, preserve anno.
    if "/ricerca/pubblicazioni" in url:
        _STRIP_PARAMS -= {"anno"}

    base, query = url.split("?", 1)
    kept = [
        p for p in query.split("&")
        if p and p.split("=")[0].lower() not in _STRIP_PARAMS
    ]
    return f"{base}?{'&'.join(kept)}" if kept else base


def _is_allowed(url: str) -> bool:
    """
    Returns True if the URL belongs to one of the authorized domains defined
    in config.ALLOWED_DOMAINS, does not match any blacklisted pattern, and
    does not contain a language-variant path segment.

    Note: easycourse.unisa.it is intentionally NOT included in
    config.ALLOWED_DOMAINS to prevent the main domain crawlers from
    following EasyCourse links found on DIEM pages. EasyCourse is crawled
    separately in Phase 4 with its own _crawl_bfs call and allowed_domain
    constraint.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    if not any(parsed.netloc == domain for domain in config.ALLOWED_DOMAINS):
        return False

    url_lower = url.lower()

    if any(pattern in url_lower for pattern in config.URL_BLACKLIST):
        return False

    # Reject language-variant URLs on any authorized domain.
    _LANG_SEGMENTS = ("/it/", "/en/", "/fr/", "/de/", "/es/", "/zh/")
    _LANG_SUFFIXES = ("/it", "/en", "/fr", "/de", "/es", "/zh")
    if any(lang in url_lower for lang in _LANG_SEGMENTS):
        return False
    if any(url_lower.endswith(suffix) for suffix in _LANG_SUFFIXES):
        return False

    return True


def _extract_links(soup: BeautifulSoup, base_url: str) -> Tuple[Set[str], Set[str]]:
    """
    Extracts and classifies all anchor links found in a parsed HTML page.

    Returns two sets:
        page_links: absolute URLs pointing to HTML pages within allowed domains.
        pdf_links:  absolute URLs pointing to PDF documents within allowed domains.
    """
    page_links: Set[str] = set()
    pdf_links: Set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()

        if not href or href.startswith("mailto:") or href.startswith("tel:"):
            continue

        if href.startswith("uploads/"):
            href = "/" + href

        absolute = urllib.parse.urljoin(base_url, href)
        absolute = _normalize_url(absolute)

        if absolute.lower().endswith(".pdf"):
            authorized_domains = list(config.ALLOWED_DOMAINS)
            if any(d in absolute for d in authorized_domains):
                pdf_links.add(absolute)
        elif _is_allowed(absolute):
            # Filter bandi links: only accept anno 2025 and 2026.
            if "/home/bandi" in absolute:
                anno_match = re.search(r"[?&]anno=(\d{4})", absolute)
                if anno_match:
                    if int(anno_match.group(1)) not in (2025, 2026):
                        continue
                else:
                    # No anno parameter — skip to avoid unfiltered bandi page.
                    continue

            # Filter pubblicazioni links: only accept anno 2024, 2025, 2026.
            if "/ricerca/pubblicazioni" in absolute:
                anno_match = re.search(r"[?&]anno=(\d{4})", absolute)
                if anno_match:
                    if int(anno_match.group(1)) not in (2024, 2025, 2026):
                        continue
                # No anno parameter — allow through (server returns latest year).

            page_links.add(absolute)

    return page_links, pdf_links

# ---------------------------------------------------------------------------
# Navigation line filter
# ---------------------------------------------------------------------------

_NAV_LINES = frozenset([
    "skip to main content", "cerca persona", "search", "english", "chinese", "français", 
    "toggle navigation", "wi-fi@unisa", "webmail",  "webmail studenti", "tutti i servizi", 
    "futuro studente", "studente iscritto", "laureato", "personale tecnico amministrativo", 
    "sitemap", "rss", "rubrica", "archivio mail allusers", "sicurezza informatica", 
    "agenda studenti", "area utente (esse3)", "helpdesk", "office365", "personaldesk",
])

def _remove_nav_lines(text: str) -> str:
    """
    Removes lines that consist entirely of known navigation elements
    from text extracted with favor_precision=False.
    """
    lines = text.split("\n")
    cleaned = [l for l in lines if l.strip().lower() not in _NAV_LINES]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


# ---------------------------------------------------------------------------
# HTML Content Extraction
# ---------------------------------------------------------------------------

def _extract_content(html: str, url: str) -> str:
    """
    Extracts the main textual content from an HTML page as Markdown.

    Uses Trafilatura with favor_precision=True as the primary strategy.
    If the result is below MIN_PAGE_LENGTH (typically pages with only
    tables and no narrative text, e.g. course detail pages on
    docenti.unisa.it), retries with favor_precision=False to capture
    structured table content that the precision mode discards.

    Args:
        html: Raw HTML string of the fetched page.
        url:  Canonical URL of the page, used by Trafilatura for context.

    Returns:
        Markdown-formatted text of the main content area, or empty string
        if no useful content was found.
    """
    def _try(favor_precision: bool) -> str:
        try:
            text = trafilatura.extract(
                html,
                url=url,
                output_format="markdown",
                include_tables=True,
                include_links=False,
                include_images=False,
                favor_precision=favor_precision,
                deduplicate=True,
            )
        except Exception:
            return ""
        if not text:
            return ""
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    text = _try(favor_precision=True)

    # Fallback: if precision mode returns too little content, retry with
    # lower precision to capture table-heavy pages (e.g. course detail
    # pages on docenti.unisa.it that have no narrative text).
    if len(text) < config.MIN_PAGE_LENGTH:
        text_fallback = _try(favor_precision=False)
        text_fallback = _remove_nav_lines(text_fallback)
        if len(text_fallback) >= config.MIN_PAGE_LENGTH:
            text = text_fallback

    return text


# ---------------------------------------------------------------------------
# content_hash Helper
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    """
    Computes a deterministic SHA-256 hash of the document text.

    Whitespace is normalised before hashing (all runs of whitespace
    collapsed to a single space) to avoid spurious hash changes caused
    by minor formatting differences between crawl runs — e.g. a page
    that adds a trailing newline or changes indentation between two
    crawls would produce a different hash without normalisation, causing
    unnecessary re-indexing of unchanged content.

    Args:
        text: Extracted text content of the document.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Whitelist Builders
# ---------------------------------------------------------------------------

def build_faculty_whitelist() -> List[str]:
    """
    Constructs the list of authorized faculty profile URLs on docenti.unisa.it.

    The procedure follows four steps:
        1. Fetch the DIEM personnel page and extract all rubrica.unisa.it
           links, collecting the matricola identifiers from their query strings.
        2. For each matricola, visit the corresponding rubrica page and look
           for a link to docenti.unisa.it.
        3. Follow the name-based URL to resolve the canonical numeric matricola
           URL via HTTP redirect.
        4. Only matricole that have a confirmed docenti.unisa.it link are
           included in the whitelist.
    """
    print(f"[Whitelist] Building faculty whitelist from: {config.DIEM_FACULTY_SEED_URL}")

    try:
        response = SESSION.get(config.DIEM_FACULTY_SEED_URL, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Whitelist] ERROR: Could not retrieve faculty seed page. {exc}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    matricola_pattern = re.compile(r"matricola=(\d+)")
    all_matricole: Set[str] = set()

    for anchor in soup.find_all("a", href=True):
        match = matricola_pattern.search(anchor["href"])
        if match:
            all_matricole.add(match.group(1))

    print(f"[Whitelist]   Found {len(all_matricole)} matricola identifier(s) on the personnel page.")

    PROFILE_SUB_PAGES = (
        "home",
        "curriculum",
        "didattica",
        "ricerca/progetti",
        "ricerca/pubblicazioni",
        "ricerca/laboratori",
    )

    _LANG_SEGMENTS = ("/it/", "/en/", "/fr/", "/de/", "/es/", "/zh/")

    authorized_urls: List[str] = []
    confirmed_faculty = 0
    not_found = 0

    for matricola in sorted(all_matricole):
        rubrica_url = f"https://rubrica.unisa.it/persone?matricola={matricola}"
        try:
            time.sleep(config.CRAWL_DELAY)
            rubrica_response = SESSION.get(rubrica_url, timeout=config.REQUEST_TIMEOUT)
            rubrica_response.raise_for_status()
        except requests.RequestException as exc:
            not_found += 1
            print(f"[Whitelist]   SKIP matricola {matricola} — rubrica unreachable: {exc}")
            continue

        rubrica_soup = BeautifulSoup(rubrica_response.text, "html.parser")

        docenti_name_url = None
        for anchor in rubrica_soup.find_all("a", href=True):
            href = anchor["href"]
            if "docenti.unisa.it" not in href:
                continue
            clean = href.rstrip("/")
            for lang in _LANG_SEGMENTS:
                if lang in clean:
                    clean = clean.split(lang)[0]
                    break
            for sub in PROFILE_SUB_PAGES:
                if clean.endswith("/" + sub):
                    clean = clean[: -(len(sub) + 1)]
                    break
            docenti_name_url = clean.rstrip("/")
            break

        if not docenti_name_url:
            not_found += 1
            print(f"[Whitelist]   SKIP matricola {matricola} "
                  f"(no docenti.unisa.it link in rubrica — non-faculty staff).")
            continue

        try:
            time.sleep(config.CRAWL_DELAY)
            redirect_response = SESSION.get(
                docenti_name_url, timeout=config.REQUEST_TIMEOUT
            )
            final_url = redirect_response.url.rstrip("/")

            for sub in PROFILE_SUB_PAGES:
                if final_url.endswith("/" + sub):
                    final_url = final_url[: -(len(sub) + 1)].rstrip("/")

            for lang in _LANG_SEGMENTS:
                if lang in final_url:
                    final_url = final_url.split(lang)[0].rstrip("/")
                    break

            canonical_base = final_url

        except requests.RequestException:
            canonical_base = docenti_name_url

        # Verify that the resolved canonical URL contains a numeric matricola
        # segment (e.g. docenti.unisa.it/004322). URLs that resolve to a
        # name-based path indicate the person belongs to a different department.
        path_segment = canonical_base.rstrip("/").split("/")[-1]
        if not path_segment.isdigit():
            numeric_url = f"https://docenti.unisa.it/{matricola}"
            try:
                test_response = SESSION.get(numeric_url, timeout=config.REQUEST_TIMEOUT)
                # A valid profile redirects (302 → 200). The general listing
                # returns 200 directly with no redirect history.
                if test_response.history and test_response.status_code == 200:
                    canonical_base = test_response.url.rstrip("/").split("/home")[0]
                    print(f"[Whitelist]   FIXED matricola {matricola} → {canonical_base}")
                else:
                    not_found += 1
                    print(f"[Whitelist]   SKIP matricola {matricola} "
                        f"(no redirect — no valid profile on docenti.unisa.it).")
                    continue
            except Exception:
                not_found += 1
                continue

        for sub_page in PROFILE_SUB_PAGES:
            authorized_urls.append(f"{canonical_base}/{sub_page}")
        confirmed_faculty += 1
        print(f"[Whitelist]   CONFIRMED faculty: matricola {matricola} -> {canonical_base}")

    print("\n[Whitelist] Faculty whitelist summary:")
    print(f"[Whitelist]   Matricole found on personnel page : {len(all_matricole)}")
    print(f"[Whitelist]   Confirmed faculty profiles        : {confirmed_faculty}")
    print(f"[Whitelist]   Skipped (no docenti.unisa.it link): {not_found}")
    print(f"[Whitelist]   Total URLs to crawl               : {len(authorized_urls)}")

    return authorized_urls


def build_courses_whitelist() -> List[str]:
    """
    Constructs the list of authorized course URLs on corsi.unisa.it.

    Fetches the DIEM course offerings page and extracts all links pointing
    to corsi.unisa.it, which are the entry points for each degree programme.
    """
    print(f"[Whitelist] Building courses whitelist from: {config.DIEM_COURSES_SEED_URL}")

    try:
        response = SESSION.get(config.DIEM_COURSES_SEED_URL, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Whitelist] ERROR: Could not retrieve courses seed page. {exc}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    authorized_urls: Set[str] = set()

    for anchor in soup.find_all("a", href=True):
        link = anchor["href"]
        if "corsi.unisa.it" in link:
            authorized_urls.add(_normalize_url(link))

    print(f"\n[Whitelist] Courses whitelist summary:")
    print(f"[Whitelist]   Total course URLs found     : {len(authorized_urls)}")
    for i, url in enumerate(sorted(authorized_urls), 1):
        print(f"[Whitelist]   {i:>3}. {url}")

    return list(authorized_urls)


# ---------------------------------------------------------------------------
# BFS Crawler
# ---------------------------------------------------------------------------

def _crawl_bfs(
    seeds: List[str],
    max_depth: int,
    allowed_domain: str,
    seed_prefixes: Set[str] = None,
) -> Tuple[List[Dict], Set[str]]:
    """
    Performs a Breadth-First Search crawl starting from the given seed URLs,
    restricted to the specified domain and bounded by max_depth.

    Each harvested document includes:
        - 'text':         Markdown-formatted content extracted by Trafilatura.
        - 'content_hash': SHA-256 of normalised text for incremental updates.
        - 'source':       Canonical URL of the page.
        - 'doc_id':       SHA-256 of the URL (stable identifier for the page).
        - 'doc_type':     Always 'html' for pages crawled by this function.

    The allowed_domain parameter is checked directly for each link found
    during BFS — only links whose netloc matches this domain are followed.
    This means EasyCourse can be crawled safely by passing
    allowed_domain='easycourse.unisa.it' without adding that domain to
    config.ALLOWED_DOMAINS, which would risk contaminating other crawl phases.

    Args:
        seeds:          Starting URLs for the BFS.
        max_depth:      Maximum number of hops from any seed URL.
        allowed_domain: Only URLs on this domain are followed.
        seed_prefixes:  Optional set of URL prefixes. When provided, only
                        links that start with one of these prefixes are
                        enqueued. Used to restrict docenti and corsi crawls
                        to the whitelist-derived URL space.

    Returns:
        documents: list of document dicts.
        pdf_urls:  set of PDF URLs discovered during the crawl.
    """
    visited: Set[str] = set()
    queue: deque = deque()
    documents: List[Dict] = []
    pdf_urls: Set[str] = set()

    for seed in seeds:
        normalized = _normalize_url(seed)
        if normalized not in visited:
            queue.append((normalized, 0))
            visited.add(normalized)

    total_visited = 0

    while queue:
        url, depth = queue.popleft()
        total_visited += 1

        print(f"[Crawler] [{total_visited}] (depth={depth}) Fetching: {url}")

        try:
            time.sleep(config.CRAWL_DELAY)
            response = SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[Crawler]   -> SKIP (request failed): {exc}")
            continue

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            print(
                f"[Crawler]   -> SKIP (non-HTML content-type: "
                f"{content_type.split(';')[0].strip()})"
            )
            continue

        canonical_url = _normalize_url(response.url)
        if not _is_allowed(canonical_url) and canonical_url.split("/")[2] not in (
            allowed_domain,
        ):
            print(f"[Crawler]   -> SKIP (redirected outside allowed domains: {canonical_url}).")
            continue
        if canonical_url != url and canonical_url in visited:
            print(f"[Crawler]   -> SKIP (redirect target already visited: {canonical_url}).")
            continue
        if canonical_url != url:
            visited.add(canonical_url)

        soup = BeautifulSoup(response.text, "html.parser")

        page_links, found_pdfs = _extract_links(soup, url)
        # For EasyCourse phase, also collect links whose domain matches
        # allowed_domain but which are not in config.ALLOWED_DOMAINS.
        if allowed_domain not in config.ALLOWED_DOMAINS:
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"].strip()
                if not href:
                    continue
                absolute = urllib.parse.urljoin(url, href)
                absolute = _normalize_url(absolute)
                parsed = urllib.parse.urlparse(absolute)
                if parsed.netloc == allowed_domain and not absolute.lower().endswith(".pdf"):
                    page_links.add(absolute)

        pdf_urls.update(found_pdfs)

        if found_pdfs:
            print(f"[Crawler]   -> {len(found_pdfs)} PDF link(s) harvested.")

        newly_enqueued = 0
        if depth < max_depth:
            for link in page_links:
                parsed = urllib.parse.urlparse(link)

                if parsed.netloc != allowed_domain:
                    continue

                if link in visited:
                    continue

                if seed_prefixes and not any(
                    link.startswith(prefix) for prefix in seed_prefixes
                ):
                    continue

                visited.add(link)
                queue.append((link, depth + 1))
                newly_enqueued += 1

        if depth < max_depth:
            print(
                f"[Crawler]   -> {newly_enqueued} new URL(s) enqueued "
                f"(queue size: {len(queue)})."
            )

        text = _extract_content(response.text, canonical_url)

        if len(text) < config.MIN_PAGE_LENGTH:
            print(f"[Crawler]   -> SKIP (page text too short: {len(text)} chars).")
            continue

        doc_id = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
        documents.append({
            "doc_id": doc_id,
            "source": canonical_url,
            "text": text,
            "content_hash": _content_hash(text),
            "doc_type": "html",
        })
        print(f"[Crawler]   -> OK ({len(text)} chars extracted, {len(documents)} docs so far).")

    return documents, pdf_urls


# ---------------------------------------------------------------------------
# PDF Downloader
# ---------------------------------------------------------------------------

def _clean_pdf_text(text: str) -> str:
    """
    Cleans text extracted from a PDF by normalising Unicode encoding artefacts.

    Applies NFKD normalisation to decompose ligatures and accented characters,
    removes non-printable control characters, and collapses excessive whitespace.

    Args:
        text: Raw text string extracted from a PDF.

    Returns:
        Cleaned text string suitable for chunking and embedding.
    """
    normalized = unicodedata.normalize("NFKD", text)
    # Remove pymupdf4llm image placeholders.
    normalized = re.sub(
        r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*\s*",
        "",
        normalized,
    )

    cleaned = "".join(
        ch for ch in normalized
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in ("\n", "\t")
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _download_pdfs(pdf_urls: Set[str]) -> List[Dict]:
    """
    Downloads PDF files discovered during the crawl and extracts their
    textual content as Markdown using pymupdf4llm.

    Each PDF is treated as a single document. Pymupdf4llm preserves the
    document structure — headings, tables, lists — as Markdown, which
    produces better chunks and richer context for the LLM.

    Each document includes a content_hash for incremental update detection.

    Args:
        pdf_urls: Set of PDF URLs discovered during the HTML crawl.

    Returns:
        List of document dicts, one per PDF file.
    """
    if not pdf_urls:
        print("[PDF] No PDF documents discovered during the crawl.")
        return []

    print(f"[PDF] Downloading {len(pdf_urls)} PDF documents...")
    documents: List[Dict] = []

    try:
        import pymupdf4llm
    except ImportError:
        print("[PDF] ERROR: pymupdf4llm is not installed. Run: pip install pymupdf4llm")
        return []

    for i, pdf_url in enumerate(sorted(pdf_urls), 1):
        try:
            print(f"[PDF] [{i}/{len(pdf_urls)}] {pdf_url}")
            response = SESSION.get(pdf_url, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()

            filename = pdf_url.split("/")[-1].split("?")[0]
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"
            filepath = os.path.join(config.PDF_DOWNLOAD_DIR, f"{i:04d}_{filename}")

            with open(filepath, "wb") as fh:
                fh.write(response.content)

            # Extract full PDF as a single Markdown document.
            md_text = pymupdf4llm.to_markdown(filepath)
            md_text = _clean_pdf_text(md_text)

            if len(md_text) < config.MIN_PAGE_LENGTH:
                print(f"[PDF]   -> SKIP (text too short after extraction: {len(md_text)} chars).")
                continue

            doc_id = hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()
            documents.append({
                "doc_id": doc_id,
                "source": pdf_url,
                "text": md_text,
                "content_hash": _content_hash(md_text),
                "doc_type": "pdf",
            })
            print(f"[PDF]   -> OK ({len(md_text)} chars extracted).")

        except Exception as exc:
            print(f"[PDF] FAILED {pdf_url} — {exc}")
            continue

    print(f"[PDF] Extracted {len(documents)} PDF document(s).")
    return documents


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

def _write_audit_log(documents: List[Dict]) -> None:
    """
    Writes the list of successfully crawled URLs to the audit log file.

    The log contains the total document count, unique source count, and
    a sorted list of all source URLs. Used for debugging and verifying
    crawl coverage between runs.
    """
    urls = sorted({doc["source"] for doc in documents})
    with open(config.CRAWL_AUDIT_PATH, "w", encoding="utf-8") as fh:
        fh.write("=== DIEM Agentic RAG: Crawl Audit Log ===\n")
        fh.write(f"Total documents: {len(documents)}\n")
        fh.write(f"Unique sources:  {len(urls)}\n\n")
        for url in urls:
            fh.write(f"{url}\n")
    print(f"[Audit] Log written to {config.CRAWL_AUDIT_PATH} ({len(urls)} unique sources).")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Executes the full crawling pipeline across four phases:

        Phase 1 — www.diem.unisa.it
            BFS from the DIEM homepage with depth=4.

        Phase 2 — docenti.unisa.it
            Builds a faculty whitelist from the DIEM personnel page,
            then BFS-crawls each confirmed faculty profile with depth=1.

        Phase 3 — corsi.unisa.it
            Builds a course whitelist from the DIEM course offerings page,
            then BFS-crawls each course entry point with depth=3.

        Phase 4 — easycourse.unisa.it (lesson timetables)
            BFS from the semester index page with depth=2.
            Covers both semesters and all degree programmes automatically.
            The seed URL must be updated in config.py each academic year.
            Note: easycourse.unisa.it is NOT in config.ALLOWED_DOMAINS —
            Phase 4 uses its own allowed_domain constraint to avoid
            contaminating other crawl phases.

        Phase 5 — PDF harvesting
            Downloads and extracts all PDF files discovered across all phases.

    All documents are deduplicated by doc_id (SHA-256 of URL) before
    serialization to RAW_DOCUMENTS_PATH.
    """
    print("=== DIEM Agentic RAG: Crawler Started ===\n")

    all_documents: List[Dict] = []
    all_pdf_urls: Set[str] = set()

    # ------------------------------------------------------------------
    # Phase 1 — DIEM main domain
    # ------------------------------------------------------------------
    print("[Phase 1] Crawling www.diem.unisa.it ...")
    target = config.CRAWL_TARGETS["diem"]
    diem_docs, diem_pdfs = _crawl_bfs(
        seeds=[target["seed"]],
        max_depth=target["depth"],
        allowed_domain=target["allowed_domain"],
    )
    all_documents.extend(diem_docs)
    all_pdf_urls.update(diem_pdfs)
    print(f"[Phase 1] Complete — {len(diem_docs)} pages, {len(diem_pdfs)} PDFs found.\n")

    # ------------------------------------------------------------------
    # Phase 2 — Faculty profiles
    # ------------------------------------------------------------------
    print("[Phase 2] Crawling docenti.unisa.it ...")
    faculty_seeds = build_faculty_whitelist()
    target = config.CRAWL_TARGETS["docenti"]
    faculty_prefixes = {_normalize_url(u) for u in faculty_seeds}

    faculty_docs, faculty_pdfs = _crawl_bfs(
        seeds=faculty_seeds,
        max_depth=target["depth"],
        allowed_domain=target["allowed_domain"],
        seed_prefixes=faculty_prefixes,
    )
    all_documents.extend(faculty_docs)
    all_pdf_urls.update(faculty_pdfs)
    print(f"[Phase 2] Complete — {len(faculty_docs)} pages, {len(faculty_pdfs)} PDFs found.\n")

    # ------------------------------------------------------------------
    # Phase 3 — Course pages
    # ------------------------------------------------------------------
    print("[Phase 3] Crawling corsi.unisa.it ...")
    courses_seeds = build_courses_whitelist()
    target = config.CRAWL_TARGETS["corsi"]
    courses_prefixes = {_normalize_url(u) for u in courses_seeds}

    courses_docs, courses_pdfs = _crawl_bfs(
        seeds=courses_seeds,
        max_depth=target["depth"],
        allowed_domain=target["allowed_domain"],
        seed_prefixes=courses_prefixes,
    )
    all_documents.extend(courses_docs)
    all_pdf_urls.update(courses_pdfs)
    print(f"[Phase 3] Complete — {len(courses_docs)} pages, {len(courses_pdfs)} PDFs found.\n")

    # ------------------------------------------------------------------
    # Phase 4 — EasyCourse lesson timetables
    # ------------------------------------------------------------------
    print("[Phase 4] Crawling easycourse.unisa.it ...")
    target = config.CRAWL_TARGETS["easycourse"]
    easycourse_docs, easycourse_pdfs = _crawl_bfs(
        seeds=[target["seed"]],
        max_depth=target["depth"],
        allowed_domain=target["allowed_domain"],
    )
    all_documents.extend(easycourse_docs)
    all_pdf_urls.update(easycourse_pdfs)
    print(f"[Phase 4] Complete — {len(easycourse_docs)} pages found.\n")

    # ------------------------------------------------------------------
    # Phase 5 — PDF harvesting
    # ------------------------------------------------------------------
    print(f"[Phase 5] Processing {len(all_pdf_urls)} unique PDF URLs...")
    pdf_docs = _download_pdfs(all_pdf_urls)
    all_documents.extend(pdf_docs)

    # ------------------------------------------------------------------
    # Deduplication by doc_id
    # ------------------------------------------------------------------
    seen_ids: Set[str] = set()
    unique_documents: List[Dict] = []
    for doc in all_documents:
        if doc["doc_id"] not in seen_ids:
            seen_ids.add(doc["doc_id"])
            unique_documents.append(doc)

    print(f"\n[Crawler] Total unique documents collected: {len(unique_documents)}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    with open(config.RAW_DOCUMENTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(unique_documents, fh, ensure_ascii=False, indent=2)
    print(f"[Crawler] Raw documents saved to {config.RAW_DOCUMENTS_PATH}")

    _write_audit_log(unique_documents)

    print("\n=== Crawler Completed ===")


if __name__ == "__main__":
    run()