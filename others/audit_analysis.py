"""
Audit analysis script for the DIEM Agentic RAG project.

Reads the crawl audit log and prints a breakdown of crawled URLs
by domain, including faculty member count and course count.

Usage:
    python audit_analysis.py
"""

import re
from collections import defaultdict

import config


def run() -> None:
    with open(config.CRAWL_AUDIT_PATH, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh.readlines() if l.strip().startswith("http")]

    by_domain = defaultdict(list)
    for url in lines:
        if "docenti.unisa.it" in url:
            by_domain["docenti.unisa.it"].append(url)
        elif "corsi.unisa.it" in url:
            by_domain["corsi.unisa.it"].append(url)
        elif "diem.unisa.it" in url:
            by_domain["diem.unisa.it"].append(url)
        elif "easycourse.unisa.it" in url:
            by_domain["easycourse.unisa.it"].append(url)
        else:
            by_domain["other"].append(url)

    # Count unique faculty members (each has multiple sub-pages).
    # The pattern captures both numeric matricola URLs (docenti.unisa.it/004367/)
    # and name-based slug URLs (docenti.unisa.it/antonio.parziale/).
    faculty_pattern = re.compile(r"docenti\.unisa\.it/([^/]+)/")
    unique_faculty = {
        faculty_pattern.search(u).group(1)
        for u in by_domain["docenti.unisa.it"]
        if faculty_pattern.search(u)
        and faculty_pattern.search(u).group(1) != "uploads"
    }

    # Count unique courses (each has multiple sub-pages).
    # Course URLs follow the pattern corsi.unisa.it/{course-code}/...
    # Technical paths such as /uploads/ are excluded from the count.
    course_pattern = re.compile(r"corsi\.unisa\.it/([^/]+)")
    unique_courses = {
        course_pattern.search(u).group(1)
        for u in by_domain["corsi.unisa.it"]
        if course_pattern.search(u)
        and course_pattern.search(u).group(1) != "uploads"
    }

    print("=== Crawl Audit Analysis ===\n")
    print(f"Total URLs crawled: {len(lines)}\n")

    print(f"[diem.unisa.it]")
    print(f"  Pages crawled : {len(by_domain['diem.unisa.it'])}\n")

    print(f"[docenti.unisa.it]")
    print(f"  Pages crawled : {len(by_domain['docenti.unisa.it'])}")
    print(f"  Unique faculty: {len(unique_faculty)}")
    for m in sorted(unique_faculty):
        print(f"    - {m}")
    print()

    print(f"[corsi.unisa.it]")
    print(f"  Pages crawled : {len(by_domain['corsi.unisa.it'])}")
    print(f"  Unique courses: {len(unique_courses)}")
    for c in sorted(unique_courses):
        print(f"    - {c}")
    print()

    if by_domain["other"]:
        print(f"[other domains] {len(by_domain['other'])} URL(s) — investigate these:")
        for u in by_domain["other"]:
            print(f"  - {u}")


if __name__ == "__main__":
    run()