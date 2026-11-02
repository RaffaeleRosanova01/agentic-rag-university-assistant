"""
Metadata extractor module for the DIEM Agentic RAG project.

Derives structured metadata from document source URLs deterministically,
without any LLM calls. The extracted metadata is stored alongside each
parent and child chunk to support domain-aware retrieval and filtering.

Supported domains:
    - www.diem.unisa.it    (main department site)
    - docenti.unisa.it     (faculty profiles)
    - corsi.unisa.it       (course pages and PDF attachments)
    - easycourse.unisa.it  (lesson timetables — static HTML)

Usage:
    from ingestion.metadata_extractor import extract_metadata

    meta = extract_metadata("https://docenti.unisa.it/004322/curriculum")
    # {
    #     "domain":       "docenti",
    #     "section_type": "curriculum",
    # }
"""

import re
from typing import Dict
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Section label maps (used only for section_type derivation)
# ---------------------------------------------------------------------------

_DOCENTI_SECTION_TYPES: Dict[str, str] = {
    "home":                             "home",
    "curriculum":                       "curriculum",
    "didattica":                        "didattica",
    "ricerca/progetti":                 "ricerca_progetti",
    "ricerca/pubblicazioni":            "ricerca_pubblicazioni",
    "ricerca/laboratori":               "ricerca_laboratori",
    "ricerca/premi-ricerca":            "ricerca_premi",
    "ricerca/brevetti":                 "ricerca_brevetti",
    "ricerca/risorse":                  "ricerca_risorse",
    "risorse":                          "risorse",
    "international/erasmus":            "international_erasmus",
    "international/traineeship":        "international_traineeship",
    "international/dottorato-con-tesi-in-cotutela": "international_dottorato",
    "international/cooperazione-internazionale":    "international_cooperazione",
    "international/eramus-teaching-docenza":        "international_teaching",
}

_DIEM_SECTION_TYPES: Dict[str, str] = {
    "dipartimento/presentazione":                               "dipartimento_presentazione",
    "dipartimento/strutture":                                   "dipartimento_strutture",
    "dipartimento/commissioni":                                 "dipartimento_commissioni",
    "dipartimento/commissione-paritetica":                      "dipartimento_commissioni",
    "dipartimento":                                             "dipartimento",
    "ricerca/aree-di-ricerca":                                  "ricerca_aree",
    "ricerca/focus":                                            "ricerca_focus",
    "ricerca/progetti-finanziati":                              "ricerca_progetti",
    "ricerca":                                                  "ricerca",
    "didattica/offerta-formativa":                              "didattica_offerta",
    "didattica/focus":                                          "didattica_focus",
    "didattica/alternanza-scuola-lavoro/progetti":              "didattica_alternanza",
    "didattica":                                                "didattica",
    "international/accordi-erasmus-plus/traineeship":           "international_traineeship",
    "international/accordi-erasmus-plus":                       "international_erasmus",
    "international":                                            "international",
    "terza-missione/trasferimento-tecnologico/spin-off":        "terzamissione_spinoff",
    "terza-missione/trasferimento-tecnologico/proprieta-intellettuale": "terzamissione_brevetti",
    "terza-missione":                                           "terzamissione",
    "home/contatti":                                            "contatti",
    "home/dati-di-monitoraggio":                                "monitoraggio",
    "home/bandi":                                               "bandi",
}

_CORSI_SECTION_TYPES: Dict[str, str] = {
    "didattica/piano-di-studi":                         "piano_studi",
    "didattica/esame-finale":                           "esame_finale",
    "didattica/regolamenti":                            "regolamenti",
    "didattica/lingua-straniera":                       "lingua_straniera",
    "didattica/strumenti-online-per-la-didattica":      "strumenti_online",
    "didattica":                                        "didattica",
    "strutture-didattiche/calendario-occupazione-spazi": "calendario_aule",
    "strutture-didattiche":                             "strutture",
    "attivita-e-servizi/tutorato":                      "tutorato",
    "attivita-e-servizi/servizi":                       "servizi",
    "attivita-e-servizi":                               "servizi",
    "attori-e-documenti-di-aq":                         "documenti_aq",
    "consultazioni-parti-interessate":                  "consultazioni",
    "comunicazioni-docenti":                            "comunicazioni",
    "internazionalizzazione":                           "internazionalizzazione",
    "immatricolazioni":                                 "immatricolazioni",
    "contatti":                                         "contatti",
    "news":                                             "news",
    "focus":                                            "focus",
}


# Every section_type value that extract_metadata() can assign, grouped by
# domain (including the fallback values used when no URL pattern matches).
# agent/tools.py uses it to validate the section filters chosen by the LLM,
# so the tool filters cannot drift from the metadata stored in ChromaDB.
SECTION_TYPES_BY_DOMAIN: Dict[str, frozenset] = {
    "docenti":    frozenset(_DOCENTI_SECTION_TYPES.values()) | {"unknown"},
    "diem":       frozenset(_DIEM_SECTION_TYPES.values()) | {"generale", "pdf_allegato"},
    "corsi":      frozenset(_CORSI_SECTION_TYPES.values()) | {"presentazione", "pdf_allegato"},
    "easycourse": frozenset({"orario_docente", "orario_lezioni", "orario_indice"}),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_metadata(source_url: str) -> Dict[str, str]:
    """
    Derives structured metadata from a document source URL.

    Extracts domain and section_type deterministically from the URL
    without any LLM calls or external lookups.

    Args:
        source_url: The canonical source URL of the document.

    Returns:
        A dict with keys:
            domain:       One of 'diem', 'docenti', 'corsi', 'easycourse'.
            section_type: A machine-readable section identifier.

    If the URL does not match any known pattern, returns domain='unknown'.
    """
    parsed = urlparse(source_url)
    netloc = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if netloc == "docenti.unisa.it":
        return _extract_docenti(path)

    if netloc == "www.diem.unisa.it":
        return _extract_diem(path, source_url)

    if netloc == "corsi.unisa.it":
        return _extract_corsi(path, source_url)

    if netloc == "easycourse.unisa.it":
        return _extract_easycourse(path)

    return {
        "domain":       "unknown",
        "section_type": "unknown",
    }


# ---------------------------------------------------------------------------
# Domain-specific extractors
# ---------------------------------------------------------------------------

def _extract_docenti(path: str) -> Dict[str, str]:
    """
    Extracts metadata from docenti.unisa.it URLs.

    Pattern: /{faculty_id}[/{sub_page}]
    """
    parts = path.split("/")
    faculty_id = parts[0] if parts and parts[0].isdigit() else None

    if faculty_id is None:
        return {
            "domain":       "docenti",
            "section_type": "unknown",
        }

    sub_path = "/".join(parts[1:]).strip("/") if len(parts) > 1 else ""
    if sub_path == "home":
        sub_path = ""

    section_type = "home"
    for key in sorted(_DOCENTI_SECTION_TYPES, key=len, reverse=True):
        if sub_path == key or sub_path.startswith(key + "/"):
            section_type = _DOCENTI_SECTION_TYPES[key]
            break

    return {
        "domain":       "docenti",
        "section_type": section_type,
    }


def _extract_diem(path: str, source_url: str) -> Dict[str, str]:
    """
    Extracts metadata from www.diem.unisa.it URLs.
    """
    if source_url.lower().endswith(".pdf"):
        return {
            "domain":       "diem",
            "section_type": "pdf_allegato",
        }

    section_type = "generale"
    for key in sorted(_DIEM_SECTION_TYPES, key=len, reverse=True):
        if path == key or path.startswith(key + "/") or path.startswith(key + "?"):
            section_type = _DIEM_SECTION_TYPES[key]
            break

    return {
        "domain":       "diem",
        "section_type": section_type,
    }


def _extract_corsi(path: str, source_url: str) -> Dict[str, str]:
    """
    Extracts metadata from corsi.unisa.it URLs.
    """
    if path.startswith("uploads/"):
        return {
            "domain":       "corsi",
            "section_type": "pdf_allegato",
        }

    parts = path.split("/")
    sub_path = "/".join(parts[1:]).strip("/") if len(parts) > 1 else ""

    section_type = "presentazione"
    for key in sorted(_CORSI_SECTION_TYPES, key=len, reverse=True):
        if sub_path == key or sub_path.startswith(key + "/") or sub_path.startswith(key + "?"):
            section_type = _CORSI_SECTION_TYPES[key]
            break

    return {
        "domain":       "corsi",
        "section_type": section_type,
    }


def _extract_easycourse(path: str) -> Dict[str, str]:
    """
    Extracts metadata from easycourse.unisa.it URLs.
    """
    if "/Docenti/" in path or "/docenti/" in path:
        return {
            "domain":       "easycourse",
            "section_type": "orario_docente",
        }

    if "/Curricula/" in path or "/curricula/" in path:
        return {
            "domain":       "easycourse",
            "section_type": "orario_lezioni",
        }

    return {
        "domain":       "easycourse",
        "section_type": "orario_indice",
    }