"""
Tools module for the DIEM Agentic RAG project.

Defines the LangChain tools exposed to the ReAct agent.

Tools:
    search_faculty        — retrieval filtered to faculty profiles (docenti).
    search_courses        — retrieval filtered to course/degree documents.
    search_department     — retrieval filtered to department pages (diem).
    compute               — deterministic mathematical calculations.
    out_of_scope_response — standard fallback for unanswerable questions.

Domain filters applied at ChromaDB level:
    search_faculty    → domain = "docenti", optional section_type filter
    search_courses    → domain IN ["corsi", "easycourse"], optional section_type filter
    search_department → domain = "diem", optional section_type filter

Each tool targets a specific subset of the knowledge base, reducing noise
from unrelated documents and improving retrieval precision.
For queries spanning multiple domains, the agent calls the relevant tools
sequentially.

Section filters chosen by the LLM are validated against the section types
that ingestion/metadata_extractor.py can actually assign. Unknown values are
ignored, and a filtered search that returns no documents is automatically
repeated with the domain filter only.
"""

import logging
from typing import Optional

from langchain_core.tools import tool

from ingestion.metadata_extractor import SECTION_TYPES_BY_DOMAIN
from retrieval.retriever import initialize as initialize_retriever
from retrieval.retriever import retrieve

logger = logging.getLogger("DIEM_Tools")

initialize_retriever()

# ---------------------------------------------------------------------------
# Metadata filters
# ---------------------------------------------------------------------------

_FILTER_FACULTY = {"domain": {"$eq": "docenti"}}

_FILTER_COURSES = {"domain": {"$in": ["corsi", "easycourse"]}}

_FILTER_DEPARTMENT = {"domain": {"$eq": "diem"}}

# Section types that can appear within each tool's domain filter.
_SECTIONS_FACULTY = SECTION_TYPES_BY_DOMAIN["docenti"]

_SECTIONS_COURSES = SECTION_TYPES_BY_DOMAIN["corsi"] | SECTION_TYPES_BY_DOMAIN["easycourse"]

_SECTIONS_DEPARTMENT = SECTION_TYPES_BY_DOMAIN["diem"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_filter(domain_filter: dict, section_type: Optional[str]) -> dict:
    """
    Combines a domain filter with an optional section_type filter.

    Args:
        domain_filter: Base ChromaDB domain filter dict.
        section_type:  Optional section_type value to add to the filter.

    Returns:
        Combined filter dict, or domain_filter if section_type is None.
    """
    if not section_type:
        return domain_filter
    return {
        "$and": [
            domain_filter,
            {"section_type": {"$eq": section_type}},
        ]
    }


def _search(
    tool_name: str,
    query: str,
    domain_filter: dict,
    section_filter: Optional[str],
    allowed_sections: frozenset,
) -> list:
    """
    Runs a filtered retrieval shared by all search tools.

    The section filter is applied only if it is a section type that exists
    within the tool's domain; otherwise it is ignored. If the filtered search
    returns no documents, the search is repeated with the domain filter only,
    so that an overly narrow section choice does not leave the agent without
    context.

    Args:
        tool_name:        Name of the calling tool, used for logging.
        query:            The search query formulated by the agent.
        domain_filter:    Base ChromaDB domain filter for the tool.
        section_filter:   Optional section_type chosen by the agent.
        allowed_sections: Section types available within the tool's domain.

    Returns:
        A list of reranked LangChain Documents (possibly empty).
    """
    if section_filter and section_filter not in allowed_sections:
        logger.warning(
            f"[{tool_name}] Unknown section_filter '{section_filter}'. "
            "Searching the whole domain instead."
        )
        section_filter = None

    docs = retrieve(query, filter=_build_filter(domain_filter, section_filter))

    if not docs and section_filter:
        logger.info(
            f"[{tool_name}] No documents with section_filter '{section_filter}'. "
            "Retrying with the domain filter only."
        )
        docs = retrieve(query, filter=domain_filter)

    return docs


def _format_docs(docs) -> str:
    if not docs:
        return (
            "No relevant documents found in the DIEM knowledge base. "
            "The requested information may not be available in the "
            "indexed sources."
        )
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Source not available")
        text = doc.page_content.strip()
        parts.append(f"[Source: {source}]\n{text}\n[End]")

    header = (
        f"Found {len(docs)} relevant documents. "
        "Use ONLY the information present in these documents to answer. "
        "When citing sources, always use the full URL from the 'Source:' field.\n\n"
    )
    return header + "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Search tools
# ---------------------------------------------------------------------------

@tool
def search_faculty(query: str, section_filter: Optional[str] = None) -> str:
    """
    Searches faculty profiles on docenti.unisa.it for information about
    DIEM professors and researchers.

    Use this tool for questions about:
    - Who a professor is, their role, background, or bio
    - Exam dates
    - Office hours and reception times (orari di ricevimento)
    - Courses taught by a specific professor
    - Research projects, laboratories, or publications of a professor
    - Contact information or resources for faculty members
    - International activities (Erasmus, co-tutela, traineeship) of a professor

    section_filter (optional): Restricts retrieval to a specific section
        type. Use this to avoid noise from unrelated sections of the same
        professor's profile. Allowed values and when to use them:
            "home"                       — identity, role, office hours
            "curriculum"                 — bio, academic background
            "didattica"                  — courses taught, exam date
            "ricerca_progetti"           — research projects
            "ricerca_pubblicazioni"      — publications
            "ricerca_laboratori"         — laboratories
            "ricerca_premi"              — awards and prizes
            "ricerca_brevetti"           — patents
            "ricerca_risorse"            — research resources
            "risorse"                    — teaching materials
            "international_erasmus"      — Erasmus activities
            "international_dottorato"    — co-tutela PhD agreements
            "international_traineeship"  — international traineeship
            "international_cooperazione" — international cooperation
            "international_teaching"     — Erasmus teaching mobility
        Leave as None to search across all faculty sections. If the
        filtered search finds nothing, the whole faculty domain is searched.

    Usage examples:
        search_faculty("Chi è Antonio Greco", section_filter="curriculum")
        search_faculty("orari ricevimento Capuano", section_filter="home")
        search_faculty("corsi insegnati Antonio Greco", section_filter="didattica")
        search_faculty("laboratori Foggia", section_filter="ricerca_laboratori")

    Args:
        query:         A precise natural language query about a faculty member.
        section_filter: Optional section_type value as described above.

    Returns:
        Formatted document excerpts from faculty profile pages.
    """
    logger.info(f"[search_faculty] Query: '{query}' | section_filter: {section_filter}")
    docs = _search("search_faculty", query, _FILTER_FACULTY, section_filter, _SECTIONS_FACULTY)
    result = _format_docs(docs)
    logger.info(f"[search_faculty] Returning {len(docs)} document(s).")
    return result


@tool
def search_courses(query: str, section_filter: Optional[str] = None) -> str:
    """
    Searches course and degree programme documents for information about
    DIEM academic offerings, regulations, and timetables.

    Use this tool for questions about:
    - Degree programmes offered by DIEM and their structure
    - Course syllabi, learning objectives, and exam modalities
    - Study plans (piani di studio) and CFU credits
    - Sessions, and timetables
    - Admission requirements and grade conversion formulas
    - Degree regulations (regolamenti didattici)
    - Internationalisation, Erasmus, or language requirements for a degree
    - Student services, tutoring, or online tools for a degree

    For grade calculations (e.g. final degree score), ALWAYS use this tool
    first to retrieve the specific formula for the student's degree programme
    before calling compute.

    section_filter (optional): Restricts retrieval to a specific section
        type. Allowed values and when to use them:
            "presentazione"          — degree overview and presentation
            "didattica"              — teaching overview of the degree
            "pdf_allegato"           — regulations, syllabi, grade formulas
            "piano_studi"            — study plans
            "esame_finale"           — final exam rules and degree grade
            "regolamenti"            — degree regulations
            "immatricolazioni"       — admission requirements
            "lingua_straniera"       — foreign language requirements
            "internazionalizzazione" — Erasmus and international for a degree
            "tutorato"               — tutoring services
            "servizi"                — student services
            "strumenti_online"       — online teaching tools
            "strutture"              — teaching facilities
            "calendario_aule"        — classroom calendars
            "contatti"               — degree programme contacts
            "documenti_aq"           — quality assurance documents
            "consultazioni"          — stakeholder consultations
            "focus"                  — news and focus articles
            "orario_lezioni"         — lesson timetables by curriculum (EasyCourse)
            "orario_indice"          — timetable index pages (EasyCourse)
            "orario_docente"         — professor timetables (EasyCourse)
        Leave as None to search across all course sections. If the
        filtered search finds nothing, all course documents are searched.

    Usage examples:
        search_courses("voto laurea Ingegneria Informatica formula",
                       section_filter="pdf_allegato")
        search_courses("piano studi secondo anno cybersecurity",
                       section_filter="piano_studi")
        search_courses("requisiti ammissione Information Engineering",
                       section_filter="immatricolazioni")
        search_courses("orario lezioni NLP magistrale",
                       section_filter="orario_lezioni")

    Args:
        query:         A precise natural language query about courses or regulations.
        section_filter: Optional section_type value as described above.

    Returns:
        Formatted document excerpts from course and regulation documents.
    """
    logger.info(f"[search_courses] Query: '{query}' | section_filter: {section_filter}")
    docs = _search("search_courses", query, _FILTER_COURSES, section_filter, _SECTIONS_COURSES)
    result = _format_docs(docs)
    logger.info(f"[search_courses] Returning {len(docs)} document(s).")
    return result


@tool
def search_department(query: str, section_filter: Optional[str] = None) -> str:
    """
    Searches department pages on www.diem.unisa.it for institutional
    information about the DIEM department.

    Use this tool for questions about:
    - Department overview, mission, and structure
    - Committees, delegates, and administrative contacts
    - Laboratories, facilities, and equipment
    - Research areas, funded projects, and focus
    - Erasmus and international mobility programmes
    - Spin-offs, patents, and technology transfer
    - Official notices (bandi) and department events
    - Department location and directions

    section_filter (optional): Restricts retrieval to a specific section
        type. Allowed values and when to use them:
            "dipartimento"               — general department pages
            "dipartimento_presentazione" — department overview
            "dipartimento_commissioni"   — committees and their members
            "dipartimento_strutture"     — laboratories and facilities
            "ricerca"                    — research overview
            "ricerca_aree"               — research areas
            "ricerca_focus"              — research focus articles
            "ricerca_progetti"           — funded research projects
            "didattica"                  — teaching overview
            "didattica_offerta"          — degree programme list
            "didattica_focus"            — teaching focus articles
            "didattica_alternanza"       — school-work alternance
            "bandi"                      — official notices and calls
            "contatti"                   — department contacts
            "monitoraggio"               — monitoring data
            "international"              — international overview
            "international_erasmus"      — Erasmus mobility
            "international_traineeship"  — international traineeship
            "terzamissione"              — third mission overview
            "terzamissione_spinoff"      — spin-off companies
            "terzamissione_brevetti"     — patents and IP
            "pdf_allegato"               — department PDF attachments
            "generale"                   — other department pages
        Leave as None to search across all department sections. If the
        filtered search finds nothing, the whole department domain is searched.

    Usage examples:
        search_department("commissione paritetica DIEM",
                          section_filter="dipartimento_commissioni")
        search_department("laboratori ricerca DIEM",
                          section_filter="dipartimento_strutture")
        search_department("bandi DIEM 2025",
                          section_filter="bandi")
        search_department("Erasmus DIEM mobilità",
                          section_filter="international_erasmus")

    Args:
        query:         A precise natural language query about the department.
        section_filter: Optional section_type value as described above.

    Returns:
        Formatted document excerpts from department pages.
    """
    logger.info(f"[search_department] Query: '{query}' | section_filter: {section_filter}")
    docs = _search("search_department", query, _FILTER_DEPARTMENT, section_filter, _SECTIONS_DEPARTMENT)
    result = _format_docs(docs)
    logger.info(f"[search_department] Returning {len(docs)} document(s).")
    return result


# ---------------------------------------------------------------------------
# Compute tool
# ---------------------------------------------------------------------------

@tool
def compute(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the numeric result.

    Use this tool after retrieving the relevant formula from the knowledge
    base with search_courses. Substitute the user's values into the formula
    exactly as written in the retrieved document, then pass the resulting
    expression here.

    The result returned is the final numeric value — do NOT divide,
    multiply, or transform it further after receiving it.

    Supported operations: +, -, *, /, **, //, %, (, )
    Supported functions: round(), abs(), min(), max()

    Examples:
        compute("4.1 * 28.8 - 8.8")
        compute("(28 * 3 + 30 * 6 + 27 * 9) / (3 + 6 + 9)")
        compute("round(105.6, 0)")

    Args:
        expression: A mathematical expression as a string.

    Returns:
        The numeric result as a string, or an error message if the
        expression is invalid or unsafe.
    """
    logger.info(f"[compute] Expression: '{expression}'")

    import ast
    import operator

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    _SAFE_FUNCTIONS = {
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            op = _OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _safe_eval(node.operand)
            op = _OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed.")
            func_name = node.func.id
            if func_name not in _SAFE_FUNCTIONS:
                raise ValueError(f"Function '{func_name}' is not allowed.")
            args = [_safe_eval(arg) for arg in node.args]
            return _SAFE_FUNCTIONS[func_name](*args)
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        result_str = str(round(result, 6)) if isinstance(result, float) else str(result)
        logger.info(f"[compute] Result: {result_str}")
        return f"Result: {result_str}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        logger.error(f"[compute] Evaluation failed: {exc}")
        return f"Error: could not evaluate expression — {exc}"


# ---------------------------------------------------------------------------
# Out-of-scope response tool
# ---------------------------------------------------------------------------

@tool
def out_of_scope_response(reason: str) -> str:
    """
    Returns a standard fallback response when the requested information
    is not available in the DIEM knowledge base.

    Use this tool ONLY when all relevant search tools returned no useful
    documents after genuine search attempts. Do NOT use this as a shortcut.

    Args:
        reason: A brief description of why the information was not found.

    Returns:
        A standard fallback message suggesting the user consult the
        official DIEM website.
    """
    logger.info(f"[out_of_scope_response] Reason: '{reason}'")
    return (
        "FALLBACK: The requested information was not found in the DIEM "
        "knowledge base. Suggest the user consult the official website at "
        "https://www.diem.unisa.it for accurate and up-to-date information."
    )


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------

DIEM_TOOLS = [
    search_faculty,
    search_courses,
    search_department,
    compute,
    out_of_scope_response,
]
