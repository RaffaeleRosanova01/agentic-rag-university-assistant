"""
Tools module for the DIEM Agentic RAG project.

Defines the LangChain tools exposed to the ReAct agent.

Tools:
    search_diem           — retrieval on the full DIEM knowledge base.
    search_easycourse     — retrieval for exam schedules and timetables.
    compute               — deterministic mathematical calculations.
    out_of_scope_response — standard fallback for unanswerable questions.

search_faculty and search_courses have been consolidated into search_diem
because without domain filtering they performed identical retrieval.
The agent model (GPT-OSS 120B) already formulates optimised queries
autonomously, making separate tools redundant.
"""

import logging

from langchain_core.tools import tool

from retrieval.retriever import initialize as initialize_retriever
from retrieval.retriever import retrieve

logger = logging.getLogger("DIEM_Tools")

# ---------------------------------------------------------------------------
# Retrieval pipeline initialisation
# ---------------------------------------------------------------------------

initialize_retriever()

# ---------------------------------------------------------------------------
# Internal formatting helper
# ---------------------------------------------------------------------------

def _format_docs(docs) -> str:
    """
    Formats a list of LangChain Documents into a structured string suitable
    for injection into the ReAct agent scratchpad.

    Document numbers are intentionally omitted to prevent the LLM from
    citing [Document N] references instead of actual source URLs.

    Args:
        docs: List of LangChain Document objects returned by retrieve().

    Returns:
        A formatted string with document blocks separated by ---,
        or a fallback message if no documents were retrieved.
    """
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
        parts.append(
            f"[Source: {source}]\n"
            f"{text}\n"
            f"[End]"
        )

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
def search_diem(query: str) -> str:
    """
    Searches the official DIEM knowledge base for any information about
    the Department of Information and Electrical Engineering and Applied
    Mathematics (DIEM) of the University of Salerno.

    Use this tool for questions about:
    - Degree programmes (which courses the DIEM offers, general overview)
    - Specific courses: syllabus, study plan, CFU credits, prerequisites,
      exam modalities, course timetables
    - Faculty members: office hours, curriculum, research, publications,
      laboratory affiliations, contact information
    - Department location, facilities, laboratories, and equipment
    - Erasmus and international mobility programmes
    - Internship opportunities and academic procedures
    - Research areas, funded projects, and spin-offs
    - Committees, delegates, and administrative contacts
    - Department regulations and official notices

    If the question covers multiple distinct topics, call this tool
    multiple times with focused queries.

    Args:
        query: A precise natural language description of the information
               needed, formulated to maximise retrieval accuracy.

    Returns:
        Formatted document excerpts with source URLs.
    """
    logger.info(f"[search_diem] Query: '{query}'")
    docs = retrieve(query)
    result = _format_docs(docs)
    logger.info(f"[search_diem] Returning {len(docs)} document(s).")
    return result


@tool
def search_easycourse(query: str) -> str:
    """
    Searches the DIEM knowledge base for exam schedules and course
    timetables from the EasyCourse platform (easycourse.unisa.it).

    Use this tool when the user asks about:
    - Exam dates and sessions (appelli d'esame)
    - Course timetables and classroom assignments
    - Exam session calendars (sessione invernale, estiva, autunnale)

    Args:
        query: A natural language query about schedules or exam dates,
               e.g. 'appelli Intelligenza Artificiale sessione estiva' or
               'orario lezioni Ingegneria Informatica secondo semestre'.

    Returns:
        Formatted document excerpts with source URLs.
    """
    logger.info(f"[search_easycourse] Query: '{query}'")
    docs = retrieve(query)
    result = _format_docs(docs)
    logger.info(f"[search_easycourse] Returning {len(docs)} document(s).")
    return result


# ---------------------------------------------------------------------------
# Compute tool
# ---------------------------------------------------------------------------

@tool
def compute(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.

    Use this tool after retrieving the relevant formula from the knowledge
    base with search_diem. Apply the formula to the numerical values
    provided by the user and pass the resulting expression here.

    Supported operations: +, -, *, /, **, //, %, (, )
    Supported functions: round(), abs(), min(), max()

    Examples:
        compute("(28 * 3 + 30 * 6 + 27 * 9) / (3 + 6 + 9)")
        compute("round(28.8 * 110 / 30, 2)")
        compute("(18 * 0.4 + 28 * 0.6)")

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

    Use this tool when:
    - search_diem returned no relevant documents for the query.
    - The question is outside the scope of the DIEM knowledge base.
    - You have already tried search_diem and found nothing useful.

    Do NOT use this tool as a shortcut to avoid searching — always attempt
    retrieval with search_diem first.

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
    search_diem,
    search_easycourse,
    compute,
    out_of_scope_response,
]