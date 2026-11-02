"""
Core module for the DIEM Agentic RAG project.

Centralizes all system initialization and pipeline logic shared between
the CLI entry point (app_CLI.py), the Chainlit web interface
(app_chainlit.py) and the evaluation script (evaluate.py).

Public API:
    build_llm(model_name)  — instantiates a LangChain chat model.
    initialize_system()    — initializes retriever, LLMs, and agent graph.
                             Returns the compiled graph.
    ask(graph, query, chat_history) — runs one query through the agent.
    ask_with_trace(graph, query, chat_history) — same as ask(), but also
                             returns the guardrail decision and the tool
                             calls made by the agent.
    update_history(chat_history, query, answer) — appends a turn and trims.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import config
from agent.graph import build_graph
from guardrails.input_guard import INJECTION_MESSAGE, OUT_OF_SCOPE_MESSAGE
from retrieval.retriever import initialize as initialize_retriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("DIEM_Core")

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = 5


# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def build_llm(model_name: str = None):
    """
    Instantiates the LLM based on the provider configured in config.py.

    Supported providers: ollama, openrouter, openai, google, huggingface.

    Args:
        model_name: Optional model identifier override. If not provided,
                    config.LLM_MODEL_NAME is used. Pass
                    config.GUARDRAIL_MODEL_NAME to build a lightweight
                    model dedicated to the output guardrail.

    Returns:
        A configured LangChain chat model instance.

    Raises:
        ValueError: if config.LLM_PROVIDER is not a supported provider.
    """
    effective_model = model_name or config.LLM_MODEL_NAME
    logger.info(
        f"Initialising LLM provider: {config.LLM_PROVIDER} "
        f"(model: {effective_model}) ..."
    )

    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=effective_model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )

    elif config.LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=effective_model,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            max_tokens=config.LLM_MAX_NEW_TOKENS,
        )

    elif config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=effective_model,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.OPENAI_API_KEY,
            max_tokens=config.LLM_MAX_NEW_TOKENS,
        )

    elif config.LLM_PROVIDER == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "The 'google' provider requires: pip install langchain-google-genai"
            ) from exc
        llm = ChatGoogleGenerativeAI(
            model=effective_model,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY,
        )

    elif config.LLM_PROVIDER == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        endpoint = HuggingFaceEndpoint(
            repo_id=effective_model,
            task="text-generation",
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            huggingfacehub_api_token=config.HF_TOKEN,
            return_full_text=False,
        )
        llm = ChatHuggingFace(llm=endpoint)

    else:
        raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")

    logger.info(f"LLM ready: {effective_model}")
    return llm


# ---------------------------------------------------------------------------
# System initialisation
# ---------------------------------------------------------------------------

def initialize_system():
    """
    Initializes all system components and returns a ready-to-use agent graph.

    Performs the following steps in order:
        1. Instantiates the primary LLM for agent reasoning and tool calling.
        2. Instantiates the guardrail LLM for input/output safety checks.
        3. Initializes the retrieval pipeline (embedding model, ChromaDB,
           cross-encoder reranker).
        4. Builds and compiles the LangGraph agent graph.

    Returns:
        graph: The compiled LangGraph CompiledGraph ready for invocation.
    """
    logger.info("Initialising primary LLM...")
    llm = build_llm()

    logger.info("Initialising guardrail LLM...")
    guardrail_llm = build_llm(model_name=config.GUARDRAIL_MODEL_NAME)

    logger.info("Initialising retrieval pipeline...")
    initialize_retriever()

    logger.info("Building agent graph...")
    graph = build_graph(llm, guardrail_llm)

    logger.info("System ready.")
    return graph


def _fix_latex(text: str) -> str:
    """
    Converts LaTeX delimiters from standard notation to Chainlit-compatible
    KaTeX notation.
 
    Chainlit uses KaTeX which requires $ and $$ as delimiters.
    GPT-class models typically output \\[...\\] for block math and
    \\(...\\) for inline math, which KaTeX does not render.
 
    Conversions applied:
        \\[...\\]  →  $$...$$   (block math)
        \\(...\\)  →  $...$     (inline math)
    """
    import re
    # Block math: \[...\] → $$...$$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Inline math: \(...\) → $...$
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text


# ---------------------------------------------------------------------------
# Agentic pipeline
# ---------------------------------------------------------------------------

def _extract_tool_trace(messages: list) -> list:
    """
    Builds a readable trace of the tool calls made during one graph run.

    Args:
        messages: The final message list from the graph state.

    Returns:
        A list of dicts, one per tool call, in execution order:
            {"name": str, "args": dict, "output": str}
    """
    outputs = {
        m.tool_call_id: str(m.content)
        for m in messages if isinstance(m, ToolMessage)
    }
    trace = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                trace.append({
                    "name": call["name"],
                    "args": call["args"],
                    "output": outputs.get(call["id"], ""),
                })
    return trace


def ask_with_trace(graph, query: str, chat_history: list) -> dict:
    """
    Runs the agentic RAG pipeline for a single user query by invoking
    the compiled LangGraph StateGraph, and returns the answer together
    with the information needed to inspect how it was produced.

    The graph handles input guardrail, tool calling, and output guardrail
    internally. This function is responsible only for constructing the
    initial state and extracting the results.

    Args:
        graph:        The compiled LangGraph StateGraph returned by
                      initialize_system().
        query:        The user query string.
        chat_history: The conversation history from previous turns,
                      as a list of LangChain BaseMessage objects.

    Returns:
        A dict with keys:
            answer:             the final answer to display to the user.
            guardrail_decision: 'PASS', 'OUT_OF_SCOPE', 'INJECTION', or
                                'ERROR' if the graph invocation failed.
            tool_calls:         list of {"name", "args", "output"} dicts.
            retrieved_context:  concatenated tool outputs of this turn.
    """
    initial_state = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "chat_history": chat_history,
        "retrieved_context": "",
        "final_answer": "",
        "guardrail_decision": "",
        "iteration_count": 0,
        "retry_count": 0,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        logger.error(f"[Core] Graph invocation failed: {exc}")
        return {
            "answer": "Si è verificato un errore di sistema. Riprova più tardi.",
            "guardrail_decision": "ERROR",
            "tool_calls": [],
            "retrieved_context": "",
        }

    final_answer = result.get("final_answer", "")
    guardrail_decision = result.get("guardrail_decision", "PASS")

    if guardrail_decision == "OUT_OF_SCOPE":
        answer = OUT_OF_SCOPE_MESSAGE
    elif guardrail_decision == "INJECTION":
        answer = INJECTION_MESSAGE
    elif not final_answer.strip():
        logger.warning("[Core] Empty final answer from graph.")
        answer = (
            "Non è stato possibile generare una risposta. "
            "Ti consiglio di consultare https://www.diem.unisa.it"
        )
    else:
        answer = _fix_latex(final_answer)

    return {
        "answer": answer,
        "guardrail_decision": guardrail_decision,
        "tool_calls": _extract_tool_trace(result.get("messages", [])),
        "retrieved_context": result.get("retrieved_context", ""),
    }


def ask(graph, query: str, chat_history: list) -> str:
    """
    Runs the agentic RAG pipeline for a single user query and returns
    only the final answer string. See ask_with_trace() for details.
    """
    return ask_with_trace(graph, query, chat_history)["answer"]


# ---------------------------------------------------------------------------
# Chat history management
# ---------------------------------------------------------------------------

def update_history(chat_history: list, query: str, answer: str) -> list:
    """
    Appends the latest query/answer pair to the chat history and trims
    it to the sliding window of MAX_HISTORY_TURNS turns.

    Args:
        chat_history: The existing conversation history.
        query:        The user query for the completed turn.
        answer:       The assistant's answer for the completed turn.

    Returns:
        The updated and trimmed chat history list.
    """
    chat_history = chat_history + [
        HumanMessage(content=query),
        AIMessage(content=answer),
    ]
    max_messages = MAX_HISTORY_TURNS * 2
    if len(chat_history) > max_messages:
        chat_history = chat_history[-max_messages:]
    return chat_history