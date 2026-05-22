"""
Nodes module for the DIEM Agentic RAG project.

Implements three nodes for the LangGraph agent graph:

    agent_node:            Invokes the LLM with bound tools. Forces a tool
                           call if the model attempts to answer without
                           retrieving documents first. Enforces the iteration
                           limit via MAX_AGENT_ITERATIONS.

    tool_executor_node:    Executes pending tool calls, accumulates retrieved
                           context in state, increments iteration_count.

    output_guardrail_node: Lightweight LLM safety check on the final answer.
                           Checks for offensive content or private personal
                           information. Allows one retry before falling back.
                           Controlled by ENABLE_OUTPUT_GUARDRAIL.
"""

import logging
from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

import config
from agent.prompts import OUTPUT_GUARDRAIL_PROMPT, build_agent_system_prompt
from agent.state import AgentState

logger = logging.getLogger("DIEM_Nodes")

# ---------------------------------------------------------------------------
# Fallback message
# ---------------------------------------------------------------------------

HALLUCINATION_FALLBACK = (
    "Mi dispiace, non sono riuscito a trovare informazioni sufficientemente "
    "affidabili per rispondere a questa domanda. "
    "Ti consiglio di consultare direttamente il sito ufficiale del DIEM: "
    "https://www.diem.unisa.it"
)


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def _has_tool_results(state: AgentState) -> bool:
    """
    Returns True if at least one ToolMessage is present in the current
    message history, indicating the agent has already retrieved documents.
    """
    return any(
        isinstance(msg, ToolMessage)
        for msg in state.get("messages", [])
    )


# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------

def build_agent_node(llm: BaseChatModel, tools: list):
    """
    Returns a node that invokes the LLM with bound tools.

    Forces a tool call if the model attempts to produce a final answer
    without having retrieved any documents first. Enforces the iteration
    limit via MAX_AGENT_ITERATIONS to prevent infinite loops.

    Args:
        llm:   A LangChain chat model instance with tool calling support.
        tools: The list of LangChain tools to bind to the LLM.

    Returns:
        A callable node function compatible with LangGraph's StateGraph.
    """
    llm_with_tools = llm.bind_tools(tools)
    default_tool_name = tools[0].name if tools else "search_diem"

    def agent_node(state: AgentState) -> Dict:
        logger.info("[AgentNode] Invoking LLM...")

        system_message = SystemMessage(content=build_agent_system_prompt())
        chat_history = state.get("chat_history", [])
        current_messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)
        query = state.get("query", "")

        full_messages = [system_message] + chat_history + current_messages

        # Force final answer when the iteration limit is reached.
        if iteration_count >= config.MAX_AGENT_ITERATIONS:
            logger.warning(
                f"[AgentNode] Iteration limit reached ({iteration_count}). "
                "Forcing final answer."
            )
            full_messages.append(SystemMessage(
                content=(
                    "You have already retrieved information from the knowledge "
                    "base. Produce your final answer now using the context "
                    "already available. Do NOT call any more tools."
                )
            ))

        response: AIMessage = llm_with_tools.invoke(full_messages)

        # Normalise list-type content from models that return structured blocks.
        if isinstance(response.content, list):
            response.content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in response.content
            ).strip()

        if response.tool_calls:
            logger.info(
                f"[AgentNode] Tool calls: "
                f"{[tc['name'] for tc in response.tool_calls]}"
            )
            return {"messages": [response]}

        # Force retrieval if no documents have been retrieved yet.
        if not _has_tool_results(state):
            logger.warning(
                "[AgentNode] Answer attempted without retrieval. "
                f"Forcing '{default_tool_name}' call."
            )
            import uuid
            forced_response = AIMessage(
                content="",
                tool_calls=[{
                    "id": str(uuid.uuid4()),
                    "name": default_tool_name,
                    "args": {"query": query},
                }],
            )
            return {"messages": [forced_response]}

        final_text = response.content
        logger.info(f"[AgentNode] Final answer ({len(final_text)} chars).")
        return {
            "messages": [response],
            "final_answer": final_text,
        }

    return agent_node


# ---------------------------------------------------------------------------
# Tool Executor Node
# ---------------------------------------------------------------------------

def build_tool_executor_node(tools: list):
    """
    Returns a node that executes all pending tool calls, accumulates
    retrieved context in state, and increments the iteration counter.

    Args:
        tools: The list of LangChain tools available to the agent.

    Returns:
        A callable node function compatible with LangGraph's StateGraph.
    """
    tool_map: Dict[str, BaseTool] = {t.name: t for t in tools}

    def tool_executor_node(state: AgentState) -> Dict:
        messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)

        ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                ai_message = msg
                break

        if ai_message is None:
            logger.warning("[ToolExecutor] No pending tool calls found.")
            return {}

        tool_messages = []
        context_parts = [state.get("retrieved_context", "")]

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            logger.info(f"[ToolExecutor] Executing '{tool_name}' with args: {tool_args}")

            if tool_name not in tool_map:
                result = f"Error: tool '{tool_name}' is not available."
                logger.error(f"[ToolExecutor] Unknown tool: '{tool_name}'")
            else:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as exc:
                    result = f"Error during '{tool_name}': {exc}"
                    logger.error(f"[ToolExecutor] Tool '{tool_name}' raised: {exc}")

            logger.info(
                f"[ToolExecutor] '{tool_name}' returned {len(str(result))} chars."
            )
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call_id)
            )
            context_parts.append(str(result))

        return {
            "messages": tool_messages,
            "retrieved_context": "\n\n".join(filter(None, context_parts)),
            "iteration_count": iteration_count + 1,
        }

    return tool_executor_node


# ---------------------------------------------------------------------------
# Output Guardrail Node
# ---------------------------------------------------------------------------

def build_output_guardrail_node(guardrail_llm: BaseChatModel):
    """
    Returns a node that validates the final answer with a lightweight LLM
    safety check — single prompt, single call, ~1-2s latency.

    Checks for:
        - Offensive or discriminatory language
        - Private personal information not publicly available
        - Content entirely unrelated to academic topics

    On BLOCK, one retry is allowed before the fallback message is returned.
    Controlled by ENABLE_OUTPUT_GUARDRAIL in config.

    --- How to add hallucination detection ---
    Add a second LLM call after the safety check using this prompt:

        Given the retrieved context and the assistant's answer, rate
        confidence that every factual claim is supported by the context.
        Reply with ONLY: HIGH, MEDIUM, or LOW.

    Pass {"context": retrieved_context, "answer": final_answer} as input.
    On LOW confidence, call _handle_guardrail_failure() with reason
    "low confidence — possible hallucination".
    On MEDIUM, log a warning but allow through.
    On HIGH, pass.

    Args:
        guardrail_llm: A lightweight LangChain chat model instance used
                       exclusively for the output safety check.

    Returns:
        A callable node function compatible with LangGraph's StateGraph.
    """
    if not config.ENABLE_OUTPUT_GUARDRAIL:
        def noop(state: AgentState) -> Dict:
            return {}
        return noop

    llm_chain = OUTPUT_GUARDRAIL_PROMPT | guardrail_llm | StrOutputParser()

    def output_guardrail_node(state: AgentState) -> Dict:
        final_answer = state.get("final_answer", "")
        retry_count = state.get("retry_count", 0)

        if not final_answer.strip():
            logger.info("[OutputGuardrail] Empty answer. Skipping.")
            return {}

        logger.info("[OutputGuardrail] Starting safety check...")

        try:
            raw = llm_chain.invoke({"response": final_answer}).strip().upper()
            logger.info(f"[OutputGuardrail] Safety check result: '{raw}'")

            if "BLOCK" in raw:
                return _handle_guardrail_failure(
                    state, retry_count,
                    reason="offensive or private content",
                )

        except Exception as exc:
            logger.error(f"[OutputGuardrail] Safety check failed: {exc}. Failing open.")

        logger.info("[OutputGuardrail] Response passed safety check.")
        return {}

    return output_guardrail_node


# ---------------------------------------------------------------------------
# Guardrail failure handler
# ---------------------------------------------------------------------------

def _handle_guardrail_failure(
    state: AgentState,
    retry_count: int,
    reason: str,
) -> Dict:
    """
    Handles a guardrail failure by either allowing a retry or returning
    the fallback message.

    Args:
        state:       The current AgentState.
        retry_count: The current retry count from state.
        reason:      A short description of why the check failed.

    Returns:
        A state update dict routing to retry or fallback.
    """
    logger.warning(f"[OutputGuardrail] Check FAILED: {reason}.")

    if retry_count < config.MAX_GUARDRAIL_RETRIES:
        logger.info("[OutputGuardrail] Retry allowed.")
        corrective = SystemMessage(
            content=(
                "Your previous answer violated the response guidelines. "
                "Please generate a new answer using ONLY information explicitly "
                "present in the retrieved context, without offensive content or "
                "private personal information."
            )
        )
        return {
            "messages": [corrective],
            "final_answer": "",
            "retry_count": retry_count + 1,
        }

    logger.warning("[OutputGuardrail] Max retries reached. Using fallback.")
    return {"final_answer": HALLUCINATION_FALLBACK}