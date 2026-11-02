"""
Nodes module for the DIEM Agentic RAG project.

Implements two nodes for the LangGraph agent graph:

    agent_node:         Invokes the LLM with bound tools. Forces a tool
                        call if the model attempts to answer without
                        retrieving documents first. Enforces the iteration
                        limit via MAX_AGENT_ITERATIONS.

    tool_executor_node: Executes pending tool calls, accumulates retrieved
                        context in state, increments iteration_count.

The input and output guardrail nodes are defined in the guardrails package.
"""

import logging
import uuid
from typing import Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

import config
from agent.prompts import build_agent_system_prompt
from agent.state import AgentState

logger = logging.getLogger("DIEM_Nodes")


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

# Returned when the model produces neither tool calls nor answer text, so that
# an empty answer never re-enters the graph.
EMPTY_ANSWER_FALLBACK = (
    "Non è stato possibile generare una risposta. "
    "Ti consiglio di consultare https://www.diem.unisa.it"
)


def _bind_with_tool_choice(
    llm: BaseChatModel, tools: list, tool_choice: str
) -> Optional[BaseChatModel]:
    """
    Binds the tools with an explicit tool_choice ("required" or "none").

    Returns None if the provider integration does not support tool_choice,
    in which case the caller falls back to provider-independent behaviour.
    """
    try:
        return llm.bind_tools(tools, tool_choice=tool_choice)
    except (NotImplementedError, TypeError, ValueError) as exc:
        logger.warning(f"[AgentNode] tool_choice='{tool_choice}' not supported: {exc}")
        return None


def _normalise_content(response: AIMessage) -> AIMessage:
    """Joins list-type content returned by some models into a single string."""
    if isinstance(response.content, list):
        response.content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in response.content
        ).strip()
    return response


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
    # Used to force a retrieval step: the model must pick a tool itself.
    llm_tool_required = _bind_with_tool_choice(llm, tools, "required")
    # Used when the iteration limit is reached: the model must answer.
    llm_tool_none = _bind_with_tool_choice(llm, tools, "none")
    search_tool_names = [t.name for t in tools if t.name.startswith("search_")]

    def _force_retrieval(full_messages: list, query: str) -> AIMessage:
        """
        Obtains a retrieval tool call when the model tried to answer
        without retrieving documents first.

        The model is re-invoked with tool_choice="required" so that it
        selects the most appropriate search tool. If the provider does not
        support tool_choice or still returns no tool call, all search tools
        are called with the original query.
        """
        if llm_tool_required is not None:
            try:
                forced = llm_tool_required.invoke(full_messages)
                if forced.tool_calls:
                    return _normalise_content(forced)
            except Exception as exc:
                logger.warning(f"[AgentNode] Forced tool call failed: {exc}")

        logger.warning(
            f"[AgentNode] Falling back to all search tools: {search_tool_names}"
        )
        return AIMessage(
            content="",
            tool_calls=[
                {"id": str(uuid.uuid4()), "name": name, "args": {"query": query}}
                for name in search_tool_names
            ],
        )

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
            response: AIMessage = (llm_tool_none or llm_with_tools).invoke(full_messages)
            if response.tool_calls:
                # The answer must be produced now: further tool calls are discarded.
                logger.warning("[AgentNode] Tool calls ignored after iteration limit.")
                response = AIMessage(content=response.content)
        else:
            response = llm_with_tools.invoke(full_messages)

        response = _normalise_content(response)

        if response.tool_calls:
            logger.info(
                f"[AgentNode] Tool calls: "
                f"{[tc['name'] for tc in response.tool_calls]}"
            )
            return {"messages": [response]}

        # Force retrieval if no documents have been retrieved yet.
        if not _has_tool_results(state):
            logger.warning("[AgentNode] Answer attempted without retrieval. Forcing a tool call.")
            return {"messages": [_force_retrieval(full_messages, query)]}

        final_text = response.content if isinstance(response.content, str) else ""
        if not final_text.strip():
            logger.warning("[AgentNode] Empty answer from the model. Using fallback message.")
            final_text = EMPTY_ANSWER_FALLBACK
            response = AIMessage(content=final_text)

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