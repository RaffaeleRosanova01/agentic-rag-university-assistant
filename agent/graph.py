"""
Graph module for the DIEM Agentic RAG project.

Defines and compiles the LangGraph StateGraph for the DIEM agent.

Flow:
    START
      │
      ▼
    input_guardrail
      │
      ├─ OUT_OF_SCOPE / INJECTION ──────────────────────────────► END
      │
      └─ PASS
           │
           ▼
         agent  ◄──────────────────────────────────────────────┐
           │                                                   │
           ├─ tool_calls                                       │
           │       │                                           │
           │       ▼                                           │
           │  tool_executor ───────────────────────────────────┘
           │  (iteration_count++)
           │
           └─ final_answer
                   │
                   ▼
           output_guardrail
                   │
                   ├─ PASS ────────────────────────────────────► END
                   │
                   └─ FAIL + retry_count < MAX_RETRIES
                           │
                           ▼
                         agent (with corrective message)
                           │
                           ▼
                   output_guardrail
                           └─ any result ──────────────────────► END
"""

import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes import build_agent_node, build_tool_executor_node
from agent.state import AgentState
from agent.tools import DIEM_TOOLS
from guardrails.input_guard import build_input_guardrail_node
from guardrails.output_guard import build_output_guardrail_node

logger = logging.getLogger("DIEM_Graph")


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_input_guardrail(
    state: AgentState,
) -> Literal["agent", "__end__"]:
    """
    Routes after input_guardrail.
    PASS → agent.
    OUT_OF_SCOPE / INJECTION → END.
    """
    decision = state.get("guardrail_decision", "PASS")
    if decision == "PASS":
        logger.info("[Router] Input guardrail passed. Routing to agent.")
        return "agent"
    logger.info(f"[Router] Input guardrail blocked ({decision}). Routing to END.")
    return END


def route_after_agent(
    state: AgentState,
) -> Literal["tool_executor", "output_guardrail"]:
    """
    Routes after agent node.
    Tool calls present → tool_executor.
    Final answer ready → output_guardrail.
    """
    messages = state.get("messages", [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )
    if last_ai is not None and last_ai.tool_calls:
        logger.info(
            f"[Router] Tool calls: "
            f"{[tc['name'] for tc in last_ai.tool_calls]}. "
            "Routing to tool_executor."
        )
        return "tool_executor"
    logger.info("[Router] Final answer ready. Routing to output_guardrail.")
    return "output_guardrail"


def route_after_output_guardrail(
    state: AgentState,
) -> Literal["agent", "__end__"]:
    """
    Routes after output_guardrail.
    Empty final_answer → retry requested → agent.
    Non-empty final_answer → done → END.
    """
    if not state.get("final_answer", "").strip():
        logger.info("[Router] Output guardrail requested retry. Routing to agent.")
        return "agent"
    logger.info("[Router] Output guardrail complete. Routing to END.")
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    llm: BaseChatModel,
    guardrail_llm: BaseChatModel,
):
    """
    Constructs and compiles the DIEM agent LangGraph StateGraph.

    Args:
        llm:           Primary LangChain chat model for agent reasoning
                       and tool calling.
        guardrail_llm: Lightweight model for input/output guardrail checks.

    Returns:
        A compiled LangGraph CompiledGraph ready for invocation.
    """
    input_guardrail_node  = build_input_guardrail_node(guardrail_llm)
    agent_node            = build_agent_node(llm, DIEM_TOOLS)
    tool_executor_node    = build_tool_executor_node(DIEM_TOOLS)
    output_guardrail_node = build_output_guardrail_node(guardrail_llm)

    graph = StateGraph(AgentState)

    graph.add_node("input_guardrail",  input_guardrail_node)
    graph.add_node("agent",            agent_node)
    graph.add_node("tool_executor",    tool_executor_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    graph.add_edge(START, "input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"agent": "agent", END: END},
    )

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tool_executor": "tool_executor",
            "output_guardrail": "output_guardrail",
        },
    )

    graph.add_edge("tool_executor", "agent")

    graph.add_conditional_edges(
        "output_guardrail",
        route_after_output_guardrail,
        {"agent": "agent", END: END},
    )

    compiled = graph.compile()
    logger.info("[Graph] DIEM agent graph compiled successfully.")
    return compiled