"""
State module for the DIEM Agentic RAG project.

Defines the AgentState TypedDict that is passed between nodes in the
LangGraph graph. Each node receives the current state, performs its
computation, and returns a partial update that LangGraph merges into
the shared state.
"""

from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the DIEM agent graph.

    Fields:
        query:
            The original user query, set at graph entry and never modified.
            Used by guardrail nodes and the agent node.

        messages:
            The full message history for the ReAct agent node, including
            system prompt, human messages, AI messages, and tool results.
            The add_messages reducer appends new messages rather than
            overwriting the existing list.

        chat_history:
            The last k (query, response) pairs from previous conversation
            turns. Injected into the agent prompt to support follow-up
            questions. Managed manually at graph entry.

        retrieved_context:
            Concatenation of all tool outputs produced during the current
            turn. Accumulated by the tool_executor node and passed to the
            output guardrail for validation.

        final_answer:
            The agent's final response to the user. Written by the agent
            node when the ReAct loop concludes and validated by the output
            guardrail before being returned to the user.

        guardrail_decision:
            Classification produced by the input guardrail node.
            One of: 'PASS', 'OUT_OF_SCOPE', 'INJECTION'.

        iteration_count:
            Number of tool call iterations completed in the current turn.
            Incremented by the tool_executor node. When it reaches
            MAX_AGENT_ITERATIONS the agent is forced to produce a final
            answer without further tool calls.

        retry_count:
            Number of output guardrail retries performed for the current
            final answer. Incremented by the output guardrail node on
            failure. When it reaches MAX_GUARDRAIL_RETRIES the guardrail
            replaces the answer with the fallback message.
    """

    query: str
    messages: Annotated[List[BaseMessage], add_messages]
    chat_history: List[BaseMessage]
    retrieved_context: str
    final_answer: str
    guardrail_decision: str
    iteration_count: int
    retry_count: int