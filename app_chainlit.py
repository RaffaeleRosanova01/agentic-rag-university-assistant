"""
Chainlit web interface for the DIEM Agentic RAG chatbot.

Run with:
    chainlit run app_chainlit.py

Chainlit handles the chat UI, message history display, streaming,
and starter suggestions natively. All system logic is delegated to core.py.

The agentic pipeline (ask_with_trace()) is synchronous and potentially slow
(10-60s). It is executed via asyncio.run_in_executor() to avoid blocking the
Chainlit event loop, which would cause WebSocket timeouts on long responses.

Each answer is preceded by a collapsible "DIEM Agent" step that shows the
input guardrail decision and every tool call actually made by the agent,
with its arguments and a summary of its result.
"""

import asyncio
import json
import re

from engineio.payload import Payload
Payload.max_decode_packets = 500

import chainlit as cl

from core import ask_with_trace, initialize_system, update_history

# ---------------------------------------------------------------------------
# System initialisation (runs once at module load)
# ---------------------------------------------------------------------------

graph = initialize_system()

# ---------------------------------------------------------------------------
# Starter suggestions — shown before the first message (like Claude/ChatGPT)
# ---------------------------------------------------------------------------

@cl.set_starters
async def set_starters():
    """
    Defines the starter suggestion cards shown to the user at the beginning
    of each conversation. Each card has a label (shown on the card) and a
    message (sent automatically when clicked).
    """
    return [
        cl.Starter(
            label="🎓 Corsi di laurea",
            message="Quali corsi di laurea offre il DIEM?",
        ),
        cl.Starter(
            label="🕐 Orari di ricevimento",
            message="Quali sono gli orari di ricevimento del professor Capuano?",
        ),
        cl.Starter(
            label="📍 Dove si trova il DIEM?",
            message="Dove si trova il DIEM?",
        ),
        cl.Starter(
            label="🌍 Erasmus e mobilità",
            message="Quali opportunità Erasmus sono disponibili al DIEM?",
        ),
        cl.Starter(
            label="👨‍💼 Direttore del dipartimento",
            message="Chi è il direttore del DIEM?",
        ),
        cl.Starter(
            label="🏠 Aule",
            message=" Dove si trova l'aula 126?",
        ),
        cl.Starter(
            label="🗳️ Organi colleggiali",
            message=" Chi sono i membri della commissione paritetica del DIEM?",
        ),
        cl.Starter(
            label="📝 Date degli esami",
            message="Quando si tiene l'esame di Natural Language Processing and LLM",
        ),
    ]


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """
    Initialises the per-session state when a new conversation starts.
    The compiled LangGraph and the semantic chat history are stored in
    the Chainlit user session so they persist across messages within
    the same conversation but are isolated between different users.
    """
    cl.user_session.set("graph", graph)
    cl.user_session.set("chat_history", [])


def _summarize_tool_output(output: str) -> str:
    """
    Produces a short, readable summary of a tool result for the UI.

    Search tools return document excerpts with "[Source: URL]" markers:
    they are summarized as the number of documents and their unique sources.
    Other tools (compute, out_of_scope_response) are shown as returned.
    """
    sources = re.findall(r"\[Source: (.+?)\]", output)
    if not sources:
        return output if len(output) <= 500 else output[:500] + "..."
    unique_sources = list(dict.fromkeys(sources))
    lines = [f"{len(sources)} document(s) retrieved from:"]
    lines += [f"- {url}" for url in unique_sources]
    return "\n".join(lines)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles an incoming user message.

    Runs the agentic pipeline in a thread executor, then shows the steps
    the agent actually performed (input guardrail decision and tool calls)
    inside a collapsible step, and finally streams the answer.
    """
    graph = cl.user_session.get("graph")
    chat_history = cl.user_session.get("chat_history")

    async with cl.Step(name="DIEM Agent", type="run") as agent_step:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, ask_with_trace, graph, message.content, chat_history
        )

        async with cl.Step(name="Input guardrail", type="tool") as guard_step:
            guard_step.output = result["guardrail_decision"]

        for call in result["tool_calls"]:
            async with cl.Step(name=call["name"], type="tool") as tool_step:
                tool_step.input = json.dumps(call["args"], ensure_ascii=False)
                tool_step.output = _summarize_tool_output(call["output"])

        n_calls = len(result["tool_calls"])
        agent_step.output = f"{n_calls} tool call(s)."

    answer = result["answer"]
    response_msg = cl.Message(content="")
    await response_msg.send()
    for char in answer:
        await response_msg.stream_token(char)
    await response_msg.update()

    chat_history = update_history(chat_history, message.content, answer)
    cl.user_session.set("chat_history", chat_history)
