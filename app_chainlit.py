"""
Chainlit web interface for the DIEM Agentic RAG chatbot.

Run with:
    chainlit run app_chainlit.py

Chainlit handles the chat UI, message history display, streaming,
and starter suggestions natively. All system logic is delegated to core.py.

The agentic pipeline (ask()) is synchronous and potentially slow (10-60s).
It is executed via asyncio.run_in_executor() to avoid blocking the Chainlit
event loop, which would cause WebSocket timeouts on long responses.
"""

import asyncio

from engineio.payload import Payload
Payload.max_decode_packets = 500

import chainlit as cl

from core import ask, initialize_system, update_history

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
            label="🔬 Aree di ricerca",
            message="Quali aree di ricerca sono attive al DIEM?",
        ),
        cl.Starter(
            label="🧪 Laboratori",
            message="Quali laboratori sono disponibili al DIEM?",
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


@cl.step(type="retrieval", name="Retrieval from DIEM Knowledge Base")
async def step_retrieval(query: str) -> str:
    """
    Displays the retrieval step in the Chainlit UI.
    Returns the query that will be used for knowledge base search.
    """
    return f"Searching DIEM knowledge base for: *{query}*"


@cl.step(type="tool", name="Tool Execution — search_diem")
async def step_tool(graph, query: str, chat_history: list) -> str:
    """
    Executes the full agentic RAG pipeline as a tool step.
    This is the main blocking call — run in a thread executor to
    avoid blocking the Chainlit event loop.
    """
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, ask, graph, query, chat_history
    )
    return answer


@cl.step(type="llm", name="Generating Response")
async def step_generate(answer: str) -> str:
    """
    Displays the generation step in the Chainlit UI.
    Returns a preview of the generated answer.
    """
    preview = answer[:120] + "..." if len(answer) > 120 else answer
    return f"Response generated ({len(answer)} chars): *{preview}*"

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles an incoming user message through three visible steps:
        1. Retrieval  — shows the query being searched
        2. Tool       — executes the agentic RAG pipeline
        3. Generation — confirms the response was generated

    The final answer is then streamed character by character.
    """
    graph = cl.user_session.get("graph")
    chat_history = cl.user_session.get("chat_history")

    async with cl.Step(name="DIEM Agent", type="run") as parent_step:
        
        async with cl.Step(name="Retrieval from DIEM Knowledge Base", type="retrieval") as s1:
            s1.output = f"Searching DIEM knowledge base for: *{message.content}*"

        async with cl.Step(name="Tool Execution — search_diem", type="tool") as s2:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None, ask, graph, message.content, chat_history
            )
            s2.output = f"Tool returned {len(answer)} chars."

        async with cl.Step(name="Generating Response", type="llm") as s3:
            preview = answer[:120] + "..." if len(answer) > 120 else answer
            s3.output = f"*{preview}*"

        parent_step.output = "Done."

    response_msg = cl.Message(content="")
    await response_msg.send()
    for char in answer:
        await response_msg.stream_token(char)
    await response_msg.update()

    chat_history = update_history(chat_history, message.content, answer)
    cl.user_session.set("chat_history", chat_history)