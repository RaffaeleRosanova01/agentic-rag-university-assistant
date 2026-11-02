"""
Input guardrail module for the DIEM Agentic RAG project.

Implements the input guardrail node for the LangGraph agent graph.
Classifies incoming user queries into one of three categories before
the ReAct agent is invoked, providing scope awareness and basic
prompt injection resistance.

The classification is performed by the lightweight guardrail LLM
(config.GUARDRAIL_MODEL_NAME), invoked with a short dedicated prompt that
requests a single-word response.
This makes the guardrail dynamic and robust to paraphrasing, mixed-language
queries, and indirect out-of-scope formulations.

Fail-open policy: if the LLM call fails for any reason, the query is
allowed through (PASS) to avoid blocking legitimate users due to
transient technical issues.
"""

import logging
from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from langchain_core.messages import HumanMessage, AIMessage

from agent.prompts import INPUT_GUARDRAIL_PROMPT
from agent.state import AgentState

logger = logging.getLogger("DIEM_InputGuardrail")

# ---------------------------------------------------------------------------
# Out-of-scope and injection response messages
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_MESSAGE = (
    "Sono l'assistente virtuale del DIEM. "
    "Posso rispondere esclusivamente a domande riguardanti i corsi di laurea, "
    "i docenti, i regolamenti, i servizi e le attività del Dipartimento di "
    "Ingegneria dell'Informazione ed Elettrica e Matematica Applicata "
    "dell'Università degli Studi di Salerno."
)

INJECTION_MESSAGE = (
    "La richiesta non può essere elaborata. "
    "Le policy di sicurezza del sistema impediscono di eseguire questo tipo di comando."
)


# ---------------------------------------------------------------------------
# Guardrail node
# ---------------------------------------------------------------------------

def build_input_guardrail_node(llm: BaseChatModel):
    """
    Returns a LangGraph node function that performs input guardrail
    classification using the provided LLM.

    The returned function follows the LangGraph node signature:
        node(state: AgentState) -> dict

    It updates the following state fields:
        guardrail_decision: 'PASS', 'OUT_OF_SCOPE', or 'INJECTION'.
        final_answer:       set only when the decision is not PASS, so that
                            the graph can route directly to END without
                            invoking the agent.

    Args:
        llm: A LangChain chat model instance used for classification.

    Returns:
        A callable node function compatible with LangGraph's StateGraph.
    """
    chain = INPUT_GUARDRAIL_PROMPT | llm | StrOutputParser()

    def input_guardrail_node(state: AgentState) -> Dict:
        query = state["query"]
        logger.info(f"[InputGuardrail] Classifying query: '{query}'")

        # Include the last exchange from chat history so the guardrail can
        # evaluate follow-up messages (e.g. corrections, clarifications) in
        # the correct conversational context.
        chat_history = state.get("chat_history", [])
        last_exchange = ""
        if len(chat_history) >= 2:
            last_human = next(
                (m.content for m in reversed(chat_history)
                 if isinstance(m, HumanMessage)), ""
            )
            last_ai = next(
                (m.content for m in reversed(chat_history)
                 if isinstance(m, AIMessage)), ""
            )
            if last_human:
                last_exchange = (
                    f"\n\nContesto della conversazione precedente:\n"
                    f"Utente: {last_human}\n"
                    f"Assistente: {last_ai[:200]}..."
                )

        query_with_context = query + last_exchange

        try:
            raw_response = chain.invoke({"query": query_with_context}).strip().upper()
            logger.info(f"[InputGuardrail] LLM response: '{raw_response}'")

            # Priority order: INJECTION > OUT_OF_SCOPE > PASS.
            # Checking for substring rather than exact match makes the
            # parser robust to minor LLM formatting variations.
            if "INJECTION" in raw_response:
                logger.warning("[InputGuardrail] Prompt injection attempt detected.")
                return {
                    "guardrail_decision": "INJECTION",
                    "final_answer": INJECTION_MESSAGE,
                }
            elif "OUT_OF_SCOPE" in raw_response:
                logger.info("[InputGuardrail] Query classified as out of scope.")
                return {
                    "guardrail_decision": "OUT_OF_SCOPE",
                    "final_answer": OUT_OF_SCOPE_MESSAGE,
                }
            else:
                logger.info("[InputGuardrail] Query passed.")
                return {"guardrail_decision": "PASS"}

        except Exception as exc:
            # Fail-open: allow the query through if classification fails.
            logger.error(f"[InputGuardrail] Classification failed: {exc}. Failing open.")
            return {"guardrail_decision": "PASS"}

    return input_guardrail_node