"""
Output guardrail module for the DIEM Agentic RAG project.

Implements a safety check on the final answer produced by the agent.
The check is performed via a single lightweight LLM call using the
OUTPUT_GUARDRAIL_PROMPT defined in agent.prompts.

The guardrail blocks responses that contain:
    - Offensive, discriminatory, or harmful language.
    - Private personal information not typically public.
    - Content entirely unrelated to university or academic topics.

On BLOCK, one retry is allowed: the agent is asked to regenerate the
answer with an explicit corrective instruction. If the retry limit is
reached, the fallback message is returned instead.

The check is skipped entirely when ENABLE_OUTPUT_GUARDRAIL is False
in config (noop node).

Special cases:
    - Empty final_answer: check is skipped.
    - Any LLM call failure: response is passed through (fail-open policy).
"""

import logging
from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

import config
from agent.prompts import OUTPUT_GUARDRAIL_PROMPT
from agent.state import AgentState

logger = logging.getLogger("DIEM_OutputGuardrail")

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


# ---------------------------------------------------------------------------
# Guardrail node
# ---------------------------------------------------------------------------

def build_output_guardrail_node(llm: BaseChatModel):
    """
    Returns a node function that performs a safety check on the final
    answer using the provided LLM.

    The check uses a single prompt (OUTPUT_GUARDRAIL_PROMPT) and a single
    LLM call. On BLOCK, one retry is allowed before the fallback message
    is returned. The node is disabled when ENABLE_OUTPUT_GUARDRAIL is
    False in config.

    Args:
        llm: A LangChain chat model instance used for the safety check.
             Should be a lightweight model (config.GUARDRAIL_MODEL_NAME).

    Returns:
        A callable node function that accepts an AgentState dict and
        returns a dict with an updated 'final_answer' and routing fields
        if the check fails, or an empty dict otherwise.
    """
    if not config.ENABLE_OUTPUT_GUARDRAIL:
        def noop(state: AgentState) -> Dict:
            return {}
        return noop

    llm_chain = OUTPUT_GUARDRAIL_PROMPT | llm | StrOutputParser()

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