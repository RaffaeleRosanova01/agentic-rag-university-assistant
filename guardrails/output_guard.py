"""
Output guardrail module for the DIEM Agentic RAG project.

Implements faithfulness-based hallucination detection as described in the
RAGAS evaluation framework. Instead of asking the LLM for a single binary
judgment over the full context, the response is decomposed into atomic
claims which are then verified individually against the retrieved context.

Faithfulness = #supported_claims / #total_claims

A response is considered grounded if its faithfulness score meets or
exceeds config.FAITHFULNESS_THRESHOLD (default: 0.8). This tolerates
minor inaccuracies (e.g. a single wrong course code among many correct
ones) without blocking the entire response.

Pipeline:
    1. Extract atomic claims from the response (LLM call, JSON output).
    2. Verify each claim against the retrieved context in a single batch
       call, receiving a JSON list of YES/NO verdicts.
    3. Compute faithfulness score and compare against threshold.

Special cases:
    - Empty retrieved context: check is skipped (conversational response).
    - No claims extracted: response is passed through (fail-open).
    - Any LLM call failure: response is passed through (fail-open policy).

The guardrail uses a dedicated lightweight LLM (config.GUARDRAIL_MODEL_NAME)
separate from the generation model to reduce cost and latency.
"""

import json
import logging
import re
from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

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
# Prompts
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_SYSTEM = """You are a claim extraction assistant.
Given an answer, extract all atomic factual claims it makes.
Each claim must be a single, self-contained statement that can be
verified independently.

Return ONLY a JSON array of strings, with no explanation or preamble.
Example output: ["Claim 1.", "Claim 2.", "Claim 3."]

If the answer contains no verifiable factual claims (e.g. it only says
information was not found), return an empty array: []"""

_CLAIM_VERIFICATION_SYSTEM = """You are a fact-checking assistant.
Given a list of claims and a set of source documents, determine whether
each claim is supported by the source documents.

Return ONLY a JSON array of "YES" or "NO" verdicts, one per claim,
in the same order as the input claims. No explanation or preamble.
Example output: ["YES", "NO", "YES"]

A claim is supported if its content can be inferred from the source
documents, even if not stated verbatim. A claim is NOT supported if it
contains specific values (codes, dates, numbers, URLs) that do not appear
anywhere in the source documents."""

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def _extract_claims(llm: BaseChatModel, answer: str) -> List[str]:
    """
    Extracts atomic factual claims from the generated answer.

    Args:
        llm:    The guardrail LLM instance.
        answer: The generated answer string.

    Returns:
        A list of atomic claim strings, or an empty list if extraction
        fails or the answer contains no verifiable claims.
    """
    messages = [
        SystemMessage(content=_CLAIM_EXTRACTION_SYSTEM),
        HumanMessage(content=f"Answer:\n{answer}"),
    ]

    try:
        raw = llm.invoke(messages).content
        if isinstance(raw, list):
            raw = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw
            )

        # Strip markdown code fences if present.
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        claims = json.loads(raw)

        if not isinstance(claims, list):
            logger.warning("[OutputGuardrail] Claim extraction returned non-list. Skipping.")
            return []

        claims = [str(c).strip() for c in claims if str(c).strip()]
        logger.info(f"[OutputGuardrail] Extracted {len(claims)} claim(s).")
        return claims

    except Exception as exc:
        logger.error(f"[OutputGuardrail] Claim extraction failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Claim verification
# ---------------------------------------------------------------------------

def _verify_claims(
    llm: BaseChatModel,
    claims: List[str],
    context: str,
) -> List[bool]:
    """
    Verifies each claim against the retrieved context in a single batch call.

    Args:
        llm:     The guardrail LLM instance.
        claims:  List of atomic claim strings to verify.
        context: The retrieved context string (synthetic source list).

    Returns:
        A list of booleans (True = supported) in the same order as claims.
        On failure, returns a list of True values (fail-open).
    """
    claims_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))

    messages = [
        SystemMessage(content=_CLAIM_VERIFICATION_SYSTEM),
        HumanMessage(content=(
            f"Source documents:\n{context}\n\n"
            f"Claims to verify:\n{claims_text}"
        )),
    ]

    try:
        raw = llm.invoke(messages).content
        if isinstance(raw, list):
            raw = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw
            )

        raw = re.sub(r"```(?:json)?", "", raw).strip()
        verdicts = json.loads(raw)

        if not isinstance(verdicts, list) or len(verdicts) != len(claims):
            logger.warning(
                f"[OutputGuardrail] Verdict count mismatch "
                f"(expected {len(claims)}, got {len(verdicts) if isinstance(verdicts, list) else 'non-list'}). "
                "Failing open."
            )
            return [True] * len(claims)

        results = [str(v).strip().upper() == "YES" for v in verdicts]

        for i, (claim, supported) in enumerate(zip(claims, results), 1):
            verdict_str = "SUPPORTED" if supported else "NOT SUPPORTED"
            logger.info(f"[OutputGuardrail]   Claim {i}: {verdict_str} — {claim[:80]}")

        return results

    except Exception as exc:
        logger.error(f"[OutputGuardrail] Claim verification failed: {exc}. Failing open.")
        return [True] * len(claims)


# ---------------------------------------------------------------------------
# Guardrail node
# ---------------------------------------------------------------------------

def build_output_guardrail_node(llm: BaseChatModel):
    """
    Returns a node function that performs faithfulness-based hallucination
    detection using the provided LLM.

    The faithfulness score is computed as:
        faithfulness = #supported_claims / #total_claims

    The response is blocked only if faithfulness < config.FAITHFULNESS_THRESHOLD,
    allowing responses with minor inaccuracies to pass through.

    Args:
        llm: A LangChain chat model instance used for claim extraction and
             verification. Should be a lightweight model (e.g. qwen3-8b).

    Returns:
        A callable node function that accepts an AgentState dict and returns
        a dict with an updated 'final_answer' field if hallucination is
        detected, or an empty dict otherwise.
    """
    import config

    def output_guardrail_node(state: AgentState) -> Dict:
        final_answer = state.get("final_answer", "")
        retrieved_context = state.get("retrieved_context", "")

        # Skip check if no context was retrieved (conversational response).
        if not retrieved_context.strip():
            logger.info(
                "[OutputGuardrail] No retrieved context. "
                "Skipping faithfulness check."
            )
            return {}

        logger.info("[OutputGuardrail] Starting faithfulness verification...")
        logger.info(
            f"[OutputGuardrail] Guardrail context length: "
            f"{len(retrieved_context)} chars"
        )

        # Step 1 — Extract atomic claims from the response.
        claims = _extract_claims(llm, final_answer)

        if not claims:
            logger.info(
                "[OutputGuardrail] No verifiable claims found. "
                "Passing response through."
            )
            return {}

        # Step 2 — Verify each claim against the retrieved context.
        verdicts = _verify_claims(llm, claims, retrieved_context)

        # Step 3 — Compute faithfulness score.
        supported = sum(verdicts)
        total = len(verdicts)
        faithfulness = supported / total

        logger.info(
            f"[OutputGuardrail] Faithfulness: {supported}/{total} = "
            f"{faithfulness:.2f} (threshold: {config.FAITHFULNESS_THRESHOLD})"
        )

        if faithfulness < config.FAITHFULNESS_THRESHOLD:
            logger.warning(
                f"[OutputGuardrail] Faithfulness {faithfulness:.2f} below "
                f"threshold {config.FAITHFULNESS_THRESHOLD}. "
                "Replacing response with fallback."
            )
            return {"final_answer": HALLUCINATION_FALLBACK}

        logger.info("[OutputGuardrail] Response verified as grounded.")
        return {}

    return output_guardrail_node