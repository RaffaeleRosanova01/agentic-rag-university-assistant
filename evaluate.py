"""
Evaluation script for the DIEM Agentic RAG project.

Standard RAG metrics (e.g. RAGAS context precision/recall) assume a single
retrieval step per question. This agent instead decides which tools to call,
may call several of them, and accumulates context across iterations, so this
script measures what matters for an agentic system:

Answer cases (questions with a reference answer and reference source URLs):
    source_hit       — at least one reference source is among the documents
                       retrieved by any tool call of the turn (retrieval).
    routing          — the agent called at least one search tool whose domain
                       contains a reference source (tool selection).
    correctness      — LLM judge: answer vs. reference answer
                       (correct = 1, partial = 0.5, incorrect = 0).
    groundedness     — LLM judge: are the answer's claims supported by the
                       retrieved context? (supported = 1, partial = 0.5,
                       unsupported = 0).
    tool_calls, latency_s — cost of the turn.

Guardrail cases (out-of-scope questions and prompt injections):
    blocked          — the input guardrail refused the question.

False refusals (answer cases blocked by the input guardrail) are reported
as well. Source metrics are computed only for reference sources that exist
in the knowledge base (data/chunked_parents.json); missing ones are listed
before the run so the dataset can be corrected.

Every case is saved as soon as it completes (evaluation/results/runs.jsonl),
so an interrupted run resumes where it stopped. Agent and judge calls are
retried with exponential backoff.

Usage:
    python evaluate.py                  # run (or resume) the full evaluation
    python evaluate.py --fresh          # discard cached results first
    python evaluate.py --no-judge       # retrieval/routing/guardrail metrics only
    python evaluate.py --only q04 g01   # run selected case ids

Outputs (in evaluation/results/):
    runs.jsonl   — per-case raw results (answer, tool calls, sources, verdicts)
    summary.json — aggregated metrics
    summary.md   — Markdown report with aggregated and per-case results
"""

import argparse
import json
import logging
import os
import re
import time
from statistics import mean
from typing import Callable, Dict, List, Optional
from urllib.parse import unquote, urlparse

import config

logger = logging.getLogger("DIEM_Eval")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "evaluation", "dataset.json")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")
RUNS_PATH = os.path.join(RESULTS_DIR, "runs.jsonl")

# Maximum characters of retrieved context passed to the groundedness judge.
MAX_JUDGE_CONTEXT_CHARS = 60_000

# Search tool responsible for each knowledge-base host.
TOOL_BY_HOST = {
    "docenti.unisa.it":    "search_faculty",
    "corsi.unisa.it":      "search_courses",
    "easycourse.unisa.it": "search_courses",
    "www.diem.unisa.it":   "search_department",
}

VERDICT_SCORES = {
    "correct": 1.0, "partial": 0.5, "incorrect": 0.0,
    "supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0,
}


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

CORRECTNESS_PROMPT = """You are grading the answer of a university department assistant
against a reference answer written by a human expert.

Judge only the factual content that answers the question:
- "correct":   the answer contains the key facts of the reference and nothing
               that contradicts it.
- "partial":   some key facts are present, but others are missing or imprecise.
- "incorrect": the key facts are missing or contradict the reference, or the
               assistant says it could not find the information.
Additional details that are not in the reference are acceptable unless they
contradict it. Question, reference and answer may be in Italian or English.

Reply with JSON only, in this exact format:
{{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}

Question:
{question}

Reference answer:
{reference}

Assistant answer:
{answer}"""

GROUNDEDNESS_PROMPT = """You are checking whether an assistant's answer is grounded in the
documents it retrieved.

- "supported":           every factual claim in the answer is stated in the context.
- "partially_supported": most claims are supported, but some are not.
- "unsupported":         important claims are not found in the context.
Statements that the information was not found, greetings and links taken from
the context do not count as unsupported claims.

Reply with JSON only, in this exact format:
{{"verdict": "supported" | "partially_supported" | "unsupported", "reason": "<one sentence>"}}

Retrieved context:
{context}

Assistant answer:
{answer}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Normalizes a URL for comparison (case, trailing slash, fragment, encoding)."""
    parsed = urlparse(unquote(url.strip()))
    path = parsed.path.rstrip("/")
    # Faculty profiles: /{id}/home and /{id} are the same page.
    if parsed.netloc.lower() == "docenti.unisa.it" and path.endswith("/home"):
        path = path[: -len("/home")]
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.netloc.lower()}{path}{query}".lower()


def url_matches(retrieved: str, reference: str) -> bool:
    """
    True if a retrieved source corresponds to a reference source: same page,
    or a sub-page of it (e.g. a faculty profile root and its /home page).
    """
    r, ref = normalize_url(retrieved), normalize_url(reference)
    if r == ref:
        return True
    # A reference to a profile home page means that page only, not its sub-pages.
    if urlparse(reference).path.rstrip("/").endswith("/home"):
        return False
    return r.startswith(ref + "/") or r.startswith(ref + "?")


def sources_from_output(output: str) -> List[str]:
    """Extracts the source URLs from a search tool output."""
    return re.findall(r"\[Source: (.+?)\]", output)


def with_retries(fn: Callable, what: str, attempts: int = 5, base_wait: float = 10.0):
    """Calls fn() with exponential backoff, re-raising after the last attempt."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            wait = base_wait * 2 ** (attempt - 1)
            print(f"[Eval]   {what} failed ({type(exc).__name__}: {str(exc)[:120]}). "
                  f"Retry {attempt}/{attempts - 1} in {wait:.0f}s...")
            time.sleep(wait)


def parse_verdict(text: str) -> Dict:
    """Extracts {"verdict", "reason"} from the judge output."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict in VERDICT_SCORES:
                return {"verdict": verdict, "reason": str(data.get("reason", ""))}
        except json.JSONDecodeError:
            pass
    return {"verdict": "unparsable", "reason": text[:300]}


def load_kb_sources() -> Optional[List[str]]:
    """Returns the source URLs present in the knowledge base, or None if unavailable."""
    if not os.path.exists(config.CHUNKED_PARENTS_PATH):
        return None
    with open(config.CHUNKED_PARENTS_PATH, "r", encoding="utf-8") as fh:
        return sorted({p["source"] for p in json.load(fh)})


def load_runs() -> Dict[str, Dict]:
    if not os.path.exists(RUNS_PATH):
        return {}
    with open(RUNS_PATH, "r", encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def save_runs(runs: Dict[str, Dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = RUNS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in runs.values():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, RUNS_PATH)


# ---------------------------------------------------------------------------
# Evaluation steps
# ---------------------------------------------------------------------------

def run_agent(graph, case: Dict) -> Dict:
    """Runs one case through the agent (previous turns first, if any)."""
    from core import ask_with_trace, update_history

    chat_history = []
    for turn in case.get("history", []):
        previous = with_retries(lambda: ask_with_trace(graph, turn, chat_history), "history turn")
        chat_history = update_history(chat_history, turn, previous["answer"])

    start = time.time()
    result = with_retries(lambda: ask_with_trace(graph, case["question"], chat_history), "agent run")
    latency = time.time() - start
    if result["guardrail_decision"] == "ERROR":
        raise RuntimeError("graph invocation failed")

    sources = [s for call in result["tool_calls"] for s in sources_from_output(call["output"])]
    return {
        "answer": result["answer"],
        "guardrail_decision": result["guardrail_decision"],
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in result["tool_calls"]],
        "retrieved_sources": list(dict.fromkeys(sources)),
        "retrieved_context": result["retrieved_context"],
        "latency_s": round(latency, 1),
    }


def score_retrieval(case: Dict, record: Dict, kb_sources: Optional[List[str]]) -> Dict:
    """Computes source_hit and routing for an answer case."""
    references = case.get("reference_sources", [])
    if kb_sources is not None:
        references = [r for r in references if any(url_matches(s, r) for s in kb_sources)]
    if not references:
        return {"evaluable_sources": [], "source_hit": None, "routing": None}

    hit = any(url_matches(s, r) for s in record["retrieved_sources"] for r in references)
    expected_tools = {TOOL_BY_HOST.get(urlparse(r).netloc.lower()) for r in references} - {None}
    called = {c["name"] for c in record["tool_calls"]}
    return {
        "evaluable_sources": references,
        "source_hit": float(hit),
        "routing": float(bool(expected_tools & called)) if expected_tools else None,
    }


def judge(judge_llm, prompt: str) -> Dict:
    response = with_retries(lambda: judge_llm.invoke(prompt), "judge call")
    content = response.content if isinstance(response.content, str) else str(response.content)
    return parse_verdict(content)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _avg(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(mean(values), 3) if values else None


def summarize(cases: List[Dict], runs: Dict[str, Dict]) -> Dict:
    answer = [runs[c["id"]] for c in cases if c["type"] == "answer" and c["id"] in runs]
    guard = [runs[c["id"]] for c in cases if c["type"] == "guardrail" and c["id"] in runs]
    passed = [r for r in answer if r["guardrail_decision"] == "PASS"]
    return {
        "answer_cases": len(answer),
        "source_hit_rate": _avg([r.get("source_hit") for r in passed]),
        "routing_accuracy": _avg([r.get("routing") for r in passed]),
        "answer_correctness": _avg([VERDICT_SCORES.get(r.get("correctness", {}).get("verdict")) for r in passed]),
        "groundedness": _avg([VERDICT_SCORES.get(r.get("groundedness", {}).get("verdict")) for r in passed]),
        "avg_tool_calls": _avg([len(r["tool_calls"]) for r in passed]),
        "avg_latency_s": _avg([r["latency_s"] for r in passed]),
        "false_refusals": len(answer) - len(passed),
        "guardrail_cases": len(guard),
        "guardrail_block_rate": _avg([float(r["guardrail_decision"] in ("OUT_OF_SCOPE", "INJECTION")) for r in guard]),
    }


def write_report(cases: List[Dict], runs: Dict[str, Dict], summary: Dict) -> None:
    def fmt(v, pct=True):
        if v is None:
            return "n/a"
        return f"{v:.0%}" if pct else f"{v}"

    lines = [
        "# Evaluation results", "",
        f"Agent model: `{config.LLM_MODEL_NAME}` · Judge model: `{config.EVAL_JUDGE_MODEL_NAME}`", "",
        "| Metric | Value |", "|---|---|",
        f"| Source hit rate (retrieval) | {fmt(summary['source_hit_rate'])} |",
        f"| Routing accuracy (tool selection) | {fmt(summary['routing_accuracy'])} |",
        f"| Answer correctness (LLM judge) | {fmt(summary['answer_correctness'])} |",
        f"| Groundedness (LLM judge) | {fmt(summary['groundedness'])} |",
        f"| Guardrail block rate | {fmt(summary['guardrail_block_rate'])} ({summary['guardrail_cases']} cases) |",
        f"| False refusals | {summary['false_refusals']} / {summary['answer_cases']} |",
        f"| Avg. tool calls per question | {fmt(summary['avg_tool_calls'], pct=False)} |",
        f"| Avg. latency per question (s) | {fmt(summary['avg_latency_s'], pct=False)} |",
        "", "## Per-case results", "",
        "| ID | Question | Tools | Source hit | Correctness | Groundedness |",
        "|---|---|---|---|---|---|",
    ]
    for c in cases:
        r = runs.get(c["id"])
        if r is None:
            continue
        tools = ", ".join(t["name"] for t in r["tool_calls"]) or "—"
        if c["type"] == "guardrail":
            lines.append(f"| {c['id']} | {c['question']} | {tools} | guardrail: {r['guardrail_decision']} | | |")
            continue
        hit = {1.0: "yes", 0.0: "no", None: "n/a"}[r.get("source_hit")]
        lines.append(
            f"| {c['id']} | {c['question']} | {tools} | {hit} | "
            f"{r.get('correctness', {}).get('verdict', '—')} | {r.get('groundedness', {}).get('verdict', '—')} |"
        )
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the DIEM agentic RAG assistant.")
    parser.add_argument("--fresh", action="store_true", help="discard cached results")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-judge metrics")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these case ids")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds to wait between cases")
    args = parser.parse_args()

    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    selected = [c for c in cases if not args.only or c["id"] in args.only]

    runs = {} if args.fresh else load_runs()
    print(f"[Eval] {len(selected)} case(s) selected, {sum(c['id'] in runs for c in selected)} already cached.")

    # Dataset check: reference sources that are not in the knowledge base.
    kb_sources = load_kb_sources()
    if kb_sources is None:
        print("[Eval] WARNING: knowledge base files not found; source metrics use all reference URLs.")
    else:
        for case in selected:
            missing = [r for r in case.get("reference_sources", [])
                       if not any(url_matches(s, r) for s in kb_sources)]
            if missing:
                print(f"[Eval] {case['id']}: reference source(s) not in the knowledge base: {missing}")

    graph = None
    judge_llm = None
    for i, case in enumerate(selected, 1):
        record = runs.get(case["id"])
        print(f"\n[Eval] [{i}/{len(selected)}] {case['id']}: {case['question'][:80]}")

        if record is None:
            if graph is None:
                from core import initialize_system
                graph = initialize_system()
            try:
                record = {"id": case["id"], **run_agent(graph, case)}
            except Exception as exc:
                print(f"[Eval]   Skipped after repeated errors: {exc}")
                continue
            runs[case["id"]] = record
            save_runs(runs)
            time.sleep(args.delay)

        # Recomputed on every run, so corrections to the dataset take effect
        # without re-running the agent.
        if case["type"] == "answer":
            record.update(score_retrieval(case, record, kb_sources))

        needs_judge = (
            case["type"] == "answer" and not args.no_judge
            and record["guardrail_decision"] == "PASS"
            and ("correctness" not in record or "groundedness" not in record)
        )
        if needs_judge:
            if judge_llm is None:
                from core import build_llm
                judge_llm = build_llm(model_name=config.EVAL_JUDGE_MODEL_NAME)
            try:
                if "correctness" not in record:
                    record["correctness"] = judge(judge_llm, CORRECTNESS_PROMPT.format(
                        question=case["question"], reference=case["reference_answer"], answer=record["answer"]))
                if "groundedness" not in record:
                    record["groundedness"] = judge(judge_llm, GROUNDEDNESS_PROMPT.format(
                        context=record["retrieved_context"][:MAX_JUDGE_CONTEXT_CHARS] or "(no context)",
                        answer=record["answer"]))
            except Exception as exc:
                print(f"[Eval]   Judge failed after retries: {exc}")
            runs[case["id"]] = record
            save_runs(runs)

        tools = [t["name"] for t in record["tool_calls"]]
        print(f"[Eval]   guardrail={record['guardrail_decision']} tools={tools} "
              f"source_hit={record.get('source_hit')} "
              f"correctness={record.get('correctness', {}).get('verdict')} "
              f"groundedness={record.get('groundedness', {}).get('verdict')}")

    save_runs(runs)
    summary = summarize(cases, runs)
    write_report(cases, runs, summary)
    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key:<22} {value}")
    print(f"\n[Eval] Report written to {os.path.join(RESULTS_DIR, 'summary.md')}")


if __name__ == "__main__":
    main()
