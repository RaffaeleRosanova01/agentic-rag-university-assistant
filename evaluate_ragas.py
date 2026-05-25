"""
RAGAS evaluation script for the DIEM Agentic RAG project.

Runs a fixed set of evaluation queries through the agent, collects responses
and retrieved contexts, computes RAGAS metrics, and produces a summary report
with charts.

Metrics computed:
    - context_precision:  fraction of retrieved contexts that are relevant
                          to answering the question (requires ground truth).
    - context_recall:     fraction of ground truth information covered by
                          retrieved contexts (requires ground truth).
    - faithfulness:       fraction of answer claims supported by retrieved
                          contexts (no ground truth required, uses LLM judge).
    - answer_relevancy:   how well the answer addresses the question
                          (no ground truth required, uses embedding similarity).

Usage:
    python evaluate_ragas.py

Output:
    - evaluation_results.json  — full per-query scores
    - evaluation_summary.png   — bar chart of mean scores per metric
    - evaluation_heatmap.png   — per-query heatmap across all metrics
"""

import json
import logging
import os
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Suppress verbose initialization logs during evaluation
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

EVAL_DATASET = [
    # --- Futuri studenti ---
    {
        "question": "Quali sono i corsi di laurea offerti dal DIEM?",
        "ground_truth": "I corsi di laurea offerti dal DIEM sono: Ingegneria dell'Informazione per la Medicina Digitale (IE128, L-8), "
        "Ingegneria Informatica (IE127, L-8), Electrical Engineering for Digital Energy (IE233, LM-28), "
        "Information Engineering for Digital Medicine (IE232, LM-32), Ingegneria Informatica magistrale (IE227, LM-32), "
        "Dottorato in Ingegneria dell'Informazione (88605), Dottorato in Photovoltaics (88608).", 
        "ground_truth_contexts": [
            "https://www.diem.unisa.it/didattica/offerta-formativa",
        ],
    },
    {
        "question": "Ho preso 18 al TOLC. Posso iscrivermi?",
        "ground_truth": "Sì, con un punteggio totale di 18/50 al TOLC-I è possibile immatricolarsi, "
        "ma ci sono due condizioni da verificare: "
        "1. Punteggio totale compreso tra 16 e 20: è richiesto che la sezione Matematica "
        "abbia un punteggio non inferiore a 8. "
        "2. Se la sezione Matematica fosse inferiore a 8, l'immatricolazione sarebbe "
        "possibile solo con OFA in Matematica di Base.",
        "ground_truth_contexts": [
            "https://corsi.unisa.it/ingegneria-informatica/immatricolazioni",
            "https://corsi.unisa.it/ingegneria-informatica-magistrale/immatricolazioni",
            "https://corsi.unisa.it/information-Engineering-for-digital-medicine/immatricolazioni",
            "https://corsi.unisa.it/electrical-engineering-for-digital-energy/immatricolazioni",],
    },
    {
        "question": (
            "Quali sono i requisiti di ammissione per entrare alla Laurea "
            "Magistrale di Information Engineering for Digital Medicine?"
        ),
        "ground_truth": "Per accedere alla Laurea Magistrale in Information Engineering for Digital Medicine sono richiesti: "
        "requisiti curriculari con un numero minimo di CFU in specifici settori disciplinari, "
        "adeguata conoscenza della lingua inglese, "
        "e il superamento di una prova di ammissione con punteggio minimo di 60 su 100, "
        "composta da valutazione dei titoli (max 20 punti) e prova scritta (max 80 punti).",
        "ground_truth_contexts": [
            "https://corsi.unisa.it/information-Engineering-for-digital-medicine/immatricolazioni",],
    },
    # --- Studenti attuali ---
    {
        "question": (
            "Quale è il programma del corso di "
            "Natural Language Processing and LLM?"
        ),
        "ground_truth": "Il corso di Natural Language Processing and Large Language Models vale 6 CFU (48 ore: 24 lezione + 24 laboratorio) "
        "ed è erogato in inglese nel secondo semestre del secondo anno del curriculum Artificial Intelligence and Intelligent Robotics "
        "della Laurea Magistrale in Ingegneria Informatica. Prerequisito: esame di Machine Learning. "
        "I contenuti principali sono: fondamenti di NLP (tokenizzazione, POS-tagging, NER, word embeddings, RNN/LSTM), "
        "fondamenti degli LLM (fine-tuning, prompt engineering, HuggingFace, LangChain), "
        "RAG e agenti (chatbot con LangChain, advanced RAG). "
        "Esame scritto e orale con voto unico. Appelli: 28/05/2026, 08/07/2026, 28/08/2026.",
        "ground_truth_contexts": [
            "https://docenti.unisa.it/026557/didattica?anno=2025&id=518961&cId=10001-2022&pId=N0*N0*S2"],
    },
    {
        "question": "Quali sono gli orari di ricevimento del Professor Capuano?",
        "ground_truth": "Il Professor Nicola Capuano riceve il mercoledi dalle 11:00 alle 13:00.",
        "ground_truth_contexts": ["https://docenti.unisa.it/004322/home"],
    },
    {
        "question": "Dove si trova l'aula 126?",
        "ground_truth": "L'aula 126 e situata nel Campus di Fisciano, edificio STEC-06 (edificio E), piano P01, stanza 007.",
        "ground_truth_contexts": ["https://corsi.unisa.it/uploads/rescue/499/2693/aule-didattiche.pdf"],
    },
    {
        "question": (
            "Quando si svolge l'esame di "
            "Artificial Intelligence for Cybersecurity?"
        ),
        "ground_truth": "L'esame di Artificial Intelligence for Cybersecurity ha tre appelli nel secondo semestre 2025/2026: "
        "16 giugno 2026 alle 09:00 (sessione ordinaria), "
        "23 luglio 2026 alle 14:00 (sessione ordinaria), "
        "3 settembre 2026 alle 09:00 (sessione di recupero).",
        "ground_truth_contexts": ["https://docenti.unisa.it/026557/didattica?id=515134&cId=9999-2022&pId=N0*N0*S2"],
    },
    {
        "question": (
            "Quando si svolgono le lezioni di "
            "Natural Language Processing and LLM?"
        ),
        "ground_truth": "Le lezioni di Natural Language Processing and Large Language Models si svolgono "
        "il martedi dalle 10:30 alle 13:30 e il giovedi dalle 11:30 alle 13:30, in Aula F (Edificio E1).",
        "ground_truth_contexts": ["https://easycourse.unisa.it/EasyCourse/Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2025-2026/1337/Curricula/INGEGNERIAINFORMATICA_CORSODILAUREAMAGISTRALE_06227_2.html",
                                  "https://easycourse.unisa.it/EasyCourse/Orario/Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata/2025-2026/1337/ttTeacherHtml.html",],
    },
    # --- Studenti internazionali ---
    {
        "question": "Chi è il responsabile dell'internazionalizzazione al DIEM?",
        "ground_truth": "Il responsabile dell'internazionalizzazione del DIEM e il Prof. Diego Gragnaniello.",
        "ground_truth_contexts": ["https://www.diem.unisa.it/uploads/rescue/472/3582/erasmus-presentazione-offerta.pdf",
                                  "https://www.diem.unisa.it/international/contatti",
                                  ],
    },
    {
        "question": (
            "Quali opportunità di mobilità internazionale sono disponibili?"
        ),
        "ground_truth": "Il DIEM offre le seguenti opportunita di mobilita internazionale: "
        "Erasmus+ per studio (fino a 12 mesi per ciclo), Erasmus+ per traineeship (tirocinio all'estero), "
        "doppio titolo con ENSICAEN (Francia), co-tutela di dottorato con universita estere, "
        "International Credit Mobility per singoli insegnamenti, Visiting Professors, "
        "mobilita per docenza e mobilita per ricerca. "
        "Tutte le opportunita sono gestite dal delegato all'internazionalizzazione.",
        "ground_truth_contexts": ["https://www.diem.unisa.it/international/international-mobility",
                                  "https://corsi.unisa.it/ingegneria-informatica/internazionalizzazione",
                                  "https://www.diem.unisa.it/international",
                                  "https://www.diem.unisa.it/uploads/rescue/472/3582/erasmus-presentazione-offerta.pdf"
                                  "https://corsi.unisa.it/ingegneria-informatica-magistrale/internazionalizzazione",
                                  "https://corsi.unisa.it/information-Engineering-for-digital-medicine/internazionalizzazione",],
    },
    # --- English ---
    {
        "question": "What is DIEM?",
        "ground_truth": "DIEM (Department of Information and Electrical Engineering and Applied Mathematics) "
        "is a department of the University of Salerno that integrates teaching, scientific research "
        "and the third mission (technology transfer and collaboration with public and private institutions). "
        "It is headed by Director Francesco Tortorella and located in Fisciano (SA).",
        "ground_truth_contexts": ["https://www.diem.unisa.it/home/contatti",
                                  "https://www.diem.unisa.it/uploads/rescue/472/734/dpd-2023-diem-1-.pdf",
                                  ],
    },
    {
        "question": "Where is DIEM located?",
        "ground_truth": "DIEM is located at Via Giovanni Paolo II, 132, 84084 Fisciano (SA), Italy.",
        "ground_truth_contexts": ["https://www.diem.unisa.it/home/contatti"],
    },
    # --- Altre ---
    {
        "question": (
            "Chi sono i membri della commissione paritetica del DIEM?"
        ),
        "ground_truth": "I membri della Commissione Paritetica Docenti-Studenti del DIEM sono: "
        "Vincenzo Auletta (Presidente), Emilia Del Pezzo (Vice-Presidente), "
        "Gabriele Di Lorenzo (Membro), Ettore Napoli (Membro), "
        "Giovanni Petrone (Segretario), Alessia Terranova (Membro).",
        "ground_truth_contexts": ["https://www.diem.unisa.it/dipartimento/commissione-paritetica"],
    },
    {
        "question": "Chi è il direttore del dipartimento?",
        "ground_truth": "Il direttore del DIEM e il Prof. Francesco Tortorella.",
        "ground_truth_contexts": [],
    },
    # --- Context chat queries (rewritten as standalone) ---
    {
        "question": "Chi è Antonio Greco?",
        "ground_truth": "Antonio Greco e un Professore Associato del DIEM dell'Universita di Salerno. "
        "E responsabile delle attivita di orientamento in entrata per Ingegneria Informatica "
        "e collabora con il laboratorio Embedded Intelligent Systems (EIS). "
        "Riceve il martedi dalle 9:00 alle 11:00 presso l'Ufficio 365, Edificio E. "
        "Email: agreco@unisa.it, telefono: 089 963 003.",
        "ground_truth_contexts": ["https://docenti.unisa.it/026557"],
    },
    {
        "question": "Quali corsi insegna Antonio Greco?",
        "ground_truth":  "Antonio Greco insegna i seguenti corsi nell'anno accademico 2025: "
        "Advanced Machine Learning (dottorato in Ingegneria dell'Informazione), "
        "Artificial Intelligence for Cybersecurity (LM Ingegneria Informatica), "
        "Autonomous and Connected Vehicles (LM Ingegneria Informatica), "
        "Medical Image Processing and Visualization (LM Information Engineering for Digital Medicine), "
        "Natural Language Processing and Large Language Models (LM Ingegneria Informatica).",
        "ground_truth_contexts": ["https://docenti.unisa.it/026557/didattica"],
    },
]


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_query(graph, query: str) -> tuple[str, list[str]]:
    """
    Runs a single query through the agent graph and returns the final
    answer and the list of retrieved context strings.

    Args:
        graph: The compiled LangGraph agent graph.
        query: The user query string.

    Returns:
        A tuple (answer, contexts) where contexts is a list of strings,
        one per retrieved parent document.
    """
    from langchain_core.messages import HumanMessage

    initial_state = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "chat_history": [],
        "retrieved_context": "",
        "final_answer": "",
        "guardrail_decision": "",
        "iteration_count": 0,
        "retry_count": 0,
    }

    result = graph.invoke(initial_state)

    answer = result.get("final_answer", "")
    raw_context = result.get("retrieved_context", "")

    # Split concatenated context blocks back into individual document strings.
    # Tool outputs are separated by "\n\n---\n\n" in _format_docs.
    contexts = []
    if raw_context:
        blocks = raw_context.split("\n\n---\n\n")
        for block in blocks:
            block = block.strip()
            if block and not block.startswith("FALLBACK"):
                contexts.append(block)

    return answer, contexts


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def build_ragas_dataset(samples: list):
    """
    Converts collected samples into a RAGAS v0.4-compatible EvaluationDataset.
 
    Each sample is converted into a SingleTurnSample using the field mapping
    required by the v0.4 API:
        question            → user_input
        answer              → response
        contexts            → retrieved_contexts
        ground_truth        → reference
 
    Args:
        samples: List of dicts with keys: question, answer, contexts,
                 ground_truth, ground_truth_contexts.
 
    Returns:
        A RAGAS EvaluationDataset instance ready for evaluation.
    """
    from ragas import EvaluationDataset, SingleTurnSample
 
    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        )
        for s in samples
    ]
    return EvaluationDataset(samples=ragas_samples)


def run_ragas_evaluation(dataset, llm_for_ragas, embeddings_for_ragas) -> dict:
    """
    Runs RAGAS evaluation on the dataset.
 
    Metrics:
        - LLMContextPrecisionWithReference  (requires reference)
        - LLMContextRecall                  (requires reference)
        - Faithfulness                      (LLM judge, no reference needed)
        - ResponseRelevancy                 (embedding similarity, no reference needed)
 
    The LLM and embeddings are wrapped in the RAGAS v0.4 wrappers before
    being passed to the metrics. Evaluation is serialized (max_workers=1)
    to stay within the rate limit of free-tier LLM providers. Each failed
    job is retried up to 10 times with a maximum wait of 60 seconds before
    being recorded as NaN.
 
    Args:
        dataset:              RAGAS EvaluationDataset instance.
        llm_for_ragas:        LangChain chat model instance used by RAGAS as judge.
        embeddings_for_ragas: LangChain embeddings instance used by RAGAS.
 
    Returns:
        RAGAS EvaluationResult object (supports .to_pandas()).
    """
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig
 
    ragas_llm = LangchainLLMWrapper(llm_for_ragas)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_for_ragas)
 
    metrics = [
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]
 
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(
            max_workers=16,
            max_wait=60,
            max_retries=5,
            timeout=300
        ),
    )
    return result

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary_bar(df, output_path: str) -> None:
    """
    Produces a bar chart of mean metric scores across all queries.

    Args:
        df:          pandas DataFrame from RAGAS result.to_pandas().
        output_path: File path for the output PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metric_cols = [
        "llm_context_precision_with_reference",
        "llm_context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    means = df[metric_cols].mean()

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    colors = colors[:len(metric_cols)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        [c.replace("_", "\n") for c in metric_cols],
        means.values,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )

    # Value labels on top of each bar.
    for bar, val in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#222222",
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (0–1)", fontsize=12)
    ax.set_title("RAGAS Evaluation — Mean Metric Scores", fontsize=14, fontweight="bold", pad=16)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.8 threshold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Bar chart saved to {output_path}")


def plot_heatmap(df, output_path: str) -> None:
    """
    Produces a per-query heatmap of all metric scores.

    Args:
        df:          pandas DataFrame from RAGAS result.to_pandas().
        output_path: File path for the output PNG.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    metric_cols = [
        "llm_context_precision_with_reference",
        "llm_context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # Shorten questions for display.
    questions = []
    for q in df["user_input"].tolist():
        short = q[:55] + "…" if len(q) > 55 else q
        questions.append(short)

    data = df[metric_cols].values
    n_queries, n_metrics = data.shape

    fig_height = max(6, n_queries * 0.42)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Colorbar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Axis labels.
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in metric_cols],
        fontsize=10,
        fontweight="bold",
    )
    ax.set_yticks(range(n_queries))
    ax.set_yticklabels(questions, fontsize=8)

    # Cell annotations.
    for i in range(n_queries):
        for j in range(n_metrics):
            val = data[i, j]
            text_color = "black" if 0.35 < val < 0.75 else "white"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold",
            )

    ax.set_title(
        "RAGAS Evaluation — Per-Query Scores",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Heatmap saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=== DIEM Agentic RAG: RAGAS Evaluation ===\n")

    results_path = "evaluation_results.json"

    # ------------------------------------------------------------------
    # Check ground truths are filled
    # ------------------------------------------------------------------
    missing_gt = [
        i for i, s in enumerate(EVAL_DATASET) if not s["ground_truth"].strip()
    ]
    if missing_gt:
        print(
            f"[Eval] WARNING: {len(missing_gt)} queries have empty ground_truth. "
            "context_precision and context_recall will be unreliable for those entries.\n"
            f"  Missing indices: {missing_gt}\n"
        )

    # ------------------------------------------------------------------
    # Run queries and collect results (skipped if cache exists)
    # ------------------------------------------------------------------
    if os.path.exists(results_path):
        print(f"[Eval] Found existing results at '{results_path}'. Skipping query execution.")
        with open(results_path, "r", encoding="utf-8") as fh:
            samples = json.load(fh)
        print(f"[Eval] Loaded {len(samples)} samples from cache.\n")
    else:
        print("[Eval] Initializing system...")
        from core import initialize_system
        graph = initialize_system()
        print("[Eval] System ready.\n")

        samples = []
        total = len(EVAL_DATASET)

        for i, entry in enumerate(EVAL_DATASET, 1):
            question = entry["question"]
            print(f"[Eval] [{i}/{total}] Query: {question[:70]}")

            try:
                answer, contexts = run_query(graph, question)
            except Exception as exc:
                print(f"[Eval]   ERROR: {exc}")
                answer = ""
                contexts = []

            samples.append({
                "question":              question,
                "answer":                answer,
                "contexts":              contexts if contexts else [""],
                "ground_truth":          entry["ground_truth"],
                "ground_truth_contexts": entry.get("ground_truth_contexts", []),
            })

            print(f"[Eval]   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"[Eval]   Contexts retrieved: {len(contexts)}")
            time.sleep(1)  # Avoid rate limiting on OpenRouter

        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(samples, fh, ensure_ascii=False, indent=2)
        print(f"\n[Eval] Raw results saved to '{results_path}'")

    # ------------------------------------------------------------------
    # RAGAS evaluation
    # ------------------------------------------------------------------
    print("\n[Eval] Running RAGAS evaluation...")

    try:
        import ragas
    except ImportError:
        print("[Eval] ERROR: ragas not installed. Run: pip install ragas")
        sys.exit(1)

    import config
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    ragas_llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME_RAGAS,
        temperature=0,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        max_tokens=4096,
        n=1,
    )
    print(f"[Eval] Using LLM judge: {config.LLM_MODEL_NAME_RAGAS} via OpenRouter.")

    ragas_embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME_RAGAS,
        cache_folder=config.MODELS_DIR,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )
    print(f"[Eval] Using embeddings: {config.EMBEDDING_MODEL_NAME_RAGAS} (local).")

    dataset = build_ragas_dataset(samples)
    ragas_result = run_ragas_evaluation(dataset, ragas_llm, ragas_embeddings)

    df = ragas_result.to_pandas()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    metric_cols = [
        "llm_context_precision_with_reference",
        "llm_context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    print("\n=== RAGAS Results Summary ===")
    print(f"{'Metric':<25} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 52)
    for col in metric_cols:
        print(
            f"{col:<25} "
            f"{df[col].mean():>8.3f} "
            f"{df[col].min():>8.3f} "
            f"{df[col].max():>8.3f}"
        )
    print("=" * 52)

    # ------------------------------------------------------------------
    # Save scores alongside raw results
    # ------------------------------------------------------------------
    df.to_csv("evaluation_scores.csv", index=False)
    print("[Eval] Scores saved to evaluation_scores.csv")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        plot_summary_bar(df, "evaluation_summary.png")
        plot_heatmap(df, "evaluation_heatmap.png")
    except ImportError:
        print("[Eval] matplotlib not installed — skipping plots. Run: pip install matplotlib")

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()