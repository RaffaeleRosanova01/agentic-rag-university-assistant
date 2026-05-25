"""
Knowledge base update script for the DIEM Agentic RAG project.

Runs the full ingestion pipeline in order:
    1. crawler     — crawls all authorized domains and harvests documents
    2. pdf_filter  — removes unwanted PDFs and deduplicates
    3. chunker     — splits documents into parent/child chunks (incremental)
    4. indexer     — embeds child chunks and upserts into ChromaDB (incremental)

Usage:
    Run once and exit:
        python update_knowledge_base.py

    Run once and then repeat weekly:
        python update_knowledge_base.py --schedule

The --schedule flag activates a weekly loop. Without it, the pipeline
runs exactly once and the process exits.
"""

import argparse
import logging
import sys
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("DIEM_Update")

WEEKLY_INTERVAL_SECONDS = 7 * 24 * 60 * 60


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> bool:
    """
    Executes the full ingestion pipeline in order.

    Returns:
        True if all steps completed without raising an exception,
        False if any step failed.
    """
    steps = [
        ("crawler",    "ingestion.crawler",    "run"),
        ("pdf_filter", "ingestion.pdf_filter", "run"),
        ("chunker",    "ingestion.chunker",    "run"),
        ("indexer",    "ingestion.indexer",    "run"),
    ]

    for step_name, module_path, func_name in steps:
        logger.info(f"Starting step: {step_name}")
        try:
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            func()
            logger.info(f"Step completed: {step_name}")
        except Exception as exc:
            logger.error(f"Step '{step_name}' failed: {exc}")
            return False

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DIEM Agentic RAG — knowledge base update script.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help=(
            "After the first run, repeat the pipeline every 7 days. "
            "Without this flag, the pipeline runs once and exits."
        ),
    )
    args = parser.parse_args()

    if not args.schedule:
        logger.info("Running pipeline once.")
        success = run_pipeline()
        if not success:
            logger.error("Pipeline failed. Exiting with error.")
            sys.exit(1)
        logger.info("Pipeline completed successfully.")
        return

    logger.info("Scheduled mode active. Pipeline will run every 7 days.")

    while True:
        start = datetime.now()
        logger.info(f"Pipeline run started at {start.strftime('%Y-%m-%d %H:%M:%S')}.")

        success = run_pipeline()

        end = datetime.now()
        duration = end - start
        if success:
            logger.info(
                f"Pipeline completed in {duration}. "
                f"Next run in 7 days."
            )
        else:
            logger.error(
                f"Pipeline failed after {duration}. "
                f"Next attempt in 7 days."
            )

        time.sleep(WEEKLY_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()