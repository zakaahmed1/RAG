import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from app.config import (
    FETCH_K,
    MIN_SIMILARITY,
    MMR_LAMBDA_MULT,
    RETRIEVAL_MODE,
    TOP_K,
)

from app.evaluation.metrics import (
    ABSTENTION_TEXT,
    evidence_recall,
    hit_at_k,
    is_abstention,
    keyword_coverage,
    reciprocal_rank,
    relevant_rank,
)

from app.generation.generator import (
    create_generator,
    generate_answer,
)

from app.retrieval.search import (
    retrieve_documents,
)

from app.retrieval.vector_store import (
    load_vector_store,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

def load_dataset(split=None):
    """
    Load the labelled evaluation dataset.

    If split is supplied, only questions belonging to
    that split are returned.

    Supported split values:
        tune
        test
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if split is None:
        return dataset

    filtered_dataset = [
        item
        for item in dataset
        if item.get("split") == split
    ]

    if not filtered_dataset:
        raise ValueError(
            f"No evaluation questions found "
            f"for split '{split}'."
        )

    return filtered_dataset


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def safe_mean(values):
    """
    Calculate a mean while ignoring None values.
    """

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return mean(values)


def format_retrieved_sources(chunks):
    """
    Convert retrieved chunks into JSON-serialisable
    source metadata.
    """

    sources = []

    for chunk in chunks:

        metadata = chunk.document.metadata

        sources.append(
            {
                "source": metadata.get(
                    "file_name"
                ),
                "page": metadata.get(
                    "page_number"
                ),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "similarity": round(
                    chunk.similarity,
                    4,
                ),
            }
        )

    return sources


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(
    *,
    mode=RETRIEVAL_MODE,
    top_k=TOP_K,
    fetch_k=FETCH_K,
    min_similarity=MIN_SIMILARITY,
    mmr_lambda=MMR_LAMBDA_MULT,
    retrieval_only=False,
    split=None,
):
    """
    Evaluate the RAG system against the labelled
    benchmark dataset.

    Parameters
    ----------
    mode:
        Retrieval strategy. Expected values are
        "similarity" or "mmr".

    top_k:
        Number of chunks returned after retrieval.

    fetch_k:
        Number of initial candidate chunks considered.

    min_similarity:
        Minimum cosine-similarity threshold.

    mmr_lambda:
        Relevance/diversity weighting used by MMR.

    retrieval_only:
        If True, evaluate retrieval without running
        the language model.

    split:
        Optional benchmark split:
            "tune"
            "test"
    """

    dataset = load_dataset(
        split=split
    )

    print(
        f"Loaded {len(dataset)} evaluation questions"
        + (
            f" from split '{split}'."
            if split
            else "."
        )
    )

    # -----------------------------------------------------
    # Load vector store
    # -----------------------------------------------------

    vector_store = load_vector_store()

    # -----------------------------------------------------
    # Load generator only when required
    # -----------------------------------------------------

    generator = None

    if not retrieval_only:
        generator = create_generator()

    results = []

    # -----------------------------------------------------
    # Evaluate each question
    # -----------------------------------------------------

    for item in dataset:

        question_id = item["id"]
        question = item["question"]
        supported = item["supported"]

        category = item.get(
            "category",
            "uncategorized",
        )

        evidence_requirement = item.get(
            "evidence_requirement"
        )

        expected_evidence = item.get(
            "expected_evidence",
            [],
        )

        expected_keywords = item.get(
            "expected_answer_keywords",
            [],
        )

        # -------------------------------------------------
        # Retrieve evidence
        # -------------------------------------------------

        chunks = retrieve_documents(
            vector_store,
            question,
            mode=mode,
            top_k=top_k,
            fetch_k=fetch_k,
            min_similarity=min_similarity,
            mmr_lambda=mmr_lambda,
        )

        # -------------------------------------------------
        # Generate answer if required
        # -------------------------------------------------

        answer = ""

        if not retrieval_only:

            if chunks:

                answer = generate_answer(
                    generator=generator,
                    question=question,
                    retrieved_chunks=chunks,
                )

            else:

                answer = ABSTENTION_TEXT

        # -------------------------------------------------
        # General retrieval information
        # -------------------------------------------------

        top_similarity = (
            chunks[0].similarity
            if chunks
            else None
        )

        sources = format_retrieved_sources(
            chunks
        )

        row = {
            "id": question_id,
            "category": category,
            "split": item.get("split"),
            "question": question,
            "supported": supported,
            "evidence_requirement":
                evidence_requirement,
            "retrieved_count": len(chunks),
            "top_similarity": top_similarity,
            "sources": sources,
            "answer": answer,
        }

        # -------------------------------------------------
        # Supported questions
        # -------------------------------------------------

        if supported:

            if evidence_requirement not in {
                "any",
                "all",
            }:
                raise ValueError(
                    f"Question {question_id} is "
                    f"supported but has invalid "
                    f"evidence_requirement: "
                    f"{evidence_requirement}"
                )

            if not expected_evidence:
                raise ValueError(
                    f"Question {question_id} is "
                    f"supported but contains no "
                    f"expected_evidence."
                )

            row["hit_at_k"] = hit_at_k(
                chunks,
                expected_evidence,
                evidence_requirement,
            )

            row["evidence_recall"] = (
                evidence_recall(
                    chunks,
                    expected_evidence,
                )
            )

            row["reciprocal_rank"] = (
                reciprocal_rank(
                    chunks,
                    expected_evidence,
                )
            )

            row["relevant_rank"] = (
                relevant_rank(
                    chunks,
                    expected_evidence,
                )
            )

            row["retrieval_rejected"] = False

            if retrieval_only:

                row["keyword_coverage"] = None
                row["answer_abstained"] = None

            else:

                row["keyword_coverage"] = (
                    keyword_coverage(
                        answer,
                        expected_keywords,
                    )
                )

                row["answer_abstained"] = (
                    is_abstention(
                        answer
                    )
                )

        # -------------------------------------------------
        # Unsupported questions
        # -------------------------------------------------

        else:

            if evidence_requirement not in {
                None,
                "none",
            }:
                raise ValueError(
                    f"Question {question_id} is "
                    f"unsupported but has "
                    f"evidence_requirement "
                    f"'{evidence_requirement}'."
                )

            row["hit_at_k"] = None
            row["evidence_recall"] = None
            row["reciprocal_rank"] = None
            row["relevant_rank"] = None
            row["keyword_coverage"] = None

            row["retrieval_rejected"] = (
                len(chunks) == 0
            )

            if retrieval_only:

                row["answer_abstained"] = None

            else:

                row["answer_abstained"] = (
                    is_abstention(
                        answer
                    )
                )

        results.append(row)

    return results


# ---------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------

def build_summary(
    results,
    retrieval_only,
):
    """
    Calculate overall and category-level metrics.
    """

    supported = [
        row
        for row in results
        if row["supported"]
    ]

    unsupported = [
        row
        for row in results
        if not row["supported"]
    ]

    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    summary = {
        "questions": len(results),

        "supported_questions":
            len(supported),

        "unsupported_questions":
            len(unsupported),

        "hit_rate_at_k":
            safe_mean(
                [
                    row["hit_at_k"]
                    for row in supported
                ]
            ),

        "evidence_recall":
            safe_mean(
                [
                    row["evidence_recall"]
                    for row in supported
                ]
            ),

        "mrr":
            safe_mean(
                [
                    row["reciprocal_rank"]
                    for row in supported
                ]
            ),

        "unsupported_rejection_accuracy":
            safe_mean(
                [
                    int(
                        row[
                            "retrieval_rejected"
                        ]
                    )
                    for row
                    in unsupported
                ]
            ),
    }

    # -----------------------------------------------------
    # Generation metrics
    # -----------------------------------------------------

    if not retrieval_only:

        summary[
            "supported_answer_keyword_coverage"
        ] = safe_mean(
            [
                row["keyword_coverage"]
                for row in supported
            ]
        )

        summary[
            "supported_answer_abstention_rate"
        ] = safe_mean(
            [
                int(
                    row["answer_abstained"]
                )
                for row in supported
            ]
        )

        summary[
            "unsupported_answer_abstention_accuracy"
        ] = safe_mean(
            [
                int(
                    row["answer_abstained"]
                )
                for row
                in unsupported
            ]
        )

    # -----------------------------------------------------
    # Category-level metrics
    # -----------------------------------------------------

    categories = sorted(
        {
            row["category"]
            for row in results
        }
    )

    category_metrics = {}

    for category in categories:

        category_rows = [
            row
            for row in results
            if row["category"] == category
        ]

        category_supported = [
            row
            for row in category_rows
            if row["supported"]
        ]

        category_unsupported = [
            row
            for row in category_rows
            if not row["supported"]
        ]

        metrics = {
            "questions":
                len(category_rows),

            "supported_questions":
                len(category_supported),

            "unsupported_questions":
                len(category_unsupported),
        }

        # ---------------------------------------------
        # Supported-category metrics
        # ---------------------------------------------

        if category_supported:

            metrics[
                "hit_rate_at_k"
            ] = safe_mean(
                [
                    row["hit_at_k"]
                    for row
                    in category_supported
                ]
            )

            metrics[
                "evidence_recall"
            ] = safe_mean(
                [
                    row["evidence_recall"]
                    for row
                    in category_supported
                ]
            )

            metrics[
                "mrr"
            ] = safe_mean(
                [
                    row["reciprocal_rank"]
                    for row
                    in category_supported
                ]
            )

            if not retrieval_only:

                metrics[
                    "answer_keyword_coverage"
                ] = safe_mean(
                    [
                        row["keyword_coverage"]
                        for row
                        in category_supported
                    ]
                )

                metrics[
                    "answer_abstention_rate"
                ] = safe_mean(
                    [
                        int(
                            row[
                                "answer_abstained"
                            ]
                        )
                        for row
                        in category_supported
                    ]
                )

        # ---------------------------------------------
        # Unsupported-category metrics
        # ---------------------------------------------

        if category_unsupported:

            metrics[
                "unsupported_rejection_accuracy"
            ] = safe_mean(
                [
                    int(
                        row[
                            "retrieval_rejected"
                        ]
                    )
                    for row
                    in category_unsupported
                ]
            )

            if not retrieval_only:

                metrics[
                    "answer_abstention_accuracy"
                ] = safe_mean(
                    [
                        int(
                            row[
                                "answer_abstained"
                            ]
                        )
                        for row
                        in category_unsupported
                    ]
                )

        category_metrics[
            category
        ] = metrics

    summary["categories"] = (
        category_metrics
    )

    return summary


# ---------------------------------------------------------
# Results persistence
# ---------------------------------------------------------

def build_result_prefix(
    split,
    retrieval_only,
):
    """
    Create a unique filename prefix so tune/test/full
    runs do not overwrite each other.
    """

    split_name = (
        split
        if split
        else "all"
    )

    mode_name = (
        "retrieval_only"
        if retrieval_only
        else "full"
    )

    return (
        f"evaluation_"
        f"{split_name}_"
        f"{mode_name}"
    )


def save_results(
    results,
    summary,
    *,
    split=None,
    retrieval_only=False,
):
    """
    Save detailed evaluation results and summary metrics.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = build_result_prefix(
        split=split,
        retrieval_only=retrieval_only,
    )

    json_path = (
        RESULTS_DIR
        / f"{prefix}_results.json"
    )

    summary_path = (
        RESULTS_DIR
        / f"{prefix}_summary.json"
    )

    csv_path = (
        RESULTS_DIR
        / f"{prefix}_results.csv"
    )

    # -----------------------------------------------------
    # Detailed JSON results
    # -----------------------------------------------------

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # -----------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # -----------------------------------------------------
    # CSV results
    # -----------------------------------------------------

    csv_rows = []

    for row in results:

        csv_row = row.copy()

        csv_row["sources"] = json.dumps(
            csv_row["sources"]
        )

        csv_rows.append(
            csv_row
        )

    if csv_rows:

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=csv_rows[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                csv_rows
            )

    return {
        "results_json": json_path,
        "summary_json": summary_path,
        "results_csv": csv_path,
    }


# ---------------------------------------------------------
# Console output
# ---------------------------------------------------------

def print_metric(
    name,
    value,
    indent=0,
):
    """
    Nicely print one metric.
    """

    prefix = " " * indent

    if isinstance(
        value,
        float,
    ):
        print(
            f"{prefix}{name}: "
            f"{value:.3f}"
        )

    else:
        print(
            f"{prefix}{name}: "
            f"{value}"
        )


def print_summary(summary):
    """
    Print overall and category-level metrics.
    """

    print(
        "\nEvaluation complete.\n"
    )

    print(
        "--- Overall metrics ---"
    )

    for key, value in summary.items():

        if key == "categories":
            continue

        print_metric(
            key,
            value,
        )

    print(
        "\n--- Category metrics ---"
    )

    for category, metrics in (
        summary["categories"].items()
    ):

        print(
            f"\n{category}"
        )

        for key, value in (
            metrics.items()
        ):

            print_metric(
                key,
                value,
                indent=2,
            )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the RAG assistant "
            "against the labelled benchmark."
        )
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Evaluate retrieval without "
            "running the language model."
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "tune",
            "test",
        ],
        default=None,
        help=(
            "Evaluate only the specified "
            "benchmark split."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------

    results = evaluate(
        retrieval_only=(
            args.retrieval_only
        ),
        split=args.split,
    )

    # -----------------------------------------------------
    # Build summary
    # -----------------------------------------------------

    summary = build_summary(
        results,
        retrieval_only=(
            args.retrieval_only
        ),
    )

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------

    paths = save_results(
        results,
        summary,
        split=args.split,
        retrieval_only=(
            args.retrieval_only
        ),
    )

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print_summary(
        summary
    )

    print(
        "\n--- Saved results ---"
    )

    for name, path in paths.items():

        print(
            f"{name}: "
            f"{path}"
        )


if __name__ == "__main__":
    main()