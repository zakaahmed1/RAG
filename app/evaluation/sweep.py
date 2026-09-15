import csv
import json
from pathlib import Path

from app.evaluation.evaluate import (
    build_run_provenance,
    build_summary,
    evaluate,
)
from app.retrieval.vector_store import (
    load_vector_store,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


# ---------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------

THRESHOLDS = [
    0.30,
    0.35,
    0.375,
    0.40,
    0.425,
    0.45,
    0.475,
    0.50,
]

TOP_K_VALUES = [
    3,
    4,
    5,
    6,
]

FETCH_K_VALUES = [
    12,
    20,
    30,
]

RETRIEVAL_MODES = [
    "similarity",
    "mmr",
]

MMR_LAMBDA = 0.7


def build_sweep_provenance(total_configurations):
    """Record the code and complete parameter grid for a sweep."""

    provenance = build_run_provenance(
        retrieval_only=True,
        split="tune",
    )
    provenance["evaluation"].update(
        {
            "type": "retrieval_parameter_sweep",
            "total_configurations": total_configurations,
            "thresholds": THRESHOLDS,
            "top_k_values": TOP_K_VALUES,
            "fetch_k_values": FETCH_K_VALUES,
            "retrieval_modes": RETRIEVAL_MODES,
            "mmr_lambda": MMR_LAMBDA,
            "mmr_threshold_backfill": True,
        }
    )
    return provenance


# ---------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------

def run_sweep():
    """
    Run retrieval-only evaluation across a grid of
    retrieval configurations using the tuning split.
    """

    sweep_results = []

    total_configurations = (
        len(THRESHOLDS)
        * len(TOP_K_VALUES)
        * len(FETCH_K_VALUES)
        * len(RETRIEVAL_MODES)
    )

    current_configuration = 0
    vector_store = load_vector_store()

    print(
        f"Running {total_configurations} "
        f"retrieval configurations...\n"
    )

    # -----------------------------------------------------
    # Parameter grid
    # -----------------------------------------------------

    for mode in RETRIEVAL_MODES:

        for top_k in TOP_K_VALUES:

            for fetch_k in FETCH_K_VALUES:

                for threshold in THRESHOLDS:

                    current_configuration += 1

                    print(
                        f"[{current_configuration}/"
                        f"{total_configurations}] "
                        f"mode={mode}, "
                        f"top_k={top_k}, "
                        f"fetch_k={fetch_k}, "
                        f"threshold={threshold:.3f}"
                    )

                    # -------------------------------------
                    # Run retrieval-only evaluation
                    # -------------------------------------

                    results = evaluate(
                        mode=mode,
                        top_k=top_k,
                        fetch_k=fetch_k,
                        min_similarity=threshold,
                        mmr_lambda=MMR_LAMBDA,
                        retrieval_only=True,
                        split="tune",
                        vector_store=vector_store,
                    )

                    # -------------------------------------
                    # Build metrics
                    # -------------------------------------

                    summary = build_summary(
                        results,
                        retrieval_only=True,
                    )

                    hit_rate = (
                        summary[
                            "hit_rate_at_k"
                        ]
                        or 0
                    )

                    evidence_recall = (
                        summary[
                            "evidence_recall"
                        ]
                        or 0
                    )

                    mrr = (
                        summary[
                            "mrr"
                        ]
                        or 0
                    )

                    rejection_accuracy = (
                        summary[
                            "unsupported_rejection_accuracy"
                        ]
                        or 0
                    )

                    # -------------------------------------
                    # Composite tuning score
                    # -------------------------------------

                    balanced_score = (
                        hit_rate
                        + evidence_recall
                        + rejection_accuracy
                    ) / 3

                    # -------------------------------------
                    # Capture category metrics
                    # -------------------------------------

                    category_metrics = summary.get(
                        "categories",
                        {},
                    )

                    cross_policy_metrics = (
                        category_metrics.get(
                            "cross_policy_multi_document",
                            {},
                        )
                    )

                    unsupported_metrics = (
                        category_metrics.get(
                            "unsupported",
                            {},
                        )
                    )

                    cross_policy_hit_rate = (
                        cross_policy_metrics.get(
                            "hit_rate_at_k"
                        )
                    )

                    cross_policy_evidence_recall = (
                        cross_policy_metrics.get(
                            "evidence_recall"
                        )
                    )

                    unsupported_category_rejection = (
                        unsupported_metrics.get(
                            "unsupported_rejection_accuracy"
                        )
                    )

                    # -------------------------------------
                    # Save row
                    # -------------------------------------

                    sweep_results.append(
                        {
                            "mode": mode,
                            "top_k": top_k,
                            "fetch_k": fetch_k,
                            "threshold": threshold,
                            "mmr_lambda": (
                                MMR_LAMBDA
                                if mode == "mmr"
                                else None
                            ),
                            "hit_rate_at_k":
                                hit_rate,
                            "evidence_recall":
                                evidence_recall,
                            "mrr":
                                mrr,
                            "unsupported_rejection_accuracy":
                                rejection_accuracy,
                            "cross_policy_hit_rate":
                                cross_policy_hit_rate,
                            "cross_policy_evidence_recall":
                                cross_policy_evidence_recall,
                            "unsupported_category_rejection_accuracy":
                                unsupported_category_rejection,
                            "balanced_score":
                                balanced_score,
                        }
                    )

    # -----------------------------------------------------
    # Rank configurations
    # -----------------------------------------------------

    sweep_results.sort(
        key=lambda row: (
            row["balanced_score"],
            row["hit_rate_at_k"],
            row["evidence_recall"],
            row["mrr"],
            row["unsupported_rejection_accuracy"],
        ),
        reverse=True,
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "retrieval_sweep.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=sweep_results[0].keys(),
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            sweep_results
        )

    provenance_path = (
        RESULTS_DIR
        / "retrieval_sweep_provenance.json"
    )
    provenance_path.write_text(
        json.dumps(
            build_sweep_provenance(
                total_configurations
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print top configurations
    # -----------------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "Top retrieval configurations"
    )

    print(
        "========================================\n"
    )

    for rank, row in enumerate(
        sweep_results[:15],
        start=1,
    ):

        print(
            f"{rank:02d}. "
            f"{row['mode']:10} "
            f"k={row['top_k']} "
            f"fetch={row['fetch_k']} "
            f"threshold="
            f"{row['threshold']:.3f} "
            f"hit="
            f"{row['hit_rate_at_k']:.3f} "
            f"recall="
            f"{row['evidence_recall']:.3f} "
            f"MRR="
            f"{row['mrr']:.3f} "
            f"rejection="
            f"{row['unsupported_rejection_accuracy']:.3f} "
            f"balanced="
            f"{row['balanced_score']:.3f}"
        )

    print(
        "\n"
        "Sweep results saved to:"
    )

    print(
        output_path
    )

    print(
        "Sweep provenance saved to:"
    )

    print(
        provenance_path
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_sweep()
