import argparse
import json

from transformers import AutoTokenizer

from app.config import (
    GENERATOR_MODEL_NAME,
    GENERATOR_MODEL_REVISION,
)
from app.evaluation.evaluate import (
    RESULTS_DIR,
    build_prompt_usage_summary,
    build_run_provenance,
    load_dataset,
)
from app.generation.generator import (
    build_context,
    build_prompt,
    prepare_generator_inputs,
)
from app.retrieval.search import (
    retrieve_documents,
)
from app.retrieval.vector_store import (
    load_vector_store,
)


def audit_prompts(split=None):
    """Measure real retrieval prompts without loading the generator model."""

    dataset = load_dataset(split=split)
    vector_store = load_vector_store()
    tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_MODEL_NAME,
        revision=GENERATOR_MODEL_REVISION,
    )

    records = []

    for item in dataset:
        chunks = retrieve_documents(
            vector_store,
            item["question"],
        )
        token_usage = None

        if chunks:
            context = build_context(chunks)
            prompt = build_prompt(
                question=item["question"],
                context=context,
            )
            _, token_usage = prepare_generator_inputs(
                tokenizer=tokenizer,
                prompt=prompt,
                context=context,
                question=item["question"],
            )

        records.append(
            {
                "id": item["id"],
                "split": item.get("split"),
                "supported": item["supported"],
                "retrieved_count": len(chunks),
                "prompt_token_usage": token_usage,
            }
        )

    summary = build_prompt_usage_summary(records)
    summary.update(
        {
            "questions": len(records),
            "retrieval_rejections": sum(
                record["retrieved_count"] == 0
                for record in records
            ),
        }
    )

    provenance = build_run_provenance(
        retrieval_only=True,
        split=split,
    )
    provenance["evaluation"]["type"] = (
        "prompt_token_audit"
    )

    return {
        "summary": summary,
        "provenance": provenance,
        "questions": records,
    }


def save_audit(report, split=None):
    """Persist a reproducible JSON prompt-audit report."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    split_name = split or "all"
    output_path = (
        RESULTS_DIR
        / f"prompt_token_audit_{split_name}.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit generator prompt truncation without "
            "loading or running the language model."
        )
    )
    parser.add_argument(
        "--split",
        choices=("tune", "test"),
        default=None,
    )
    args = parser.parse_args()

    report = audit_prompts(
        split=args.split
    )
    output_path = save_audit(
        report,
        split=args.split,
    )

    print(
        json.dumps(
            report["summary"],
            indent=2,
        )
    )
    print(f"Saved prompt audit: {output_path}")


if __name__ == "__main__":
    main()
