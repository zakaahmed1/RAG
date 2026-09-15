import re

from app.generation.generator import (
    build_prompt,
    prepare_generator_inputs,
)


class FakeTokenizer:
    """Small tokenizer double with deterministic word offsets."""

    truncation_side = "right"

    def __init__(self, model_max_length):
        self.model_max_length = model_max_length
        self.vocabulary = {}

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=False,
        return_tensors=None,
    ):
        matches = list(re.finditer(r"\S+", text))
        tokens = [match.group() for match in matches]
        token_ids = [
            self.vocabulary.setdefault(
                token,
                len(self.vocabulary) + 1,
            )
            for token in tokens
        ]
        offsets = [
            (match.start(), match.end())
            for match in matches
        ]

        if add_special_tokens:
            token_ids.append(0)
            offsets.append((0, 0))

        if truncation:
            token_ids = token_ids[:self.model_max_length]
            offsets = offsets[:self.model_max_length]

        result = {
            "input_ids": (
                [token_ids]
                if return_tensors
                else token_ids
            )
        }

        if return_offsets_mapping:
            result["offset_mapping"] = offsets

        return result


def test_prompt_diagnostics_detect_context_and_question_truncation():
    tokenizer = FakeTokenizer(model_max_length=40)
    context = " ".join(
        f"context-{index}"
        for index in range(80)
    )
    question = "Can the employee wait until tomorrow?"
    prompt = build_prompt(
        question=question,
        context=context,
    )

    _, diagnostics = prepare_generator_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        context=context,
        question=question,
    )

    assert diagnostics["prompt_truncated"] is True
    assert diagnostics["truncated_tokens"] > 0
    assert diagnostics["context_fully_retained"] is False
    assert diagnostics["question_fully_retained"] is False


def test_prompt_diagnostics_confirm_fully_retained_sections():
    tokenizer = FakeTokenizer(model_max_length=500)
    context = "Lost devices must be reported within two hours."
    question = "When must a lost device be reported?"
    prompt = build_prompt(
        question=question,
        context=context,
    )

    inputs, diagnostics = prepare_generator_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        context=context,
        question=question,
    )

    assert diagnostics["prompt_truncated"] is False
    assert diagnostics["truncated_tokens"] == 0
    assert diagnostics["context_fully_retained"] is True
    assert diagnostics["question_fully_retained"] is True
    assert (
        diagnostics["tokens_after_truncation"]
        == len(inputs["input_ids"][0])
    )
