from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import (
    GENERATOR_MODEL_NAME,
    MAX_NEW_TOKENS,
)


def create_generator():
    """
    Load the tokenizer and sequence-to-sequence language model.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATOR_MODEL_NAME
    )

    return tokenizer, model


def build_context(retrieved_chunks):
    """
    Build source-aware context from retrieved chunks.
    """

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        document = chunk.document
        metadata = document.metadata

        file_name = metadata.get(
            "file_name",
            "Unknown source",
        )

        page_number = metadata.get(
            "page_number"
        )

        if page_number is not None:
            source = (
                f"{file_name}, page {page_number}"
            )
        else:
            source = file_name

        context_parts.append(
            f"[Source {index}: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(
        context_parts
    )


def build_prompt(question, context):
    """
    Construct a grounded RAG prompt.
    """

    prompt = f"""
You are a grounded knowledge assistant.

Answer the user's question using only the supplied context.

Do not use outside knowledge.

Do not invent facts that are not present in the context.

If the context does not contain enough information to answer
the question, respond:

"I could not find sufficient information in the supplied documents."

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt


def generate_answer(
    generator,
    question,
    retrieved_chunks,
):
    """
    Generate an answer using retrieved document context.
    """

    tokenizer, model = generator

    context = build_context(
        retrieved_chunks
    )

    prompt = build_prompt(
        question=question,
        context=context,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return answer.strip()