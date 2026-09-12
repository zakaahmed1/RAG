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


def build_context(retrieved_documents):
    """
    Combine retrieved document chunks into a single
    context string for the language model.
    """

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    return context


def build_prompt(question, context):
    """
    Construct a grounded RAG prompt using retrieved context.
    """

    prompt = f"""
Answer the question using only the context provided below.

If the context does not contain enough information to answer
the question, say that you do not have enough information.

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
    retrieved_documents,
):
    """
    Generate an answer using retrieved document context.
    """

    tokenizer, model = generator

    context = build_context(
        retrieved_documents
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