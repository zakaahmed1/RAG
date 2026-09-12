from transformers import pipeline

from app.config import (
    GENERATOR_MODEL_NAME,
    MAX_NEW_TOKENS,
)


def create_generator():
    """
    Load the Hugging Face text generation pipeline.
    """

    generator = pipeline(
        "text2text-generation",
        model=GENERATOR_MODEL_NAME,
    )

    return generator


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
    Construct a grounded RAG prompt using the retrieved context.
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


def generate_answer(generator, question, retrieved_documents):
    """
    Generate an answer using retrieved document context.
    """

    context = build_context(retrieved_documents)

    prompt = build_prompt(
        question=question,
        context=context,
    )

    result = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    answer = result[0]["generated_text"]

    return answer.strip()