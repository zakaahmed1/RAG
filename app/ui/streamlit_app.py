import os

import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_BASE_URL = os.getenv(
    "RAG_API_BASE_URL",
    "http://127.0.0.1:8000",
)

API_KEY = os.getenv(
    "RAG_API_KEY",
    "",
).strip()


def api_headers():
    """
    Return authentication headers when configured.
    """

    if not API_KEY:
        return {}

    return {
        "X-API-Key": API_KEY
    }

REQUEST_TIMEOUT_SECONDS = 120


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# API helpers
# ---------------------------------------------------------

def get_health():
    """
    Check whether the FastAPI service is reachable.
    """

    try:

        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


def get_status():
    """
    Retrieve the active RAG service configuration.
    """

    try:

        response = requests.get(
            f"{API_BASE_URL}/status",
            headers=api_headers(),
            timeout=5,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


def query_rag(question: str):
    """
    Send a question to the FastAPI RAG endpoint.
    """

    response = requests.post(
        f"{API_BASE_URL}/query",
        json={
            "question": question
        },
        headers=api_headers(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header(
        "System Status"
    )

    health = get_health()

    if health:

        st.success(
            "API online"
        )

        status = get_status()

        if status:

            st.write(
                f"**Ready:** "
                f"{'Yes' if status['ready'] else 'No'}"
            )

            st.write(
                f"**Retrieval:** "
                f"{status['retrieval_mode']}"
            )

            st.write(
                f"**Top K:** "
                f"{status['top_k']}"
            )

            st.write(
                f"**Fetch K:** "
                f"{status['fetch_k']}"
            )

            st.write(
                f"**Threshold:** "
                f"{status['min_similarity']}"
            )

            with st.expander(
                "Models"
            ):

                st.write(
                    "**Embedding model**"
                )

                st.code(
                    status[
                        "embedding_model"
                    ]
                )

                st.write(
                    "**Generator model**"
                )

                st.code(
                    status[
                        "generator_model"
                    ]
                )

    else:

        st.error(
            "API offline"
        )

        st.caption(
            "Start the FastAPI service "
            "before submitting questions."
        )

    st.divider()

    st.caption(
        f"API: {API_BASE_URL}"
    )

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

st.title(
    "📚 RAG Knowledge Assistant"
)

st.write(
    "Ask questions about the indexed document knowledge base. "
    "Answers are grounded in retrieved source material."
)

st.caption(
    "Sources and semantic similarity scores are shown "
    "for every retrieved answer."
)

st.caption(
    "Generated answers should be verified against "
    "the cited source material before being used "
    "for high-impact decisions."
)

st.divider()


# ---------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("metadata")
        ):

            metadata = message[
                "metadata"
            ]

            if metadata.get(
                "retrieval_rejected"
            ):

                st.info(
                    "No document chunks met the "
                    "configured retrieval threshold."
                )

            sources = metadata.get(
                "sources",
                [],
            )

            if sources:

                with st.expander(
                    "Sources",
                    expanded=False,
                ):

                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):

                        file_name = source.get(
                            "file",
                            "Unknown source",
                        )

                        page = source.get(
                            "page"
                        )

                        chunk_id = source.get(
                            "chunk_id"
                        )

                        similarity = source.get(
                            "similarity"
                        )

                        if page is not None:

                            source_name = (
                                f"{file_name}, "
                                f"page {page}"
                            )

                        else:

                            source_name = (
                                file_name
                            )

                        st.markdown(
                            f"**{index}. "
                            f"{source_name}**"
                        )

                        details = []

                        if chunk_id is not None:

                            details.append(
                                f"Chunk {chunk_id}"
                            )

                        if similarity is not None:

                            details.append(
                                "Similarity: "
                                f"{similarity:.4f}"
                            )

                        if details:

                            st.caption(
                                " | ".join(
                                    details
                                )
                            )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about the documents..."
)


# ---------------------------------------------------------
# Process a new question
# ---------------------------------------------------------

if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # -----------------------------------------------------
    # Store and display user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # -----------------------------------------------------
    # Call RAG API
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the knowledge base..."
        ):

            try:

                result = query_rag(
                    question
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to the RAG API. "
                    "Make sure FastAPI is running."
                )

                st.stop()

            except requests.exceptions.Timeout:

                st.error(
                    "The RAG request timed out."
                )

                st.stop()

            except requests.exceptions.HTTPError as exc:

                status_code = (
                    exc.response.status_code
                    if exc.response
                    else None
                )

                if status_code == 401:

                    st.error(
                        "API authentication failed. "
                    )

                if status_code == 422:

                    st.error(
                        "The question was rejected "
                        "by API validation."
                    )

                elif status_code == 503:

                    st.error(
                        "The RAG service is not ready."
                    )

                else:

                    st.error(
                        "The API returned an error."
                    )

                st.stop()

            except requests.RequestException:

                st.error(
                    "An unexpected API communication "
                    "error occurred."
                )

                st.stop()

    # -----------------------------------------------------
    # Render answer
    # -----------------------------------------------------

        answer = result[
            "answer"
        ]

        st.markdown(
            answer
        )

        if result[
            "retrieval_rejected"
        ]:

            st.info(
                "No document chunks met the "
                "configured retrieval threshold."
            )

        sources = result.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                "Sources",
                expanded=True,
            ):

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    file_name = source.get(
                        "file",
                        "Unknown source",
                    )

                    page = source.get(
                        "page"
                    )

                    chunk_id = source.get(
                        "chunk_id"
                    )

                    similarity = source.get(
                        "similarity"
                    )

                    if page is not None:

                        source_name = (
                            f"{file_name}, "
                            f"page {page}"
                        )

                    else:

                        source_name = (
                            file_name
                        )

                    st.markdown(
                        f"**{index}. "
                        f"{source_name}**"
                    )

                    details = []

                    if chunk_id is not None:

                        details.append(
                            f"Chunk {chunk_id}"
                        )

                    if similarity is not None:

                        details.append(
                            "Similarity: "
                            f"{similarity:.4f}"
                        )

                    if details:

                        st.caption(
                            " | ".join(
                                details
                            )
                        )

    # -----------------------------------------------------
    # Store assistant response
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "metadata": {
                "retrieval_rejected":
                    result[
                        "retrieval_rejected"
                    ],
                "retrieved_count":
                    result[
                        "retrieved_count"
                    ],
                "sources":
                    sources,
            },
        }
    )