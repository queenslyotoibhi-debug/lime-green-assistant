"""Gemini embedding calls shared by the index build and the query path.

Both callers must use the same model and the same asymmetric task types, or the
stored vectors and the query vector end up in different spaces and cosine
similarity stops meaning anything. Keeping the call in one module is what makes
that guarantee checkable.
"""

from pathlib import Path
import os
import re
import time
import tomllib

import requests


SECRETS_FILE = (
    Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
)


MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT = 120

# Asymmetric retrieval: the corpus is embedded as documents, the user's question
# as a query. This replaces the "Instruct:" prefix the Ollama build used.
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"

API_KEY_VARIABLE = "GEMINI_API_KEY"

# The free tier caps requests per minute, so a corpus rebuild will be throttled
# partway through and a busy demo can be throttled mid-question. Both recover by
# waiting, so retry here rather than making every caller handle it.
MAX_ATTEMPTS = 8
FALLBACK_RETRY_SECONDS = 20.0
MAX_RETRY_SECONDS = 90.0


def get_api_key():
    api_key = os.environ.get(API_KEY_VARIABLE, "").strip()

    # Fall back to the same secrets file Streamlit reads, so a local index rebuild
    # and a local app run need the key in exactly one place.
    if not api_key and SECRETS_FILE.is_file():
        with SECRETS_FILE.open("rb") as file:
            api_key = str(tomllib.load(file).get(API_KEY_VARIABLE, "")).strip()

    if not api_key:
        raise RuntimeError(
            f"{API_KEY_VARIABLE} is not set. Create a free key at "
            "https://aistudio.google.com/apikey, then set it as an environment "
            "variable locally or as a Streamlit secret when deployed."
        )

    return api_key


def retry_delay_seconds(response):
    """Prefer the server's own RetryInfo over a guess."""
    try:
        details = response.json()["error"].get("details", [])
    except Exception:
        return None

    for detail in details:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
        if match:
            return min(float(match.group(1)) + 1.0, MAX_RETRY_SECONDS)

    return None


def post_with_retry(payload, progress=None):
    delay = FALLBACK_RETRY_SECONDS

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = requests.post(
            f"{API_BASE}/models/{MODEL}:batchEmbedContents",
            headers={"x-goog-api-key": get_api_key()},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429 and attempt < MAX_ATTEMPTS:
            wait = retry_delay_seconds(response) or delay
            if progress:
                progress(f"  rate limited, waiting {wait:.0f}s "
                         f"(attempt {attempt}/{MAX_ATTEMPTS - 1})")
            time.sleep(wait)
            delay = min(delay * 2, MAX_RETRY_SECONDS)
            continue

        if not response.ok:
            raise RuntimeError(
                f"Gemini embedding request failed with HTTP "
                f"{response.status_code}: {response.text[:400]}"
            )

        return response

    raise RuntimeError(
        f"Gemini embedding request was rate limited {MAX_ATTEMPTS} times in a row. "
        "The free-tier quota may be exhausted for now; try again later."
    )


def embed_texts(texts, task_type, progress=None):
    """Embed a batch of texts and return one vector per input, in order."""
    if not texts:
        return []

    response = post_with_retry(
        {
            "requests": [
                {
                    "model": f"models/{MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in texts
            ]
        },
        progress=progress,
    )

    embeddings = response.json().get("embeddings")

    if not isinstance(embeddings, list):
        raise ValueError(
            "The Gemini response does not contain an embeddings list."
        )

    if len(embeddings) != len(texts):
        raise ValueError(
            "Gemini returned an unexpected number of embeddings: "
            f"expected {len(texts)}, found {len(embeddings)}."
        )

    vectors = []

    for position, embedding in enumerate(embeddings, start=1):
        values = embedding.get("values")

        if not isinstance(values, list):
            raise ValueError(
                f"Embedding {position} has no values array."
            )

        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unexpected embedding dimensions in item {position}: "
                f"expected {EMBEDDING_DIMENSIONS}, found {len(values)}."
            )

        vectors.append(values)

    return vectors


def embed_query(query):
    [vector] = embed_texts([query], QUERY_TASK_TYPE)
    return vector
