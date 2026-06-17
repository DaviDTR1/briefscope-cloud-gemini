"""
Embedding provider for the CLOUD-GEMINI variant.

The embedding backend is FIXED to Google Gemini for this plugin variant,
regardless of which LLM the user selects in Ajustes (the LLM is
provider-agnostic; embeddings are not, because the vector store must use one
consistent embedding space).

Returns a LangChain ``Embeddings`` object, so RAG code stays provider-neutral.
"""
from __future__ import annotations

from typing import Any

from app import config

# Fixed embedding provider for this variant. The API key is shared with the
# Google LLM option, but it is required here even if another LLM is chosen.
EMBEDDING_PROVIDER = "google"
EMBEDDING_KEY = "google_api_key"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"


def build_embeddings() -> Any:
    api_key = config.get(EMBEDDING_KEY, "")
    if not api_key:
        raise ValueError(
            "[Error: API key de Google no configurada (necesaria para los "
            "embeddings de esta variante). Ve a Ajustes.]"
        )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = config.get("embedding_model", "") or DEFAULT_EMBEDDING_MODEL
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
