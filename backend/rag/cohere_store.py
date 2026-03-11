"""
cohere_store.py — Shared Cohere embeddings + AstraDB vector store.

Imported by:
  • ingest/multimodal_pdf_ingest.py  (write path — stores multimodal chunks)
  • rag/retriever.py                  (read path  — searched on every query)

We bypass CohereEmbeddings from langchain-cohere entirely because older
versions don't accept input_type in the constructor. Instead we subclass
langchain_core.embeddings.Embeddings and call the Cohere SDK directly,
passing input_type per call — which is how Cohere v3 models must be used.

Cohere embed-english-v3.0  →  1024-dim  →  AstraDB collection "crag_multimodal"
OpenAI text-embedding-3-small → 1536-dim → AstraDB collection "crag"  (retriever.py)

.env keys required:
  COHERE_API_KEY
  ASTRA_DB_API_ENDPOINT
  ASTRA_DB_APPLICATION_TOKEN
"""

import os
import logging
from typing import List
from dotenv import load_dotenv

import cohere
from langchain_core.embeddings import Embeddings
from langchain_astradb import AstraDBVectorStore

load_dotenv()

logger = logging.getLogger(__name__)

COHERE_EMBEDDING_MODEL = "embed-english-v3.0"


# ─── Custom Embeddings wrapper ────────────────────────────────────────────────

class CohereDocEmbeddings(Embeddings):
    """
    Embeddings for ingestion — input_type='search_document'.
    Used when storing PDF chunks in AstraDB.
    """
    def __init__(self):
        self._client = cohere.Client(os.getenv("COHERE_API_KEY"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embed(
            texts=texts,
            model=COHERE_EMBEDDING_MODEL,
            input_type="search_document",
        )
        return [list(e) for e in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        # embed_documents is used at ingest time; this is a fallback
        response = self._client.embed(
            texts=[text],
            model=COHERE_EMBEDDING_MODEL,
            input_type="search_document",
        )
        return list(response.embeddings[0])


class CohereQueryEmbeddings(Embeddings):
    """
    Embeddings for retrieval — input_type='search_query'.
    Used when querying AstraDB at runtime.
    """
    def __init__(self):
        self._client = cohere.Client(os.getenv("COHERE_API_KEY"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embed(
            texts=texts,
            model=COHERE_EMBEDDING_MODEL,
            input_type="search_query",
        )
        return [list(e) for e in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embed(
            texts=[text],
            model=COHERE_EMBEDDING_MODEL,
            input_type="search_query",
        )
        return list(response.embeddings[0])


# ─── Instantiate embeddings ───────────────────────────────────────────────────

cohere_embeddings_doc   = CohereDocEmbeddings()
cohere_embeddings_query = CohereQueryEmbeddings()


# ─── AstraDB vector stores ────────────────────────────────────────────────────
# Both point at the same "crag_multimodal" collection.
# cohere_vector_store  → write path (ingestion, search_document embeddings)
# cohere_query_store   → read path  (retrieval, search_query embeddings)

cohere_vector_store = AstraDBVectorStore(
    embedding=cohere_embeddings_doc,
    collection_name="crag_multimodal",
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
)

cohere_query_store = AstraDBVectorStore(
    embedding=cohere_embeddings_query,
    collection_name="crag_multimodal",
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
)

logger.info(
    f"Cohere vector store initialised — model={COHERE_EMBEDDING_MODEL}, "
    f"collection=crag_multimodal"
)