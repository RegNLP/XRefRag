from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .eval import run_evaluation
from .fusion import RRFFusion
from .rerank import CrossEncoderReranker

__all__ = [
    "run_evaluation",
    "BM25Retriever",
    "DenseRetriever",
    "RRFFusion",
    "CrossEncoderReranker",
]
"""
IR module: Indexing and retrieval for XRefRAG.

Implements multiple retrieval methods:
  - BM25 (sparse/lexical)
  - Dense retrieval (E5, BGE)
  - RRF fusion
  - Cross-encoder reranking
"""

from .types import RetrievalRun, SearchResult

__all__ = [
    "SearchResult",
    "RetrievalRun",
    "BM25Retriever",
    "DenseRetriever",
    "RRFFusion",
    "CrossEncoderReranker",
]
