"""Domain enumerations."""

from enum import StrEnum


class LegalDocumentType(StrEnum):
    LAW = "law"
    DECREE = "decree"
    CIRCULAR = "circular"
    RESOLUTION = "resolution"
    DECISION = "decision"
    OTHER = "other"


class RetrievalSource(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    FUSED = "fused"
    RERANKED = "reranked"
