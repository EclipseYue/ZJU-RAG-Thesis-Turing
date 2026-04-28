from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class RetrievedEvidence:
    """Framework-neutral evidence item returned by a retriever."""

    text: str
    score: float = 0.0
    source: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Framework-neutral answer object used by experiments."""

    answer: str
    evidence: List[RetrievedEvidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Framework-neutral verifier output."""

    supported: bool
    score: float
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrieverAdapter(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedEvidence]:
        ...


class GeneratorAdapter(Protocol):
    def generate(self, query: str, evidence: List[RetrievedEvidence]) -> GenerationResult:
        ...


class VerifierAdapter(Protocol):
    def verify(self, claim: str, evidence: List[RetrievedEvidence]) -> VerificationResult:
        ...
