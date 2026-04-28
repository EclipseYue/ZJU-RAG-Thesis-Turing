from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rererank_v1.adapters.contracts import GenerationResult, RetrievedEvidence
from rererank_v1.hetero_data import EvidenceUnit
from rererank_v1.paths import repo_root


@dataclass
class LlamaIndexTextConfig:
    """Configuration for the Route A text-only LlamaIndex baseline."""

    embed_model: str = "local:sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    cache_dir: Optional[str] = None
    local_files_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class LlamaIndexTextBaseline:
    """Minimal text-only baseline wrapper for Route A.

    The import of LlamaIndex is delayed so the current repo remains usable before
    the new baseline dependency is installed on the experiment server.
    """

    def __init__(
        self,
        corpus: Iterable[EvidenceUnit],
        config: Optional[LlamaIndexTextConfig] = None,
    ) -> None:
        self.config = config or LlamaIndexTextConfig()
        self._documents = list(corpus)
        self._retriever = self._build_retriever(self._documents)

    @staticmethod
    def dependency_hint() -> str:
        return (
            "Install Route A dependencies with `pip install llama-index "
            "llama-index-embeddings-huggingface`."
        )

    def _build_retriever(self, corpus: List[EvidenceUnit]):
        try:
            from llama_index.core import Document, Settings, VectorStoreIndex
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as exc:
            raise ImportError(self.dependency_hint()) from exc

        cache_dir = self.config.cache_dir
        if cache_dir == "auto":
            hf_hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
            cache_dir = str(hf_hub_cache if hf_hub_cache.exists() else repo_root() / "config" / "cache" / "llama_index")
        elif cache_dir is None:
            cache_dir = str(repo_root() / "config" / "cache" / "llama_index")
        elif not Path(cache_dir).is_absolute():
            cache_dir = str((repo_root() / cache_dir).resolve())
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embed_model.replace("local:", ""),
            cache_folder=cache_dir,
            local_files_only=self.config.local_files_only,
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        documents = [
            Document(
                text=item.content,
                metadata={
                    "source": item.source,
                    **item.metadata,
                },
            )
            for item in corpus
            if item.source == "text"
        ]
        index = VectorStoreIndex.from_documents(documents)
        return index.as_retriever(similarity_top_k=self.config.top_k)

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedEvidence]:
        if top_k is not None and top_k != self.config.top_k:
            self._retriever.similarity_top_k = top_k

        source_nodes = self._retriever.retrieve(question)
        return [
            RetrievedEvidence(
                text=getattr(node.node, "text", ""),
                score=float(getattr(node, "score", 0.0) or 0.0),
                source=getattr(node.node, "metadata", {}).get("source", "text"),
                metadata=dict(getattr(node.node, "metadata", {}) or {}),
            )
            for node in source_nodes
        ]

    def query(self, question: str) -> GenerationResult:
        evidence = self.retrieve(question)
        return GenerationResult(
            answer="",
            evidence=evidence,
            metadata={
                "baseline": "llamaindex_text",
                "top_k": self.config.top_k,
            },
        )
