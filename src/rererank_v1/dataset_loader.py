import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning("`datasets` library not installed. Cannot load real HF datasets.")

from .hetero_data import EvidenceUnit


def _build_text_corpus_from_context(
    sampled,
    dataset_name: str,
) -> Dict[str, Any]:
    queries = []
    corpus: List[EvidenceUnit] = []
    doc_ids = set()

    for item in sampled:
        supporting_titles = list(dict.fromkeys(item["supporting_facts"]["title"]))
        queries.append({
            "id": item["id"],
            "query": item["question"],
            "answer": item["answer"],
            "type": item.get("type", "unknown"),
            "supporting_facts": item["supporting_facts"],
            "supporting_titles": supporting_titles,
            "dataset": dataset_name,
        })

        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            doc_key = f"{dataset_name}::{title}"
            if doc_key in doc_ids:
                continue

            doc_ids.add(doc_key)
            text_content = f"Title: {title}. {' '.join(sentences)}"
            corpus.append(EvidenceUnit(
                content=text_content,
                source="text",
                metadata={
                    "title": title,
                    "dataset": dataset_name,
                    "example_id": item["id"],
                },
            ))

    logger.info(
        "Loaded %s queries and %s unique evidence units from %s.",
        len(queries),
        len(corpus),
        dataset_name,
    )
    return {"queries": queries, "corpus": corpus}

def _build_hetero_corpus_from_context(
    sampled,
    dataset_name: str,
) -> Dict[str, Any]:
    queries = []
    corpus: List[EvidenceUnit] = []
    doc_ids = set()

    for item in sampled:
        supporting_titles = list(dict.fromkeys(item["supporting_facts"]["title"]))
        queries.append({
            "id": item["id"],
            "query": item["question"],
            "answer": item["answer"],
            "type": item.get("type", "unknown"),
            "supporting_facts": item["supporting_facts"],
            "supporting_titles": supporting_titles,
            "dataset": dataset_name,
        })

        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            doc_key = f"{dataset_name}::{title}"
            if doc_key in doc_ids:
                continue
            doc_ids.add(doc_key)
            
            # Baseline Text Evidence
            text_content = f"Title: {title}. {' '.join(sentences)}"
            corpus.append(EvidenceUnit(
                content=text_content,
                source="text",
                metadata={"title": title, "dataset": dataset_name, "example_id": item["id"]},
            ))
            
            # Simulated LLM Extraction: Heterogeneous Data
            # In a real pipeline, an LLM extracts these from the text content.
            # We simulate this here by creating synthetic table and graph units from the same text.
            
            # 1. Simulate Graph Triple extraction (e.g. Entity -> relation -> Entity)
            if len(sentences) > 0 and len(sentences[0]) > 20:
                words = sentences[0].split()
                if len(words) > 5:
                    subj = " ".join(words[:2]).strip(".,")
                    obj = " ".join(words[-2:]).strip(".,")
                    triple_str = f"Graph Fact: {subj} -> related_to -> {obj}"
                    corpus.append(EvidenceUnit(
                        content=triple_str,
                        source="graph",
                        metadata={"title": title, "dataset": dataset_name, "type": "synthetic_extraction"}
                    ))
            
            # 2. Simulate Table Row extraction
            if len(sentences) > 1 and len(sentences[1]) > 20:
                words = sentences[1].split()
                if len(words) > 5:
                    key = words[0].strip(".,")
                    val = words[-1].strip(".,")
                    row_str = f"Table: {title} Profile | {key}: {val}"
                    corpus.append(EvidenceUnit(
                        content=row_str,
                        source="table",
                        metadata={"title": title, "dataset": dataset_name, "type": "synthetic_extraction"}
                    ))

    logger.info(
        "Loaded %s queries and %s unique evidence units (including text, table, graph) from %s.",
        len(queries),
        len(corpus),
        dataset_name,
    )
    return {"queries": queries, "corpus": corpus}

def load_hotpotqa_sample(split: str = "validation", num_samples: int = 100, use_hetero: bool = False) -> Dict[str, Any]:
    """
    Loads a sample of the HotpotQA dataset for multi-hop reasoning.
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("Please install `datasets` via `pip install datasets`")
        
    logger.info(f"Loading {num_samples} samples from HotpotQA ({split} split)...")
    dataset = load_dataset("hotpot_qa", 'distractor', split=split)
    sampled = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    if use_hetero:
        return _build_hetero_corpus_from_context(sampled, dataset_name="hotpotqa")
    return _build_text_corpus_from_context(sampled, dataset_name="hotpotqa")


def load_2wiki_sample(split: str = "validation", num_samples: int = 100, use_hetero: bool = False) -> Dict[str, Any]:
    """
    Loads a sample of the 2WikiMultihopQA dataset for cross-dataset transfer evaluation.
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("Please install `datasets` via `pip install datasets`")

    logger.info("Loading %s samples from 2WikiMultihopQA (%s split)...", num_samples, split)
    dataset = load_dataset("framolfese/2WikiMultihopQA", split=split)
    sampled = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    if use_hetero:
        return _build_hetero_corpus_from_context(sampled, dataset_name="2wikimultihopqa")
    return _build_text_corpus_from_context(sampled, dataset_name="2wikimultihopqa")


def load_multihop_sample(dataset_name: str, split: str = "validation", num_samples: int = 100, use_hetero: bool = False) -> Dict[str, Any]:
    """
    Unified loader for supported multi-hop QA datasets.
    """
    normalized = dataset_name.lower()
    loaders = {
        "hotpotqa": load_hotpotqa_sample,
        "hotpot_qa": load_hotpotqa_sample,
        "2wiki": load_2wiki_sample,
        "2wikimultihopqa": load_2wiki_sample,
    }

    loader = loaders.get(normalized)
    if loader is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return loader(split=split, num_samples=num_samples, use_hetero=use_hetero)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        for loader_name in ("hotpotqa", "2wiki"):
            data = load_multihop_sample(loader_name, num_samples=2)
            print(f"[{loader_name}] Sample Query: {data['queries'][0]['query']}")
            print(f"[{loader_name}] Sample Answer: {data['queries'][0]['answer']}")
            print(f"[{loader_name}] Sample Corpus size: {len(data['corpus'])}")
            print(f"[{loader_name}] Sample Document: {data['corpus'][0].content[:120]}...")
    except ImportError as e:
        print(e)
