import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from datasets import DownloadConfig, load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning("`datasets` library not installed. Cannot load real HF datasets.")
    DownloadConfig = None

from .hetero_data import EvidenceUnit
from .paths import data_dir, repo_root


def _build_hybridqa_corpus(
    sampled: List[Dict[str, Any]],
    dataset_name: str,
) -> Dict[str, Any]:
    queries = []
    corpus: List[EvidenceUnit] = []
    doc_ids = set()

    for item in sampled:
        qid = item.get("question_id", item.get("id", ""))
        answer_text = item.get("answer-text", item.get("answer", ""))
        table_id = item.get("table_id", "unknown_table")

        answer_nodes = item.get("answer-node", [])
        supporting_titles = set()
        supporting_titles.add(table_id)
        for node in answer_nodes:
            if isinstance(node, list) and len(node) >= 4:
                wiki_link = node[2]
                if wiki_link:
                    entity = wiki_link.split("/wiki/")[-1].replace("_", " ")
                    supporting_titles.add(entity)

        queries.append({
            "id": qid,
            "query": item["question"],
            "answer": answer_text,
            "type": "hybridqa",
            "supporting_facts": {"title": sorted(supporting_titles)},
            "supporting_titles": sorted(supporting_titles),
            "table_id": table_id,
            "dataset": dataset_name,
        })

        header = item.get("table", {}).get("header", [])
        rows = item.get("table", {}).get("rows", [])

        if header and rows:
            for row_idx, row in enumerate(rows):
                doc_key = f"{dataset_name}::{table_id}::row{row_idx}"
                if doc_key in doc_ids:
                    continue
                doc_ids.add(doc_key)
                cells = ", ".join(f"{h}: {c}" for h, c in zip(header, row))
                content = f"Table: {table_id} | Row {row_idx} | {cells}"
                corpus.append(EvidenceUnit(
                    content=content,
                    source="table",
                    metadata={
                        "title": table_id,
                        "dataset": dataset_name,
                        "table_id": table_id,
                        "row_idx": row_idx,
                        "example_id": qid,
                    },
                ))

        for node in answer_nodes:
            if not isinstance(node, list) or len(node) < 4:
                continue
            cell_text = str(node[0])
            row_col = node[1]
            wiki_link = node[2]
            node_type = node[3]

            entity = ""
            if wiki_link:
                entity = wiki_link.split("/wiki/")[-1].replace("_", " ")

            if isinstance(row_col, list) and len(row_col) == 2:
                doc_key = f"{dataset_name}::{table_id}::r{row_col[0]}c{row_col[1]}"
            else:
                doc_key = f"{dataset_name}::{table_id}::{cell_text}"

            if doc_key in doc_ids:
                continue
            doc_ids.add(doc_key)

            if node_type == "passage" and entity:
                content = f"Passage about {entity}: {cell_text}"
                source = "text"
                title = entity
            else:
                r, c = (row_col if isinstance(row_col, list) and len(row_col) == 2 else [0, 0])
                col_name = header[c] if header and c < len(header) else f"col{c}"
                content = f"Table: {table_id} | {col_name}: {cell_text}"
                source = "table"
                title = table_id

            corpus.append(EvidenceUnit(
                content=content,
                source=source,
                metadata={
                    "title": title,
                    "dataset": dataset_name,
                    "table_id": table_id,
                    "cell_text": cell_text,
                    "wiki_link": wiki_link or "",
                    "node_type": node_type,
                    "example_id": qid,
                },
            ))

    logger.info(
        "Loaded %s queries and %s evidence units from %s.",
        len(queries),
        len(corpus),
        dataset_name,
    )
    return {"queries": queries, "corpus": corpus}


def _graph_triple_title(triple: Dict[str, Any]) -> str:
    head = str(triple.get("head", "")).strip()
    relation = str(triple.get("relation", triple.get("predicate", ""))).strip()
    tail = str(triple.get("tail", "")).strip()
    return f"{head} --{relation}-> {tail}"


def _build_graphqa_corpus(
    sampled: List[Dict[str, Any]],
    dataset_name: str,
) -> Dict[str, Any]:
    queries = []
    corpus: List[EvidenceUnit] = []
    doc_ids = set()

    for item_idx, item in enumerate(sampled):
        qid = str(item.get("id", item.get("question_id", f"graph_{item_idx}")))
        triples = item.get("triples", item.get("graph", []))
        if not isinstance(triples, list):
            triples = []

        triple_titles = [_graph_triple_title(triple) for triple in triples if isinstance(triple, dict)]
        supporting_titles = item.get("supporting_titles")
        if not supporting_titles:
            supporting_indices = item.get("supporting_triples", item.get("supporting_indices", []))
            if isinstance(supporting_indices, list) and supporting_indices:
                supporting_titles = [
                    triple_titles[idx]
                    for idx in supporting_indices
                    if isinstance(idx, int) and 0 <= idx < len(triple_titles)
                ]
            else:
                supporting_titles = triple_titles

        queries.append({
            "id": qid,
            "query": item.get("question", item.get("query", "")),
            "answer": item.get("answer", ""),
            "type": "graphqa",
            "supporting_facts": {"title": supporting_titles},
            "supporting_titles": supporting_titles,
            "dataset": dataset_name,
        })

        for triple_idx, triple in enumerate(triples):
            if not isinstance(triple, dict):
                continue
            title = _graph_triple_title(triple)
            doc_key = f"{dataset_name}::{qid}::{triple_idx}::{title}"
            if doc_key in doc_ids:
                continue
            doc_ids.add(doc_key)
            source = str(triple.get("source", dataset_name))
            qid_value = str(triple.get("qid", triple.get("id", "")))
            content = f"[Graph Fact] {title}"
            corpus.append(EvidenceUnit(
                content=content,
                source="graph",
                metadata={
                    "title": title,
                    "dataset": dataset_name,
                    "example_id": qid,
                    "head": str(triple.get("head", "")),
                    "relation": str(triple.get("relation", triple.get("predicate", ""))),
                    "tail": str(triple.get("tail", "")),
                    "source": source,
                    "qid": qid_value,
                },
            ))

    logger.info(
        "Loaded %s queries and %s graph evidence units from %s.",
        len(queries),
        len(corpus),
        dataset_name,
    )
    return {"queries": queries, "corpus": corpus}


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

def _dataset_aliases(dataset_name: str) -> List[str]:
    normalized = dataset_name.lower()
    aliases = {
        "hotpotqa": ["hotpotqa", "hotpot_qa"],
        "hotpot_qa": ["hotpotqa", "hotpot_qa"],
        "2wiki": ["2wiki", "2wikimultihopqa", "2wiki_multihopqa"],
        "2wikimultihopqa": ["2wiki", "2wikimultihopqa", "2wiki_multihopqa"],
        "hybridqa": ["hybridqa", "hybrid_qa"],
        "hybrid_qa": ["hybridqa", "hybrid_qa"],
    }
    return aliases.get(normalized, [normalized])


def _candidate_local_data_dirs(explicit_dir: Optional[str] = None) -> List[Path]:
    candidates: List[Path] = []
    seen = set()
    raw_values = [
        explicit_dir,
        os.getenv("RERERANK_LOCAL_DATA_DIR"),
        os.getenv("RAG_LOCAL_DATA_DIR"),
        os.getenv("RAG_DATA_DIR"),
        str(data_dir() / "datasets"),
        str(data_dir() / "raw"),
        str(repo_root() / "datasets"),
    ]
    for value in raw_values:
        if not value:
            continue
        path = Path(value).expanduser()
        resolved = path if path.is_absolute() else (repo_root() / path).resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        candidates.append(resolved)
    return candidates


def _find_local_dataset_file(dataset_name: str, split: str, local_data_dir: Optional[str] = None) -> Optional[Path]:
    extensions = (".json", ".jsonl")
    for base_dir in _candidate_local_data_dirs(local_data_dir):
        for alias in _dataset_aliases(dataset_name):
            patterns = [
                base_dir / alias / f"{split}{ext}" for ext in extensions
            ] + [
                base_dir / f"{alias}_{split}{ext}" for ext in extensions
            ]
            for path in patterns:
                if path.exists():
                    return path
    return None


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported local dataset format: {path}")


def _load_local_sample(
    dataset_name: str,
    split: str,
    num_samples: int,
    use_hetero: bool,
    local_data_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    local_file = _find_local_dataset_file(dataset_name, split, local_data_dir=local_data_dir)
    if local_file is None:
        return None

    logger.info("Loading %s samples from local file: %s", num_samples, local_file)
    records = _load_json_records(local_file)[:num_samples]
    if use_hetero:
        return _build_hetero_corpus_from_context(records, dataset_name=dataset_name)
    return _build_text_corpus_from_context(records, dataset_name=dataset_name)


def _load_hf_split(
    dataset_id: str,
    config_name: Optional[str],
    split: str,
    cache_dir: Optional[str],
    offline: bool,
):
    kwargs: Dict[str, Any] = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if offline and DownloadConfig is not None:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)

    if config_name is None:
        return load_dataset(dataset_id, **kwargs)
    return load_dataset(dataset_id, config_name, **kwargs)


def load_hotpotqa_sample(
    split: str = "validation",
    num_samples: int = 100,
    use_hetero: bool = False,
    local_data_dir: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    offline: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Loads a sample of the HotpotQA dataset for multi-hop reasoning.
    """
    local_bundle = _load_local_sample(
        "hotpotqa",
        split=split,
        num_samples=num_samples,
        use_hetero=use_hetero,
        local_data_dir=local_data_dir,
    )
    if local_bundle is not None:
        return local_bundle

    if not HF_DATASETS_AVAILABLE:
        raise ImportError("Please install `datasets` via `pip install datasets`")

    offline_mode = bool(
        os.getenv("HF_DATASETS_OFFLINE") == "1"
        or os.getenv("HF_HUB_OFFLINE") == "1"
        or offline
    )
    logger.info(
        "Loading %s samples from HotpotQA (%s split)%s...",
        num_samples,
        split,
        " [offline]" if offline_mode else "",
    )
    dataset = _load_hf_split("hotpot_qa", "distractor", split, cache_dir=hf_cache_dir, offline=offline_mode)
    sampled = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    if use_hetero:
        return _build_hetero_corpus_from_context(sampled, dataset_name="hotpotqa")
    return _build_text_corpus_from_context(sampled, dataset_name="hotpotqa")


def load_2wiki_sample(
    split: str = "validation",
    num_samples: int = 100,
    use_hetero: bool = False,
    local_data_dir: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    offline: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Loads a sample of the 2WikiMultihopQA dataset for cross-dataset transfer evaluation.
    """
    local_bundle = _load_local_sample(
        "2wikimultihopqa",
        split=split,
        num_samples=num_samples,
        use_hetero=use_hetero,
        local_data_dir=local_data_dir,
    )
    if local_bundle is not None:
        return local_bundle

    if not HF_DATASETS_AVAILABLE:
        raise ImportError("Please install `datasets` via `pip install datasets`")

    offline_mode = bool(
        os.getenv("HF_DATASETS_OFFLINE") == "1"
        or os.getenv("HF_HUB_OFFLINE") == "1"
        or offline
    )
    logger.info(
        "Loading %s samples from 2WikiMultihopQA (%s split)%s...",
        num_samples,
        split,
        " [offline]" if offline_mode else "",
    )
    dataset = _load_hf_split("framolfese/2WikiMultihopQA", None, split, cache_dir=hf_cache_dir, offline=offline_mode)
    sampled = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    if use_hetero:
        return _build_hetero_corpus_from_context(sampled, dataset_name="2wikimultihopqa")
    return _build_text_corpus_from_context(sampled, dataset_name="2wikimultihopqa")


def load_hybridqa_sample(
    split: str = "validation",
    num_samples: int = 100,
    use_hetero: bool = False,
    local_data_dir: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    offline: Optional[bool] = None,
) -> Dict[str, Any]:
    local_bundle = _load_local_hybridqa(
        split=split,
        num_samples=num_samples,
        local_data_dir=local_data_dir,
    )
    if local_bundle is not None:
        return local_bundle

    if not HF_DATASETS_AVAILABLE:
        raise ImportError("Please install `datasets` via `pip install datasets`")

    offline_mode = bool(
        os.getenv("HF_DATASETS_OFFLINE") == "1"
        or os.getenv("HF_HUB_OFFLINE") == "1"
        or offline
    )
    logger.info(
        "Loading %s samples from HybridQA (%s split)%s...",
        num_samples,
        split,
        " [offline]" if offline_mode else "",
    )
    dataset = _load_hf_split("hybrid_qa", None, split, cache_dir=hf_cache_dir, offline=offline_mode)
    sampled = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    return _build_hybridqa_corpus(sampled, dataset_name="hybridqa")


def _load_local_hybridqa(
    split: str,
    num_samples: int,
    local_data_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for base_dir in _candidate_local_data_dirs(local_data_dir):
        for alias in ("hybridqa", "hybrid_qa"):
            for ext in (".json", ".jsonl"):
                path = base_dir / alias / f"{split}{ext}"
                if path.exists():
                    logger.info("Loading %s HybridQA samples from local file: %s", num_samples, path)
                    records = _load_json_records(path)[:num_samples]
                    return _build_hybridqa_corpus(records, dataset_name="hybridqa")
    return None


def _load_local_graphqa(
    split: str,
    num_samples: int,
    local_data_dir: Optional[str] = None,
    dataset_name: str = "wikidata_graph",
) -> Optional[Dict[str, Any]]:
    aliases = (dataset_name, "wikidata_graph", "neo4j_graph", "graphqa")
    for base_dir in _candidate_local_data_dirs(local_data_dir):
        for alias in aliases:
            for ext in (".jsonl", ".json"):
                path = base_dir / alias / f"{split}{ext}"
                if path.exists():
                    logger.info("Loading %s graph QA samples from local file: %s", num_samples, path)
                    records = _load_json_records(path)[:num_samples]
                    return _build_graphqa_corpus(records, dataset_name=dataset_name)
    return None


def load_wikidata_graph_sample(
    split: str = "validation",
    num_samples: int = 100,
    use_hetero: bool = False,
    local_data_dir: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    offline: Optional[bool] = None,
) -> Dict[str, Any]:
    local_bundle = _load_local_graphqa(
        split=split,
        num_samples=num_samples,
        local_data_dir=local_data_dir,
        dataset_name="wikidata_graph",
    )
    if local_bundle is not None:
        return local_bundle
    raise FileNotFoundError(
        "Wikidata/Neo4j graph smoke requires a local JSON/JSONL file at "
        "data/datasets/wikidata_graph/validation.jsonl. "
        "Each record should include question, answer, and triples."
    )


def load_multihop_sample(
    dataset_name: str,
    split: str = "validation",
    num_samples: int = 100,
    use_hetero: bool = False,
    local_data_dir: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    offline: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Unified loader for supported multi-hop QA datasets.
    """
    normalized = dataset_name.lower()
    loaders = {
        "hotpotqa": load_hotpotqa_sample,
        "hotpot_qa": load_hotpotqa_sample,
        "2wiki": load_2wiki_sample,
        "2wikimultihopqa": load_2wiki_sample,
        "hybridqa": load_hybridqa_sample,
        "hybrid_qa": load_hybridqa_sample,
        "wikidata_graph": load_wikidata_graph_sample,
        "neo4j_graph": load_wikidata_graph_sample,
        "graphqa": load_wikidata_graph_sample,
    }

    loader = loaders.get(normalized)
    if loader is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return loader(
        split=split,
        num_samples=num_samples,
        use_hetero=use_hetero,
        local_data_dir=local_data_dir,
        hf_cache_dir=hf_cache_dir,
        offline=offline,
    )

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
