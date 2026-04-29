from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.baselines.llamaindex_text import LlamaIndexTextBaseline, LlamaIndexTextConfig
from rererank_v1.dataset_loader import load_multihop_sample
from rererank_v1.llm_generator import llm_generate_answer
from rererank_v1.paths import repo_root, results_dir


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counter: Dict[str, int] = {}
    truth_counter: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counter[token] = pred_counter.get(token, 0) + 1
    for token in truth_tokens:
        truth_counter[token] = truth_counter.get(token, 0) + 1

    common = sum(min(pred_counter.get(t, 0), truth_counter.get(t, 0)) for t in pred_counter)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_private_overrides() -> Dict[str, Any]:
    local_override = repo_root() / "experiments" / "configs" / "local_api_overrides.json"
    if not local_override.exists():
        return {}
    return load_json(local_override)


def retrieved_to_generator_results(items: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "text": item.text,
            "score": item.score,
            "metadata": item.metadata,
            "source": item.source,
        }
        for item in items
    ]


def extract_titles(results: List[Dict[str, Any]], top_k: int) -> List[str]:
    titles = []
    for item in results[:top_k]:
        title = (item.get("metadata", {}) or {}).get("title")
        if title:
            titles.append(title)
    return titles


def evaluate_query(
    baseline: LlamaIndexTextBaseline,
    query_item: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    top_k = int(config["baseline"].get("top_k", 5))
    start = time.perf_counter()
    evidence = baseline.retrieve(query_item["query"], top_k=top_k)
    retrieved_results = retrieved_to_generator_results(evidence)
    answer = llm_generate_answer(
        query_item["query"],
        retrieved_results,
        backend=config.get("generator_backend", "heuristic"),
        model=config.get("generator_model", "deepseek-v4-flash"),
        api_key=config.get("generator_api_key"),
        base_url=config.get("generator_base_url"),
        answer_mode=config.get("answer_mode", "concise"),
    )
    latency_ms = (time.perf_counter() - start) * 1000

    supporting_titles = set(query_item.get("supporting_titles", []))
    predicted_titles = extract_titles(retrieved_results, top_k=top_k)
    support_hits = len(supporting_titles.intersection(predicted_titles))
    support_recall = support_hits / len(supporting_titles) if supporting_titles else 0.0
    support_all_hit = 1.0 if supporting_titles and support_hits == len(supporting_titles) else 0.0

    return {
        "id": query_item.get("id"),
        "query": query_item["query"],
        "answer": query_item.get("answer", ""),
        "predicted_answer": answer,
        "exact_match": exact_match_score(answer, query_item.get("answer", "")),
        "f1_score": f1_score(answer, query_item.get("answer", "")),
        "support_recall": support_recall,
        "support_all_hit": support_all_hit,
        "predicted_titles": predicted_titles,
        "supporting_titles": sorted(supporting_titles),
        "latency_ms": latency_ms,
        "retrieved": retrieved_results,
    }


def summarize(config: Dict[str, Any], details: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "Config": config["baseline"]["name"],
        "Route": config.get("route", "A"),
        "Stage": config.get("stage", "A1_text_baseline"),
        "Dataset": config["dataset"],
        "GeneratorBackend": config.get("generator_backend", "heuristic"),
        "GeneratorModel": config.get("generator_model", ""),
        "AnswerMode": config.get("answer_mode", "concise"),
        "Samples": len(details),
        "SupportRecall@K": round(mean(item["support_recall"] for item in details) * 100, 2),
        "SupportAllHit@K": round(mean(item["support_all_hit"] for item in details) * 100, 2),
        "ExactMatch": round(mean(item["exact_match"] for item in details) * 100, 2),
        "F1_Score": round(mean(item["f1_score"] for item in details) * 100, 2),
        "Avg_Latency_ms": round(mean(item["latency_ms"] for item in details), 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Route A mature-framework baseline experiments.")
    parser.add_argument("--preset", default="experiments/presets/route_a_hotpotqa.json")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output-name", default="route_a_hotpotqa_baseline.json")
    parser.add_argument("--generator-backend", default=None)
    parser.add_argument("--generator-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset_path = Path(args.preset)
    if not preset_path.is_absolute():
        preset_path = repo_root() / preset_path
    config = load_json(preset_path)
    config.update(load_local_private_overrides())
    if args.samples is not None:
        config["samples"] = args.samples
    if args.generator_backend is not None:
        config["generator_backend"] = args.generator_backend
    if args.generator_model is not None:
        config["generator_model"] = args.generator_model

    config.setdefault("generator_backend", "deepseek")
    config.setdefault("generator_model", "deepseek-v4-flash")

    bundle = load_multihop_sample(
        config["dataset"],
        split=config.get("split", "validation"),
        num_samples=int(config.get("samples", 100)),
        use_hetero=False,
        local_data_dir=config.get("local_data_dir"),
        offline=bool(config.get("offline", False)),
    )
    baseline_config = LlamaIndexTextConfig(
        embed_model=config["baseline"].get("embed_model", "BAAI/bge-small-en-v1.5"),
        top_k=int(config["baseline"].get("top_k", 5)),
        chunk_size=int(config["baseline"].get("chunk_size", 512)),
        chunk_overlap=int(config["baseline"].get("chunk_overlap", 64)),
        cache_dir=config["baseline"].get("cache_dir"),
        local_files_only=bool(config["baseline"].get("local_files_only", config.get("offline", True))),
    )
    baseline = LlamaIndexTextBaseline(bundle["corpus"], config=baseline_config)

    details = [evaluate_query(baseline, item, config) for item in bundle["queries"]]
    matrix = [summarize(config, details)]
    payload = {
        "run_info": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "preset": str(preset_path),
            "dataset": config["dataset"],
            "split": config.get("split", "validation"),
            "samples": len(details),
            "generator_backend": config.get("generator_backend", "heuristic"),
            "generator_model": config.get("generator_model", ""),
            "answer_mode": config.get("answer_mode", "concise"),
            "uses_real_api": config.get("generator_backend", "heuristic") != "heuristic",
        },
        "matrix": matrix,
        "details": details,
    }

    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / args.output_name, "w", encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=2)
    report_name = f"{Path(args.output_name).stem}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_dir / report_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Effective generator: backend={config.get('generator_backend', 'heuristic')} "
        f"model={config.get('generator_model', '') or '<default>'}"
    )
    print(json.dumps(matrix[0], ensure_ascii=False, indent=2))
    print(f"Saved matrix: {out_dir / args.output_name}")
    print(f"Saved report: {out_dir / report_name}")


if __name__ == "__main__":
    main()
