import json
import math
import os
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.dataset_loader import load_multihop_sample
from rererank_v1.rag_pipeline import RAGPipeline


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DOCS_IMG_DIR = ROOT / "docs" / "images"
THESIS_IMG_DIR = ROOT / "paper" / "zjuthesis" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)
THESIS_IMG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text or ""))))


def tokenize(text: str):
    return [tok for tok in re.findall(r"\w+", normalize_answer(text)) if len(tok) > 1]


def extract_title(result) -> str:
    metadata = result.get("metadata", {}) or {}
    if metadata.get("title"):
        return metadata["title"]

    match = re.match(r"Title:\s*(.*?)\.\s", result.get("text", ""))
    return match.group(1) if match else ""


def answer_present(answer: str, results) -> bool:
    answer_norm = normalize_answer(answer)
    if answer_norm in {"yes", "no", "noanswer", ""}:
        return False
    return any(answer_norm in normalize_answer(item.get("text", "")) for item in results)


def lexical_retrieve(query: str, corpus, top_k: int = 5):
    q_terms = Counter(tokenize(query))
    ranked = []
    for idx, item in enumerate(corpus):
        doc_terms = Counter(tokenize(item.content))
        overlap = sum((q_terms & doc_terms).values())
        if overlap == 0:
            continue
        ranked.append({
            "id": idx,
            "text": item.content,
            "source": item.source,
            "metadata": item.metadata,
            "score": float(overlap),
            "rank": 0,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[:top_k]
    for i, item in enumerate(top, start=1):
        item["rank"] = i
    return top


def snapshot_stats(rag: RAGPipeline):
    return dict(rag.stats)


def diff_stats(before, after):
    return {
        "total_tokens": after["total_tokens"] - before["total_tokens"],
        "total_latency": after["total_latency"] - before["total_latency"],
        "retrieval_calls": after["retrieval_calls"] - before["retrieval_calls"],
        "reranker_calls": after["reranker_calls"] - before["reranker_calls"],
    }


def run_variant(rag: RAGPipeline, query: str, corpus, variant: str, top_k: int, threshold: float):
    if variant == "lexical":
        start = time.time()
        results = lexical_retrieve(query, corpus, top_k=top_k)
        elapsed = time.time() - start
        return {
            "results": results,
            "chain": [],
            "stats": {
                "total_tokens": len(query.split()),
                "total_latency": elapsed,
                "retrieval_calls": 1,
                "reranker_calls": 0,
            },
        }

    before = snapshot_stats(rag)
    if variant == "dense_only":
        results = rag._retrieve(query, top_k=top_k)
        chain = []
    elif variant == "dense_rerank":
        results = rag._rerank(query, rag._retrieve(query, top_k=top_k))
        chain = []
    elif variant == "adaptive":
        results = rag.search(query, top_k=top_k, prf_threshold=threshold, active_retrieval=True)
        chain = []
    elif variant == "full_chain":
        search_res = rag.search_with_chain(query, top_k=top_k, prf_threshold=threshold)
        results = search_res["results"]
        chain = search_res["chain"]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    stats = diff_stats(before, rag.stats)
    return {"results": results[:top_k], "chain": chain, "stats": stats}


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 42):
    rng = random.Random(seed)
    if not values:
        return [0.0, 0.0]

    samples = []
    n = len(values)
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(resample))
    samples.sort()
    return [round(samples[int(0.025 * n_boot)], 4), round(samples[int(0.975 * n_boot)], 4)]


def paired_bootstrap_delta(values_a, values_b, n_boot: int = 1000, seed: int = 42):
    rng = random.Random(seed)
    if not values_a or len(values_a) != len(values_b):
        return {"delta": 0.0, "ci95": [0.0, 0.0]}

    n = len(values_a)
    deltas = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample_a = mean(values_a[i] for i in idxs)
        sample_b = mean(values_b[i] for i in idxs)
        deltas.append(sample_a - sample_b)
    deltas.sort()
    return {
        "delta": round(mean(values_a) - mean(values_b), 4),
        "ci95": [round(deltas[int(0.025 * n_boot)], 4), round(deltas[int(0.975 * n_boot)], 4)],
    }


def classify_failure(record):
    if record["support_hit"] == 0:
        return "retrieval_miss"
    if record["support_hit"] < len(record["support_titles"]):
        return "partial_evidence"
    if not record["answer_presence"] and normalize_answer(record["answer"]) not in {"yes", "no"}:
        return "answer_not_in_topk"
    if record["adaptive_calls"] > 1 and not record["support_all_hit"]:
        return "over_retrieval_without_gain"
    return "other"


def aggregate_records(records):
    support_all = [r["support_all_hit"] for r in records]
    support_recall = [r["support_recall"] for r in records]
    answer_presence = [r["answer_presence"] for r in records if r["answer_presence_applicable"]]
    latency_ms = [r["latency_ms"] for r in records]
    tokens = [r["tokens"] for r in records]
    retrieval_calls = [r["retrieval_calls"] for r in records]
    reranker_calls = [r["reranker_calls"] for r in records]

    return {
        "n": len(records),
        "SupportAllHit@5": round(mean(support_all) * 100, 2),
        "SupportRecall@5": round(mean(support_recall) * 100, 2),
        "AnswerPresence@5": round(mean(answer_presence) * 100, 2) if answer_presence else None,
        "AvgLatencyMs": round(mean(latency_ms), 2),
        "AvgTokens": round(mean(tokens), 2),
        "AvgRetrievalCalls": round(mean(retrieval_calls), 2),
        "AvgRerankerCalls": round(mean(reranker_calls), 2),
        "SupportAllHit@5_CI95": bootstrap_ci(support_all),
        "SupportRecall@5_CI95": bootstrap_ci(support_recall),
        "AnswerPresence@5_CI95": bootstrap_ci(answer_presence) if answer_presence else None,
    }


def run_dataset_study(dataset_name: str, num_samples: int, threshold: float, top_k: int = 5):
    data = load_multihop_sample(dataset_name, num_samples=num_samples)
    corpus = data["corpus"]
    queries = data["queries"]

    rag = RAGPipeline(use_v6_reranker=True)
    rag.add_evidence_units(corpus)

    variants = ["lexical", "dense_only", "dense_rerank", "adaptive", "full_chain"]
    per_variant = {variant: [] for variant in variants}

    for idx, item in enumerate(queries, start=1):
        print(f"[{dataset_name}] {idx}/{len(queries)} :: {item['query'][:90]}")
        support_titles = set(item["supporting_titles"])
        query_type = item.get("type", "unknown")

        for variant in variants:
            out = run_variant(rag, item["query"], corpus, variant, top_k=top_k, threshold=threshold)
            results = out["results"]
            retrieved_titles = [extract_title(r) for r in results]
            support_hit = len(support_titles.intersection(retrieved_titles))
            answer_presence_flag = answer_present(item["answer"], results)
            answer_presence_applicable = normalize_answer(item["answer"]) not in {"yes", "no", ""}

            record = {
                "id": item["id"],
                "query": item["query"],
                "answer": item["answer"],
                "query_type": query_type,
                "support_titles": sorted(support_titles),
                "retrieved_titles": retrieved_titles,
                "support_hit": support_hit,
                "support_recall": support_hit / max(len(support_titles), 1),
                "support_all_hit": int(support_hit == len(support_titles)),
                "answer_presence": int(answer_presence_flag),
                "answer_presence_applicable": answer_presence_applicable,
                "latency_ms": out["stats"]["total_latency"] * 1000,
                "tokens": out["stats"]["total_tokens"],
                "retrieval_calls": out["stats"]["retrieval_calls"],
                "reranker_calls": out["stats"]["reranker_calls"],
                "adaptive_calls": out["stats"]["retrieval_calls"],
            }
            if variant == "full_chain":
                record["failure_type"] = classify_failure(record)
            per_variant[variant].append(record)

    summary = {variant: aggregate_records(records) for variant, records in per_variant.items()}

    bucket_stats = {}
    for variant, records in per_variant.items():
        bucket_stats[variant] = {}
        type_to_records = defaultdict(list)
        for record in records:
            type_to_records[record["query_type"]].append(record)
        for query_type, bucket in type_to_records.items():
            bucket_stats[variant][query_type] = aggregate_records(bucket)

    error_distribution = Counter(r["failure_type"] for r in per_variant["full_chain"])
    total_failures = max(sum(error_distribution.values()), 1)
    error_distribution = {
        key: {
            "count": value,
            "ratio_percent": round((value / total_failures) * 100, 2),
        }
        for key, value in error_distribution.items()
    }

    significance = {
        "adaptive_vs_dense_rerank_support_all": paired_bootstrap_delta(
            [r["support_all_hit"] for r in per_variant["adaptive"]],
            [r["support_all_hit"] for r in per_variant["dense_rerank"]],
        ),
        "full_chain_vs_dense_rerank_support_all": paired_bootstrap_delta(
            [r["support_all_hit"] for r in per_variant["full_chain"]],
            [r["support_all_hit"] for r in per_variant["dense_rerank"]],
        ),
    }

    return {
        "dataset": dataset_name,
        "num_samples": num_samples,
        "threshold": threshold,
        "summary": summary,
        "bucket_stats": bucket_stats,
        "error_distribution": error_distribution,
        "significance": significance,
        "records": per_variant,
    }


def run_threshold_sweep(dataset_name: str, num_samples: int, thresholds, top_k: int = 5):
    data = load_multihop_sample(dataset_name, num_samples=num_samples)
    corpus = data["corpus"]
    queries = data["queries"]

    rag = RAGPipeline(use_v6_reranker=True)
    rag.add_evidence_units(corpus)

    sweep = []
    for threshold in thresholds:
        records = []
        for idx, item in enumerate(queries, start=1):
            print(f"[sweep {dataset_name} @ {threshold}] {idx}/{len(queries)}")
            out = run_variant(rag, item["query"], corpus, "adaptive", top_k=top_k, threshold=threshold)
            support_titles = set(item["supporting_titles"])
            retrieved_titles = [extract_title(r) for r in out["results"]]
            support_hit = len(support_titles.intersection(retrieved_titles))
            records.append({
                "support_all_hit": int(support_hit == len(support_titles)),
                "support_recall": support_hit / max(len(support_titles), 1),
                "latency_ms": out["stats"]["total_latency"] * 1000,
                "tokens": out["stats"]["total_tokens"],
                "retrieval_calls": out["stats"]["retrieval_calls"],
            })

        sweep.append({
            "threshold": threshold,
            "SupportAllHit@5": round(mean(r["support_all_hit"] for r in records) * 100, 2),
            "SupportRecall@5": round(mean(r["support_recall"] for r in records) * 100, 2),
            "AvgLatencyMs": round(mean(r["latency_ms"] for r in records), 2),
            "AvgTokens": round(mean(r["tokens"] for r in records), 2),
            "AvgRetrievalCalls": round(mean(r["retrieval_calls"] for r in records), 2),
            "SupportAllHit@5_CI95": bootstrap_ci([r["support_all_hit"] for r in records]),
        })
    return sweep


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_cross_dataset(summary_data):
    variants = ["lexical", "dense_only", "dense_rerank", "adaptive", "full_chain"]
    labels = {
        "lexical": "Lexical",
        "dense_only": "Dense",
        "dense_rerank": "Dense+Rerank",
        "adaptive": "Adaptive",
        "full_chain": "Adaptive+CoVe",
    }
    datasets = list(summary_data.keys())
    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, dataset in enumerate(datasets):
        y = [summary_data[dataset]["summary"][variant]["SupportAllHit@5"] for variant in variants]
        ax.bar(x + (i - 0.5) * width, y, width=width, label=dataset)

    ax.set_ylabel("SupportAllHit@5 (%)")
    ax.set_title("Cross-dataset Support Coverage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[v] for v in variants], rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    for target in (DOCS_IMG_DIR / "cross_dataset_support_hit.png", THESIS_IMG_DIR / "cross_dataset_support_hit.png"):
        plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_sweep(sweep):
    thresholds = [item["threshold"] for item in sweep]
    support = [item["SupportAllHit@5"] for item in sweep]
    latency = [item["AvgLatencyMs"] for item in sweep]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, support, marker="o", linewidth=2.5, color="#1f77b4", label="SupportAllHit@5")
    ax1.set_xlabel("Adaptive Threshold")
    ax1.set_ylabel("SupportAllHit@5 (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, latency, marker="s", linewidth=2.5, linestyle="--", color="#d62728", label="Latency")
    ax2.set_ylabel("Avg Latency (ms)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Threshold Sensitivity: Coverage vs Latency")
    fig.tight_layout()

    for target in (DOCS_IMG_DIR / "threshold_sensitivity.png", THESIS_IMG_DIR / "threshold_sensitivity.png"):
        plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.environ["FORCE_MOCK"] = "0"

    config = {
        "hotpot_samples": 80,
        "wiki2_samples": 80,
        "sweep_samples": 50,
        "top_k": 5,
        "threshold": 0.8,
        "thresholds": [0.7, 0.8, 0.9, 0.95],
    }

    started = time.time()
    hotpot = run_dataset_study("hotpotqa", num_samples=config["hotpot_samples"], threshold=config["threshold"], top_k=config["top_k"])
    wiki2 = run_dataset_study("2wiki", num_samples=config["wiki2_samples"], threshold=config["threshold"], top_k=config["top_k"])
    sweep = run_threshold_sweep("hotpotqa", num_samples=config["sweep_samples"], thresholds=config["thresholds"], top_k=config["top_k"])

    result = {
        "config": config,
        "runtime_seconds": round(time.time() - started, 2),
        "datasets": {
            "hotpotqa": hotpot,
            "2wikimultihopqa": wiki2,
        },
        "threshold_sweep": sweep,
    }

    save_json(RESULTS_DIR / "research_extension_results.json", result)
    plot_cross_dataset(result["datasets"])
    plot_threshold_sweep(sweep)

    compact = {
        "config": config,
        "runtime_seconds": result["runtime_seconds"],
        "datasets": {
            name: {
                "summary": payload["summary"],
                "error_distribution": payload["error_distribution"],
                "significance": payload["significance"],
            }
            for name, payload in result["datasets"].items()
        },
        "threshold_sweep": sweep,
    }
    save_json(RESULTS_DIR / "research_extension_summary.json", compact)
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
