import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.dataset_loader import load_multihop_sample
from rererank_v1.paths import repo_root, results_dir
from rererank_v1.rag_pipeline import RAGPipeline


def extract_title(result):
    metadata = result.get("metadata", {}) or {}
    return metadata.get("title", "")


def paired_bootstrap_delta(values_a, values_b, n_boot=1000, seed=42):
    rng = random.Random(seed)
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


def evaluate_variant(rag, query_item, variant, top_k, prf_threshold):
    if variant == "dense_rerank":
        results = rag._rerank(query_item["query"], rag._retrieve(query_item["query"], top_k=top_k))
    elif variant == "adaptive":
        results = rag.search(
            query_item["query"],
            top_k=top_k,
            active_retrieval=True,
            prf_threshold=prf_threshold,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    support_titles = set(query_item["supporting_titles"])
    retrieved_titles = [extract_title(item) for item in results[:top_k]]
    support_hit = len(support_titles.intersection(retrieved_titles))
    return {
        "id": query_item["id"],
        "query_type": query_item.get("type", "unknown"),
        "support_all_hit": int(support_hit == len(support_titles)),
        "support_recall": support_hit / max(len(support_titles), 1),
    }


def summarize(records):
    return {
        "n": len(records),
        "SupportAllHit@5": round(mean(item["support_all_hit"] for item in records) * 100, 2),
        "SupportRecall@5": round(mean(item["support_recall"] for item in records) * 100, 2),
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Compare Dense+Rerank vs Adaptive gains by query bucket.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wiki"], help="Dataset name.")
    parser.add_argument("--split", default="validation", help="Dataset split.")
    parser.add_argument("--samples", type=int, default=500, help="Number of evaluated samples.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k.")
    parser.add_argument("--prf-threshold", type=float, default=0.8, help="Adaptive retrieval threshold.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap resamples.")
    parser.add_argument("--device", default=None, help="Runtime device.")
    parser.add_argument("--output-name", default="bucket_gain_study.json", help="Output JSON filename.")
    return parser


def load_config(args):
    if not args.config:
        return vars(args)

    config_path = Path(args.config).expanduser()
    candidate_paths = [config_path]
    if config_path.is_absolute():
        candidate_paths.append(repo_root() / "experiments" / "configs" / config_path.name)
    else:
        candidate_paths.append((repo_root() / config_path).resolve())
        candidate_paths.append(repo_root() / "experiments" / "configs" / config_path.name)

    resolved_path = next((path for path in candidate_paths if path.exists()), None)
    if resolved_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Could not find config file. Searched: {searched}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    merged = vars(args).copy()
    merged.update(config)
    return merged


def main():
    args = build_argparser().parse_args()
    config = load_config(args)
    os.environ["FORCE_MOCK"] = "0"

    data = load_multihop_sample(
        config["dataset"],
        split=config["split"],
        num_samples=int(config["samples"]),
        use_hetero=True,
    )
    rag = RAGPipeline(device=config.get("device"), use_v6_reranker=True)
    rag.add_evidence_units(data["corpus"])

    per_variant = {"dense_rerank": [], "adaptive": []}
    for query_item in data["queries"]:
        for variant in per_variant:
            per_variant[variant].append(
                evaluate_variant(
                    rag,
                    query_item,
                    variant=variant,
                    top_k=int(config["top_k"]),
                    prf_threshold=float(config["prf_threshold"]),
                )
            )

    bucket_view = {}
    for query_type in sorted({item.get("type", "unknown") for item in data["queries"]}):
        bucket_view[query_type] = {}
        for variant, records in per_variant.items():
            bucket_records = [item for item in records if item["query_type"] == query_type]
            bucket_view[query_type][variant] = summarize(bucket_records)

        dense_hits = [item["support_all_hit"] for item in per_variant["dense_rerank"] if item["query_type"] == query_type]
        adaptive_hits = [item["support_all_hit"] for item in per_variant["adaptive"] if item["query_type"] == query_type]
        if dense_hits and adaptive_hits:
            bucket_view[query_type]["adaptive_minus_dense"] = paired_bootstrap_delta(
                adaptive_hits,
                dense_hits,
                n_boot=int(config["bootstrap_samples"]),
            )

    output = {
        "config": {
            "dataset": config["dataset"],
            "split": config["split"],
            "samples": config["samples"],
            "top_k": config["top_k"],
            "prf_threshold": config["prf_threshold"],
            "bootstrap_samples": config["bootstrap_samples"],
        },
        "overall": {
            variant: summarize(records) for variant, records in per_variant.items()
        },
        "bucket_stats": bucket_view,
    }
    out_path = results_dir() / config["output_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved_to": str(out_path), "buckets": list(bucket_view.keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
