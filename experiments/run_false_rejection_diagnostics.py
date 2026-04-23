import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.cove_verifier import CoVeVerifier
from rererank_v1.dataset_loader import load_multihop_sample
from rererank_v1.paths import repo_root, results_dir
from rererank_v1.rag_pipeline import RAGPipeline, heuristic_generate_answer


def load_local_private_overrides():
    local_override = repo_root() / "experiments" / "configs" / "local_api_overrides.json"
    if not local_override.exists():
        return {}
    with open(local_override, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc((text or "").lower())))


def answer_present_in_results(answer: str, results) -> bool:
    answer_norm = normalize_answer(answer)
    if answer_norm in {"", "yes", "no", "noanswer"}:
        return False
    return any(answer_norm in normalize_answer(item.get("text", "")) for item in results)


def summarize_variant(records):
    return {
        "samples": len(records),
        "rejection_rate": round(mean(1.0 if item["rejected"] else 0.0 for item in records) * 100, 2),
        "false_rejection_rate": round(mean(1.0 if item["false_rejection"] else 0.0 for item in records) * 100, 2),
        "unsafe_accept_rate": round(mean(1.0 if item["unsafe_accept"] else 0.0 for item in records) * 100, 2),
        "avg_confidence": round(mean(item["avg_confidence"] for item in records), 4),
        "reason_buckets": dict(Counter(item["reason_bucket"] for item in records if item["reason_bucket"])),
    }


def diagnose_reason(gold_answer: str, results, verification):
    if not results:
        return "empty_retrieval"
    if not answer_present_in_results(gold_answer, results):
        return "evidence_missing"
    if (verification or {}).get("avg_confidence", 0.0) < 0.35:
        return "low_verify_confidence"
    return "partial_support_reject"


def run_variant(rag, query_item, top_k, prf_threshold, verifier_threshold, config):
    search_res = rag.search_with_chain(query_item["query"], top_k=top_k, prf_threshold=prf_threshold)
    results = search_res["results"]
    chain = search_res["chain"]
    draft = heuristic_generate_answer(query_item["query"], results)
    verifier = CoVeVerifier(
        confidence_threshold=verifier_threshold,
        backend=config.get("verifier_backend", "heuristic") if not config.get("real_cove", False) else config.get("verifier_backend", "moonshot"),
        model=config.get("verifier_model", "moonshot-v1-8k"),
        api_key=config.get("verifier_api_key"),
        base_url=config.get("verifier_base_url"),
    )
    verification = verifier.evaluate_answer(draft, chain)

    rejected = verification.get("status") == "REJECTED"
    false_rejection = rejected and normalize_answer(query_item["answer"]) != "noanswer"
    reason_bucket = diagnose_reason(query_item["answer"], results, verification) if false_rejection else None
    unsafe_accept = (not rejected) and not answer_present_in_results(query_item["answer"], results)

    return {
        "id": query_item["id"],
        "query": query_item["query"],
        "answer": query_item["answer"],
        "draft_answer": draft,
        "rejected": rejected,
        "false_rejection": false_rejection,
        "unsafe_accept": unsafe_accept,
        "avg_confidence": round(float(verification.get("avg_confidence", 0.0)), 4),
        "reason_bucket": reason_bucket,
        "retrieved_titles": [item.get("metadata", {}).get("title", "") for item in results[:top_k]],
        "verification": verification,
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Run false-rejection diagnostics for CoVe variants.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wiki"], help="Dataset name.")
    parser.add_argument("--split", default="validation", help="Dataset split.")
    parser.add_argument("--samples", type=int, default=200, help="Number of evaluated samples.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k.")
    parser.add_argument("--prf-threshold", type=float, default=0.8, help="Adaptive retrieval threshold.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9], help="Verifier thresholds to compare.")
    parser.add_argument("--hetero", action="store_true", help="Use heterogeneous corpus instead of text-only corpus.")
    parser.add_argument("--device", default=None, help="Runtime device.")
    parser.add_argument("--output-name", default="false_rejection_diagnostics.json", help="Output JSON filename.")
    parser.add_argument("--local-data-dir", default=None, help="Directory containing offline dataset JSON/JSONL files.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache dir.")
    parser.add_argument("--offline", action="store_true", help="Use local files / cache only and avoid network dataset fetches.")
    parser.add_argument("--verifier-backend", default="heuristic", choices=["heuristic", "openai", "moonshot", "siliconflow"], help="Verification backend.")
    parser.add_argument("--verifier-model", default="moonshot-v1-8k", help="Verification model name.")
    parser.add_argument("--verifier-api-key", default=None, help="Optional explicit verifier API key.")
    parser.add_argument("--verifier-base-url", default=None, help="Optional explicit verifier base URL.")
    parser.add_argument("--real-cove", action="store_true", help="Force real LLM-based CoVe verification.")
    return parser


def load_config(args):
    merged = vars(args).copy()
    merged.update(load_local_private_overrides())
    if not args.config:
        return merged

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
    merged.update(config)
    return merged


def main():
    args = build_argparser().parse_args()
    config = load_config(args)
    os.environ["FORCE_MOCK"] = "0"
    if config.get("real_cove", False) and config.get("verifier_backend", "heuristic") == "heuristic":
        config["verifier_backend"] = "moonshot"

    data = load_multihop_sample(
        config["dataset"],
        split=config["split"],
        num_samples=int(config["samples"]),
        use_hetero=bool(config.get("hetero", False)),
        local_data_dir=config.get("local_data_dir"),
        hf_cache_dir=config.get("hf_cache_dir"),
        offline=bool(config.get("offline", False)),
    )
    rag = RAGPipeline(device=config.get("device"), use_v6_reranker=True)
    rag.add_evidence_units(data["corpus"])

    variants = {}
    for threshold in config["thresholds"]:
        key = f"cove_t{threshold:.2f}"
        records = [
            run_variant(
                rag,
                query_item,
                top_k=int(config["top_k"]),
                prf_threshold=float(config["prf_threshold"]),
                verifier_threshold=float(threshold),
                config=config,
            )
            for query_item in data["queries"]
        ]
        variants[key] = {
            "threshold": threshold,
            "summary": summarize_variant(records),
            "records": records,
        }

    output = {
        "config": {
            "dataset": config["dataset"],
            "split": config["split"],
            "samples": config["samples"],
            "top_k": config["top_k"],
            "prf_threshold": config["prf_threshold"],
            "hetero": config.get("hetero", False),
            "thresholds": config["thresholds"],
            "real_cove": bool(config.get("real_cove", False)),
            "verifier_backend": config.get("verifier_backend", "heuristic"),
            "verifier_model": config.get("verifier_model", "moonshot-v1-8k"),
        },
        "variants": variants,
    }
    out_path = results_dir() / config["output_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved_to": str(out_path), "variants": list(variants.keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
