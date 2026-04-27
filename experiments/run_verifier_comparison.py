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
from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.llm_generator import llm_generate_answer, heuristic_generate_answer


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


def compute_f1(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    common = Counter(gold_tokens) & Counter(pred_tokens)
    overlap = sum(common.values())
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class OverlapVerifier:
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def evaluate_answer(self, generated_answer: str, evidence_chain):
        evidence_text = " ".join(node.get("text", "").lower() for node in evidence_chain)
        tokens = {tok for tok in re.findall(r"\w+", generated_answer.lower()) if len(tok) > 3}
        if not tokens:
            confidence = 1.0
        else:
            confidence = sum(1 for tok in tokens if tok in evidence_text) / len(tokens)
        status = "ACCEPTED" if confidence >= self.threshold else "REJECTED"
        return {
            "status": status,
            "reason": "Accepted by overlap verifier." if status == "ACCEPTED" else "Rejected by overlap verifier.",
            "avg_confidence": confidence,
            "claims": [],
        }


def build_verifier(name, threshold, config):
    if name == "cove":
        return CoVeVerifier(
            confidence_threshold=threshold,
            backend=config.get("verifier_backend", "heuristic") if not config.get("real_cove", False) else config.get("verifier_backend", "deepseek"),
            model=config.get("verifier_model", "deepseek-v4-flash"),
            api_key=config.get("verifier_api_key"),
            base_url=config.get("verifier_base_url"),
        )
    if name == "overlap":
        return OverlapVerifier(threshold=threshold)
    raise ValueError(f"Unknown verifier: {name}")


def summarize(records):
    return {
        "samples": len(records),
        "F1": round(mean(item["f1"] for item in records) * 100, 2),
        "No_Answer_Rate": round(mean(1.0 if item["rejected"] else 0.0 for item in records) * 100, 2),
        "False_Rejection_Rate": round(mean(1.0 if item["false_rejection"] else 0.0 for item in records) * 100, 2),
        "Unsafe_Accept_Rate": round(mean(1.0 if item["unsafe_accept"] else 0.0 for item in records) * 100, 2),
        "Avg_Verify_Confidence": round(mean(item["avg_confidence"] for item in records), 4),
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Compare CoVe and lightweight verifiers on the same retrieval chain.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wiki"], help="Dataset name.")
    parser.add_argument("--split", default="validation", help="Dataset split.")
    parser.add_argument("--samples", type=int, default=200, help="Number of evaluated samples.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k.")
    parser.add_argument("--prf-threshold", type=float, default=0.8, help="Adaptive retrieval threshold.")
    parser.add_argument("--device", default=None, help="Runtime device.")
    parser.add_argument("--output-name", default="verifier_comparison.json", help="Output JSON filename.")
    parser.add_argument("--local-data-dir", default=None, help="Directory containing offline dataset JSON/JSONL files.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache dir.")
    parser.add_argument("--offline", action="store_true", help="Use local files / cache only and avoid network dataset fetches.")
    parser.add_argument("--verifier-backend", default="heuristic", choices=["heuristic", "openai", "deepseek", "moonshot", "siliconflow"], help="Verification backend.")
    parser.add_argument("--verifier-model", default="deepseek-v4-flash", help="Verification model name.")
    parser.add_argument("--verifier-api-key", default=None, help="Optional explicit verifier API key.")
    parser.add_argument("--verifier-base-url", default=None, help="Optional explicit verifier base URL.")
    parser.add_argument("--real-cove", action="store_true", help="Force real LLM-based CoVe verification for cove_* variants.")
    return parser


def load_config(args):
    defaults = {
        "verifiers": [
            {"name": "cove_soft", "mode": "cove", "threshold": 0.35},
            {"name": "cove_standard", "mode": "cove", "threshold": 0.50},
            {"name": "cove_strict", "mode": "cove", "threshold": 0.90},
            {"name": "overlap_soft", "mode": "overlap", "threshold": 0.35},
        ]
    }
    merged = vars(args).copy()
    merged.update(defaults)
    merged.update(load_local_private_overrides())
    if args.config:
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
            merged.update(json.load(f))
    return merged


def main():
    args = build_argparser().parse_args()
    config = load_config(args)
    os.environ["FORCE_MOCK"] = "0"
    if config.get("real_cove", False) and config.get("verifier_backend", "heuristic") == "heuristic":
        config["verifier_backend"] = "deepseek"

    data = load_multihop_sample(
        config["dataset"],
        split=config["split"],
        num_samples=int(config["samples"]),
        use_hetero=True,
        local_data_dir=config.get("local_data_dir"),
        hf_cache_dir=config.get("hf_cache_dir"),
        offline=bool(config.get("offline", False)),
    )
    rag = RAGPipeline(device=config.get("device"), use_v6_reranker=True)
    rag.add_evidence_units(data["corpus"])

    variants = {}
    for verifier_cfg in config["verifiers"]:
        verifier = build_verifier(verifier_cfg["mode"], float(verifier_cfg["threshold"]), config)
        records = []
        for query_item in data["queries"]:
            search_res = rag.search_with_chain(
                query_item["query"],
                top_k=int(config["top_k"]),
                prf_threshold=float(config["prf_threshold"]),
            )
            # Use LLM generator if configured, otherwise fallback to heuristic
            generator_backend = config.get("generator_backend", "heuristic")
            if generator_backend and generator_backend != "heuristic":
                draft = llm_generate_answer(
                    query_item["query"],
                    search_res["results"],
                    model=config.get("generator_model", "deepseek-v4-flash"),
                    backend=generator_backend,
                    api_key=config.get("generator_api_key"),
                    base_url=config.get("generator_base_url"),
                )
            else:
                draft = heuristic_generate_answer(query_item["query"], search_res["results"])
            verification = verifier.evaluate_answer(draft, search_res["chain"])
            rejected = verification.get("status") == "REJECTED"
            final_answer = "No-Answer" if rejected else draft
            f1 = 0.0 if rejected else compute_f1(query_item["answer"], final_answer)
            false_rejection = rejected and normalize_answer(query_item["answer"]) != "noanswer"
            unsafe_accept = (not rejected) and f1 < 0.1
            records.append({
                "id": query_item["id"],
                "f1": f1,
                "rejected": rejected,
                "false_rejection": false_rejection,
                "unsafe_accept": unsafe_accept,
                "avg_confidence": float(verification.get("avg_confidence", 0.0)),
            })

        variants[verifier_cfg["name"]] = {
            "mode": verifier_cfg["mode"],
            "threshold": verifier_cfg["threshold"],
            "summary": summarize(records),
        }

    output = {
        "config": {
            "dataset": config["dataset"],
            "split": config["split"],
            "samples": config["samples"],
            "top_k": config["top_k"],
            "prf_threshold": config["prf_threshold"],
            "verifiers": config["verifiers"],
            "real_cove": bool(config.get("real_cove", False)),
            "verifier_backend": config.get("verifier_backend", "heuristic"),
            "verifier_model": config.get("verifier_model", "deepseek-v4-flash"),
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
