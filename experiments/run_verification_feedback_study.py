import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.cove_verifier import CoVeVerifier
from rererank_v1.dataset_loader import load_multihop_sample
from rererank_v1.llm_generator import heuristic_generate_answer, llm_generate_answer
from rererank_v1.paths import repo_root, results_dir
from rererank_v1.rag_pipeline import RAGPipeline

STOPWORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "were", "was",
    "are", "is", "did", "does", "do", "the", "a", "an", "of", "in", "on",
    "for", "to", "by", "and", "or", "with", "from", "that", "this", "these",
    "those", "same", "held", "position", "located", "director", "woman",
    "man", "film", "series", "book", "books", "city", "country",
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_private_overrides() -> Dict[str, Any]:
    local_override = repo_root() / "experiments" / "configs" / "local_api_overrides.json"
    return load_json(local_override) if local_override.exists() else {}


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def f1_score(gold: str, pred: str) -> float:
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


def exact_match(gold: str, pred: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(pred))


def extract_titles(results: List[Dict[str, Any]], top_k: int) -> List[str]:
    titles = []
    for item in results[:top_k]:
        title = (item.get("metadata", {}) or {}).get("title")
        if title:
            titles.append(title)
    return titles


def snapshot_stats(rag: RAGPipeline) -> Dict[str, float]:
    return {
        "total_tokens": float(rag.stats.get("total_tokens", 0)),
        "total_latency": float(rag.stats.get("total_latency", 0.0)),
        "retrieval_calls": float(rag.stats.get("retrieval_calls", 0)),
        "reranker_calls": float(rag.stats.get("reranker_calls", 0)),
    }


def stats_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in before}


def build_verifier(variant: Dict[str, Any], config: Dict[str, Any]) -> CoVeVerifier:
    return CoVeVerifier(
        confidence_threshold=float(variant.get("threshold", config.get("cove_threshold", 0.5))),
        backend=config.get("verifier_backend", "heuristic") if not config.get("real_cove", False) else config.get("verifier_backend", "deepseek"),
        model=config.get("verifier_model", "deepseek-v4-flash"),
        api_key=config.get("verifier_api_key"),
        base_url=config.get("verifier_base_url"),
        decision_policy=variant.get("decision_policy", "soft"),
        min_claim_confidence=variant.get("min_claim_confidence"),
    )


def generate_answer(query: str, results: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    backend = config.get("generator_backend", "heuristic")
    if backend == "heuristic":
        return heuristic_generate_answer(query, results)
    return llm_generate_answer(
        query,
        results,
        backend=backend,
        model=config.get("generator_model", "deepseek-v4-flash"),
        api_key=config.get("generator_api_key"),
        base_url=config.get("generator_base_url"),
        answer_mode=config.get("answer_mode", "concise"),
    )


def failed_claim_text(verification: Dict[str, Any], fallback: str) -> str:
    claims = [
        item.get("claim", "")
        for item in verification.get("claims", [])
        if not item.get("supported", False)
    ]
    text = " ".join(claim for claim in claims if claim)
    return text[:240] if text else fallback[:240]


def useful_text_fragment(text: str) -> bool:
    normalized = normalize_answer(text)
    if not normalized or normalized in {"noanswer", "no answer", "unknown"}:
        return False
    if normalized.startswith(("we need", "need to", "i need", "answer based")):
        return False
    return len(normalized.split()) >= 2


def extract_query_terms(*texts: str) -> List[str]:
    terms: List[str] = []
    seen = set()

    def add(term: str) -> None:
        clean = re.sub(r"\s+", " ", term).strip(" '\".,;:!?()[]{}")
        key = clean.lower()
        if clean and key not in seen and key not in STOPWORDS:
            terms.append(clean)
            seen.add(key)

    for text in texts:
        for quoted in re.findall(r'"([^"]{2,80})"', text or ""):
            add(quoted)
        for phrase in re.findall(r"\b[A-Z][A-Za-z0-9'’-]*(?:\s+[A-Z][A-Za-z0-9'’-]*){0,5}", text or ""):
            add(phrase)
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9'’-]{4,}\b", text or ""):
            if token.lower() not in STOPWORDS:
                add(token)
    return terms[:10]


def build_retry_query(
    query: str,
    verification: Dict[str, Any],
    draft: str,
    results: List[Dict[str, Any]],
    strategy: str,
    top_k: int,
) -> str:
    if strategy == "claim_concat":
        retry_hint = failed_claim_text(verification, draft)
        return f"{query} {retry_hint}".strip()

    failed_claims = [
        item.get("claim", "")
        for item in verification.get("claims", [])
        if not item.get("supported", False) and useful_text_fragment(item.get("claim", ""))
    ]
    retrieved_titles = extract_titles(results, top_k=3)
    terms = extract_query_terms(query, draft, " ".join(failed_claims), " ".join(retrieved_titles))

    parts = [query]
    if terms:
        parts.append("Key entities: " + "; ".join(terms[:8]))
    if retrieved_titles:
        parts.append("Retrieved evidence titles: " + "; ".join(retrieved_titles[:3]))
    if failed_claims:
        parts.append("Missing or weak claim: " + " ".join(failed_claims)[:160])
    parts.append("Find the missing bridge evidence and answer entity.")
    return " ".join(parts)[:500]


def evaluate_item(
    rag: RAGPipeline,
    query_item: Dict[str, Any],
    variant: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    top_k = int(config.get("top_k", 5))
    prf_threshold = float(config.get("prf_threshold", 0.8))
    before = snapshot_stats(rag)

    search = rag.search_with_chain(query_item["query"], top_k=top_k, prf_threshold=prf_threshold)
    draft = generate_answer(query_item["query"], search["results"], config)
    verifier = build_verifier(variant, config)
    verification = verifier.evaluate_answer(draft, search["chain"])
    used_feedback = False
    final_results = search["results"]
    final_verification = verification
    final_answer = draft
    retry_query = ""

    if verification.get("status") == "REJECTED" and variant.get("feedback_retry", False):
        used_feedback = True
        retry_query = build_retry_query(
            query_item["query"],
            verification,
            draft,
            search["results"],
            strategy=variant.get("feedback_strategy", "claim_concat"),
            top_k=top_k,
        )
        retry_search = rag.search_with_chain(retry_query, top_k=top_k, prf_threshold=prf_threshold)
        retry_draft = generate_answer(query_item["query"], retry_search["results"], config)
        retry_verification = verifier.evaluate_answer(retry_draft, retry_search["chain"])
        final_results = retry_search["results"]
        final_answer = retry_draft
        final_verification = retry_verification

    rejected = final_verification.get("status") == "REJECTED"
    pred = "No-Answer" if rejected else final_answer
    after = snapshot_stats(rag)
    deltas = stats_delta(before, after)
    supporting_titles = set(query_item.get("supporting_titles", []))
    predicted_titles = extract_titles(final_results, top_k)
    support_hits = len(supporting_titles.intersection(predicted_titles))

    return {
        "id": query_item.get("id"),
        "query": query_item["query"],
        "answer": query_item.get("answer", ""),
        "draft_answer": draft,
        "predicted_answer": pred,
        "rejected": rejected,
        "used_feedback": used_feedback,
        "feedback_strategy": variant.get("feedback_strategy", "none"),
        "retry_query": retry_query,
        "exact_match": 0.0 if rejected else exact_match(query_item.get("answer", ""), pred),
        "f1": 0.0 if rejected else f1_score(query_item.get("answer", ""), pred),
        "support_recall": support_hits / len(supporting_titles) if supporting_titles else 0.0,
        "support_all_hit": bool(supporting_titles and support_hits == len(supporting_titles)),
        "predicted_titles": predicted_titles,
        "supporting_titles": sorted(supporting_titles),
        "avg_confidence": float(final_verification.get("avg_confidence", 0.0)),
        "min_confidence": float(final_verification.get("min_confidence", 0.0)),
        "unsupported_count": int(final_verification.get("unsupported_count", 0)),
        "verification_reason": final_verification.get("reason", ""),
        "stats_delta": deltas,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "Samples": len(records),
        "ExactMatch": round(mean(item["exact_match"] for item in records) * 100, 2),
        "F1_Score": round(mean(item["f1"] for item in records) * 100, 2),
        "SupportRecall@K": round(mean(item["support_recall"] for item in records) * 100, 2),
        "SupportAllHit@K": round(mean(1.0 if item["support_all_hit"] else 0.0 for item in records) * 100, 2),
        "No_Answer_Rate_Percent": round(mean(1.0 if item["rejected"] else 0.0 for item in records) * 100, 2),
        "Feedback_Rate_Percent": round(mean(1.0 if item["used_feedback"] else 0.0 for item in records) * 100, 2),
        "Avg_Verify_Confidence": round(mean(item["avg_confidence"] for item in records), 4),
        "Avg_Latency_ms": round(mean(item["stats_delta"]["total_latency"] for item in records) * 1000, 2),
        "Avg_Retrieval_Calls": round(mean(item["stats_delta"]["retrieval_calls"] for item in records), 2),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare hard verification, soft verification, and verification-aware retry.")
    parser.add_argument("--config", default="experiments/configs/verification_feedback_study.json")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--real-cove", action="store_true")
    return parser


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    config = load_json(config_path)
    config.update(load_local_private_overrides())
    if args.samples is not None:
        config["samples"] = args.samples
    if args.output_name is not None:
        config["output_name"] = args.output_name
    if args.real_cove:
        config["real_cove"] = True
    return config


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args)
    os.environ["FORCE_MOCK"] = "0"
    if config.get("real_cove", False) and config.get("verifier_backend", "heuristic") == "heuristic":
        config["verifier_backend"] = "deepseek"

    data = load_multihop_sample(
        config["dataset"],
        split=config.get("split", "validation"),
        num_samples=int(config.get("samples", 100)),
        use_hetero=bool(config.get("hetero", False)),
        local_data_dir=config.get("local_data_dir"),
        hf_cache_dir=config.get("hf_cache_dir"),
        offline=bool(config.get("offline", False)),
    )
    rag = RAGPipeline(device=config.get("device"), use_v6_reranker=True)
    rag.add_evidence_units(data["corpus"])

    variants = {}
    for variant in config["variants"]:
        records = [evaluate_item(rag, item, variant, config) for item in data["queries"]]
        variants[variant["name"]] = {
            "config": variant,
            "summary": summarize(records),
            "records": records,
        }

    payload = {
        "config": {
            key: value
            for key, value in config.items()
            if key not in {"generator_api_key", "verifier_api_key"}
        },
        "variants": variants,
    }
    out_path = results_dir() / config.get("output_name", "verification_feedback_study.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved_to": str(out_path), "variants": list(variants.keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
