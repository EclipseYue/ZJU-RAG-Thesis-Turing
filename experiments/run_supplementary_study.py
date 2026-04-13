import collections
import json
import os
import re
import string
from pathlib import Path
from statistics import mean

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.dataset_loader import load_multihop_sample
from rererank_v1.rag_pipeline import RAGPipeline, heuristic_generate_answer


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_em(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def answer_present_in_results(gold: str, results) -> bool:
    gold_norm = normalize_answer(gold)
    if gold_norm in {"", "yes", "no", "noanswer"}:
        return False
    return any(gold_norm in normalize_answer(r.get("text", "")) for r in results)


def cot_only_answer(query: str) -> str:
    # A lightweight no-retrieval baseline used to measure retrieval value gap.
    return f"让我们逐步思考：{query}。但在没有外部证据时，无法给出可靠答案。"


def bootstrap_ci(values, n_boot=500, seed=42):
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        means.append(arr[idx].mean())
    means = np.sort(np.array(means))
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def run_variant(rag, variant: str, query: str, top_k: int = 5):
    if variant == "naive_rag":
        results = rag._retrieve(query, top_k=top_k)
        answer = heuristic_generate_answer(query, results)
        return results, answer, None

    if variant == "dense_rerank":
        results = rag._rerank(query, rag._retrieve(query, top_k=top_k))
        answer = heuristic_generate_answer(query, results)
        return results, answer, None

    if variant == "adaptive_prf":
        results = rag.search(query, top_k=top_k, active_retrieval=True, prf_threshold=0.85)
        answer = heuristic_generate_answer(query, results)
        return results, answer, None

    if variant == "adaptive_cove":
        res = rag.search_with_chain(query, top_k=top_k, prf_threshold=0.85)
        results = res["results"]
        draft = heuristic_generate_answer(query, results)
        cove_res = rag.verify_answer(draft, res["chain"])
        final_answer = "No-Answer" if cove_res["status"] == "REJECTED" else draft
        return results, final_answer, cove_res

    if variant == "adaptive_cove_strict":
        from rererank_v1.cove_verifier import CoVeVerifier
        res = rag.search_with_chain(query, top_k=top_k, prf_threshold=0.85)
        results = res["results"]
        top_texts = [r.get("text", "") for r in results[:3]]
        draft = " ".join([heuristic_generate_answer(query, results)] + top_texts)[:420]
        
        # Manually create a strict verifier
        strict_verifier = CoVeVerifier(confidence_threshold=0.90)
        cove_res = strict_verifier.evaluate_answer(draft, res["chain"])
        final_answer = "No-Answer" if cove_res["status"] == "REJECTED" else draft
        return results, final_answer, cove_res

    if variant == "ircot_lite":
        # Simulates an iterative retrieval baseline
        results = rag._retrieve(query, top_k=top_k)
        if results:
            expanded_query = query + " " + results[0].get("text", "")[:50]
            results.extend(rag._retrieve(expanded_query, top_k=top_k))
        results = rag._rerank(query, results)
        answer = heuristic_generate_answer(query, results)
        return results, answer, None

    if variant == "cot_only":
        return [], cot_only_answer(query), None

    raise ValueError(f"Unknown variant: {variant}")


def evaluate_dataset(dataset_name: str, num_samples: int = 100, top_k: int = 5):
    # Load pure text corpus
    data_text = load_multihop_sample(dataset_name, num_samples=num_samples, use_hetero=False)
    rag_text = RAGPipeline(use_v6_reranker=True)
    rag_text.add_evidence_units(data_text["corpus"])

    # Load heterogeneous corpus (simulated extraction)
    data_hetero = load_multihop_sample(dataset_name, num_samples=num_samples, use_hetero=True)
    rag_hetero = RAGPipeline(use_v6_reranker=True)
    rag_hetero.add_evidence_units(data_hetero["corpus"])

    variants = [
        "naive_rag",
        "dense_rerank",
        "ircot_lite",
        "adaptive_prf",
        "adaptive_cove",
        "adaptive_cove_strict",
        "cot_only",
        "hetero_dense_rerank",
        "hetero_adaptive_prf",
        "hetero_adaptive_cove",
    ]
    rows = {v: {"f1": [], "em": [], "no_answer": []} for v in variants}

    cove_reject_analysis = {
        "total": 0,
        "rejected": 0,
        "false_rejection": 0,
        "unsafe_accept": 0,
        "reason_buckets": collections.Counter(),
    }

    for idx, item in enumerate(data_text["queries"], start=1):
        if idx % 20 == 0:
            print(f"[{dataset_name}] processed {idx}/{len(data_text['queries'])}")
        gold = item["answer"]
        query = item["query"]

        for variant in variants:
            # Route to appropriate RAG pipeline
            is_hetero = variant.startswith("hetero_")
            current_rag = rag_hetero if is_hetero else rag_text
            base_variant = variant.replace("hetero_", "") if is_hetero else variant

            results, pred, cove_res = run_variant(current_rag, base_variant, query, top_k=top_k)
            f1 = compute_f1(gold, pred)
            em = compute_em(gold, pred)
            no_answer_flag = int(normalize_answer(pred) == "noanswer")

            rows[variant]["f1"].append(f1)
            rows[variant]["em"].append(em)
            rows[variant]["no_answer"].append(no_answer_flag)

            if variant in {"adaptive_cove", "adaptive_cove_strict"}:
                cove_reject_analysis["total"] += 1
                if no_answer_flag == 1:
                    cove_reject_analysis["rejected"] += 1
                    if normalize_answer(gold) != "noanswer":
                        cove_reject_analysis["false_rejection"] += 1
                        if not results:
                            cove_reject_analysis["reason_buckets"]["empty_retrieval"] += 1
                        elif not answer_present_in_results(gold, results):
                            cove_reject_analysis["reason_buckets"]["evidence_missing"] += 1
                        elif (cove_res or {}).get("avg_confidence", 0.0) < 0.35:
                            cove_reject_analysis["reason_buckets"]["low_verify_confidence"] += 1
                        else:
                            cove_reject_analysis["reason_buckets"]["partial_support_reject"] += 1
                else:
                    if f1 < 0.1:
                        cove_reject_analysis["unsafe_accept"] += 1

    summary_rows = []
    for variant in variants:
        f1_values = rows[variant]["f1"]
        em_values = rows[variant]["em"]
        na_values = rows[variant]["no_answer"]
        summary_rows.append({
            "Variant": variant,
            "F1": round(mean(f1_values) * 100, 2),
            "EM": round(mean(em_values) * 100, 2),
            "No_Answer_Rate": round(mean(na_values) * 100, 2),
            "F1_CI95": bootstrap_ci(f1_values),
            "EM_CI95": bootstrap_ci(em_values),
        })

    return {
        "dataset": dataset_name,
        "num_samples": len(data_text["queries"]),
        "summary": summary_rows,
        "cove_reject_analysis": {
            "total": cove_reject_analysis["total"],
            "rejected": cove_reject_analysis["rejected"],
            "rejected_rate": round(
                100 * cove_reject_analysis["rejected"] / max(cove_reject_analysis["total"], 1), 2
            ),
            "false_rejection": cove_reject_analysis["false_rejection"],
            "false_rejection_rate": round(
                100 * cove_reject_analysis["false_rejection"] / max(cove_reject_analysis["total"], 1), 2
            ),
            "unsafe_accept": cove_reject_analysis["unsafe_accept"],
            "reason_buckets": dict(cove_reject_analysis["reason_buckets"]),
        },
    }


def main():
    os.environ["FORCE_MOCK"] = "0"

    config = {
        "hotpot_samples": 100,
        "wiki2_samples": 100,
        "top_k": 5,
    }

    hotpot = evaluate_dataset("hotpotqa", num_samples=config["hotpot_samples"], top_k=config["top_k"])
    wiki2 = evaluate_dataset("2wiki", num_samples=config["wiki2_samples"], top_k=config["top_k"])

    out = {
        "config": config,
        "results": {
            "hotpotqa": hotpot,
            "2wikimultihopqa": wiki2,
        },
    }

    out_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "supplementary_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved supplementary study to: {out_file}")
    print(json.dumps(out["results"]["hotpotqa"]["cove_reject_analysis"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
