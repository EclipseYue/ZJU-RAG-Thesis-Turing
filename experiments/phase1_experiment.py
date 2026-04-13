
import sys
import os
import time
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.benchmark_data import DOCUMENTS, TEST_CASES
from rererank_v1.metrics import calculate_mrr, calculate_ndcg, calculate_precision

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def run_phase1_experiment():
    logger.info("=" * 60)
    logger.info("🧪 Phase 1: Multi-hop Reasoning & Active Retrieval Baseline")
    logger.info("=" * 60)
    
    os.environ["FORCE_MOCK"] = "1"
    rag = RAGPipeline(use_v6_reranker=False)
    rag.add_documents(DOCUMENTS)
    
    results_stats = {
        "total_queries": 0,
        "mrr_final": [],
        "ndcg_final": [],
        "precision_final": [],
        "latency": [],
        "tokens": []
    }
    
    def get_relevant_ranks(results_list, relevant_doc_ids):
        ranks = []
        for doc_id in relevant_doc_ids:
            found_rank = next((i + 1 for i, item in enumerate(results_list) if item.get("id") == doc_id), 0)
            if found_rank > 0:
                ranks.append(found_rank)
        return sorted(ranks)
    
    for case in TEST_CASES:
        results_stats["total_queries"] += 1
        logger.info(f"\nQuery: {case.query}")
        
        rag.stats["total_latency"] = 0.0
        rag.stats["total_tokens"] = 0
        
        final_results = rag.search(case.query, top_k=5, active_retrieval=True)
        
        ranks_final = get_relevant_ranks(final_results, case.relevant_doc_ids)
        mrr = calculate_mrr(ranks_final)
        ndcg = calculate_ndcg(ranks_final, k=5)
        p3 = calculate_precision(ranks_final, k=3)
        
        results_stats["mrr_final"].append(mrr)
        results_stats["ndcg_final"].append(ndcg)
        results_stats["precision_final"].append(p3)
        results_stats["latency"].append(rag.stats["total_latency"])
        results_stats["tokens"].append(rag.stats["total_tokens"])
        
    avg_mrr = sum(results_stats["mrr_final"]) / len(TEST_CASES)
    avg_ndcg = sum(results_stats["ndcg_final"]) / len(TEST_CASES)
    avg_p3 = sum(results_stats["precision_final"]) / len(TEST_CASES)
    
    latencies = sorted(results_stats["latency"])
    p50_lat = latencies[int(len(latencies)*0.5)]
    p95_lat = latencies[int(len(latencies)*0.95)]
    avg_tokens = sum(results_stats["tokens"]) / len(TEST_CASES)
    
    logger.info("=" * 60)
    logger.info("📊 Phase 1 Results Summary")
    logger.info(f"MRR:         {avg_mrr:.4f}")
    logger.info(f"NDCG@5:      {avg_ndcg:.4f}")
    logger.info(f"Precision@3: {avg_p3:.4f}")
    logger.info(f"Avg Tokens:  {avg_tokens:.1f}")
    logger.info(f"p50 Latency: {p50_lat:.3f}s")
    logger.info(f"p95 Latency: {p95_lat:.3f}s")
    
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "research_history.json")
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        history = {"iterations": [], "plan": []}
        
    history["iterations"].append({
        "id": len(history.get("iterations", [])) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Phase 1: Multi-hop Reasoning & Active Retrieval Baseline",
        "metrics": {
            "mrr_final": avg_mrr,
            "ndcg_final": avg_ndcg,
            "p50_latency": p50_lat,
            "p95_latency": p95_lat,
            "avg_tokens": avg_tokens
        }
    })
    
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Results saved to {data_file}")

if __name__ == "__main__":
    run_phase1_experiment()
