import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.benchmark_data import DOCUMENTS, TEST_CASES
from rererank_v1.paths import data_dir, docs_dir
import logging
import os

# Suppress pipeline logs for cleaner benchmark output
logging.getLogger("rag_pipeline").setLevel(logging.ERROR)

def calculate_mrr(rank_list):
    """Calculate Mean Reciprocal Rank."""
    if not rank_list:
        return 0.0
    return np.mean([1.0 / r if r > 0 else 0.0 for r in rank_list])

def calculate_ndcg(rank_list, k=5):
    """Calculate Normalized Discounted Cumulative Gain @ K."""
    if not rank_list:
        return 0.0
    
    # DCG
    dcg = 0.0
    for i, rank in enumerate(rank_list):
        if rank <= k:
            # Re-rank list is not available here, so we assume rank_list contains ranks of RELEVANT docs
            # This is a simplified NDCG where we assume binary relevance (1 or 0)
            # and that we found 'len(rank_list)' relevant docs at positions 'rank'
            dcg += 1.0 / np.log2(rank + 1)
            
    # IDCG (Ideal DCG) - assuming all relevant docs are at the top
    idcg = 0.0
    num_relevant = len(rank_list)
    for i in range(min(num_relevant, k)):
        idcg += 1.0 / np.log2(i + 1 + 1)
        
    if idcg == 0: return 0.0
    return dcg / idcg

def calculate_precision(rank_list, k=3):
    """Calculate Precision @ K."""
    if not rank_list:
        return 0.0
    # Count how many relevant docs are in top K
    relevant_in_top_k = sum(1 for r in rank_list if r <= k)
    return relevant_in_top_k / k

import json
from datetime import datetime

def save_benchmark_result(metrics, description="Benchmark Run"):
    """Save results to JSON for the dashboard."""
    data_file = data_dir() / "research_history.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing data
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"iterations": [], "plan": []}
    
    # Create new iteration entry
    new_iteration = {
        "id": len(data["iterations"]) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "metrics": {
            "mrr_retrieval": metrics['mrr_retrieval'],
            "mrr_final": metrics['mrr_final'],
            "ndcg_retrieval": metrics['ndcg_retrieval'],
            "ndcg_final": metrics['ndcg_final']
        }
    }
    
    data["iterations"].append(new_iteration)
    
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Results saved to {data_file} for visualization.")

def run_benchmark():
    print("="*60)
    print("🧪 STARTING EXTENDED RERANKER RESEARCH BENCHMARK (V2)")
    print("="*60)
    print(f"📚 Corpus Size: {len(DOCUMENTS)} documents")
    print(f"🔍 Test Queries: {len(TEST_CASES)}")
    
    # Force mock mode for consistency if real models aren't available
    os.environ['FORCE_MOCK'] = '1'
    rag = RAGPipeline()
    rag.add_documents(DOCUMENTS)
    
    results_stats = {
        'total_queries': 0,
        # Metrics storage: lists of metric values per query
        'mrr_retrieval': [], 'mrr_rerank': [], 'mrr_final': [],
        'ndcg_retrieval': [], 'ndcg_rerank': [], 'ndcg_final': [],
        'p3_retrieval': [], 'p3_rerank': [], 'p3_final': [],
        
        'scores_relevant': [],
        'scores_irrelevant': [],
        'prf_triggered': 0,
        'no_answer_total': 0,
        'no_answer_safe': 0
    }

    print("\nProcessing Queries...")
    print("-" * 60)
    
    for case in TEST_CASES:
        results_stats['total_queries'] += 1
        print(f"Query: {case.query}")
        
        # --- Helper to extract ranks of relevant docs ---
        def get_relevant_ranks(results_list):
            ranks = []
            for doc_id in case.relevant_doc_ids:
                # Rank is index + 1
                # Try to find doc_id in results
                found_rank = next((i+1 for i, item in enumerate(results_list) if item['id'] == doc_id), 0)
                if found_rank > 0:
                    ranks.append(found_rank)
            return sorted(ranks)

        # 1. Vector Retrieval
        retrieved_docs = rag._retrieve(case.query, top_k=20) # Retrieve more to allow reranker to work
        ranks_retrieval = get_relevant_ranks(retrieved_docs)
        
        # 2. Reranking
        reranked_docs = rag._rerank(case.query, retrieved_docs)
        ranks_rerank = get_relevant_ranks(reranked_docs)
        
        # Collect scores
        for doc in reranked_docs:
            score = doc['rerank_score']
            if doc['id'] in case.relevant_doc_ids:
                results_stats['scores_relevant'].append(score)
            else:
                results_stats['scores_irrelevant'].append(score)

        if len(case.relevant_doc_ids) == 0:
            results_stats['no_answer_total'] += 1
            top_rerank_score = reranked_docs[0]['rerank_score'] if reranked_docs else 1.0
            if top_rerank_score < 0.5:
                results_stats['no_answer_safe'] += 1

        # 3. Full Pipeline (PRF/RRF)
        final_results = rag.search(case.query, top_k=10)
        ranks_final = get_relevant_ranks(final_results)
        
        if final_results and 'sources' in final_results[0]:
             results_stats['prf_triggered'] += 1

        # --- Calculate Metrics per Query ---
        # MRR
        results_stats['mrr_retrieval'].append(calculate_mrr(ranks_retrieval))
        results_stats['mrr_rerank'].append(calculate_mrr(ranks_rerank))
        results_stats['mrr_final'].append(calculate_mrr(ranks_final))
        
        # NDCG@5
        results_stats['ndcg_retrieval'].append(calculate_ndcg(ranks_retrieval, k=5))
        results_stats['ndcg_rerank'].append(calculate_ndcg(ranks_rerank, k=5))
        results_stats['ndcg_final'].append(calculate_ndcg(ranks_final, k=5))
        
        # Precision@3
        results_stats['p3_retrieval'].append(calculate_precision(ranks_retrieval, k=3))
        results_stats['p3_rerank'].append(calculate_precision(ranks_rerank, k=3))
        results_stats['p3_final'].append(calculate_precision(ranks_final, k=3))

        best_rank_final = ranks_final[0] if ranks_final else 0
        print(f"  -> Top Relevant Rank: Final={best_rank_final} | P@3={results_stats['p3_final'][-1]:.2f} | NDCG@5={results_stats['ndcg_final'][-1]:.2f}")

    # --- Generate Statistics ---
    print("\n" + "="*60)
    print("📊 EXTENDED RESEARCH STATISTICS (V2)")
    print("="*60)

    # Average Metrics
    avg_mrr = {k: np.mean(v) for k, v in results_stats.items() if k.startswith('mrr_')}
    avg_ndcg = {k: np.mean(v) for k, v in results_stats.items() if k.startswith('ndcg_')}
    avg_p3 = {k: np.mean(v) for k, v in results_stats.items() if k.startswith('p3_')}

    # Score Analysis
    avg_pos = np.mean(results_stats['scores_relevant']) if results_stats['scores_relevant'] else 0
    avg_neg = np.mean(results_stats['scores_irrelevant']) if results_stats['scores_irrelevant'] else 0
    
    # Prepare metrics dict for saving
    final_metrics = {
        'mrr_retrieval': avg_mrr['mrr_retrieval'],
        'mrr_final': avg_mrr['mrr_final'],
        'ndcg_retrieval': avg_ndcg['ndcg_retrieval'],
        'ndcg_final': avg_ndcg['ndcg_final']
    }
    
    # Save to JSON
    save_benchmark_result(final_metrics, description="Iteration V5: Constraint & No-Answer Safety")

    print(f"{'Metric':<20} | {'Vector Only':<12} | {'+ Reranker':<12} | {'+ PRF/RRF':<12}")
    print(f"-"*66)
    print(f"{'MRR':<20} | {avg_mrr['mrr_retrieval']:.4f}       | {avg_mrr['mrr_rerank']:.4f}       | {avg_mrr['mrr_final']:.4f}")
    print(f"{'NDCG@5':<20} | {avg_ndcg['ndcg_retrieval']:.4f}       | {avg_ndcg['ndcg_rerank']:.4f}       | {avg_ndcg['ndcg_final']:.4f}")
    print(f"{'Precision@3':<20} | {avg_p3['p3_retrieval']:.4f}       | {avg_p3['p3_rerank']:.4f}       | {avg_p3['p3_final']:.4f}")
    
    print("\n📈 Robustness Analysis")
    print(f"  - Separation Gap:              {avg_pos - avg_neg:.4f} (Target: > 0.5)")
    print(f"  - PRF Trigger Rate:            {results_stats['prf_triggered']}/{len(TEST_CASES)}")
    if results_stats['no_answer_total'] > 0:
        no_answer_acc = results_stats['no_answer_safe'] / results_stats['no_answer_total']
        print(f"  - No-Answer Safety:            {results_stats['no_answer_safe']}/{results_stats['no_answer_total']} ({no_answer_acc:.2%})")

    generate_v2_report(avg_mrr, avg_ndcg, avg_p3, avg_pos, avg_neg, results_stats)

def generate_v2_report(mrr, ndcg, p3, score_pos, score_neg, stats):
    out_dir = docs_dir() / "research_docs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "RERANKER_RESEARCH_REPORT_V2.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 重排序器研究报告 V5：约束查询与无答案安全\n\n")
        f.write("## 1. 摘要\n")
        f.write("本阶段在更困难的检索条件下评估RAG流水线的鲁棒性，覆盖语义歧义查询、带否定/约束条件的查询，以及无答案（No-Answer）安全场景，重点观察重排序、PRF触发与融合策略在真实分布下的收益与风险。\n\n")

        f.write("## 2. 核心指标汇总\n")
        f.write("| 指标 | 阶段1：向量检索 | 阶段2：重排序 | 阶段3：全流程（PRF+RRF） | 相对变化（全流程 vs 向量检索） |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **MRR** | {mrr['mrr_retrieval']:.4f} | {mrr['mrr_rerank']:.4f} | {mrr['mrr_final']:.4f} | **{((mrr['mrr_final']-mrr['mrr_retrieval'])/mrr['mrr_retrieval']*100):.1f}%** |\n")
        f.write(f"| **NDCG@5** | {ndcg['ndcg_retrieval']:.4f} | {ndcg['ndcg_rerank']:.4f} | {ndcg['ndcg_final']:.4f} | **{((ndcg['ndcg_final']-ndcg['ndcg_retrieval'])/ndcg['ndcg_retrieval']*100):.1f}%** |\n")
        f.write(f"| **Precision@3** | {p3['p3_retrieval']:.4f} | {p3['p3_rerank']:.4f} | {p3['p3_final']:.4f} | **{((p3['p3_final']-p3['p3_retrieval'])/p3['p3_retrieval']*100):.1f}%** |\n\n")

        f.write("## 3. 歧义查询现象与分析\n")
        f.write("测试集包含“Python / Amazon / Apple / Bank”等多义词查询。向量检索在语义空间上容易将不同义项混合，导致Top结果中出现与意图不一致的候选；交叉编码器重排序能够显式建模查询与文档的交互，从而对干扰项进行惩罚，提高Top-$k$结果的相关性。当前阶段的Precision@3为")
        f.write(f"`{p3['p3_final']:.4f}`，反映了在歧义场景下对干扰项的过滤能力。\n\n")

        f.write("## 4. 置信度与PRF触发概况\n")
        f.write(f"* **平均相关分**：`{score_pos:.4f}`\n")
        f.write(f"* **平均非相关分**：`{score_neg:.4f}`\n")
        f.write(f"* **分离间隔（Gap）**：`{score_pos - score_neg:.4f}`\n")
        f.write(f"* **PRF触发次数**：`{stats['prf_triggered']}/{len(TEST_CASES)}`\n")
        if stats['no_answer_total'] > 0:
            no_answer_acc = stats['no_answer_safe'] / stats['no_answer_total']
            f.write(f"* **无答案安全**：`{stats['no_answer_safe']}/{stats['no_answer_total']}`（`{no_answer_acc:.2%}`）\n")

if __name__ == "__main__":
    run_benchmark()
