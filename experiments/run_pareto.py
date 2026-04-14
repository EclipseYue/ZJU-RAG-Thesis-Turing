import argparse
import os
import sys
import json
from pathlib import Path

# Ensure src module is reachable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.dataset_loader import load_multihop_sample
from run_all import evaluate_query, summarize_results
import matplotlib.pyplot as plt
import seaborn as sns

def run_pareto_experiment(dataset="hotpotqa", samples=100, thresholds=[0.6, 0.7, 0.8, 0.9, 0.95], device="cuda"):
    os.environ["FORCE_MOCK"] = "0"
    print(f"🚀 Running Pareto Frontier Experiment on {dataset} (N={samples})")
    
    hetero_bundle = load_multihop_sample(dataset, split="validation", num_samples=samples, use_hetero=True)
    queries = hetero_bundle["queries"]
    corpus = hetero_bundle["corpus"]
    
    rag = RAGPipeline(device=device, use_v6_reranker=True)
    rag.add_evidence_units(corpus)
    
    pareto_results = []
    
    for thresh in thresholds:
        print(f"\n--- Testing Adaptive Threshold: {thresh} ---")
        config = {
            "name": f"Adaptive(tau={thresh})",
            "hetero": True,
            "adaptive": True,
            "cove": False
        }
        
        # Monkey patch the internal threshold just for this run if needed
        # In rag_pipeline, prf_threshold is passed to search()
        query_results = []
        for q in queries:
            # We override evaluate_query logic manually for custom thresholding
            before = {"total_tokens": rag.stats.get("total_tokens", 0), "total_latency": rag.stats.get("total_latency", 0.0)}
            results = rag.search(q["query"], top_k=5, prf_threshold=thresh, active_retrieval=True)
            after = {"total_tokens": rag.stats.get("total_tokens", 0), "total_latency": rag.stats.get("total_latency", 0.0)}
            
            # Simple metrics calculation for Pareto
            support_titles = set(q.get("supporting_titles", []))
            pred_titles = set()
            for r in results:
                t = r.get("metadata", {}).get("title")
                if t: pred_titles.add(t)
            
            recall = len(support_titles.intersection(pred_titles)) / len(support_titles) if support_titles else 0.0
            
            query_results.append({
                "support_recall": recall,
                "stats_delta": {
                    "total_tokens": after["total_tokens"] - before["total_tokens"],
                    "total_latency": after["total_latency"] - before["total_latency"]
                }
            })
            
        avg_recall = sum(r["support_recall"] for r in query_results) / len(query_results) * 100
        avg_tokens = sum(r["stats_delta"]["total_tokens"] for r in query_results) / len(query_results)
        avg_latency = sum(r["stats_delta"]["total_latency"] for r in query_results) / len(query_results) * 1000
        
        res = {
            "Threshold": thresh,
            "SupportRecall": avg_recall,
            "Avg_Tokens": avg_tokens,
            "Avg_Latency_ms": avg_latency
        }
        pareto_results.append(res)
        print(res)

    # Save Results
    out_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pareto_results.json", "w") as f:
        json.dump(pareto_results, f, indent=2)
        
    # Plot Pareto Curve
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    
    tokens = [r["Avg_Tokens"] for r in pareto_results]
    recalls = [r["SupportRecall"] for r in pareto_results]
    labels = [f"$\\tau$={r['Threshold']}" for r in pareto_results]
    
    plt.plot(tokens, recalls, marker='o', linewidth=2, color='tab:blue')
    for i, label in enumerate(labels):
        plt.annotate(label, (tokens[i], recalls[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.title("Pareto Frontier: Cost vs Coverage (Adaptive Retrieval)", fontsize=14)
    plt.xlabel("Average Tokens Cost", fontsize=12)
    plt.ylabel("Support Evidence Recall (%)", fontsize=12)
    
    plot_path = Path(__file__).resolve().parent.parent / "paper" / "zjuthesis" / "figures" / "pareto_frontier.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Pareto chart saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_pareto_experiment(samples=args.samples, device=args.device)
