import argparse
import os
import sys
import json
from pathlib import Path

# Ensure src module is reachable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.llm_generator import llm_generate_answer
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

def load_popqa_sample(num_samples=100):
    print(f"Loading {num_samples} samples from PopQA...")
    dataset = load_dataset("akariasai/PopQA", split="test")
    # Sort by popularity to get the lowest popularity (long-tail) entities
    dataset = dataset.sort("s_pop").select(range(num_samples))
    
    queries = []
    corpus = []
    for item in dataset:
        queries.append({
            "id": item["id"],
            "query": item["question"],
            "answer": eval(item["possible_answers"])[0] if isinstance(item["possible_answers"], str) else item["possible_answers"][0],
            "subject": item["subj"]
        })
        # Synthetic corpus for PopQA since it's just questions
        # In a real scenario, this would search Wikipedia. Here we add synthetic correct/distractor docs
        corpus.append(f"The entity {item['subj']} is associated with {eval(item['possible_answers'])[0] if isinstance(item['possible_answers'], str) else item['possible_answers'][0]}.")
        corpus.append(f"The entity {item['subj']} is completely unrelated to anything meaningful.")
        
    return {"queries": queries, "corpus": corpus}

def run_popqa_long_tail_experiment(samples=100, device="cuda"):
    os.environ["FORCE_MOCK"] = "0"
    print(f"🚀 Running Long-Tail Knowledge (PopQA) Experiment (N={samples})")
    
    data = load_popqa_sample(samples)
    queries = data["queries"]
    
    rag = RAGPipeline(device=device, use_v6_reranker=True)
    rag.add_documents(data["corpus"])
    
    results = []
    cove_rejections = 0
    
    for q in queries:
        # Standard Search
        docs = rag.search(q["query"], top_k=2, active_retrieval=False)
        
        # We test the CoVe mechanism here
        # Intentionally feed a hallucinated answer to CoVe to see if it catches it
        hallucinated_answer = f"{q['subject']} is related to Paris, France."
        
        # Need chain format for CoVe
        chain = [{"text": d["text"], "score": d.get("rerank_score", 0.0)} for d in docs]
        
        verification = rag.verify_answer(hallucinated_answer, chain)
        
        if verification["status"] == "REJECTED":
            cove_rejections += 1
            
        results.append({
            "query": q["query"],
            "true_answer": q["answer"],
            "hallucinated": hallucinated_answer,
            "cove_status": verification["status"]
        })
        
    rejection_rate = (cove_rejections / samples) * 100
    print(f"\n✅ PopQA Long-Tail Experiment Complete.")
    print(f"🎯 CoVe Hallucination Rejection Rate on Long-Tail: {rejection_rate:.2f}%")
    
    out_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    with open(out_dir / "popqa_results.json", "w") as f:
        json.dump({"RejectionRate": rejection_rate, "Details": results}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_popqa_long_tail_experiment(samples=args.samples, device=args.device)
