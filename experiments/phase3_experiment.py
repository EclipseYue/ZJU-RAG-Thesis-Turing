import os
import sys
import json
from pathlib import Path
import logging

# Ensure src module is reachable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.dataset_loader import load_hotpotqa_sample, HF_DATASETS_AVAILABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase3_experiment():
    """
    Phase 3: Evidence Chain (Graph of Thoughts) on HotpotQA
    Demonstrates connecting retrieved evidence into a reasoning chain
    before passing to a generator LLM.
    """
    os.environ['FORCE_MOCK'] = '1'
    
    print("\n" + "="*60)
    print("🚀 Phase 3: Evidence Chain (Graph of Thoughts)")
    print("="*60)

    if not HF_DATASETS_AVAILABLE:
        print("❌ Error: `datasets` library is not installed.")
        print("Please run `pip install datasets` to execute Phase 3.")
        return

    # Load real multi-hop dataset (HotpotQA)
    try:
        data = load_hotpotqa_sample(num_samples=20)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    rag = RAGPipeline()
    rag.add_evidence_units(data['corpus'])

    results_log = []
    
    # Select first 3 queries for demonstration
    test_queries = data['queries'][:3]
    
    for i, q in enumerate(test_queries):
        print(f"\n" + "-"*50)
        print(f"🧐 [Query {i+1}]: {q['query']}")
        print(f"🎯 [Expected Answer]: {q['answer']}")
        print(f"🏷️  [Type]: {q['type'].upper()}")
        print("-" * 50)
        
        # Execute search with chain building
        res = rag.search_with_chain(q['query'], top_k=3)
        
        print("\n🔗 [Reasoning Chain Constructed]:")
        print(res['chain_str'])
        
        results_log.append({
            "query": q['query'],
            "expected_answer": q['answer'],
            "type": q['type'],
            "chain_hierarchy": res['chain']
        })

    # Save to data/results/
    out_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "phase3_results.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*60)
    print(f"✅ Phase 3 Experiment completed.")
    print(f"📁 Evidence chains and structural results saved to: {out_file}")
    print("="*60)

if __name__ == "__main__":
    run_phase3_experiment()
