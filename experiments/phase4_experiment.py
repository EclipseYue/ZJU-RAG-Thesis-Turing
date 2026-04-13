import os
import sys
import json
from pathlib import Path
import logging

# Ensure src module is reachable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.hetero_data import build_hetero_corpus, HETERO_TEST_CASES, TEXT_DATA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase4_ablation_study():
    """
    Phase 4: CoVe + No-Answer Safety + Ablation Matrix
    Runs the full pipeline across different configurations to generate
    the final ablation table for the paper.
    """
    # NOTE: Set this to '0' to download and use real HF models.
    # Set to '1' if you want to run quickly with mock models.
    os.environ['FORCE_MOCK'] = '0'
    
    print("\n" + "="*70)
    print("🛡️ Phase 4: Verification & Ablation Matrix (Full System)")
    print("="*70)

    # Prepare datasets
    text_corpus = [{"text": item['text'], "source": "text"} for item in TEXT_DATA]
    hetero_corpus = build_hetero_corpus()
    
    # Define Ablation Matrix
    configurations = [
        {"name": "A Baseline", "hetero": False, "adaptive": False, "cove": False},
        {"name": "B +Hetero", "hetero": True, "adaptive": False, "cove": False},
        {"name": "C +Adaptive", "hetero": True, "adaptive": True, "cove": False},
        {"name": "D +CoVe (Full)", "hetero": True, "adaptive": True, "cove": True},
    ]

    # Sample queries including a hallucination trigger
    test_queries = [
        # Normal query
        {"query": "Who founded SpaceX and in what year?", "hallucination_trigger": "Elon Musk founded SpaceX in 1990."},
        # Multi-hop query
        {"query": "What industry is the company founded by Elon Musk in 2002 involved in?", "hallucination_trigger": "It is a food company."},
        # No-Answer Safety trigger (Information not in corpus)
        {"query": "What is the capital of France?", "hallucination_trigger": "The capital of France is Paris."}
    ]

    results_matrix = []

    for config in configurations:
        print(f"\n" + "-"*50)
        print(f"⚙️ Running Config: {config['name']}")
        print("-" * 50)
        
        # Init Pipeline
        rag = RAGPipeline()
        if config['hetero']:
            rag.add_evidence_units(hetero_corpus)
        else:
            rag.add_documents([item['text'] for item in text_corpus])
            
        config_stats = {"cost_tokens": 0, "latency": 0.0, "cove_rejections": 0}
        
        for q_data in test_queries:
            query = q_data['query']
            mock_llm_answer = q_data['hallucination_trigger']
            
            # 1. Retrieval & Chain
            if config['cove']:
                # Need chain for CoVe
                res = rag.search_with_chain(query, top_k=2, prf_threshold=0.8 if config['adaptive'] else 1.1)
                chain = res['chain']
            else:
                # Standard search
                res = rag.search(query, top_k=2, active_retrieval=config['adaptive'])
                chain = [] # Not used
                
            # Accumulate cost
            config_stats['cost_tokens'] += rag.stats['total_tokens']
            config_stats['latency'] += rag.stats['total_latency']
            
            # 2. Verification (CoVe)
            if config['cove']:
                cove_res = rag.verify_answer(mock_llm_answer, chain)
                if cove_res['status'] == 'REJECTED':
                    config_stats['cove_rejections'] += 1
                    
        # Log results
        results_matrix.append({
            "Config": config['name'],
            "Hetero": "✅" if config['hetero'] else "❌",
            "Adaptive": "✅" if config['adaptive'] else "❌",
            "CoVe": "✅" if config['cove'] else "❌",
            "Avg_Tokens": int(config_stats['cost_tokens'] / len(test_queries)),
            "Avg_Latency_ms": round((config_stats['latency'] / len(test_queries)) * 1000, 2),
            "No_Answer_Rate": f"{(config_stats['cove_rejections'] / len(test_queries))*100:.0f}%"
        })

    # Save and Print Matrix
    out_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ablation_matrix.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_matrix, f, ensure_ascii=False, indent=2)

    print("\n\n" + "="*70)
    print("📊 Final Ablation Matrix (LaTeX ready)")
    print("="*70)
    print(f"{'Model':<15} | {'Hetero':<6} | {'Adaptive':<8} | {'CoVe':<6} | {'Cost(Tokens)':<12} | {'Latency(ms)':<12} | {'No-Answer Rate':<15}")
    print("-" * 85)
    for row in results_matrix:
        print(f"{row['Config']:<15} | {row['Hetero']:<6} | {row['Adaptive']:<8} | {row['CoVe']:<6} | {row['Avg_Tokens']:<12} | {row['Avg_Latency_ms']:<12} | {row['No_Answer_Rate']:<15}")
    
    print(f"\n📁 Ablation results saved to: {out_file}")

if __name__ == "__main__":
    run_phase4_ablation_study()
