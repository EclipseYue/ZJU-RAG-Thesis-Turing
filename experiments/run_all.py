import os
import sys
import json
import time
from pathlib import Path
import logging

# Ensure src module is reachable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.hetero_data import build_hetero_corpus, TEXT_DATA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_automated_ablation_with_tracking(use_wandb: bool = False):
    """
    Automated Master Runner for Phase 4 Ablation.
    Supports integration with Weights & Biases (wandb) for visualization.
    """
    # 1. Initialization for Visualization Website
    if use_wandb:
        try:
            import wandb
            wandb.login()
        except ImportError:
            logger.warning("wandb is not installed. Falling back to local logging. Run `pip install wandb`.")
            use_wandb = False

    os.environ['FORCE_MOCK'] = '0' # Use real models for automation by default
    logger.info("🚀 Starting Automated Experiment Runner for Full Ablation Matrix")

    # 2. Setup Datasets
    text_corpus = [{"text": item['text'], "source": "text"} for item in TEXT_DATA]
    hetero_corpus = build_hetero_corpus()
    
    # Ablation Configurations
    configurations = [
        {"name": "A_Baseline", "hetero": False, "adaptive": False, "cove": False},
        {"name": "B_Hetero", "hetero": True, "adaptive": False, "cove": False},
        {"name": "C_Adaptive", "hetero": True, "adaptive": True, "cove": False},
        {"name": "D_CoVe_Full", "hetero": True, "adaptive": True, "cove": True},
    ]

    # Extensible Test Queries for Automation (Can load from HF datasets here)
    test_queries = [
        {"query": "Who founded SpaceX and in what year?", "hallucination_trigger": "Elon Musk founded SpaceX in 1990."},
        {"query": "What industry is the company founded by Elon Musk in 2002 involved in?", "hallucination_trigger": "It is a food company."},
        {"query": "What is the capital of France?", "hallucination_trigger": "The capital of France is Paris."}
    ]

    results_matrix = []
    
    # 3. Execution Loop
    for config in configurations:
        if use_wandb:
            # Initialize a new run for each config variant
            wandb.init(
                project="rag-ablation-study",
                name=config['name'],
                config=config,
                reinit=True
            )
            
        logger.info(f"⚙️ Running Config: {config['name']}")
        
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
            
            # Retrieval Phase
            if config['cove']:
                res = rag.search_with_chain(query, top_k=2, prf_threshold=0.8 if config['adaptive'] else 1.1)
                chain = res['chain']
            else:
                res = rag.search(query, top_k=2, active_retrieval=config['adaptive'])
                chain = []
                
            config_stats['cost_tokens'] += rag.stats['total_tokens']
            config_stats['latency'] += rag.stats['total_latency']
            
            # Verification Phase
            if config['cove']:
                cove_res = rag.verify_answer(mock_llm_answer, chain)
                if cove_res['status'] == 'REJECTED':
                    config_stats['cove_rejections'] += 1
                    
        # Calculate Averages
        avg_tokens = int(config_stats['cost_tokens'] / len(test_queries))
        avg_latency = round((config_stats['latency'] / len(test_queries)) * 1000, 2)
        no_answer_rate = (config_stats['cove_rejections'] / len(test_queries)) * 100
        
        # 4. Log to Visualization Dashboard (WandB)
        if use_wandb:
            wandb.log({
                "Avg_Tokens": avg_tokens,
                "Avg_Latency_ms": avg_latency,
                "No_Answer_Rate": no_answer_rate
            })
            wandb.finish()
            
        # Log Locally
        results_matrix.append({
            "Config": config['name'],
            "Hetero": config['hetero'],
            "Adaptive": config['adaptive'],
            "CoVe": config['cove'],
            "Avg_Tokens": avg_tokens,
            "Avg_Latency_ms": avg_latency,
            "No_Answer_Rate_Percent": round(no_answer_rate, 2)
        })

    # 5. Local Result Persistence
    out_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "automated_ablation.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_matrix, f, ensure_ascii=False, indent=2)
        
    logger.info(f"✅ Automated Run Completed. Results saved to {out_file}")
    if use_wandb:
        logger.info("📊 View your live dashboards and charts at: https://wandb.ai")

if __name__ == "__main__":
    # Note: Pass use_wandb=True to enable Weights & Biases dashboard tracking.
    run_automated_ablation_with_tracking(use_wandb=False)
