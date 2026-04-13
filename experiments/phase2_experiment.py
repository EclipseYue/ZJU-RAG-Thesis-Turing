import os
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from rererank_v1.rag_pipeline import RAGPipeline
from rererank_v1.hetero_data import build_hetero_corpus, HETERO_TEST_CASES, TEXT_DATA

def run_phase2_experiment():
    """
    Phase 2: Heterogeneous Knowledge Experiment
    Ablation Matrix:
    - Model A: Text-only (Baseline)
    - Model B: Heterogeneous (Text + Table + Graph)
    """
    os.environ['FORCE_MOCK'] = '1'
    print("\n" + "="*50)
    print("🚀 Phase 2: Heterogeneous Knowledge Retrieval")
    print("="*50)

    # ---------------------------------------------------------
    # Baseline (Text-only)
    # ---------------------------------------------------------
    print("\n--- Model A: Text-Only Baseline ---")
    rag_baseline = RAGPipeline()
    # Add only text
    rag_baseline.add_documents([item['text'] for item in TEXT_DATA])
    
    # ---------------------------------------------------------
    # Heterogeneous (Text + Table + Graph)
    # ---------------------------------------------------------
    print("\n--- Model B: Heterogeneous Corpus ---")
    rag_hetero = RAGPipeline()
    hetero_corpus = build_hetero_corpus()
    rag_hetero.add_evidence_units(hetero_corpus)

    # ---------------------------------------------------------
    # Evaluation Loop
    # ---------------------------------------------------------
    for idx, case in enumerate(HETERO_TEST_CASES):
        query = case["query"]
        print(f"\n[Test Case {idx+1}]: {query}")
        print(f"Description: {case['description']}")
        
        # Baseline Search
        print("\n  [Baseline Results (Text Only)]")
        res_base = rag_baseline.search(query, top_k=2, active_retrieval=False)
        for i, res in enumerate(res_base):
            score = res.get('rrf_score', res.get('rerank_score'))
            print(f"  {i+1}. [text] {res['text']} (Score: {score:.4f})")

        # Heterogeneous Search
        print("\n  [Heterogeneous Results (Text+Table+Graph)]")
        res_hetero = rag_hetero.search(query, top_k=3, active_retrieval=False)
        for i, res in enumerate(res_hetero):
            score = res.get('rrf_score', res.get('rerank_score'))
            source = res.get('source', 'unknown')
            print(f"  {i+1}. [{source}] {res['text']} (Score: {score:.4f})")

if __name__ == "__main__":
    run_phase2_experiment()
