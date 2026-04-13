from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class EvidenceUnit:
    """
    Unified interface for heterogeneous knowledge (Text, Table, Graph).
    This is the core abstraction for Phase 2.
    """
    content: str       # Serialized string representation for reranker/LLM
    source: str        # 'text', 'table', or 'graph'
    metadata: Dict[str, Any]  # Original structured data

# ==========================================
# Heterogeneous Dataset (Text, Table, Graph)
# Note: For full-scale experiments, replace these synthetic
# datasets with loaders for HotpotQA, 2WikiMultihopQA, or MuSiQue.
# ==========================================

# 1. Text Data
TEXT_DATA = [
    {"id": "t1", "text": "Elon Musk founded SpaceX in 2002 to reduce space transportation costs."},
    {"id": "t2", "text": "Tesla, Inc. is an American multinational automotive and clean energy company."},
    {"id": "t3", "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."},
]

# 2. Table Data (Structured Rows)
TABLE_DATA = [
    {
        "id": "tab1",
        "title": "Tech Company Founders",
        "rows": [
            {"Company": "SpaceX", "Founder": "Elon Musk", "Year": 2002},
            {"Company": "Tesla", "Founder": "Martin Eberhard, Marc Tarpenning", "Year": 2003},
            {"Company": "Apple", "Founder": "Steve Jobs, Steve Wozniak", "Year": 1976}
        ]
    }
]

# 3. Graph Data (Knowledge Graph Triples)
GRAPH_DATA = [
    {"id": "g1", "subject": "Elon Musk", "predicate": "founder_of", "object": "SpaceX"},
    {"id": "g2", "subject": "Elon Musk", "predicate": "CEO_of", "object": "Tesla"},
    {"id": "g3", "subject": "SpaceX", "predicate": "industry", "object": "Aerospace"},
]

# Helper to convert raw data into EvidenceUnits
def build_hetero_corpus() -> List[EvidenceUnit]:
    corpus = []
    
    # Process Text
    for item in TEXT_DATA:
        corpus.append(EvidenceUnit(
            content=item["text"],
            source="text",
            metadata={"id": item["id"]}
        ))
        
    # Process Tables (Row Serialization)
    for table in TABLE_DATA:
        title = table["title"]
        for i, row in enumerate(table["rows"]):
            # Serialize row into string
            row_str = f"Table: {title} | " + ", ".join([f"{k}: {v}" for k, v in row.items()])
            corpus.append(EvidenceUnit(
                content=row_str,
                source="table",
                metadata={"id": table["id"], "row_idx": i, "raw_row": row}
            ))
            
    # Process Graph (Triple Serialization)
    for edge in GRAPH_DATA:
        # Serialize triple
        triple_str = f"Graph Fact: {edge['subject']} -> {edge['predicate']} -> {edge['object']}"
        corpus.append(EvidenceUnit(
            content=triple_str,
            source="graph",
            metadata={"id": edge["id"], "triple": edge}
        ))
        
    return corpus

# Test Queries for Phase 2
HETERO_TEST_CASES = [
    {
        "query": "Who founded SpaceX and in what year?",
        "expected_sources": ["text", "table", "graph"],
        "description": "Information exists across all modalities. Table has exact year and founder."
    },
    {
        "query": "What industry is the company founded by Elon Musk in 2002 involved in?",
        "expected_sources": ["text", "table", "graph"],
        "description": "Multi-hop query requiring linking Table/Text (SpaceX=2002) to Graph (SpaceX industry)."
    }
]
