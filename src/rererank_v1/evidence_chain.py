import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EvidenceNode:
    def __init__(self, evidence_dict: Dict[str, Any]):
        """
        Represents a node in the reasoning graph (Evidence Chain).
        """
        self.evidence = evidence_dict
        self.children: List['EvidenceNode'] = []
        # Score could be RRF score or CrossEncoder score
        self.score = evidence_dict.get('rrf_score', evidence_dict.get('rerank_score', 0.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node and its children to a dictionary for serialization."""
        return {
            "id": self.evidence.get('id', -1),
            "text": self.evidence.get('text', ''),
            "source": self.evidence.get('source', 'text'),
            "score": self.score,
            "children": [child.to_dict() for child in self.children]
        }

def build_reasoning_graph(evidences: List[Dict[str, Any]], similarity_threshold: float = 0.5) -> List[EvidenceNode]:
    """
    Construct a reasoning graph/chain from retrieved evidences.
    In a full production system, this uses an LLM to determine logical links
    between pieces of evidence (e.g., entity overlap or causal relation).
    Here, we use a simple heuristic to simulate building a chain: 
    linking high-confidence nodes sequentially if they share context.
    
    Args:
        evidences: Ranked list of retrieved documents.
        similarity_threshold: Minimum score required to be included in the chain.
        
    Returns:
        A list of root EvidenceNodes representing the start of reasoning chains.
    """
    if not evidences:
        return []
        
    # Convert dictionaries to EvidenceNode objects
    nodes = [EvidenceNode(e) for e in evidences if e.get('rrf_score', e.get('rerank_score', 0.0)) > similarity_threshold]
    
    if not nodes:
        # Fallback to top-1 if all scores are below threshold
        nodes = [EvidenceNode(evidences[0])]
        
    # Simple chain building: Connect nodes sequentially. 
    # For a true "Graph of Thoughts" approach, we might branch out.
    # Here we build a linear reasoning chain for simplicity: A -> B -> C
    root = nodes[0]
    current = root
    
    for next_node in nodes[1:]:
        # Simulate an LLM deciding that next_node connects to current logically
        current.children.append(next_node)
        current = next_node
            
    return [root]

def format_chain_for_llm(nodes: List[EvidenceNode], depth: int = 1) -> str:
    """
    Formats the graph/chain into a prompt-friendly string for the final LLM generator.
    """
    if not nodes:
        return ""
    
    result = ""
    for i, node in enumerate(nodes):
        prefix = "  " * (depth - 1) + f"[{depth}] "
        source = node.evidence.get('source', 'text')
        text = node.evidence.get('text', '').replace('\n', ' ').strip()
        
        # Truncate text for readability in logs
        if len(text) > 150:
            text = text[:147] + "..."
            
        result += f"{prefix}(Source: {source} | Confidence: {node.score:.2f}) {text}\n"
        if node.children:
            result += format_chain_for_llm(node.children, depth + 1)
            
    return result
