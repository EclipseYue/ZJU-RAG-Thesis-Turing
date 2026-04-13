import logging
from typing import List, Dict, Any, Tuple
import re

logger = logging.getLogger(__name__)

class CoVeVerifier:
    """
    Phase 4: Chain-of-Verification (CoVe) Module.
    Designed to prevent hallucination by fact-checking generated claims
    against the retrieved evidence. Supports 'No-Answer' safety.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.threshold = confidence_threshold

    def extract_claims(self, generated_answer: str) -> List[str]:
        """
        Simulate an LLM extracting factual claims from an answer.
        In a real system, this would be an LLM prompt: "Extract facts from: {answer}"
        Here, we use a simple sentence splitter for demonstration.
        """
        # Basic heuristic: split by periods to represent claims
        sentences = [s.strip() for s in re.split(r'[.!?]+', generated_answer) if len(s.strip()) > 10]
        if not sentences:
            return [generated_answer]
        return sentences

    def verify_claim(self, claim: str, evidence_chain: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Verify a single claim against the evidence chain.
        Returns: (is_supported, confidence_score)
        """
        # Flatten the evidence text for simple string matching simulation
        # In a real system, an NLI (Natural Language Inference) model or LLM 
        # would score Entailment vs Contradiction.
        
        # Flatten evidence texts
        def _get_texts(nodes):
            texts = []
            for n in nodes:
                texts.append(n.get('text', '').lower())
                if 'children' in n and n['children']:
                    texts.extend(_get_texts(n['children']))
            return texts
            
        evidence_texts = _get_texts(evidence_chain)
        full_context = " ".join(evidence_texts)
        
        claim_lower = claim.lower()
        
        # Simulation logic:
        # If key nouns from the claim appear in the evidence, we consider it supported.
        words = set(re.findall(r'\b\w{4,}\b', claim_lower))
        
        if not words:
            return True, 1.0 # Trivial claim
            
        match_count = sum(1 for w in words if w in full_context)
        confidence = match_count / len(words)
        
        # Add a slight boost if the evidence chain itself had high retrieval confidence
        chain_score = max([n.get('score', 0.0) for n in evidence_chain], default=0.0)
        final_confidence = (confidence * 0.7) + (chain_score * 0.3)
        
        is_supported = final_confidence >= self.threshold
        return is_supported, final_confidence

    def evaluate_answer(self, generated_answer: str, evidence_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the full CoVe pipeline on a generated answer.
        """
        if not evidence_chain:
            return {
                "status": "REJECTED",
                "reason": "No evidence retrieved (No-Answer Safety)",
                "claims": []
            }
            
        claims = self.extract_claims(generated_answer)
        verification_results = []
        
        all_supported = True
        total_confidence = 0.0
        
        for claim in claims:
            is_supported, conf = self.verify_claim(claim, evidence_chain)
            verification_results.append({
                "claim": claim,
                "supported": is_supported,
                "confidence": conf
            })
            if not is_supported:
                all_supported = False
            total_confidence += conf
            
        avg_confidence = total_confidence / len(claims) if claims else 0.0
        
        if avg_confidence >= self.threshold:
            status = "ACCEPTED"
            if all_supported:
                reason = "All claims supported by evidence."
            else:
                reason = "Partially supported by evidence (Soft acceptance)."
        else:
            status = "REJECTED"
            reason = "Hallucination detected (Failed CoVe checks). Triggering No-Answer."
            
        return {
            "status": status,
            "reason": reason,
            "avg_confidence": avg_confidence,
            "claims": verification_results
        }
