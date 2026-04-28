import logging
from typing import List, Dict, Any, Tuple
import re

from .llm_backends import (
    build_openai_compat_client,
    create_chat_completion,
    extract_message_text,
    resolve_openai_compat_config,
)

logger = logging.getLogger(__name__)

class CoVeVerifier:
    """
    Phase 4: Chain-of-Verification (CoVe) Module.
    Designed to prevent hallucination by fact-checking generated claims
    against the retrieved evidence. Supports 'No-Answer' safety.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        backend: str = "heuristic",
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.threshold = confidence_threshold
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

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

    def _flatten_evidence(self, evidence_chain: List[Dict[str, Any]]) -> List[str]:
        def _get_texts(nodes):
            texts = []
            for n in nodes:
                texts.append(n.get('text', '').lower())
                if 'children' in n and n['children']:
                    texts.extend(_get_texts(n['children']))
            return texts

        return _get_texts(evidence_chain)

    def _verify_claim_with_llm(self, claim: str, evidence_chain: List[Dict[str, Any]]) -> Tuple[bool, float]:
        config = resolve_openai_compat_config(
            model=self.model,
            provider=self.backend,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        if not config:
            logger.warning("Verifier backend is not configured. Falling back to heuristic verification.")
            return self._verify_claim_heuristic(claim, evidence_chain)

        try:
            client = build_openai_compat_client(config)
        except RuntimeError as exc:
            logger.error(str(exc))
            return self._verify_claim_heuristic(claim, evidence_chain)

        evidence_text = "\n".join(f"- {text}" for text in self._flatten_evidence(evidence_chain)[:20])
        prompt = (
            "Given a claim and retrieved evidence, decide whether the claim is supported.\n"
            "Return JSON with keys: label, confidence, reason.\n"
            "label must be one of SUPPORTED, INSUFFICIENT, CONTRADICTED.\n"
            f"Claim: {claim}\n"
            f"Evidence:\n{evidence_text}\n"
        )
        try:
            response = create_chat_completion(
                client,
                config,
                model=config.model,
                messages=[
                    {"role": "system", "content": "You are a strict factual verifier. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0] if getattr(response, "choices", None) else None
            content = extract_message_text(getattr(choice, "message", None))
            import json
            payload = json.loads(content)
            confidence = float(payload.get("confidence", 0.0))
            label = str(payload.get("label", "INSUFFICIENT")).upper()
            return label == "SUPPORTED" and confidence >= self.threshold, confidence
        except Exception as exc:
            logger.error("LLM verification failed via provider=%s model=%s: %s", config.provider, config.model, exc)
            return self._verify_claim_heuristic(claim, evidence_chain)

    def _verify_claim_heuristic(self, claim: str, evidence_chain: List[Dict[str, Any]]) -> Tuple[bool, float]:
        evidence_texts = self._flatten_evidence(evidence_chain)
        full_context = " ".join(evidence_texts)
        claim_lower = claim.lower()
        words = set(re.findall(r'\b\w{4,}\b', claim_lower))

        if not words:
            return True, 1.0

        match_count = sum(1 for w in words if w in full_context)
        confidence = match_count / len(words)
        chain_score = max([n.get('score', 0.0) for n in evidence_chain], default=0.0)
        final_confidence = (confidence * 0.7) + (chain_score * 0.3)
        is_supported = final_confidence >= self.threshold
        return is_supported, final_confidence

    def verify_claim(self, claim: str, evidence_chain: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Verify a single claim against the evidence chain.
        Returns: (is_supported, confidence_score)
        """
        if self.backend and self.backend != "heuristic":
            return self._verify_claim_with_llm(claim, evidence_chain)
        return self._verify_claim_heuristic(claim, evidence_chain)

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
