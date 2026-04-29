import logging
from typing import List, Dict, Any, Tuple
import re
import json

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
        decision_policy: str = "soft",
        min_claim_confidence: float | None = None,
    ):
        self.threshold = confidence_threshold
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.decision_policy = decision_policy
        self.min_claim_confidence = min_claim_confidence

    def _parse_llm_verification_payload(self, content: str) -> Dict[str, Any]:
        """
        Parse verifier output from OpenAI-compatible providers.

        Some providers return strict JSON, while others may wrap it in Markdown
        fences or add a short natural-language prefix. The verifier should be
        robust to those formatting differences because they are not semantic
        verification failures.
        """
        text = (content or "").strip()
        if not text:
            raise ValueError("empty verifier response")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return json.loads(fenced.group(1))

        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start : end + 1])

        label_match = re.search(r"\b(SUPPORTED|INSUFFICIENT|CONTRADICTED)\b", text, flags=re.IGNORECASE)
        confidence_match = re.search(r"(?:confidence|score)\D{0,20}([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
        if label_match:
            return {
                "label": label_match.group(1).upper(),
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                "reason": text[:240],
            }

        raise ValueError(f"cannot parse verifier response: {text[:120]}")

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
            "Return only a JSON object, without Markdown fences or extra text.\n"
            'Example: {"label":"SUPPORTED","confidence":0.82,"reason":"..."}\n'
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
            payload = self._parse_llm_verification_payload(content)
            confidence = float(payload.get("confidence", 0.0))
            label = str(payload.get("label", "INSUFFICIENT")).upper()
            # LLM confidence is confidence in the predicted label, not always
            # confidence that the claim is supported. Convert it into a support
            # confidence before soft aggregation.
            support_confidence = confidence if label == "SUPPORTED" else max(0.0, 1.0 - confidence)
            return label == "SUPPORTED" and support_confidence >= self.threshold, support_confidence
        except Exception as exc:
            logger.warning(
                "LLM verification parse/call failed via provider=%s model=%s: %s. Falling back to heuristic verification.",
                config.provider,
                config.model,
                exc,
            )
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
        if (generated_answer or "").strip().lower() in {"no-answer", "no answer", "unknown"}:
            return {
                "status": "REJECTED",
                "reason": "Generator abstained before verification.",
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "unsupported_count": 1,
                "decision_policy": self.decision_policy,
                "claims": [
                    {
                        "claim": generated_answer,
                        "supported": False,
                        "confidence": 0.0,
                    }
                ],
            }

        if not evidence_chain:
            return {
                "status": "REJECTED",
                "reason": "No evidence retrieved (No-Answer Safety)",
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "unsupported_count": 0,
                "decision_policy": self.decision_policy,
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
        
        min_confidence = min((item["confidence"] for item in verification_results), default=avg_confidence)
        unsupported_count = sum(1 for item in verification_results if not item["supported"])
        if self.min_claim_confidence is not None and min_confidence < self.min_claim_confidence:
            status = "REJECTED"
            reason = "At least one claim is below the minimum claim confidence."
        elif self.decision_policy == "hard" and not all_supported:
            status = "REJECTED"
            reason = "At least one claim is unsupported under hard verification."
        elif avg_confidence >= self.threshold:
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
            "min_confidence": min_confidence,
            "unsupported_count": unsupported_count,
            "decision_policy": self.decision_policy,
            "claims": verification_results
        }
