import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Fallback heuristic if API is not configured or fails
def heuristic_generate_answer(query: str, results) -> str:
    """Legacy heuristic generator for quick CPU/mock tests."""
    import re
    if not results:
        return "No-Answer"
    q_words = set(re.findall(r"\b[A-Z][a-z]+\b", query))
    candidates = []
    for doc in results[:4]:
        text = doc.get("text", "")
        source = doc.get("source", "text")
        if source == "graph" and "->" in text:
            parts = text.split("->")
            if len(parts) >= 3:
                ans_candidate = parts[-1].strip()
                if any(w in parts[0] for w in q_words):
                    candidates.append((5.0, ans_candidate))
        elif source == "table" and "|" in text and ":" in text:
            parts = text.split("|")[-1].split(":")
            if len(parts) >= 2:
                ans_candidate = parts[-1].strip()
                candidates.append((4.0, ans_candidate))
        for sent in re.split(r"[.!?。；;]", text):
            sent = sent.strip()
            if len(sent) < 8: continue
            s_tokens = set(re.findall(r"\w+", sent.lower()))
            q_tokens = set(re.findall(r"\w+", query.lower()))
            overlap = len(q_tokens & s_tokens)
            if overlap > 0: candidates.append((overlap * 0.5, sent))
    if not candidates:
        return results[0].get("text", "")[:120]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1][:200]


def llm_generate_answer(query: str, results: List[Dict[str, Any]], model: str = "Qwen/Qwen2.5-7B-Instruct") -> str:
    """
    Real LLM Generator using OpenAI-compatible API (e.g., DeepSeek, vLLM local LLaMA-3).
    Requires environment variables:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (Optional, defaults to DeepSeek if not provided)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # If no API key is provided, fallback to the fast heuristic generator
    if not api_key:
        return heuristic_generate_answer(query, results)
        
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Please install openai package: pip install openai")
        return heuristic_generate_answer(query, results)

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build the context from retrieved evidence
    context_str = ""
    for i, doc in enumerate(results[:5]):
        source = doc.get("source", "text")
        text = doc.get("text", "")
        context_str += f"[{i+1}] (Source: {source}) {text}\n"

    if not context_str:
        context_str = "No relevant context found."

    prompt = (
        "You are an expert QA assistant. Answer the user's question STRICTLY based on the provided Context.\n"
        "If the Context does not contain enough information to answer the question, output exactly 'No-Answer'.\n"
        "Keep your answer concise and factual (e.g., just the entity name, date, or short phrase if possible).\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise, concise QA assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for factual QA
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()
        
        # Normalize LLM "I don't know" variations to our standard rejection signal
        if "no-answer" in answer.lower() or "not contain enough information" in answer.lower():
            return "No-Answer"
            
        return answer
    except Exception as e:
        logger.error(f"LLM API Call failed: {e}")
        return heuristic_generate_answer(query, results)
