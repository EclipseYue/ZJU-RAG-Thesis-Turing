import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

from .llm_backends import build_openai_compat_client, create_chat_completion, resolve_openai_compat_config

logger = logging.getLogger(__name__)


def _postprocess_answer(query: str, answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return "No-Answer"

    lowered = text.lower()
    if lowered in {"yes", "yes.", "yes!", "yes,"}:
        return "Yes"
    if lowered in {"no", "no.", "no!", "no,"}:
        return "No"
    if "no-answer" in lowered or "not enough information" in lowered:
        return "No-Answer"

    query_lower = query.lower()
    if any(query_lower.startswith(prefix) for prefix in ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ")):
        if re.search(r"\byes\b", lowered):
            return "Yes"
        if re.search(r"\bno\b", lowered):
            return "No"

    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"^(answer\s*:\s*)", "", text, flags=re.IGNORECASE).strip()

    # Prefer the first short clause over an explanatory sentence.
    first_clause = re.split(r"[.;\n]", text)[0].strip()
    if first_clause:
        text = first_clause

    # Compress common definitional patterns to their subject/entity phrase.
    for pattern in (
        r"^(.{1,120}?)\s+is\s+",
        r"^(.{1,120}?)\s+was\s+",
        r"^(.{1,120}?)\s+are\s+",
        r"^(.{1,120}?)\s+were\s+",
    ):
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ,.-")
            if candidate:
                text = candidate
                break

    # For "who/what/where/when" questions, keep the answer terse.
    if len(text.split()) > 12:
        comma_clause = text.split(",")[0].strip()
        if comma_clause:
            text = comma_clause
    if len(text.split()) > 12:
        text = " ".join(text.split()[:12]).strip(" ,.-")

    return text if text else "No-Answer"

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
        return _postprocess_answer(query, results[0].get("text", "")[:120])
    candidates.sort(key=lambda x: x[0], reverse=True)
    return _postprocess_answer(query, candidates[0][1][:200])


def llm_generate_answer(
    query: str,
    results: List[Dict[str, Any]],
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    backend: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Real LLM Generator using OpenAI-compatible API (e.g., DeepSeek, vLLM local LLaMA-3).
    Requires environment variables:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (Optional, defaults to DeepSeek if not provided)
    """
    if backend == "heuristic":
        return heuristic_generate_answer(query, results)

    config = resolve_openai_compat_config(
        model=model,
        provider=backend,
        api_key=api_key,
        base_url=base_url,
    )

    if not config:
        logger.info("LLM backend is not configured. Falling back to heuristic generator.")
        return heuristic_generate_answer(query, results)

    try:
        client = build_openai_compat_client(config)
    except RuntimeError as exc:
        logger.error(str(exc))
        return heuristic_generate_answer(query, results)

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
        "Return only the final answer, not an explanation.\n"
        "Prefer the shortest correct answer possible: a single entity, date, number, place, organization, or short phrase.\n"
        "For yes/no questions, output exactly 'Yes' or 'No'.\n"
        "Do not quote long evidence sentences. Do not restate the question. Do not add reasoning.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    try:
        response = create_chat_completion(
            client,
            config,
            model=config.model,
            messages=[
                {"role": "system", "content": "You are a precise, concise QA assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for factual QA
            max_tokens=100
        )
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None) if choice is not None else None
        raw_content = getattr(message, "content", "") if message is not None else ""
        if raw_content is None:
            raw_content = ""
        answer = str(raw_content).strip()

        if not answer:
            logger.warning(
                "LLM returned empty content via provider=%s model=%s. Falling back to heuristic generator.",
                config.provider,
                config.model,
            )
            fallback_answer = heuristic_generate_answer(query, results).strip()
            return fallback_answer if fallback_answer else "No-Answer"

        return _postprocess_answer(query, answer)
    except Exception as e:
        logger.error("LLM API call failed via provider=%s model=%s: %s", config.provider, config.model, e)
        fallback_answer = heuristic_generate_answer(query, results).strip()
        return fallback_answer if fallback_answer else "No-Answer"
