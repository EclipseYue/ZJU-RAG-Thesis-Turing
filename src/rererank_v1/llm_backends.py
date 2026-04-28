import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)
_LAST_REQUEST_TS: dict[str, float] = {}


@dataclass
class OpenAICompatConfig:
    provider: str
    api_key: str
    base_url: str
    model: str


def _pick(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def resolve_openai_compat_config(
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[OpenAICompatConfig]:
    """
    Resolve a compatible LLM backend configuration from explicit args or env vars.

    Supported providers:
    - auto: prefer explicit OPENAI_* envs, then Moonshot/Kimi, then SiliconFlow.
    - deepseek: uses DEEPSEEK_API_KEY and https://api.deepseek.com
    - moonshot: uses MOONSHOT_API_KEY / KIMI_API_KEY and https://api.moonshot.cn/v1
    - siliconflow: uses SILICONFLOW_API_KEY or OPENAI_API_KEY and https://api.siliconflow.cn/v1
    - openai: uses OPENAI_API_KEY / OPENAI_BASE_URL
    """
    provider = (provider or os.getenv("RERERANK_LLM_PROVIDER") or "auto").lower()

    if provider == "deepseek":
        resolved_key = _pick(api_key, os.getenv("DEEPSEEK_API_KEY"))
        resolved_url = _pick(base_url, os.getenv("DEEPSEEK_BASE_URL"), "https://api.deepseek.com")
        if not resolved_key:
            return None
        return OpenAICompatConfig("deepseek", resolved_key, resolved_url, model)

    if provider == "moonshot":
        resolved_key = _pick(api_key, os.getenv("MOONSHOT_API_KEY"), os.getenv("KIMI_API_KEY"))
        resolved_url = _pick(base_url, os.getenv("MOONSHOT_BASE_URL"), "https://api.moonshot.cn/v1")
        if not resolved_key:
            return None
        return OpenAICompatConfig("moonshot", resolved_key, resolved_url, model)

    if provider == "siliconflow":
        resolved_key = _pick(api_key, os.getenv("SILICONFLOW_API_KEY"), os.getenv("OPENAI_API_KEY"))
        resolved_url = _pick(base_url, os.getenv("SILICONFLOW_BASE_URL"), os.getenv("OPENAI_BASE_URL"), "https://api.siliconflow.cn/v1")
        if not resolved_key:
            return None
        return OpenAICompatConfig("siliconflow", resolved_key, resolved_url, model)

    if provider == "openai":
        resolved_key = _pick(api_key, os.getenv("OPENAI_API_KEY"))
        resolved_url = _pick(base_url, os.getenv("OPENAI_BASE_URL"), "https://api.openai.com/v1")
        if not resolved_key:
            return None
        return OpenAICompatConfig("openai", resolved_key, resolved_url, model)

    # auto
    return (
        resolve_openai_compat_config(model=model, provider="openai", api_key=api_key, base_url=base_url)
        or resolve_openai_compat_config(model=model, provider="deepseek", api_key=api_key, base_url=base_url)
        or resolve_openai_compat_config(model=model, provider="moonshot", api_key=api_key, base_url=base_url)
        or resolve_openai_compat_config(model=model, provider="siliconflow", api_key=api_key, base_url=base_url)
    )


def build_openai_compat_client(config: OpenAICompatConfig):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Please install openai package: pip install openai") from exc
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env %s=%r. Falling back to %s.", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int env %s=%r. Falling back to %s.", name, value, default)
        return default


def _provider_rate_key(config: OpenAICompatConfig) -> str:
    return f"{config.provider}::{config.base_url}"


def _respect_rate_limit(config: OpenAICompatConfig) -> None:
    min_interval = _env_float("RERERANK_LLM_MIN_INTERVAL_SEC", 3.2)
    if min_interval <= 0:
        return
    key = _provider_rate_key(config)
    last_ts = _LAST_REQUEST_TS.get(key)
    now = time.monotonic()
    if last_ts is not None:
        sleep_for = min_interval - (now - last_ts)
        if sleep_for > 0:
            logger.info(
                "LLM rate limiter sleeping %.2fs before provider=%s model=%s",
                sleep_for,
                config.provider,
                config.model,
            )
            time.sleep(sleep_for)
    _LAST_REQUEST_TS[key] = time.monotonic()


def _retry_delay(attempt_idx: int) -> float:
    base_delay = _env_float("RERERANK_LLM_BACKOFF_BASE_SEC", 5.0)
    max_delay = _env_float("RERERANK_LLM_BACKOFF_MAX_SEC", 60.0)
    return min(base_delay * (2 ** attempt_idx), max_delay)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text or "overloaded" in text


def create_chat_completion(client, config: OpenAICompatConfig, **kwargs):
    max_retries = _env_int("RERERANK_LLM_MAX_RETRIES", 6)
    for attempt in range(max_retries + 1):
        try:
            _respect_rate_limit(config)
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt >= max_retries or not _is_rate_limit_error(exc):
                raise
            delay = _retry_delay(attempt)
            logger.warning(
                "LLM request rate-limited for provider=%s model=%s (attempt %s/%s). Sleeping %.2fs. Error=%s",
                config.provider,
                config.model,
                attempt + 1,
                max_retries + 1,
                delay,
                exc,
            )
            time.sleep(delay)


def _collect_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_collect_text(item))
        return parts
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "content", "reasoning_content", "output_text", "arguments", "value"):
            if key in value:
                parts.extend(_collect_text(value.get(key)))
        return parts

    # Pydantic/OpenAI SDK objects
    for attr in ("text", "content", "reasoning_content", "output_text", "arguments", "value"):
        if hasattr(value, attr):
            parts = _collect_text(getattr(value, attr))
            if parts:
                return parts
    return []


def extract_message_text(message: Any) -> str:
    """
    Best-effort extraction for OpenAI-compatible message payloads.

    Compatible providers sometimes return:
    - message.content as plain string
    - message.content as structured list/dicts
    - blank content but populated reasoning_content
    - tool/function call arguments carrying the only text payload
    """
    if message is None:
        return ""

    candidates: list[str] = []
    for attr in ("content", "reasoning_content", "output_text", "refusal"):
        if hasattr(message, attr):
            candidates.extend(_collect_text(getattr(message, attr)))

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        candidates.extend(_collect_text(tool_calls))

    function_call = getattr(message, "function_call", None)
    if function_call:
        candidates.extend(_collect_text(function_call))

    deduped: list[str] = []
    seen = set()
    for item in candidates:
        normalized = item.strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return "\n".join(deduped).strip()
