import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


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
    - moonshot: uses MOONSHOT_API_KEY / KIMI_API_KEY and https://api.moonshot.cn/v1
    - siliconflow: uses SILICONFLOW_API_KEY or OPENAI_API_KEY and https://api.siliconflow.cn/v1
    - openai: uses OPENAI_API_KEY / OPENAI_BASE_URL
    """
    provider = (provider or os.getenv("RERERANK_LLM_PROVIDER") or "auto").lower()

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
        or resolve_openai_compat_config(model=model, provider="moonshot", api_key=api_key, base_url=base_url)
        or resolve_openai_compat_config(model=model, provider="siliconflow", api_key=api_key, base_url=base_url)
    )


def build_openai_compat_client(config: OpenAICompatConfig):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Please install openai package: pip install openai") from exc
    return OpenAI(api_key=config.api_key, base_url=config.base_url)

