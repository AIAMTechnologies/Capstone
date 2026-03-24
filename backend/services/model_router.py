import os
from dataclasses import dataclass
from typing import Any, Optional

from audit_logger import log_event
from settings_store import get_setting_json

DEFAULT_TASK_MODEL_MAP = {
    "bulk_scoring": "gpt-4.1-nano",
    "email_analysis": "gpt-4.1-mini",
    "reasoning": "gpt-4.1-mini",
    "realtime": "gpt-4.1-mini",
}

DEFAULT_FALLBACK_MAP = {
    "gpt-4.1-nano": ["gpt-4o-mini"],
    "gpt-4.1-mini": ["gpt-4o-mini"],
    "gpt-4.1": ["gpt-4.1-mini", "gpt-4o-mini"],
    "gpt-4o-mini": [],
}


@dataclass
class RoutedCompletion:
    response: Any
    model_used: str
    provider: str
    task_type: str
    attempts: list[dict[str, str]]


def _get_task_model_map() -> dict[str, str]:
    configured = get_setting_json("model_router_mapping", DEFAULT_TASK_MODEL_MAP)
    return configured if isinstance(configured, dict) else DEFAULT_TASK_MODEL_MAP


def _get_fallback_map() -> dict[str, list[str]]:
    configured = get_setting_json("model_router_fallback_mapping", DEFAULT_FALLBACK_MAP)
    return configured if isinstance(configured, dict) else DEFAULT_FALLBACK_MAP


def _get_primary_model(task_type: str, override_model: Optional[str] = None) -> str:
    if override_model:
        return override_model
    mapping = _get_task_model_map()
    return str(mapping.get(task_type, DEFAULT_TASK_MODEL_MAP.get(task_type, "gpt-4.1-mini")))


def _get_candidate_models(task_type: str, override_model: Optional[str] = None) -> list[str]:
    primary = _get_primary_model(task_type, override_model=override_model)
    fallback_map = _get_fallback_map()
    candidates = [primary]
    for model in fallback_map.get(primary, []):
        if model not in candidates:
            candidates.append(model)
    return candidates


def _should_use_max_completion_tokens(model: str) -> bool:
    return model.startswith(("gpt-5", "o3", "o4"))


def _build_request_kwargs(model: str, messages: list[dict], max_tokens: int, temperature: float):
    request_kwargs = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    if _should_use_max_completion_tokens(model):
        request_kwargs["max_completion_tokens"] = max_tokens
    else:
        request_kwargs["max_tokens"] = max_tokens
        request_kwargs["temperature"] = temperature
    return request_kwargs


def _get_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    return OpenAI(api_key=api_key)


def _is_azure_enabled() -> bool:
    return os.getenv("AZURE_OPENAI_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _get_azure_deployment_name(model: str) -> Optional[str]:
    env_key = f"AZURE_OPENAI_DEPLOYMENT_{model.upper().replace('-', '_').replace('.', '_')}"
    value = os.getenv(env_key, "").strip()
    return value or None


def _get_azure_client():
    from openai import AzureOpenAI

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
    if not endpoint or not api_key:
        return None
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )


def _log_fallback(task_type: str, from_model: str, to_model: str, provider: str, attempts: list[dict[str, str]]) -> None:
    log_event(
        event_type="MODEL_ROUTER_FALLBACK",
        entity_type="ai_model",
        actor="system",
        model_used=to_model,
        payload={
            "task_type": task_type,
            "from_model": from_model,
            "to_model": to_model,
            "provider": provider,
            "attempts": attempts,
        },
    )


def route_json_completion(
    task_type: str,
    messages: list[dict],
    max_tokens: int = 300,
    temperature: float = 0.1,
    override_model: Optional[str] = None,
) -> RoutedCompletion:
    attempts: list[dict[str, str]] = []
    candidates = _get_candidate_models(task_type, override_model=override_model)
    primary = candidates[0]

    client = _get_openai_client()
    last_exc: Optional[Exception] = None

    for index, model in enumerate(candidates):
        try:
            response = client.chat.completions.create(
                **_build_request_kwargs(model, messages, max_tokens, temperature)
            )
            if index > 0:
                _log_fallback(task_type, primary, model, "openai", attempts)
            return RoutedCompletion(
                response=response,
                model_used=model,
                provider="openai",
                task_type=task_type,
                attempts=attempts,
            )
        except Exception as exc:
            attempts.append({"provider": "openai", "model": model, "error": str(exc)})
            last_exc = exc

    if _is_azure_enabled():
        azure_client = _get_azure_client()
        if azure_client:
            for model in candidates:
                deployment_name = _get_azure_deployment_name(model)
                if not deployment_name:
                    continue
                try:
                    response = azure_client.chat.completions.create(
                        **_build_request_kwargs(deployment_name, messages, max_tokens, temperature)
                    )
                    _log_fallback(task_type, primary, model, "azure_openai", attempts)
                    return RoutedCompletion(
                        response=response,
                        model_used=model,
                        provider="azure_openai",
                        task_type=task_type,
                        attempts=attempts,
                    )
                except Exception as exc:
                    attempts.append({"provider": "azure_openai", "model": model, "error": str(exc)})
                    last_exc = exc

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Model router could not resolve a provider for task_type={task_type}")
