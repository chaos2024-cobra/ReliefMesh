"""Reusable Gemini client (official `google-genai` SDK).

- Reads GEMINI_* configuration from the shared settings; the API key is
  never logged, returned or persisted.
- Tolerates a missing SDK/key: the backend starts normally and only AI
  endpoints return a clear configuration error.
- Translates upstream failures into typed application errors
  (auth/model/network -> AiUpstreamError, quota/rate limit -> AiQuotaError).
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    AiNotConfiguredError,
    AiQuotaError,
    AiUpstreamError,
)

logger = logging.getLogger("reliefmesh.ai.client")

# Imported lazily/optionally so the app runs even without the package.
try:  # pragma: no cover - exercised implicitly
    from google import genai
    from google.genai import types as genai_types

    _SDK_ERROR_BASE: tuple[type[Exception], ...] | None = None
    for candidate in ("APIError", "ClientError", "ServerError"):
        if hasattr(genai_errors := __import__(
            "google.genai.errors", fromlist=["errors"]), candidate
        ):
            _SDK_ERROR_BASE = _SDK_ERROR_BASE or ()
            _SDK_ERROR_BASE += (getattr(genai_errors, candidate),)
    if _SDK_ERROR_BASE is None:  # pragma: no cover - defensive
        _SDK_ERROR_BASE = (Exception,)
except ImportError:  # pragma: no cover - SDK not installed
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    _SDK_ERROR_BASE = None


class GeminiClient:
    """Thin wrapper around `google.genai.Client` for content generation."""

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 60.0) -> None:
        if not api_key or not api_key.strip():
            raise AiNotConfiguredError(
                "GEMINI_API_KEY is not configured; AI endpoints are unavailable.",
                details={"hint": "Set GEMINI_API_KEY in backend/.env"},
            )
        if genai is None:  # pragma: no cover - guarded by ImportError branch
            raise AiNotConfiguredError(
                "The google-genai SDK is not installed.",
                details={"hint": "pip install google-genai"},
            )
        self.model = model
        # The SDK builds on httpx; timeout applies per request.
        self._client = genai.Client(api_key=api_key, http_options={"timeout": timeout_seconds * 1000})

    def generate_response(self, contents: list[Any], config: Any) -> Any:
        """Call `models.generate_content` with typed upstream error mapping."""
        try:
            return self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except _SDK_ERROR_BASE as exc:  # type: ignore[misc]
            raise _translate_sdk_error(exc) from exc


def _translate_sdk_error(exc: Exception) -> AiUpstreamError | AiQuotaError:
    """Map an upstream SDK failure onto our typed application errors."""
    message = type(exc).__name__
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()
    # Never echo full payloads; a short class/status summary is enough.
    if (status in (401, 403)) or "api key" in text or "unauthenticated" in text or "permission" in text:
        return AiUpstreamError(
            "Gemini rejected the credentials (authentication failed).",
            details={"upstream_status": status},
        )
    if isinstance(status, int) and status == 429 or "quota" in text or "rate limit" in text:
        return AiQuotaError(
            "Gemini quota/rate limit exhausted. Try again shortly.",
            details={"upstream_status": status},
        )
    if isinstance(status, int) and status == 404 or "model" in text and "not found" in text:
        return AiUpstreamError(
            "The configured Gemini model is unavailable.",
            details={"upstream_status": status, "model": settings.gemini_model},
        )
    if "timeout" in text or "timed out" in text or "connection" in text:
        return AiUpstreamError(
            "Gemini request timed out or the network is unreachable.",
            details={"upstream_status": status},
        )
    logger.warning("Gemini upstream error: %s (status=%s)", message, status)
    return AiUpstreamError(
        "Gemini upstream is temporarily unavailable.",
        details={"upstream_status": status},
    )


def _type_str(type_enum: Any) -> str:
    """Normalise a gemini `Type` enum member (or bare string) to a lowercase
    JSON-schema type name (OpenAI-compatible APIs require lowercase)."""
    value = getattr(type_enum, "value", type_enum)
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _schema_to_json(value: Any) -> dict:
    """Convert a `google.genai.types.Schema` (or plain dict) to JSON Schema."""
    if value is None:
        return {"type": "object"}
    if isinstance(value, dict):
        base: dict[str, Any] = {}
        if "type" in value:
            base["type"] = _type_str(value["type"])
        for key in ("description", "required", "enum"):
            if key in value and value[key] is not None:
                base[key] = value[key]
        if "properties" in value:
            base["properties"] = {
                k: _schema_to_json(v) for k, v in (value["properties"] or {}).items()
            }
        if "items" in value:
            base["items"] = _schema_to_json(value["items"])
        return base
    properties = getattr(value, "properties", None)
    converted: dict[str, Any] = {"type": _type_str(getattr(value, "type", "OBJECT"))}
    description = getattr(value, "description", None)
    if description:
        converted["description"] = description
    enum = getattr(value, "enum", None)
    if enum:
        converted["enum"] = [str(item) for item in enum]
    required = getattr(value, "required", None)
    if required:
        converted["required"] = list(required)
    if properties:
        converted["properties"] = {
            k: _schema_to_json(v) for k, v in properties.items()
        }
    items = getattr(value, "items", None)
    if items is not None:
        converted["items"] = _schema_to_json(items)
    return converted


def _system_instruction_text(config: Any) -> str:
    instruction = getattr(config, "system_instruction", None)
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    parts = getattr(instruction, "parts", None) or []
    return "\n".join(p for p in (getattr(part, "text", None) for part in parts) if p)


class GroqClient:
    """OpenAI-compatible provider (Groq) behind the Gemini client interface.

    Accepts `google.genai` shaped `contents`/`config`, converts them to the
    OpenAI chat-completions format for Groq, and returns a Gemini-shaped
    response object so the shared manual function-calling loop keeps working
    unchanged across providers.
    """

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 60.0,
                 base_url: str = "https://api.groq.com/openai/v1") -> None:
        import json as _json  # local import keeps module import cheap

        if not api_key or not api_key.strip():
            raise AiNotConfiguredError(
                "GROQ_API_KEY is not configured; AI endpoints are unavailable.",
                details={"hint": "Set GROQ_API_KEY in backend/.env"},
            )
        import httpx

        self.model = model
        self._json = _json
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=httpx.Timeout(timeout_seconds),
        )

    # ------------------------------------------------------------------ tools
    @staticmethod
    def _tools_from_config(config: Any) -> list[dict]:
        tools = getattr(config, "tools", None) or []
        out: list[dict] = []
        for tool in tools:
            for decl in getattr(tool, "function_declarations", None) or []:
                parameters = _schema_to_json(getattr(decl, "parameters", None))
                out.append({
                    "type": "function",
                    "function": {
                        "name": decl.name,
                        "description": getattr(decl, "description", "") or "",
                        "parameters": parameters,
                    },
                })
        return out

    # -------------------------------------------------------------- messages
    def _messages_from_contents(self, contents: list[Any]) -> list[dict]:
        """Map genai `Content` objects onto OpenAI chat messages.

        Call-ids follow the conversation order: the orchestrator always sends
        the assistant turn (tool calls) directly before the matching
        function-response parts, so a FIFO queue ties ids deterministically.
        """
        messages: list[dict] = []
        pending_ids: list[str] = []
        call_no = 0

        for content in contents:
            role_raw = getattr(content, "role", "user")
            role = "assistant" if role_raw in ("model", "assistant") else "user"
            parts = getattr(content, "parts", None) or []

            text_bits: list[str] = []
            tool_calls: list[dict] = []
            for part in parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_bits.append(text)
                    continue
                call = getattr(part, "function_call", None)
                if call is not None:
                    name = getattr(call, "name", "")
                    args = getattr(call, "args", None) or {}
                    try:
                        arguments = self._json.dumps(dict(args))
                    except (TypeError, ValueError):
                        arguments = "{}"
                    call_no += 1
                    call_id = f"call_{call_no:04d}"
                    pending_ids.append(call_id)
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": str(name), "arguments": arguments},
                    })

            responses = [
                getattr(part, "function_response", None)
                for part in parts
            ]
            responses = [r for r in responses if r is not None]
            if responses:
                for fr in responses:
                    call_id = pending_ids.pop(0) if pending_ids else f"call_{call_no:04d}"
                    payload = getattr(fr, "response", None) or {}
                    if not isinstance(payload, str):
                        payload = self._json.dumps(payload)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": payload,
                    })
                continue

            message: dict[str, Any] = {"role": role, "content": "\n".join(text_bits) or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        return messages

    # ------------------------------------------------------------------- call
    def generate_response(self, contents: list[Any], config: Any) -> Any:
        """Call Groq chat-completions and shape the result like a Gemini response."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_from_contents(contents),
        }
        temperature = getattr(config, "temperature", None)
        if temperature is not None:
            payload["temperature"] = temperature
        max_tokens = getattr(config, "max_output_tokens", None)
        if max_tokens:
            payload["max_tokens"] = max_tokens
        tools = self._tools_from_config(config)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        system_text = _system_instruction_text(config)
        if system_text:
            payload["messages"].insert(0, {"role": "system", "content": system_text})

        import types as _types

        try:
            response = self._http.post("/chat/completions", json=payload)
        except Exception as exc:  # network layer
            raise _translate_transport_error(exc) from exc

        if response.status_code >= 400:
            raise _translate_http_error(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise AiUpstreamError(
                "Groq returned a malformed response.",
                details={"hint": "non-JSON payload"},
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise AiUpstreamError("Gemini returned no candidates.")
        choice = choices[0]
        message = choice.get("message") or {}
        parts: list[Any] = []
        content_text = message.get("content")
        if isinstance(content_text, str) and content_text.strip():
            parts.append(_types.SimpleNamespace(text=content_text))
        for tool_call in message.get("tool_calls") or []:
            fn = (tool_call.get("function") or {})
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or "{}"
            try:
                args = self._json.loads(raw_args)
            except ValueError:
                args = {"_raw": str(raw_args)[:200]}
            parts.append(_types.SimpleNamespace(
                function_call=_types.SimpleNamespace(name=name, args=args),
            ))
        if not parts:
            raise AiUpstreamError("Gemini returned an empty response.")

        finish_map = {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "tool_calls": "STOP",
        }
        finish = finish_map.get(str(choice.get("finish_reason", "stop")).lower(), "STOP")
        candidate = _types.SimpleNamespace(
            content=_types.SimpleNamespace(role="model", parts=parts),
            finish_reason=finish,
        )
        usage = data.get("usage") or {}
        return _types.SimpleNamespace(
            candidates=[candidate],
            usage_metadata=_types.SimpleNamespace(
                prompt_token_count=usage.get("prompt_tokens"),
                candidates_token_count=usage.get("completion_tokens"),
                total_token_count=usage.get("total_tokens"),
            ),
            model_version=data.get("model"),
        )


def _translate_http_error(response: Any) -> AiUpstreamError | AiQuotaError:
    """Map a Groq HTTP failure onto the typed application errors."""
    status = response.status_code
    try:
        detail = response.json()
    except ValueError:
        detail = {}
    text = str(detail.get("error", {}).get("message", "") or detail)[:300]
    if status in (401, 403):
        return AiUpstreamError(
            "The AI provider rejected the credentials (authentication failed).",
            details={"upstream_status": status},
        )
    if status == 429:
        return AiQuotaError(
            "Groq quota/rate limit exhausted. Try again shortly.",
            details={"upstream_status": status},
        )
    if status == 404 or "model" in text.lower() and "not" in text.lower():
        return AiUpstreamError(
            "The configured Groq model is unavailable.",
            details={"upstream_status": status, "model": settings.groq_model, "upstream": text},
        )
    return AiUpstreamError(
        "The AI provider upstream is temporarily unavailable.",
        details={"upstream_status": status, "upstream": text},
    )


def _translate_transport_error(exc: Exception) -> AiUpstreamError:
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text or "connect" in text:
        return AiUpstreamError(
            "AI request timed out or the network is unreachable.",
            details={"type": type(exc).__name__},
        )
    return AiUpstreamError(
        "The AI provider upstream is temporarily unavailable.",
        details={"type": type(exc).__name__},
    )


def get_gemini_client() -> GeminiClient:
    """Build (or reuse) the process-level provider client (gemini | groq).

    Raises `AiNotConfiguredError` when the key is missing/empty or AI is
    disabled — the API knows how to turn that into HTTP 503 with a clear
    message while the rest of the application keeps serving.
    """
    global _CLIENT_SINGLETON
    provider = settings.ai_provider.strip().lower()
    if not settings.ai_ready:
        raise AiNotConfiguredError(
            "AI integration is not configured/enabled for the active provider.",
            details={
                "ai_enabled": settings.gemini_enabled,
                "provider": provider or "gemini",
                "key_configured": False,
            },
        )
    cached = _CLIENT_SINGLETON
    if cached is not None:
        return cached
    if provider == "groq":
        client = GroqClient(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            timeout_seconds=settings.gemini_timeout_seconds,
            base_url=settings.groq_base_url,
        )
    else:
        client = GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=settings.gemini_timeout_seconds,
        )
    _CLIENT_SINGLETON = client
    return client


_CLIENT_SINGLETON: GeminiClient | None = None


def reset_gemini_client() -> None:
    """Clear the cached client (used by tests to force reconfiguration)."""
    global _CLIENT_SINGLETON
    _CLIENT_SINGLETON = None
