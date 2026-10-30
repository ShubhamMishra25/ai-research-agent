"""Structured LLM invocation with retries and token-usage extraction."""
from typing import Any, Dict, List, Tuple

from app.config import Settings
from app.utils.logging_config import setup_logger
from app.utils.retry import with_retries

logger = setup_logger("utils.llm")


def _split_structured_result(result: Any) -> Tuple[Any, Dict[str, Any]]:
    """Support both ``include_raw=True`` dicts and plain parsed models."""
    if isinstance(result, dict) and "parsed" in result:
        parsed = result.get("parsed")
        if parsed is None:
            raise ValueError(f"Structured output parsing failed: {result.get('parsing_error')}")
        return parsed, _extract_usage(result.get("raw"))
    return result, {}


def _extract_usage(message: Any) -> Dict[str, Any]:
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage:
        return {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def invoke_structured(
    llm: Any,
    schema: Any,
    messages: List[Any],
    *,
    settings: Settings,
    description: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Invoke a structured-output LLM with bounded retries.

    Returns ``(parsed_output, usage)``. Usage is empty when the provider does
    not expose token metadata.
    """
    try:
        structured = llm.with_structured_output(schema, include_raw=True)
    except TypeError:
        structured = llm.with_structured_output(schema)

    result = with_retries(
        lambda: structured.invoke(messages),
        attempts=settings.max_retries + 1,
        backoff_seconds=settings.retry_backoff_seconds,
        description=description,
    )
    return _split_structured_result(result)
