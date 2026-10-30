"""Per-node latency and token accounting for the research workflow."""
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict

from app.state import ResearchState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def track_node(node_name: str) -> Callable:
    """Record wall-clock latency (and optional metrics) for a graph node.

    Nodes may return a private ``_metrics`` dict, which is merged into the
    node's metadata entry and removed from the public result.
    """

    def decorator(func: Callable[[ResearchState], Dict[str, Any]]):
        @wraps(func)
        def wrapper(state: ResearchState) -> Dict[str, Any]:
            started = time.perf_counter()
            result = func(state)
            elapsed_ms = (time.perf_counter() - started) * 1000

            result = dict(result)
            extra = result.pop("_metrics", {}) or {}
            metadata = dict(state.get("metadata") or {})
            metrics = list(metadata.get("metrics") or [])
            entry: Dict[str, Any] = {
                "node": node_name,
                "latency_ms": round(elapsed_ms, 2),
            }
            entry.update(extra)
            metrics.append(entry)
            metadata["metrics"] = metrics
            result["metadata"] = metadata
            return result

        return wrapper

    return decorator
