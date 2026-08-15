"""Privacy-safe, environment-gated Sentry boundary for the FastAPI service."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any

import sentry_sdk

_DSN_RE = re.compile(r"^https://[A-Za-z0-9_]+@[A-Za-z0-9.-]+/\d+$")
_SAFE_ENV_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SAFE_RELEASE_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_SAFE_TAG_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,64}$")
_ALLOWED_KINDS = frozenset({"server_error", "startup"})


def is_valid_sentry_dsn(value: str | None) -> bool:
    """Return whether *value* has the deliberately narrow DSN form we accept."""

    return value is not None and _DSN_RE.fullmatch(value) is not None


def _safe_value(value: object, pattern: re.Pattern[str], fallback: str) -> str:
    return value if isinstance(value, str) and pattern.fullmatch(value) else fallback


def _sanitize_event(
    event: dict[str, Any], _hint: dict[str, Any]
) -> dict[str, Any] | None:
    """Build a new event from a small allowlist; fail closed for unknown shapes."""

    exception = event.get("exception")
    values = exception.get("values") if isinstance(exception, dict) else None
    if not isinstance(values, list) or not values:
        return None

    safe_values: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        safe_value_data: dict[str, Any] = {"type": "Error"}
        stacktrace = value.get("stacktrace")
        frames = stacktrace.get("frames") if isinstance(stacktrace, dict) else None
        if isinstance(frames, list):
            safe_frames: list[dict[str, int | str]] = []
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                safe_frame: dict[str, int | str] = {"filename": "?", "function": "?"}
                if isinstance(frame.get("lineno"), int):
                    safe_frame["lineno"] = frame["lineno"]
                if isinstance(frame.get("colno"), int):
                    safe_frame["colno"] = frame["colno"]
                safe_frames.append(safe_frame)
            if safe_frames:
                safe_value_data["stacktrace"] = {"frames": safe_frames}
        safe_values.append(safe_value_data)

    if not safe_values:
        return None

    safe_event: dict[str, Any] = {
        "exception": {"values": safe_values},
    }
    for field in ("event_id", "platform", "environment", "release", "level"):
        value = event.get(field)
        if isinstance(value, str) and _SAFE_TAG_RE.fullmatch(value):
            safe_event[field] = value

    tags = event.get("tags")
    safe_tags: dict[str, str] = {}
    if isinstance(tags, dict):
        for field in ("service", "runtime", "error_kind"):
            value = tags.get(field)
            if isinstance(value, str) and _SAFE_TAG_RE.fullmatch(value):
                safe_tags[field] = value
    safe_event["tags"] = safe_tags
    return safe_event


class SentryReporter:
    """Small synchronous facade that keeps Sentry off the application contract."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def capture(self, error: BaseException, kind: str) -> None:
        if not self.enabled:
            return
        safe_kind = kind if kind in _ALLOWED_KINDS else "server_error"
        try:
            with sentry_sdk.new_scope() as scope:
                scope.set_tag("service", "tm-ai-server")
                scope.set_tag("runtime", "python-fastapi")
                scope.set_tag("error_kind", safe_kind)
                sentry_sdk.capture_exception(error)
        except Exception:  # noqa: BLE001
            # Observability must never break the API error path.
            return

    def flush(self, timeout: float = 1.0) -> bool:
        if not self.enabled:
            return True
        try:
            sentry_sdk.get_client().flush(timeout=timeout)
            return True
        except Exception:  # pragma: no cover - best effort only  # noqa: BLE001
            return False

    def close(self, timeout: float = 1.0) -> bool:
        if not self.enabled:
            return True
        try:
            sentry_sdk.get_client().close(timeout=timeout)
            return True
        except Exception:  # pragma: no cover - best effort only  # noqa: BLE001
            return False


class NoopSentryReporter(SentryReporter):
    def __init__(self) -> None:
        super().__init__(enabled=False)


def create_sentry_reporter(
    env: Mapping[str, str] | None = None,
    transport: Callable[..., Any] | Any | None = None,
) -> SentryReporter:
    """Create an enabled reporter only when the explicit DSN contract is satisfied."""

    source = os.environ if env is None else env
    dsn = source.get("TM_AI_SENTRY_DSN")
    if not is_valid_sentry_dsn(dsn):
        return NoopSentryReporter()

    environment = _safe_value(
        source.get("TM_AI_SENTRY_ENVIRONMENT", "local"), _SAFE_ENV_RE, "local"
    )
    release = _safe_value(
        source.get("TM_AI_SENTRY_RELEASE", "local"), _SAFE_RELEASE_RE, "local"
    )
    options: dict[str, Any] = {
        "dsn": dsn,
        "environment": environment,
        "release": release,
        "default_integrations": False,
        "send_default_pii": False,
        "traces_sample_rate": 0.0,
        "profiles_sample_rate": 0.0,
        "enable_logs": False,
        "before_send": _sanitize_event,
    }
    if transport is not None:
        options["transport"] = transport

    try:
        sentry_sdk.init(**options)
    except Exception:  # noqa: BLE001
        return NoopSentryReporter()
    return SentryReporter(enabled=True)
