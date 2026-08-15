"""Sentry boundary tests use a real SDK client and a fake in-memory transport."""

from __future__ import annotations

import json
from typing import Any

import pytest
import tm_llm.app as app_module
from fastapi.testclient import TestClient
from sentry_sdk.transport import Transport
from tm_llm.sentry import create_sentry_reporter, is_valid_sentry_dsn

_CAPTURED_BATCHES: list[list[Any]] = []


class CaptureTransport(Transport):
    def __init__(self, options=None):
        super().__init__(options)
        self.envelopes = []
        _CAPTURED_BATCHES.append(self.envelopes)

    def capture_envelope(self, envelope):
        self.envelopes.append(envelope)


def test_dsn_gate_rejects_unsafe_forms():
    assert is_valid_sentry_dsn("https://public@example.com/1")
    assert not is_valid_sentry_dsn("http://public@example.com/1")
    assert not is_valid_sentry_dsn("HTTPS://public@example.com/1")
    assert not is_valid_sentry_dsn("https://user:password@example.com/1")
    assert not is_valid_sentry_dsn("https://public@example.com/1?token=secret")
    assert not is_valid_sentry_dsn("https://public%5Fkey@example.com/1")


def test_noop_for_malformed_dsn():
    reporter = create_sentry_reporter(
        {"TM_AI_SENTRY_DSN": "not-a-dsn"}, transport=CaptureTransport
    )
    assert not reporter.enabled
    reporter.capture(RuntimeError("should-not-send"), "server_error")
    assert reporter.flush()
    assert reporter.close()


def test_real_sdk_envelope_is_privacy_safe():
    reporter = create_sentry_reporter(
        {
            "TM_AI_SENTRY_DSN": "https://public@example.com/1",
            "TM_AI_SENTRY_ENVIRONMENT": "staging",
            "TM_AI_SENTRY_RELEASE": "tm-ai-test-abc123",
        },
        transport=CaptureTransport,
    )
    try:
        assert reporter.enabled
        reporter.capture(
            RuntimeError(
                "authorization=secretAuthorization Cookie=secretCookie "
                "10.0.0.42 query=secretQuery body=secretBody nested=secretNested"
            ),
            "server_error",
        )
        assert reporter.flush()
        envelopes = _CAPTURED_BATCHES[-1]
        assert len(envelopes) == 1
        event = envelopes[0].get_event()
        assert event is not None
        serialized = json.dumps(event, sort_keys=True)
        for sentinel in (
            "secretAuthorization",
            "secretCookie",
            "10.0.0.42",
            "secretQuery",
            "secretBody",
            "secretNested",
        ):
            assert sentinel not in serialized
        assert event["environment"] == "staging"
        assert event["release"] == "tm-ai-test-abc123"
        assert event["tags"]["service"] == "tm-ai-server"
        assert event["tags"]["error_kind"] == "server_error"
        assert "message" not in event
        assert "request" not in event
        assert "user" not in event
        assert "contexts" not in event
        assert "extra" not in event
    finally:
        assert reporter.close()


async def _sentry_test_error():
    raise RuntimeError("secretSentinel from unexpected route")


app_module.app.add_api_route(
    "/__sentry_test_error", _sentry_test_error, methods=["GET"]
)


def test_expected_4xx_and_404_are_not_captured(monkeypatch):
    reporter = create_sentry_reporter(
        {"TM_AI_SENTRY_DSN": "https://public@example.com/1"}, transport=CaptureTransport
    )
    monkeypatch.setattr(app_module, "reporter", reporter)
    try:
        with TestClient(app_module.app, raise_server_exceptions=False) as client:
            assert client.get("/missing-route").status_code == 404
            assert client.post("/move", json={}).status_code == 422
            assert client.get("/__sentry_test_error").status_code == 500
            assert reporter.flush()
        envelopes = _CAPTURED_BATCHES[-1]
        assert len(envelopes) == 1
        event = envelopes[0].get_event()
        assert event is not None
        assert "secretSentinel" not in json.dumps(event)
    finally:
        reporter.close()


@pytest.mark.asyncio
async def test_startup_failure_is_captured_and_re_raised(monkeypatch):
    reporter = create_sentry_reporter(
        {"TM_AI_SENTRY_DSN": "https://public@example.com/1"}, transport=CaptureTransport
    )
    monkeypatch.setattr(app_module, "reporter", reporter)
    monkeypatch.setattr(app_module.config, "OPENROUTER_API_KEY", "configured")

    def fail_startup():
        raise RuntimeError("startup-secret")

    monkeypatch.setattr(app_module, "ensure_client", fail_startup)
    try:
        with pytest.raises(RuntimeError, match="startup-secret"):
            async with app_module.lifespan(app_module.app):
                pass
        envelopes = _CAPTURED_BATCHES[-1]
        assert len(envelopes) == 1
        event = envelopes[0].get_event()
        assert event is not None
        assert "startup-secret" not in json.dumps(event)
        assert event["tags"]["error_kind"] == "startup"
    finally:
        reporter.close()
