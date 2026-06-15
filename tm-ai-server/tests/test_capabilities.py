"""Tests for the model capability table and the global OPENROUTER_THINKING override."""
import pytest

from tm_llm import config, openrouter


@pytest.fixture(autouse=True)
def reset_caps_and_thinking():
    """Each test runs with a clean capability cache and the override restored after."""
    saved = config.OPENROUTER_THINKING
    openrouter._capabilities.clear()
    yield
    config.OPENROUTER_THINKING = saved
    openrouter._capabilities.clear()


MODEL = "deepseek/deepseek-v4-flash"  # thinking=True in the static table


def test_thinking_auto_honours_table():
    config.OPENROUTER_THINKING = "auto"
    assert openrouter.get_capabilities(MODEL)["thinking"] is True


def test_thinking_off_forces_disabled():
    config.OPENROUTER_THINKING = "off"
    assert openrouter.get_capabilities(MODEL)["thinking"] is False


def test_thinking_on_forces_enabled():
    config.OPENROUTER_THINKING = "on"
    # gpt-4o-mini is thinking=False in the table; "on" overrides it.
    assert openrouter.get_capabilities("openai/gpt-4o-mini")["thinking"] is True


def test_thinking_override_leaves_caching_untouched():
    config.OPENROUTER_THINKING = "off"
    caps = openrouter.get_capabilities("anthropic/claude-sonnet-4-6")
    assert caps["thinking"] is False and caps["caching"] is True


def test_provider_routing_pins_explicit_provider():
    """OPENROUTER_PROVIDER pins one provider with no fallback (warm cache, no slow drift)."""
    saved = config.OPENROUTER_PROVIDER
    try:
        config.OPENROUTER_PROVIDER = "Cloudflare"
        r = openrouter._provider_routing("deepseek/deepseek-v4-flash")
        assert r == {"order": ["Cloudflare"], "allow_fallbacks": False, "require_parameters": True}
        # comma-separated → ordered preference; applies regardless of model vendor
        config.OPENROUTER_PROVIDER = "Cloudflare, DeepSeek"
        r = openrouter._provider_routing("anthropic/claude-opus-4-7")
        assert r["order"] == ["Cloudflare", "DeepSeek"] and r["allow_fallbacks"] is False
    finally:
        config.OPENROUTER_PROVIDER = saved


def test_provider_routing_default_throughput_for_open_models():
    saved = config.OPENROUTER_PROVIDER
    try:
        config.OPENROUTER_PROVIDER = ""
        # an open model with no per-model pin falls back to throughput sorting
        assert openrouter._provider_routing("deepseek/deepseek-v4-pro") == {
            "sort": "throughput", "require_parameters": True}
        # Anthropic/OpenAI resolve to a single host already → no provider block
        assert openrouter._provider_routing("anthropic/claude-opus-4-7") is None
        assert openrouter._provider_routing("openai/gpt-4o-mini") is None
    finally:
        config.OPENROUTER_PROVIDER = saved


def test_provider_routing_per_model_pin():
    """A model in _MODEL_PROVIDER is pinned (with fallback) so its prompt cache stays warm,
    instead of floating across providers under throughput sorting."""
    saved = config.OPENROUTER_PROVIDER
    try:
        config.OPENROUTER_PROVIDER = ""
        r = openrouter._provider_routing("deepseek/deepseek-v4-flash")
        assert r == {"order": ["GMICloud", "Baidu"], "allow_fallbacks": True,
                     "require_parameters": True}
    finally:
        config.OPENROUTER_PROVIDER = saved


def test_explicit_provider_overrides_per_model_pin():
    """A global OPENROUTER_PROVIDER takes precedence over a per-model pin."""
    saved = config.OPENROUTER_PROVIDER
    try:
        config.OPENROUTER_PROVIDER = "Cloudflare"
        r = openrouter._provider_routing("deepseek/deepseek-v4-flash")
        assert r == {"order": ["Cloudflare"], "allow_fallbacks": False,
                     "require_parameters": True}
    finally:
        config.OPENROUTER_PROVIDER = saved


def test_gemini_3_1_flash_lite_thinking_off():
    """gemini-3.1-flash-lite must match its flash-lite sibling (no reasoning), not fall through
    to the unknown-model default of thinking=True."""
    config.OPENROUTER_THINKING = "auto"
    caps = openrouter.get_capabilities("google/gemini-3.1-flash-lite")
    assert caps["thinking"] is False and caps["caching"] is False
