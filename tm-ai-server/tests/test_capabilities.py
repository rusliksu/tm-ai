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
