"""
tests/test_llm_provider.py
──────────────────────────
Tests for the provider factory and message normalisation.

The content-block cases matter more than they look: Gemini 3.x returns a list
of blocks instead of a string, which silently turns a chat reply into a Python
repr in the UI if nothing normalises it.
"""

import pytest

from agent import llm as llm_module
from agent.llm import DEFAULT_MODELS, message_text, resolve_model, resolve_provider


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch):
    for var in (
        "LLM_PROVIDER", "LLM_MODEL", "GROQ_MODEL",
        "GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# ── message_text ──────────────────────────────────────────────────────────────

def test_plain_string_passes_through():
    assert message_text("Hello") == "Hello"


def test_gemini_style_content_blocks_are_flattened():
    content = [{"type": "text", "text": "PO-100002 is approved.", "extras": {"signature": "..."}}]
    assert message_text(content) == "PO-100002 is approved."


def test_multiple_blocks_are_joined():
    content = [{"type": "text", "text": "First. "}, {"type": "text", "text": "Second."}]
    assert message_text(content) == "First. Second."


def test_non_text_blocks_are_dropped():
    content = [
        {"type": "thinking", "thinking": "internal reasoning"},
        {"type": "text", "text": "Visible answer."},
    ]
    assert message_text(content) == "Visible answer."


def test_list_of_bare_strings():
    assert message_text(["a", "b"]) == "ab"


@pytest.mark.parametrize("empty", ["", [], None])
def test_empty_content_becomes_empty_string(empty):
    assert message_text(empty) == ""


# ── resolve_provider ──────────────────────────────────────────────────────────

def test_explicit_provider_wins_over_present_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert resolve_provider() == "groq"


def test_provider_is_detected_from_whichever_key_is_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert resolve_provider() == "groq"


def test_gemini_is_preferred_when_both_keys_exist(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "y")
    assert resolve_provider() == "gemini"


def test_falls_back_to_ollama_when_no_key_is_set():
    assert resolve_provider() == "ollama"


def test_blank_key_does_not_count_as_configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "   ")
    assert resolve_provider() == "ollama"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gpt5")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "GEMINI")
    assert resolve_provider() == "gemini"


# ── resolve_model ─────────────────────────────────────────────────────────────

def test_each_provider_has_a_default_model():
    for provider, model in DEFAULT_MODELS.items():
        assert resolve_model(provider) == model


def test_llm_model_overrides_the_default(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    assert resolve_model("gemini") == "some-other-model"


def test_legacy_groq_model_var_still_honoured(monkeypatch):
    """Pre-abstraction configs set GROQ_MODEL; don't silently ignore them."""
    monkeypatch.setenv("GROQ_MODEL", "mixtral-8x7b-32768")
    assert resolve_model("groq") == "mixtral-8x7b-32768"
    assert resolve_model("gemini") == DEFAULT_MODELS["gemini"]


# ── build_llm ─────────────────────────────────────────────────────────────────

def test_missing_key_explains_how_to_fix_it(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(RuntimeError) as exc:
        llm_module.build_llm()

    message = str(exc.value)
    assert "GOOGLE_API_KEY" in message
    assert "aistudio.google.com" in message
    assert "LLM_PROVIDER=ollama" in message


def test_ollama_needs_no_key(monkeypatch):
    """Should fail on the missing package, never on a missing credential."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    try:
        llm_module.build_llm()
    except RuntimeError as exc:
        assert "langchain-ollama" in str(exc)
