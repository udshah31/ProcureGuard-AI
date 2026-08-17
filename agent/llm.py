"""
agent/llm.py
────────────
Provider-agnostic LLM factory.

The agent only needs a chat model that supports tool calling, so the provider
is a config choice rather than a code choice. Keeping it swappable means a
deprecated model or an expired key is a one-line fix, and the eval suite can be
run across several models to compare them.

Configuration:
    LLM_PROVIDER   groq | gemini | ollama   (auto-detected from keys if unset)
    LLM_MODEL      overrides the provider default
    LLM_TEMPERATURE / MAX_NEW_TOKENS

Ollama needs no API key and runs locally; the hosted providers need their key.
"""

import logging
import os

log = logging.getLogger(__name__)

DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3.6-flash",
    "ollama": "llama3.2:3b",
}
# Pinned deliberately: 'gemini-flash-latest' survives deprecations but moves
# under you, which makes eval scores incomparable between runs.

# Reasoning models spend this budget on internal thinking before emitting
# anything, so a budget sized for a non-thinking model comes back empty with
# finish_reason=MAX_TOKENS — no text and no tool call.
DEFAULT_MAX_TOKENS: dict[str, int] = {
    "groq": 1024,
    "gemini": 8192,
    "ollama": 1024,
}

# Provider → the env var holding its key. Ollama is local, so it has none.
KEY_VARS: dict[str, tuple[str, ...]] = {
    "groq": ("GROQ_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "ollama": (),
}

SIGNUP_URLS: dict[str, str] = {
    "groq": "https://console.groq.com",
    "gemini": "https://aistudio.google.com/apikey",
    "ollama": "https://ollama.com/download",
}


def _read_key(provider: str) -> str:
    for var in KEY_VARS[provider]:
        value = os.getenv(var, "").strip()
        if value:
            return value
    return ""


def resolve_provider() -> str:
    """Explicit LLM_PROVIDER wins; otherwise pick whichever key is present."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{explicit}'. "
                f"Choose one of: {', '.join(DEFAULT_MODELS)}."
            )
        return explicit

    for provider in ("gemini", "groq"):
        if _read_key(provider):
            return provider
    return "ollama"


def resolve_model(provider: str) -> str:
    explicit = os.getenv("LLM_MODEL", "").strip()
    if explicit:
        return explicit
    if provider == "groq":
        legacy = os.getenv("GROQ_MODEL", "").strip()   # pre-abstraction config
        if legacy:
            return legacy
    return DEFAULT_MODELS[provider]


def resolve_max_tokens(provider: str) -> int:
    explicit = os.getenv("MAX_NEW_TOKENS", "").strip()
    if explicit:
        return int(explicit)
    return DEFAULT_MAX_TOKENS[provider]


def describe() -> str:
    """Human-readable 'provider/model', for logs and the health endpoint."""
    provider = resolve_provider()
    return f"{provider}/{resolve_model(provider)}"


def message_text(content) -> str:
    """
    Flatten message content to plain text.

    Gemini 3.x and Anthropic return a list of content blocks rather than a
    string, so anything rendering a reply has to normalise first or it ends up
    displaying a Python repr.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content) if content else ""


PACKAGES: dict[str, str] = {
    "groq": "langchain-groq",
    "gemini": "langchain-google-genai",
    "ollama": "langchain-ollama",
}


def _import_provider(provider: str, module: str, attr: str):
    try:
        return getattr(__import__(module, fromlist=[attr]), attr)
    except ImportError as exc:
        raise RuntimeError(
            f"The '{provider}' provider needs {PACKAGES[provider]}. "
            f"Install it with: pip install {PACKAGES[provider]}"
        ) from exc


def build_llm(provider: str | None = None, model: str | None = None):
    """Return a tool-calling chat model for the configured provider."""
    provider = (provider or resolve_provider()).lower()
    model = model or resolve_model(provider)
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens = resolve_max_tokens(provider)

    if provider != "ollama" and not _read_key(provider):
        raise RuntimeError(
            f"{KEY_VARS[provider][0]} is not set. Get a free key at "
            f"{SIGNUP_URLS[provider]} and add it to your .env file, or set "
            f"LLM_PROVIDER=ollama to run locally without a key."
        )

    log.info("Initialising LLM: %s/%s", provider, model)

    if provider == "groq":
        ChatGroq = _import_provider(provider, "langchain_groq", "ChatGroq")
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=_read_key("groq"),
        )

    if provider == "gemini":
        ChatGoogleGenerativeAI = _import_provider(
            provider, "langchain_google_genai", "ChatGoogleGenerativeAI"
        )
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=_read_key("gemini"),
        )

    ChatOllama = _import_provider(provider, "langchain_ollama", "ChatOllama")
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_predict=max_tokens,
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )
