# agent package
#
# Attributes resolve lazily (PEP 562) so importing a light submodule such as
# agent.guard_rules — which is stdlib-only — does not drag LangChain in.
# Demo mode depends on that: it reuses the real guard layer while running
# without the LLM stack.

__all__ = ["TOOLS", "TOOL_MAP", "AgentState"]


def __getattr__(name: str):
    if name in ("TOOLS", "TOOL_MAP"):
        from agent import tools

        return getattr(tools, name)
    if name == "AgentState":
        from agent.state import AgentState

        return AgentState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
