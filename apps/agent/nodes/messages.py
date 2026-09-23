"""Small shared helpers for reading node input from conversation state."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from apps.agent.state import ConversationState


def last_human_text(state: ConversationState) -> str:
    """The most recent customer message's text, or "" if there is none."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""
