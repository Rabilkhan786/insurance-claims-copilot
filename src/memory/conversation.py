"""In-process LangChain conversation memory, scoped by client session."""
from uuid import uuid4

from config import settings


class ConversationMemory:
    def __init__(self) -> None:
        from langchain_core.chat_history import InMemoryChatMessageHistory

        self._history_type = InMemoryChatMessageHistory
        self._sessions: dict[str, object] = {}

    def session_id(self, session_id: str | None) -> str:
        if session_id and session_id.strip():
            return session_id.strip()
        return str(uuid4())

    def get(self, session_id: str):
        history = self._sessions.setdefault(session_id, self._history_type())
        if len(history.messages) > settings.memory_max_messages:
            history.messages = history.messages[-settings.memory_max_messages:]
        return history

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
