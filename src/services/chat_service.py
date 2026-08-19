"""Application service wiring memory, retrieval, and answer generation."""
import logging
import time

logger = logging.getLogger(__name__)

HISTORY_MESSAGES_FOR_RETRIEVAL = 4


class ChatService:
    def __init__(self, retriever, memory, generator) -> None:
        self.retriever = retriever
        self.memory = memory
        self.generator = generator

    def chat(
        self,
        question: str,
        requested_session_id: str | None = None,
    ) -> dict[str, str | None]:
        started_at = time.perf_counter()
        session_id = self.memory.session_id(requested_session_id)
        history = self.memory.get(session_id)

        try:
            logger.info(
                "user_question=%r session_id=%s",
                question,
                session_id,
            )

            retrieval_query = self._build_retrieval_query(question, history)
            hits = self.retriever.retrieve(retrieval_query)
            context = self._build_context(hits)

            llm_started_at = time.perf_counter()
            answer = self.generator.answer(question, context, history)
            logger.info(
                "llm_seconds=%.3f",
                time.perf_counter() - llm_started_at,
            )

            history.add_user_message(question)
            history.add_ai_message(answer)

            logger.info(
                "total_response_seconds=%.3f",
                time.perf_counter() - started_at,
            )
            return {"answer": answer, "session_id": session_id, "error": None}

        except Exception:
            logger.exception("chat_failed session_id=%s", session_id)
            return {
                "answer": None,
                "session_id": session_id,
                "error": "Something went wrong. Please try again.",
            }

    def _build_retrieval_query(self, question: str, history) -> str:
        """Prefix the question with recent chat turns for better retrieval."""
        recent_messages = history.messages[-HISTORY_MESSAGES_FOR_RETRIEVAL:]
        recent_text = [
            message.content
            for message in recent_messages
            if isinstance(message.content, str)
        ]
        return "\n".join([*recent_text, question]).strip()

    def _build_context(self, hits: list[dict]) -> str:
        chunks = [self._hit_text(hit) for hit in hits]
        return "\n\n".join(chunks)

    @staticmethod
    def _hit_text(hit: dict) -> str:
        metadata = hit.get("metadata", {})
        return metadata.get("text") or metadata.get("chunk_text") or ""
