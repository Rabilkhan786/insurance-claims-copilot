"""Groq adapter using the existing constrained insurance-answer prompt."""
from config import settings

SYSTEM_PROMPT = """You are a friendly insurance assistant.
Answer ONLY from CONTEXT. Do not guess.
If the answer is absent, reply exactly: Sorry, I don't know that.
Is there any other insurance-related question you would like to talk about?
Keep the answer polite, clear, well-polished, within five lines.
Do not mention documents, context, sources, or internal information.

CONTEXT:
{context}"""


class GroqAnswerGenerator:
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required for chat")

        from langchain_groq import ChatGroq

        self.model = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

    def answer(self, question: str, context: str, history) -> str:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ])
        messages = prompt.format_messages(
            context=context,
            chat_history=history.messages,
            question=question,
        )
        return self.model.invoke(messages).content
