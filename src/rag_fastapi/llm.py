"""Step 4 of RAG: generate an answer grounded in the retrieved chunks."""

from typing import Protocol

import anthropic

from .store import Hit

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided in the \
user's message. Cite the sources you rely on inline as [1], [2], etc. If the sources do \
not contain the answer, say that plainly instead of guessing. Keep answers concise."""


class Generator(Protocol):
    def answer(self, question: str, hits: list[Hit]) -> str: ...


class RefusalError(RuntimeError):
    pass


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f'<source index="{i}" name="{h.chunk.source}">\n{h.chunk.text}\n</source>'
        for i, h in enumerate(hits, start=1)
    )
    return f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"


class ClaudeGenerator:
    def __init__(self, model: str, effort: str = "medium"):
        # Credentials come from ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self._client = anthropic.Anthropic()
        self._model = model
        self._effort = effort

    def answer(self, question: str, hits: list[Hit]) -> str:
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": self._effort},
            # If a safety classifier declines, retry server-side on Anthropic's
            # recommended fallback model instead of failing the request.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
        )
        if response.stop_reason == "refusal":
            raise RefusalError("Claude declined to answer this question.")
        return "".join(b.text for b in response.content if b.type == "text").strip()
