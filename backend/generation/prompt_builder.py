"""
Prompt builder for citation-grounded QA.
Constructs structured prompts that instruct the LLM to:
  1. Answer from provided context only
  2. Cite the source for every factual claim
  3. Acknowledge when the answer is not in the context
"""
from __future__ import annotations

from backend.retrieval.vector_store import SearchResult

SYSTEM_PROMPT = """\
You are ScholarRAG, an expert academic research assistant.
Your job is to answer questions based ONLY on the research paper excerpts provided below.

Rules:
1. Ground every factual claim in the provided context.
2. After each claim, cite the source using [Source N] notation.
3. If the answer is not found in the context, say: "I cannot find sufficient information in the provided papers to answer this question."
4. Do NOT fabricate information or cite sources you haven't been given.
5. Be concise but complete. Use bullet points for lists.
"""

CONTEXT_TEMPLATE = """\
=== Context Excerpt {idx} ===
Source: {source}
---
{text}
"""

QA_TEMPLATE = """\
{context_block}

=== Question ===
{question}

=== Answer ===
Answer the question based solely on the excerpts above. Cite sources as [Source N].
"""


class PromptBuilder:
    """Assembles prompts from retrieved chunks and user query."""

    def build(
        self,
        question: str,
        context_chunks: list[SearchResult],
    ) -> tuple[str, str]:
        """
        Build system + user prompt pair.

        Returns:
            (system_prompt, user_prompt)
        """
        context_parts = []
        for idx, chunk in enumerate(context_chunks, start=1):
            source_label = self._format_source(chunk, idx)
            context_parts.append(
                CONTEXT_TEMPLATE.format(idx=idx, source=source_label, text=chunk.text.strip())
            )

        context_block = "\n".join(context_parts)
        user_prompt = QA_TEMPLATE.format(
            context_block=context_block, question=question
        )

        return SYSTEM_PROMPT, user_prompt

    def build_sources_list(self, chunks: list[SearchResult]) -> list[dict]:
        """Return structured source metadata for API response."""
        sources = []
        for idx, chunk in enumerate(chunks, start=1):
            sources.append(
                {
                    "index": idx,
                    "source": chunk.source,
                    "score": round(chunk.score, 4),
                    "snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "metadata": chunk.metadata,
                }
            )
        return sources

    @staticmethod
    def _format_source(chunk: SearchResult, idx: int) -> str:
        meta = chunk.metadata
        parts = [f"[Source {idx}]"]
        if title := meta.get("title"):
            parts.append(title)
        if authors := meta.get("authors"):
            if isinstance(authors, list):
                parts.append(", ".join(authors[:3]))
        if published := meta.get("published"):
            parts.append(str(published)[:10])
        return " | ".join(parts)
