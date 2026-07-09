"""Unit tests for PromptBuilder."""
from __future__ import annotations

from backend.generation.prompt_builder import PromptBuilder
from backend.retrieval.vector_store import SearchResult


def make_results(n: int = 3) -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id=f"chunk_{i}",
            text=f"This is context text number {i} about deep learning.",
            source=f"paper_{i}.pdf",
            metadata={"title": f"Paper {i}", "authors": [f"Author {i}"]},
            score=0.9 - i * 0.1,
        )
        for i in range(1, n + 1)
    ]


builder = PromptBuilder()


def test_build_returns_tuple():
    system, user = builder.build("What is deep learning?", make_results())
    assert isinstance(system, str)
    assert isinstance(user, str)


def test_system_prompt_has_rules():
    system, _ = builder.build("question", make_results(1))
    assert "cite" in system.lower() or "source" in system.lower()


def test_user_prompt_has_context():
    _, user = builder.build("question?", make_results(2))
    assert "Context Excerpt 1" in user
    assert "Context Excerpt 2" in user


def test_user_prompt_has_question():
    question = "What is BERT?"
    _, user = builder.build(question, make_results(1))
    assert question in user


def test_sources_list_structure():
    results = make_results(3)
    sources = builder.build_sources_list(results)
    assert len(sources) == 3
    for src in sources:
        assert "index" in src
        assert "source" in src
        assert "score" in src
        assert "snippet" in src


def test_no_context():
    system, user = builder.build("question", [])
    assert isinstance(system, str)
    assert isinstance(user, str)
