"""
LLM client supporting OpenAI and Anthropic APIs.
Uses tenacity for retry logic on transient errors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import get_settings
from backend.core.exceptions import GenerationError

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class LLMResponse:
    """Response from the LLM."""

    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """
    Unified LLM client for OpenAI and Anthropic.
    Automatically selects backend from settings.
    """

    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self._client = self._init_client()

    def _init_client(self):  # noqa: ANN201
        if self.provider == "openai":
            import openai
            return openai.OpenAI(api_key=settings.openai_api_key)
        elif self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            raise GenerationError(f"Unknown LLM provider: {self.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User query + context
            max_tokens: Max response tokens
            temperature: Sampling temperature (low = deterministic)

        Returns:
            LLMResponse with answer + token usage
        """
        try:
            if self.provider == "openai":
                return self._generate_openai(system_prompt, user_prompt, max_tokens, temperature)
            else:
                return self._generate_anthropic(system_prompt, user_prompt, max_tokens, temperature)
        except Exception as exc:
            raise GenerationError(f"LLM generation failed: {exc}") from exc

    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            answer=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return LLMResponse(
            answer=answer,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
