import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from DeAction.utils import create_client, parse_model_spec


class BaseGuardrail(ABC):
    """Common functionality shared by guardrail implementations."""

    SUPPORTED_PROVIDERS = {"azure", "together", "aws", "openai", "local"}

    def __init__(
        self,
        *,
        model: str = "gpt-5",
        model_reasoning_effort: Optional[str] = None,
        max_retry: int = 3,
        logger: Optional[logging.Logger] = None,
        guardrail_name: Optional[str] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.guardrail_name = guardrail_name or self.__class__.__name__
        self.max_retry = max_retry
        self.client = None
        self.model_reasoning_effort = model_reasoning_effort
        self.provider, self.model = self._parse_model_spec(model)

        self.logger.info(
            "[%s] Initializing client with provider %s and model %s",
            self.guardrail_name,
            self.provider,
            self.model,
        )
        self.client = self._initialize_client()

    def _parse_model_spec(self, model_spec: str) -> Tuple[str, str]:
        provider, model_name = parse_model_spec(model_spec)
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        return provider, model_name

    def _create_client(self, provider: str, model: str):
        return create_client(provider, model)

    def _initialize_client(self):
        """Instantiate the underlying LLM client for the configured provider."""
        return self._create_client(self.provider, self.model)

    def _ensure_client(self) -> None:
        """Lazily instantiate the client when needed."""
        if self.client is None:
            self.client = self._initialize_client()

    def get_max_retry(self) -> int:
        return self.max_retry

    def supports_narrative_memory(self) -> bool:
        """Whether this guardrail can optionally consume narrative memory context."""
        return False

    @abstractmethod
    def check_once(
        self,
        instruction: str,
        response: str,
        obs: Dict,
        agent_history: Dict,
        max_retries: int = 1,
    ) -> Tuple[bool, str]:
        """Run a single guardrail evaluation and return (is_misaligned, explanation)."""
        raise NotImplementedError
