import json
import logging
import threading
from typing import Any, Dict, Optional, Tuple
from DeAction.utils import create_client, encode_image, parse_model_spec


NARRATIVE_SUMMARY_SYSTEM_PROMPT = """
You are an expert in computer usage responsible for describing a `pyaotugui` computer action and what happened after it is taken. You will analyze the before and after screenshots given an action and provide a concise narrative that explains the action and the meaningful changes observed."""


class NarrativeSummaryGenerator:
    """Generate per-action narratives to summarize trajectory steps."""

    def __init__(
        self,
        model: str,
        logger: Optional[logging.Logger] = None,
        max_retries: int = 2,
    ) -> None:
        self.logger = (logger or logging.getLogger(__name__)).getChild("NarrativeSummary")
        self.provider, self.model = parse_model_spec(model)
        self.client = create_client(self.provider, self.model)
        self.max_retries = max(1, max_retries)
        self._client_lock = threading.Lock()

    def _build_messages(
        self,
        *,
        action: str,
        pre_image_b64: str,
        post_image_b64: str,
    ) -> Dict[str, Any]:
        """Create model input messages with before/after screenshots."""
        user_text = (
            f"Executed Action:\n{action}\n\n"
            "You are provided with two screenshots:\n"
            "- The first image shows the state immediately before the action.\n"
            "- The second image shows the state immediately after the action.\n\n"
            "Describe the observed transition. Respond with a JSON object containing:\n"
            '- "narrative": a concise description (1-2 sentences) explaning the action and what changed.'
        )

        message_content = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{pre_image_b64}",
                    "detail": "high",
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{post_image_b64}",
                    "detail": "high",
                },
            },
        ]

        return {
            "system": {"role": "system", "content": NARRATIVE_SUMMARY_SYSTEM_PROMPT},
            "user": {"role": "user", "content": message_content},
        }

    def _request_narrative(self, messages: Dict[str, Any]) -> Dict[str, Any]:
        """Call the configured model provider and parse the JSON response."""
        system_message = messages["system"]
        user_message = messages["user"]

        with self._client_lock:
            if self.provider == "azure":
                response_obj = self.client.chat.completions.create(
                    model=self.model,
                    messages=[system_message, user_message],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "narrative_summary",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "narrative": {"type": "string"},
                                },
                                "required": ["narrative"],
                            }
                        },
                    },
                )
                response_text = response_obj.choices[0].message.content
                return json.loads(response_text)

            if self.provider == "together":
                system_message = dict(system_message)
                system_message["content"] += (
                    "\n\nIMPORTANT: Respond with a valid JSON object containing "
                    '"narrative" (string).'
                )
                response_obj = self.client.chat.completions.create(
                    model=self.model,
                    messages=[system_message, user_message],
                    temperature=0.1,
                    max_tokens=256
                )
                response_text = response_obj.choices[0].message.content
                return self._extract_json(response_text)

            if self.provider == "aws":
                claude_messages = []
                for msg in [user_message]:
                    claude_msg = {"role": msg["role"], "content": []}
                    for chunk in msg["content"]:
                        if chunk["type"] == "text":
                            claude_msg["content"].append(
                                {"type": "text", "text": chunk["text"]}
                            )
                        elif chunk["type"] == "image_url":
                            image_data = chunk["image_url"]["url"].replace(
                                "data:image/png;base64,", ""
                            )
                            claude_msg["content"].append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_data,
                                    },
                                }
                            )
                    claude_messages.append(claude_msg)

                system_text = (
                    NARRATIVE_SUMMARY_SYSTEM_PROMPT
                    + "\n\nPlease respond with a JSON object containing "
                    '"narrative" (string).'
                )
                response_obj = self.client.messages.create(
                    model=self.model,
                    system=system_text,
                    max_tokens=256,
                    messages=claude_messages,
                )
                response_text = response_obj.content[0].text
                return self._extract_json(response_text)

            if self.provider in ("openai", "local"):
                response_obj = self.client.chat.completions.create(
                    model=self.model,
                    messages=[system_message, user_message],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "narrative_summary",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "narrative": {"type": "string"},
                                },
                                "required": ["narrative"],
                            }
                        },
                    },
                )
                response_text = response_obj.choices[0].message.content
                return self._extract_json(response_text)

            raise ValueError(f"Unsupported provider for narrative summary: {self.provider}")

    def generate_narrative(
        self,
        *,
        action: str,
        step_idx: int,
        pre_screenshot: bytes,
        post_screenshot: bytes,
    ) -> Dict[str, Any]:
        """Generate structured narrative summary for a single action."""
        if pre_screenshot is None or post_screenshot is None:
            raise ValueError("Both pre and post screenshots are required for narrative summary")

        pre_b64 = encode_image(pre_screenshot)
        post_b64 = encode_image(post_screenshot)
        messages = self._build_messages(
            action=action,
            pre_image_b64=pre_b64,
            post_image_b64=post_b64,
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    "Generating narrative for step %s with model %s|%s (attempt %s)",
                    step_idx,
                    self.provider,
                    self.model,
                    attempt + 1,
                )
                payload = self._request_narrative(messages)
                narrative_text = payload.get("narrative")
                if not isinstance(narrative_text, str) or not narrative_text.strip():
                    raise ValueError("Narrative response missing 'narrative' field")

                return {
                    "step_idx": step_idx,
                    "action": action,
                    "narrative": narrative_text.strip(),
                    "model": f"{self.provider}|{self.model}",
                }
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Narrative generation attempt %s for step %s failed: %s",
                    attempt + 1,
                    step_idx,
                    exc,
                )
        raise RuntimeError(
            f"Narrative generation failed for step {step_idx} after {self.max_retries} attempts"
        ) from last_error

    @staticmethod
    def _extract_json(raw_text: str) -> Dict[str, Any]:
        """Attempt to parse a JSON object from raw model output."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            import re

            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise
