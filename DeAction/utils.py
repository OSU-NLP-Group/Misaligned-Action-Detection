import base64
import os
from typing import Tuple

import httpx
from anthropic import AnthropicBedrock
from openai import AzureOpenAI, OpenAI


def parse_model_spec(model_spec: str) -> Tuple[str, str]:
    """Split provider|model strings into components; default to Azure when unspecified."""
    provider = "azure"
    model_name = model_spec
    if model_spec and "|" in model_spec:
        provider_part, model_part = model_spec.split("|", 1)
        provider = provider_part.strip().lower()
        model_name = model_part.strip()
    else:
        model_name = model_spec.strip()
    return provider, model_name


def create_client(provider: str, model: str):
    """Instantiate an LLM client for the specified provider/model."""
    timeout = httpx.Timeout(60, read=60, write=60, connect=60)

    if provider == "azure":
        return AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            timeout=timeout,
        )

    if provider == "together":
        return OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.ai/v1",
            timeout=timeout,
        )

    if provider == "openai":
        client_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "timeout": timeout,
        }
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        return OpenAI(**client_kwargs)

    if provider == "local":
        base_url = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1")
        return OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=timeout,
        )

    if provider == "aws":
        if "claude" not in model.lower():
            raise ValueError(f"AWS provider only supports Claude models, got: {model}")
        return AnthropicBedrock(
            aws_region=os.getenv("AWS_REGION"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_key=os.getenv("AWS_SECRET_KEY"),
        )

    raise ValueError(f"Unsupported provider: {provider}")


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


COORDINATE_ACTION_LABELS = {
    "click": "click",
    "doubleClick": "double-click",
    "rightClick": "right-click",
    "middleClick": "middle-click",
    "moveTo": "move cursor",
    "moveRel": "move cursor (relative)",
    "dragTo": "drag",
    "dragRel": "drag (relative)",
    "mouseDown": "mouse down",
    "mouseUp": "mouse up",
}

ANNOTATION_COLORS = [
    (239, 71, 111),   # pink/red
    (63, 193, 201),   # teal
    (255, 196, 0),    # amber
    (87, 117, 144),   # slate
    (27, 153, 139),   # green
    (244, 162, 97),   # orange
]
