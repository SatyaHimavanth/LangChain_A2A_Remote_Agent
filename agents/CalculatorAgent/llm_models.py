from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from .logger import get_logger
logger = get_logger(__name__)

load_dotenv()


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _build_model():
    model_source = _clean(os.getenv("MODEL_SOURCE", "azure_openai"))
    model_name = _clean(os.getenv("CHAT_MODEL_NAME"))

    if model_source == "azure_openai":
        deployment = _clean(os.getenv("CHAT_DEPLOYMENT_NAME"))

        logger.info("Using Azure OpenAI model=%s deployment=%s", model_name, deployment or model_name)
        return init_chat_model(
            model=model_name,
            model_provider=model_source,
            azure_deployment=deployment,
        )

    logger.info("Using chat model=%s", model_name)
    return init_chat_model(model=model_name, model_provider=model_source)


llm = _build_model()
