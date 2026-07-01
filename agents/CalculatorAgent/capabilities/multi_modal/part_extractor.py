"""Convert A2A message parts into LangChain content blocks."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from a2a.utils.errors import ContentTypeNotSupportedError
from a2a.types.a2a_pb2 import Message, Part
from google.protobuf import json_format

logger = logging.getLogger(__name__)

_IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/gif", "image/webp")

MULTIMODAL_INPUT_TYPES: list[str] = [
    "text",
    "text/plain",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/json",
]
TEXT_OUTPUT_TYPES: list[str] = ["text", "text/plain"]


def validate_message_content_types(
    message: Message,
    allowed_types: list[str] | tuple[str, ...],
) -> None:
    """Raise a protocol error when a message part uses an undeclared MIME type."""
    allowed = set(allowed_types)
    for part in message.parts:
        variant = part.WhichOneof("content")
        mime = part.media_type or _default_mime_for_variant(variant)
        if variant == "raw" and _is_image_mime(mime):
            mime = _canonical_image_mime(mime)
        if mime not in allowed:
            raise ContentTypeNotSupportedError(
                f"Input content type {mime!r} is not supported. "
                f"Supported input types: {', '.join(sorted(allowed))}."
            )


def extract_langchain_content(message: Message) -> list[dict[str, Any]]:
    """Convert every A2A part in a user message into LangChain content blocks."""
    if not message or not message.parts:
        return [{"type": "text", "text": ""}]

    blocks: list[dict[str, Any]] = []
    for part in message.parts:
        block = _convert_part(part)
        if block is not None:
            blocks.append(block)

    return blocks or [{"type": "text", "text": ""}]


def _convert_part(part: Part) -> dict[str, Any] | None:
    """Dispatch by the protobuf oneof variant used by the A2A Part."""
    variant = part.WhichOneof("content")
    if variant == "text":
        return {"type": "text", "text": part.text}
    if variant == "raw":
        return _convert_raw(part)
    if variant == "url":
        return _convert_url(part)
    if variant == "data":
        return _convert_data(part)

    logger.warning("Unrecognized A2A Part content variant %r; skipping.", variant)
    return None


def _convert_raw(part: Part) -> dict[str, Any]:
    """Convert image bytes to ``image_url`` and other binaries to text."""
    mime = part.media_type or "application/octet-stream"
    if _is_image_mime(mime):
        encoded = base64.b64encode(part.raw).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        }

    name = part.filename or "unnamed"
    size = len(part.raw)
    return {
        "type": "text",
        "text": f"[Attached file: {name!r} ({mime}, {size} bytes); binary content omitted]",
    }


def _convert_url(part: Part) -> dict[str, Any]:
    """Convert remote image URLs to ``image_url`` and other URLs to text."""
    mime = part.media_type or ""
    if _is_image_mime(mime):
        return {"type": "image_url", "image_url": {"url": part.url}}
    return {"type": "text", "text": f"[Resource URL: {part.url} ({mime or 'unknown type'})]"}


def _convert_data(part: Part) -> dict[str, str]:
    """Serialize structured JSON data into a compact text block for the LLM."""
    try:
        data = json_format.MessageToDict(part.data)
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    except Exception:
        logger.exception("Failed to serialize structured A2A data part.")
        text = "[Structured data could not be serialized]"
    return {"type": "text", "text": text}


def _is_image_mime(mime: str) -> bool:
    return any(mime.startswith(prefix) for prefix in _IMAGE_MIME_PREFIXES)


def _default_mime_for_variant(variant: str | None) -> str:
    if variant == "text":
        return "text/plain"
    if variant == "data":
        return "application/json"
    return "application/octet-stream"


def _canonical_image_mime(mime: str) -> str:
    for prefix in _IMAGE_MIME_PREFIXES:
        if mime.startswith(prefix):
            return prefix
    return mime
