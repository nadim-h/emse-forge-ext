"""Async client for the model endpoint, and extraction of code and JSON from replies."""
from __future__ import annotations

import re

import json_repair
from openai import AsyncOpenAI

from pipeline import config as C


def client() -> AsyncOpenAI:
    """The vLLM endpoint; the client retries a failed call twice with backoff."""
    return AsyncOpenAI(base_url=C.VLLM_BASE_URL, api_key="EMPTY", timeout=3600.0, max_retries=2)


async def complete(cli: AsyncOpenAI, messages: list[dict]) -> tuple[str, bool]:
    """One chat completion as (text, truncated)."""
    r = await cli.chat.completions.create(
        model=C.PROFILE.model_name, messages=messages, max_tokens=C.MAX_TOKENS,
        extra_body=C.PROFILE.extra_body, **C.PROFILE.sampling)
    ch = r.choices[0]
    return ch.message.content or "", ch.finish_reason == "length"


async def complete_parsed(cli: AsyncOpenAI, messages: list[dict], parse, *,
                          retry_on_parse_miss: bool = False):
    """Resample a truncated reply (and, with `retry_on_parse_miss`, one `parse` rejects)
    until `parse(text)` is truthy, at most `TRUNCATION_RETRIES` times in all;
    returns (parsed or None, attempts used).
    """
    for i in range(1, C.TRUNCATION_RETRIES + 1):
        text, truncated = await complete(cli, messages)
        if parsed := parse(text):
            return parsed, i
        if not (truncated or retry_on_parse_miss):
            break
    return None, i


# --------------------------------------------------------- reply parsing ---
# The closing fence is required, so a reply cut off mid-answer yields no block.
_FENCE = re.compile(
    r"(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n`]*)\r?\n(?P<body>.*?)(?P=fence)",
    re.DOTALL,
)
_LANG_TAGS = {
    "cpp": ("cpp", "c++", "cxx", "cc", "c"),
    "java": ("java",),
}
_JSON_TAGS = ("json", "jsonc", "json5")


def _blocks(text: str) -> list[tuple[str, str]]:
    """Every non-empty fenced block as (first word of its info string, stripped body)."""
    out = []
    for m in _FENCE.finditer(text):
        if body := m.group("body").strip():
            info = m.group("info").lower().split()
            out.append((info[0] if info else "", body))
    return out


def strip_thinking(text: str) -> str:
    """Drop an inline `<think>...</think>` trace (Qwen); vLLM splits off the others'."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def extract_code(text: str, lang: str) -> str | None:
    """The last fenced block tagged `lang`, else the last fenced block of any tag."""
    blocks = _blocks(strip_thinking(text))
    tagged = [b for tag, b in blocks if tag in _LANG_TAGS[lang]]
    found = tagged or [b for _, b in blocks]
    return found[-1] if found else None


def extract_json(text: str):
    """The first JSON array/object found in the ```json blocks, then in any block
    (last block first in both), then in the bare reply."""
    text = strip_thinking(text)
    blocks = _blocks(text)
    tagged = [b for tag, b in blocks if tag in _JSON_TAGS]
    for cand in [*reversed(tagged), *reversed([b for _, b in blocks]), text]:
        for opener, closer in (("[", "]"), ("{", "}")):
            s, e = cand.find(opener), cand.rfind(closer)
            if s != -1 and e > s and (obj := json_repair.loads(cand[s:e + 1])):
                return obj
    return None
