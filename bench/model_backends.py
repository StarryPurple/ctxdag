"""Shared model backends for correctness evaluation (text and agent)."""

from __future__ import annotations

import json
import os

import requests


def load_tokenize(path: str | None = None):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        path or "data/models/qwen3-4b-awq/tokenizer.json"
    )
    return lambda s: tokenizer.encode(s).ids, tokenizer.decode


def _post_chat(base: str, key: str | None, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = requests.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=300)
    body = resp.json()
    if "error" in body or "detail" in body:
        raise RuntimeError(f"model error: {body.get('error') or body.get('detail')}")
    return body


def make_text_model(
    backend: str = "scripted",
    base: str | None = None,
    max_tokens: int = 512,
):
    """Return ``callable(prompt) -> str``."""
    if backend == "local":
        url = base or os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:30000/v1")
        model = os.environ.get("LOCAL_MODEL_NAME", "data/models/qwen3-4b-awq")

        def text(prompt: str) -> str:
            body = _post_chat(
                url,
                None,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
            )
            return body["choices"][0]["message"]["content"].strip()

        return text
    if backend == "api":
        url = base or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.environ["OPENAI_API_KEY"]
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        def text(prompt: str) -> str:
            body = _post_chat(
                url,
                key,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
            )
            return body["choices"][0]["message"]["content"].strip()

        return text

    def scripted(prompt: str) -> str:
        return prompt[:80]

    return scripted


def make_agent_model(backend: str = "scripted"):
    """Return ``callable(messages, tools) -> message-dict`` (OpenAI style)."""
    if backend == "local":
        url = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:30000/v1")
        model = os.environ.get("LOCAL_MODEL_NAME", "data/models/qwen3-4b-awq")

        def agent(messages: list[dict], tools: list[dict] | None) -> dict:
            body = _post_chat(
                url,
                None,
                {
                    "model": model,
                    "messages": messages,
                    "tools": tools or [],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            )
            return body["choices"][0]["message"].model_dump() if hasattr(
                body["choices"][0]["message"], "model_dump"
            ) else dict(body["choices"][0]["message"])

        return agent
    if backend == "api":
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.environ["OPENAI_API_KEY"]
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        def agent(messages: list[dict], tools: list[dict] | None) -> dict:
            body = _post_chat(
                url,
                key,
                {
                    "model": model,
                    "messages": messages,
                    "tools": tools or [],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            )
            return dict(body["choices"][0]["message"])

        return agent

    def scripted(messages: list[dict], tools: list[dict] | None) -> dict:
        return {"role": "assistant", "content": "回答：已完成。"}

    return scripted


def message_to_action(message: dict):
    """Extract a tau-bench Action from an OpenAI-style assistant message."""
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    calls = message.get("tool_calls") or []
    if calls and calls[0].get("function"):
        fn = calls[0]["function"]
        return Action(name=fn["name"], kwargs=json.loads(fn.get("arguments") or "{}"))
    content = (message.get("content") or "").strip()
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
