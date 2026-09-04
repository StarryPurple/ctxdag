"""Shared model backends for correctness evaluation (text and agent)."""

from __future__ import annotations

import json
import os

import requests


def load_tokenize(path: str | None = None):
    if path == "char":
        return lambda s: list(s), lambda ids: "".join(ids)
    from tokenizers import Tokenizer

    if not path:
        path = os.environ.get("LOCAL_TOKENIZER_PATH")
    if not path or not os.path.exists(path):
        model_name = os.environ.get("LOCAL_MODEL_NAME")
        if not model_name:
            base = os.environ.get(
                "LOCAL_BASE_URL", "http://127.0.0.1:30000/v1"
            )
            model_name = _discover_local_model(base)
        candidate = os.path.join(model_name, "tokenizer.json")
        if os.path.exists(candidate):
            path = candidate
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"tokenizer not found: {path!r}; "
            "set LOCAL_TOKENIZER_PATH or LOCAL_MODEL_NAME"
        )
    tokenizer = Tokenizer.from_file(path)
    return lambda s: tokenizer.encode(s).ids, tokenizer.decode


def _discover_local_model(base: str) -> str:
    """Return the first model id served by an OpenAI-compatible endpoint."""
    try:
        resp = requests.get(f"{base}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") or []
        if models and models[0].get("id"):
            return models[0]["id"]
    except Exception:
        pass
    return "data/models/qwen3-4b-awq"


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
    seed: int = 0,
):
    """Return ``callable(prompt) -> str``."""
    if backend == "local":
        url = base or os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:30000/v1")
        model = os.environ.get("LOCAL_MODEL_NAME") or _discover_local_model(url)

        def text(prompt: str) -> str:
            body = _post_chat(
                url,
                None,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "seed": seed,
                },
            )
            usage = body.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            text.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
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
                    "seed": seed,
                },
            )
            usage = body.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            text.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            return body["choices"][0]["message"]["content"].strip()

        return text

    def scripted(prompt: str) -> str:
        return prompt[:80]

    return scripted


def make_chat_model(
    backend: str = "scripted",
    base: str | None = None,
    max_tokens: int = 512,
    seed: int = 0,
):
    """Return ``callable(messages) -> str`` while preserving message boundaries."""
    if backend in {"local", "api"}:
        if backend == "local":
            url = base or os.environ.get(
                "LOCAL_BASE_URL", "http://127.0.0.1:30000/v1"
            )
            key = None
            model = (
                os.environ.get("LOCAL_MODEL_NAME") or _discover_local_model(url)
            )
        else:
            url = base or os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            )
            key = os.environ["OPENAI_API_KEY"]
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        def chat(messages: list[dict]) -> str:
            body = _post_chat(
                url,
                key,
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "seed": seed,
                },
            )
            usage = body.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            chat.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            return body["choices"][0]["message"]["content"].strip()

        return chat

    def scripted(messages: list[dict]) -> str:
        return messages[-1].get("content", "")[:80]

    return scripted


def make_agent_model(
    backend: str = "scripted",
    base: str | None = None,
    max_tokens: int = 512,
    seed: int = 0,
):
    """Return ``callable(messages, tools) -> message-dict`` (OpenAI style)."""
    if backend == "local":
        url = base or os.environ.get(
            "LOCAL_BASE_URL", "http://127.0.0.1:30000/v1"
        )
        model = os.environ.get("LOCAL_MODEL_NAME") or _discover_local_model(url)

        def agent(
            messages: list[dict],
            tools: list[dict] | None,
            tool_choice: str | None = None,
        ) -> dict:
            body = _post_chat(
                url,
                None,
                {
                    "model": model,
                    "messages": messages,
                    "tools": tools or [],
                    "tool_choice": tool_choice or ("auto" if tools else "none"),
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "seed": seed,
                },
            )
            usage = body.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            agent.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            return dict(body["choices"][0]["message"])

        return agent
    if backend == "api":
        url = base or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        key = os.environ["OPENAI_API_KEY"]
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        def agent(
            messages: list[dict],
            tools: list[dict] | None,
            tool_choice: str | None = None,
        ) -> dict:
            body = _post_chat(
                url,
                key,
                {
                    "model": model,
                    "messages": messages,
                    "tools": tools or [],
                    "tool_choice": tool_choice or ("auto" if tools else "none"),
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "seed": seed,
                },
            )
            usage = body.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            agent.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            return dict(body["choices"][0]["message"])

        return agent

    def scripted(
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | None = None,
    ) -> dict:
        del tool_choice
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
