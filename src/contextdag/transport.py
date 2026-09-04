"""Serialization adapters for runtime control actions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .control import ControlError, RequireContext, ReturnAnswer


@dataclass(frozen=True)
class ToolRequest:
    call_id: str
    action: RequireContext


class ToolTransport:
    """Map ContextDAG control actions to OpenAI-compatible tool messages."""

    @staticmethod
    def tools(authorized_ids: tuple[str, ...] | None = None) -> list[dict]:
        answer_tool = {
            "type": "function",
            "function": {
                "name": "return_answer",
                "description": (
                    "Return the final answer when the full visible nodes contain "
                    "sufficient evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
        if authorized_ids == ():
            return [answer_tool]
        node_id_schema = {"type": "string"}
        if authorized_ids is not None:
            node_id_schema["enum"] = list(authorized_ids)
        return [
            {
                "type": "function",
                "function": {
                    "name": "require_context",
                    "description": (
                        "Load context nodes needed to answer. Use only node_ids "
                        "allowed by the parameter schema."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_ids": {
                                "type": "array",
                                "items": node_id_schema,
                                "minItems": 1,
                            }
                        },
                        "required": ["node_ids"],
                        "additionalProperties": False,
                    },
                },
            },
            answer_tool,
        ]

    @staticmethod
    def parse(message: dict) -> ToolRequest | ReturnAnswer:
        calls = message.get("tool_calls") or []
        if not calls:
            return ReturnAnswer(str(message.get("content") or ""))
        if len(calls) != 1:
            raise ControlError("exactly one control tool call is allowed")
        call = calls[0]
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raise ControlError("invalid control tool arguments") from exc
        if function.get("name") == "return_answer":
            if set(arguments) != {"text"} or not isinstance(
                arguments["text"], str
            ):
                raise ControlError("return_answer arguments do not match schema")
            return ReturnAnswer(arguments["text"])
        if function.get("name") != "require_context":
            raise ControlError("unsupported control tool")
        if set(arguments) != {"node_ids"} or not isinstance(
            arguments["node_ids"], list
        ):
            raise ControlError("require_context arguments do not match schema")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ControlError("control tool call needs an id")
        return ToolRequest(
            call_id=call_id,
            action=RequireContext(tuple(arguments["node_ids"])),
        )

    @staticmethod
    def result(call_id: str, rendered_nodes: str) -> dict:
        if not call_id:
            raise ControlError("tool result needs a call id")
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "require_context",
            "content": rendered_nodes,
        }
