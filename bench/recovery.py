"""Model-in-the-loop require recovery used by protocol evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from contextdag import (
    ControlError,
    DependencyError,
    ReturnAnswer,
    Session,
    ToolRequest,
    ToolTransport,
    parse_directives,
    strip_tags,
)


@dataclass(frozen=True)
class RecoveryResult:
    initial_output: str
    final_output: str
    initial_prompt_tokens: int
    recovery_prompt_tokens: int
    total_prompt_tokens: int
    require_rounds: int
    requested_node_ids: tuple[str, ...]
    required_node_ids: tuple[str, ...]
    incremental_node_tokens: int
    calls: tuple[dict, ...]
    failure_type: str | None


def run_recovery(
    session: Session,
    model,
    tokenize,
    prompt_suffix: str,
    max_require_rounds: int,
    allow_recovery: bool = True,
) -> RecoveryResult:
    """Call a model and fulfill authorized require directives for bounded rounds."""

    initial_output = ""
    final_output = ""
    initial_prompt_tokens = 0
    recovery_prompt_tokens = 0
    require_rounds = 0
    requested_node_ids: list[str] = []
    required_node_ids: list[str] = []
    incremental_node_tokens = 0
    calls: list[dict] = []
    seen_requests: set[str] = set()
    failure_type: str | None = None

    while True:
        if session.current_context is None:
            raise RuntimeError("session must be expanded before recovery")
        suffix = prompt_suffix() if callable(prompt_suffix) else prompt_suffix
        prompt = session.current_context.text + suffix
        prompt_tokens = len(tokenize(prompt))
        if require_rounds == 0:
            initial_prompt_tokens = prompt_tokens
        else:
            recovery_prompt_tokens += prompt_tokens
        started = perf_counter()
        output = model(prompt)
        latency_seconds = perf_counter() - started
        usage = getattr(model, "last_usage", {}) or {}
        calls.append(
            {
                "logical_prompt_tokens": prompt_tokens,
                "api_prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_seconds": latency_seconds,
            }
        )
        if require_rounds == 0:
            initial_output = output
        directives = parse_directives(output)
        requested_node_ids.extend(directives.requires)
        if not directives.requires:
            final_output = strip_tags(output)
            break
        if not allow_recovery:
            final_output = strip_tags(output)
            failure_type = "require_not_executed"
            break
        if require_rounds >= max_require_rounds:
            final_output = strip_tags(output)
            failure_type = "max_require_rounds"
            break
        duplicate = next(
            (node_id for node_id in directives.requires if node_id in seen_requests),
            None,
        )
        if duplicate is not None:
            final_output = strip_tags(output)
            failure_type = "repeated_require"
            break
        try:
            for node_id in directives.requires:
                before = set(session.current_node_ids)
                session.require(node_id)
                added = set(session.current_node_ids) - before
                incremental_node_tokens += sum(
                    len(tokenize(session.read(added_id))) for added_id in added
                )
                seen_requests.add(node_id)
                if node_id not in before:
                    required_node_ids.append(node_id)
        except DependencyError:
            final_output = strip_tags(output)
            failure_type = "rejected_require"
            break
        require_rounds += 1

    return RecoveryResult(
        initial_output=initial_output,
        final_output=final_output,
        initial_prompt_tokens=initial_prompt_tokens,
        recovery_prompt_tokens=recovery_prompt_tokens,
        total_prompt_tokens=initial_prompt_tokens + recovery_prompt_tokens,
        require_rounds=require_rounds,
        requested_node_ids=tuple(requested_node_ids),
        required_node_ids=tuple(required_node_ids),
        incremental_node_tokens=incremental_node_tokens,
        calls=tuple(calls),
        failure_type=failure_type,
    )



def run_message_recovery(
    session: Session,
    model,
    tokenize,
    initial_suffix,
    followup_suffix,
    max_require_rounds: int,
    allow_recovery: bool = True,
) -> RecoveryResult:
    """Recover by appending messages, leaving every earlier byte unchanged."""
    if session.current_context is None:
        raise RuntimeError("session must be expanded before recovery")

    initial = initial_suffix() if callable(initial_suffix) else initial_suffix
    messages = [{"role": "user", "content": session.current_context.text + initial}]
    initial_output = ""
    final_output = ""
    initial_prompt_tokens = 0
    recovery_prompt_tokens = 0
    require_rounds = 0
    requested_node_ids: list[str] = []
    required_node_ids: list[str] = []
    incremental_node_tokens = 0
    calls: list[dict] = []
    seen_requests: set[str] = set()
    failure_type: str | None = None

    while True:
        prompt_tokens = sum(
            len(tokenize(message.get("content", ""))) for message in messages
        )
        if require_rounds == 0:
            initial_prompt_tokens = prompt_tokens
        else:
            recovery_prompt_tokens += prompt_tokens
        started = perf_counter()
        output = model(messages)
        latency_seconds = perf_counter() - started
        usage = getattr(model, "last_usage", {}) or {}
        calls.append(
            {
                "logical_prompt_tokens": prompt_tokens,
                "api_prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_seconds": latency_seconds,
                "message_count": len(messages),
            }
        )
        if require_rounds == 0:
            initial_output = output
        directives = parse_directives(output)
        requested_node_ids.extend(directives.requires)
        if not directives.requires:
            final_output = strip_tags(output)
            break
        if not allow_recovery:
            final_output = strip_tags(output)
            failure_type = "require_not_executed"
            break
        if require_rounds >= max_require_rounds:
            final_output = strip_tags(output)
            failure_type = "max_require_rounds"
            break
        duplicate = next(
            (node_id for node_id in directives.requires if node_id in seen_requests),
            None,
        )
        if duplicate is not None:
            final_output = strip_tags(output)
            failure_type = "repeated_require"
            break

        before = set(session.current_node_ids)
        try:
            for node_id in directives.requires:
                session.require(node_id)
                seen_requests.add(node_id)
                if node_id not in before:
                    required_node_ids.append(node_id)
        except DependencyError:
            final_output = strip_tags(output)
            failure_type = "rejected_require"
            break

        added = set(session.current_node_ids) - before
        ordered_added = [
            node_id for node_id in session.current_context.order if node_id in added
        ]
        incremental_node_tokens += sum(
            len(tokenize(session.read(node_id))) for node_id in ordered_added
        )
        delta = "\n\n".join(session.read(node_id) for node_id in ordered_added)
        followup = followup_suffix() if callable(followup_suffix) else followup_suffix
        messages.append({"role": "assistant", "content": output})
        messages.append(
            {"role": "user", "content": "Requested context:\n\n" + delta + followup}
        )
        require_rounds += 1

    return RecoveryResult(
        initial_output=initial_output,
        final_output=final_output,
        initial_prompt_tokens=initial_prompt_tokens,
        recovery_prompt_tokens=recovery_prompt_tokens,
        total_prompt_tokens=initial_prompt_tokens + recovery_prompt_tokens,
        require_rounds=require_rounds,
        requested_node_ids=tuple(requested_node_ids),
        required_node_ids=tuple(required_node_ids),
        incremental_node_tokens=incremental_node_tokens,
        calls=tuple(calls),
        failure_type=failure_type,
    )



def run_tool_recovery(
    session: Session,
    model,
    tokenize,
    prompt_suffix,
    max_require_rounds: int,
    allow_recovery: bool = True,
) -> RecoveryResult:
    """Recover through native tool calls while preserving message prefixes."""
    if session.current_context is None:
        raise RuntimeError("session must be expanded before recovery")
    suffix = prompt_suffix() if callable(prompt_suffix) else prompt_suffix
    messages = [{"role": "user", "content": session.current_context.text + suffix}]
    authorized_ids = session.current_context.catalog
    initial_output = ""
    final_output = ""
    initial_prompt_tokens = 0
    recovery_prompt_tokens = 0
    require_rounds = 0
    requested_node_ids: list[str] = []
    required_node_ids: list[str] = []
    incremental_node_tokens = 0
    calls: list[dict] = []
    seen_requests: set[str] = set()
    failure_type: str | None = None

    while True:
        prompt_tokens = sum(
            len(tokenize(str(message.get("content") or "")))
            for message in messages
        )
        if require_rounds == 0:
            initial_prompt_tokens = prompt_tokens
        else:
            recovery_prompt_tokens += prompt_tokens
        started = perf_counter()
        tool_specs = ToolTransport.tools(authorized_ids)
        message = model(messages, tool_specs, "required")
        latency_seconds = perf_counter() - started
        usage = getattr(model, "last_usage", {}) or {}
        calls.append(
            {
                "logical_prompt_tokens": prompt_tokens,
                "api_prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_seconds": latency_seconds,
                "message_count": len(messages),
            }
        )
        content = str(message.get("content") or "")
        try:
            action = ToolTransport.parse(message)
        except ControlError as exc:
            calls[-1]["control_error"] = str(exc)
            final_output = content
            failure_type = "invalid_tool_call"
            break
        if isinstance(action, ReturnAnswer):
            if require_rounds == 0:
                initial_output = action.text
            final_output = action.text
            break
        if not isinstance(action, ToolRequest):
            raise RuntimeError("unsupported transport action")
        requested_node_ids.extend(action.action.node_ids)
        if not allow_recovery:
            failure_type = "require_not_executed"
            break
        if require_rounds >= max_require_rounds:
            failure_type = "max_require_rounds"
            break
        if any(node_id in seen_requests for node_id in action.action.node_ids):
            failure_type = "repeated_require"
            break

        before = set(session.current_node_ids)
        try:
            session.require_many(action.action.node_ids)
        except DependencyError:
            failure_type = "rejected_require"
            break
        seen_requests.update(action.action.node_ids)
        required_node_ids.extend(
            node_id for node_id in action.action.node_ids if node_id not in before
        )
        added = set(session.current_node_ids) - before
        ordered_added = [
            node_id for node_id in session.current_context.order if node_id in added
        ]
        rendered = "\n\n".join(session.read(node_id) for node_id in ordered_added)
        incremental_node_tokens += sum(
            len(tokenize(session.read(node_id))) for node_id in ordered_added
        )
        if require_rounds + 1 >= max_require_rounds:
            state_note = (
                "Runtime note: requested context is now available. This was "
                "the final retrieval round; call return_answer now."
            )
        else:
            state_note = (
                "Runtime note: requested context is now available. Do not "
                "request the same node ids again."
            )
        messages.append(dict(message))
        messages.append(
            ToolTransport.result(action.call_id, rendered + "\n\n" + state_note)
        )
        require_rounds += 1

    return RecoveryResult(
        initial_output=initial_output,
        final_output=final_output,
        initial_prompt_tokens=initial_prompt_tokens,
        recovery_prompt_tokens=recovery_prompt_tokens,
        total_prompt_tokens=initial_prompt_tokens + recovery_prompt_tokens,
        require_rounds=require_rounds,
        requested_node_ids=tuple(requested_node_ids),
        required_node_ids=tuple(required_node_ids),
        incremental_node_tokens=incremental_node_tokens,
        calls=tuple(calls),
        failure_type=failure_type,
    )
