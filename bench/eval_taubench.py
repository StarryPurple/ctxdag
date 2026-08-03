"""Level-2 correctness: task success on tau-bench under the protocol.

Three agents solve the same retail tasks:
  oracle   - replays the ground-truth action sequence (validates the harness,
             must reach reward 1.0)
  baseline - vanilla tau-bench loop with the full message history
  protocol - the same loop, but history is stored as nodes and rendered as
             dependency sets (ContextDAG Session)

The litellm user simulator is replaced by a scripted user, so no external
model/network is needed for the environment; the agent model comes from
``--backend`` (scripted for pipeline checks, local/api for real runs).

Usage:
  PYTHONPATH=src .venv/bin/python bench/eval_taubench.py --backend scripted --limit 2
  PYTHONPATH=src .venv/bin/python bench/eval_taubench.py --backend local --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

# litellm stub: the tau-bench user simulator imports it, but we replace the
# user with a scripted one, so the stub only needs to exist.
_litellm = types.ModuleType("litellm")
_litellm.completion = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("litellm stub: LLM user simulator unavailable")
)
sys.modules.setdefault("litellm", _litellm)

sys.path.insert(0, "data/tau-bench")
sys.path.insert(0, "src")
sys.path.insert(0, "bench")

from contextdag import Session  # noqa: E402
from tau_bench.envs.retail.env import MockRetailDomainEnv  # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402

from model_backends import (  # noqa: E402
    load_tokenize,
    make_agent_model,
    message_to_action,
)


class ScriptedUser:
    """Deterministic user stand-in; terminates on the first respond."""

    def reset(self, instruction: str | None = None) -> str:
        return f"开始任务。"

    def step(self, content: str) -> str:
        return "收到。任务完成，再见。###STOP###"

    def get_total_cost(self) -> float:
        return 0.0


def new_env() -> MockRetailDomainEnv:
    env = MockRetailDomainEnv(user_strategy="human")
    env.user = ScriptedUser()
    return env


def oracle_solve(env, task_index: int, tokenize, max_steps: int = 30):
    env.reset(task_index=task_index)
    prompt_tokens = 0
    steps = 0
    done = False
    reward = 0.0
    for action in list(env.task.actions):
        if steps >= max_steps:
            break
        prompt_tokens += len(tokenize(action.name))
        resp = env.step(action)
        reward = resp.reward
        done = resp.done
        steps += 1
        if done:
            break
    if not done:
        outputs = getattr(env.task, "outputs", []) or []
        content = "完成。" + (" " + " ".join(outputs) if outputs else "")
        resp = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": content}))
        reward = resp.reward
    return {"reward": reward, "steps": steps, "prompt_tokens": prompt_tokens}


def baseline_solve(
    env, task_index: int, model, tokenize, max_steps: int = 30, verbose: bool = False
):
    env.reset(task_index=task_index)
    messages: list[dict] = [
        {"role": "system", "content": env.wiki},
        {"role": "user", "content": env.task.instruction},
    ]
    prompt_tokens = 0
    steps = 0
    reward = 0.0
    done = False
    for _ in range(max_steps):
        prompt_tokens += len(tokenize("\n".join(m["content"] for m in messages)))
        message = model(messages, env.tools_info)
        action = message_to_action(message)
        if verbose:
            print(f"    baseline step {steps + 1}: {action.name} {str(action.kwargs)[:80]}")
        messages.append(message)
        resp = env.step(action)
        if verbose:
            print(f"      -> {resp.observation[:100]}")
        reward = resp.reward
        done = resp.done
        steps += 1
        if action.name != RESPOND_ACTION_NAME:
            calls = message.get("tool_calls") or []
            messages.append(
                {
                    "role": "tool",
                    "name": action.name,
                    "content": resp.observation,
                }
            )
        else:
            messages.append({"role": "user", "content": resp.observation})
        if done:
            break
    return {"reward": reward, "steps": steps, "prompt_tokens": prompt_tokens}


def protocol_solve(
    env, task_index: int, model, tokenize, max_steps: int = 30, verbose: bool = False
):
    env.reset(task_index=task_index)
    session = Session()
    first = session.register(content=env.task.instruction, refs=())
    prev = first.id
    prompt_tokens = 0
    steps = 0
    reward = 0.0
    done = False
    for _ in range(max_steps):
        ctx = session.expand(refs=[prev])
        prompt_tokens += len(tokenize(ctx.text))
        messages = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": ctx.text},
        ]
        message = model(messages, env.tools_info)
        action = message_to_action(message)
        if verbose:
            print(f"    protocol step {steps + 1}: {action.name} {str(action.kwargs)[:80]}")
        action_node = session.register(
            content=json.dumps(message, ensure_ascii=False),
            refs=(prev,),
        )
        resp = env.step(action)
        if verbose:
            print(f"      -> {resp.observation[:100]}")
        reward = resp.reward
        done = resp.done
        steps += 1
        obs_node = session.register(
            content=f"[{action.name}] {resp.observation}", refs=(action_node.id,)
        )
        prev = obs_node.id
        if done:
            break
    return {"reward": reward, "steps": steps, "prompt_tokens": prompt_tokens}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "api", "scripted"], default="scripted")
    parser.add_argument("--env", choices=["retail", "airline"], default="retail")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out", default="tmp/eval_taubench.json")
    parser.add_argument("--verbose", action="store_true", help="打印每步动作")
    parser.add_argument("--tokenizer", default="data/models/qwen3-4b-awq/tokenizer.json")
    args = parser.parse_args()

    tokenize, _ = load_tokenize(args.tokenizer)
    model = make_agent_model(args.backend)
    if args.env == "airline":
        from tau_bench.envs.airline.env import MockAirlineDomainEnv

        env_type = MockAirlineDomainEnv
    else:
        env_type = MockRetailDomainEnv

    env = env_type(user_strategy="human")
    env.user = ScriptedUser()
    tasks = env.tasks[: args.limit]

    Path("tmp").mkdir(exist_ok=True)
    rows = []
    n_tasks = len(tasks)
    t0 = time.time()
    for idx in range(n_tasks):
        print(
            f"[{time.strftime('%H:%M:%S')}] task {idx + 1}/{n_tasks} 开始",
            flush=True,
        )
        row = {"task_index": idx}
        row["oracle"] = oracle_solve(env, idx, tokenize, args.max_steps)
        row["baseline"] = baseline_solve(
            env, idx, model, tokenize, args.max_steps, args.verbose
        )
        row["protocol"] = protocol_solve(
            env, idx, model, tokenize, args.max_steps, args.verbose
        )
        rows.append(row)
        with open(args.out, "w") as f:
            json.dump(
                {"rows": rows, "n_done": len(rows), "n_total": n_tasks},
                f,
                ensure_ascii=False,
                indent=2,
            )
        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (n_tasks - idx - 1)
        print(
            f"[{time.strftime('%H:%M:%S')}] task {idx + 1}/{n_tasks} 完成: "
            f"oracle={row['oracle']['reward']} baseline={row['baseline']['reward']} "
            f"protocol={row['protocol']['reward']} "
            f"elapsed={elapsed:.0f}s eta~{eta:.0f}s",
            flush=True,
        )

    def aggregate(key):
        rewards = [r[key]["reward"] for r in rows]
        steps = [r[key]["steps"] for r in rows]
        tokens = [r[key]["prompt_tokens"] for r in rows]
        return {
            "success": sum(1 for v in rewards if v == 1.0) / len(rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "mean_steps": sum(steps) / len(steps),
            "mean_tokens": sum(tokens) / len(tokens),
        }

    report = {
        "backend": args.backend,
        "n": len(rows),
        "aggregate": {k: aggregate(k) for k in ("oracle", "baseline", "protocol")},
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\naggregate:")
    for k, v in report["aggregate"].items():
        print(f"  {k}: {v}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
