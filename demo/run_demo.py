"""End-to-end protocol demo with a real or scripted model.

Shows the complete flow with debug information: the expanded context the
model sees (dependency set + catalog), the model's raw tagged output,
parsed directives, registered nodes, page faults, and the final DAG.

Backends:
  --backend local      SGLang server at 127.0.0.1:30000 (Qwen3-4B-AWQ)
  --backend api        OpenAI-compatible API (OPENAI_API_KEY, OPENAI_BASE_URL)
  --backend scripted   deterministic canned replies (no network)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, "src")
from contextdag import Session, parse_directives


HOST = "127.0.0.1"
PORT = 30000
SERVER_LOG = "tmp/server.log"


def server_alive() -> bool:
    for endpoint in ("/health", "/get_model_info"):
        try:
            r = requests.get(f"http://{HOST}:{PORT}{endpoint}", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            continue
    return False


def start_server() -> subprocess.Popen:
    probe = subprocess.run(
        [sys.executable, "-c", "import sglang"],
        capture_output=True,
    )
    if probe.returncode != 0:
        raise FileNotFoundError(
            "sglang is not installed in this environment; run:\n"
            "  uv sync --extra engine\n"
            "or use --backend api / --backend scripted."
        )
    env = os.environ.copy()
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    cmd = [
        sys.executable,
        "-m", "sglang.launch_server",
        "--model-path", "data/models/qwen3-4b-awq",
        "--port", str(PORT), "--host", HOST,
        "--mem-fraction-static", "0.7", "--trust-remote-code",
    ]
    os.makedirs("tmp", exist_ok=True)
    log = open(SERVER_LOG, "a", encoding="utf-8")
    print(f"[demo] SGLang not running; starting it (log: {SERVER_LOG}) ...")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_until_ready(timeout: int = 480) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if server_alive():
            return True
        time.sleep(2)
    return False


def parse_chat_response(resp: requests.Response, endpoint: str) -> str:
    """Extract chat content from a chat-completions response; raise the
    server's real error instead of a bare KeyError when the shape is wrong."""
    try:
        body = resp.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError(
            f"{endpoint} returned non-JSON (status {resp.status_code}): "
            f"{resp.text[:300]!r}"
        ) from None
    if not isinstance(body, dict):
        raise RuntimeError(
            f"{endpoint} unexpected body type {type(body).__name__} "
            f"(status {resp.status_code}): {str(body)[:300]}"
        )
    error = body.get("error")
    if error is not None:
        raise RuntimeError(f"{endpoint} server error: {error}")
    detail = body.get("detail")
    if detail is not None:
        raise RuntimeError(
            f"{endpoint} error (status {resp.status_code}): {detail}"
        )
    choices = body.get("choices")
    if not choices:
        raise RuntimeError(
            f"{endpoint} response has no 'choices' (status {resp.status_code}); "
            f"body keys: {sorted(body)}"
        )
    content = choices[0].get("message", {}).get("content")
    if content is None:
        raise RuntimeError(
            f"{endpoint} choice has no message.content: {body}"
        )
    return content.strip()


def local_generate(prompt: str) -> str:
    resp = requests.post(
        "http://127.0.0.1:30000/v1/chat/completions",
        json={
            "model": "data/models/qwen3-4b-awq",
            "messages": [
                {"role": "system", "content": "你是上下文感知的助手。"},
                {"role": "user", "content": prompt},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 1024,
            "temperature": 0.3,
        },
        timeout=180,
    )
    return parse_chat_response(resp, "local sglang")


def api_generate(prompt: str) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 220,
            "temperature": 0.3,
        },
        timeout=180,
    )
    return parse_chat_response(resp, "api")


class ScriptedModel:
    """Deterministic replies that exercise ref, require, and no-tag paths."""

    def __init__(self, ids: dict[str, str]) -> None:
        self._ids = ids
        self._stage = 0

    def generate(self, prompt: str) -> str:
        if self._stage == 0:
            self._stage = 1
            return "已核实订单：金额 $120，符合退款条件。"
        if self._stage == 1:
            self._stage = 2
            return (
                f"需要确认换货约束。<require={self._ids['policy']}> "
                "确认后执行换货为型号 A。"
            )
        return (
            f"<ref={self._ids['n1']},{self._ids['n2']}> "
            "综合结论：退款 $120 并换货为型号 A。"
        )


def section(title: str, body: str = "") -> None:
    print(f"\n===== {title} =====")
    if body:
        print(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "api", "scripted"], default="scripted")
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.backend == "local":
        if args.start_server and not server_alive():
            try:
                start_server()
            except FileNotFoundError as exc:
                print(f"[demo] {exc}")
                sys.exit(1)
            if not wait_until_ready():
                try:
                    tail = "".join(open(SERVER_LOG, encoding="utf-8").readlines()[-25:])
                except OSError:
                    tail = ""
                print(f"[demo] server failed to start; last log lines:\n{tail}")
                if "flashinfer" in tail and "No such file" in tail:
                    print(
                        "[demo] hint: stale flashinfer JIT cache; clear it and retry:\n"
                        "  rm -rf ~/.cache/flashinfer"
                    )
                sys.exit(1)
            print("[demo] server ready.")
        model = local_generate
    elif args.backend == "api":
        model = api_generate
    else:
        scripted = ScriptedModel({})
        model = scripted.generate

    session = Session()

    # -- setup: root brief + catalog candidate (policy node) --------------
    brief = session.register(
        content="客户要求：退款订单 #X（$120），并换货为型号 A。",
        refs=[],
    )
    policy = session.register(
        content="换货政策：30 天内可换新款；退款需在确认订单后发起。",
        refs=[],
        meta={"summary": "换货与退款政策"},
    )
    section("准备", f"brief = {brief.id}\npolicy = {policy.id} (meta.summary)")
    if args.backend == "scripted":
        scripted._ids["policy"] = policy.id

    # -- worker 1 ---------------------------------------------------------
    section("Worker-1 展开依赖集（含目录）")
    ctx1 = session.expand(refs=[brief.id], candidates=[policy.id])
    print(ctx1.text)
    reply1 = model(ctx1.text)
    section("Worker-1 模型输出", reply1)
    section("解析", str(parse_directives(reply1)))
    node1 = session.register_declared(reply1, default_refs=[brief.id])
    print(f"-> 节点 {node1.id} refs={node1.refs} page_faults={session.page_faults}")
    if args.backend == "scripted":
        scripted._ids["n1"] = node1.id

    # -- worker 2 (require path) ------------------------------------------
    section("Worker-2 展开 node1 依赖集（引用传递，brief 自动带入）")
    ctx2 = session.expand(refs=[node1.id], candidates=[policy.id])
    print(ctx2.text)
    reply2 = model(ctx2.text)
    section("Worker-2 模型输出（含 require）", reply2)
    section("解析", str(parse_directives(reply2)))
    node2 = session.register_declared(reply2, default_refs=[node1.id])
    print(f"-> 节点 {node2.id} refs={node2.refs} page_faults={session.page_faults}")
    if args.backend == "scripted":
        scripted._ids["n2"] = node2.id

    # -- orchestrator (union + ref declared in text) ----------------------
    section("编排者 展开并集依赖集")
    ctx3 = session.expand(refs=[node1.id, node2.id])
    print(ctx3.text)
    reply3 = model(ctx3.text)
    section("编排者模型输出（含 ref）", reply3)
    section("解析", str(parse_directives(reply3)))
    final = session.register_declared(reply3, default_refs=[node1.id, node2.id])
    print(f"-> 节点 {final.id} refs={final.refs}")

    # -- final DAG --------------------------------------------------------
    section("最终 DAG（id -> refs）")
    for node in session.registry.nodes():
        head = node.content.splitlines()[0][:40]
        print(f"  {node.id} -> {list(node.refs)}  | {head!r}")


if __name__ == "__main__":
    main()
