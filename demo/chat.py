"""Interactive chat over the protocol, with tagged internal views.

Usage:
  python demo/chat.py --backend local --verbose
  python demo/chat.py --backend api --verbose
  python demo/chat.py --backend scripted --once "你好"
"""

from __future__ import annotations

import argparse
import re
import sys

import requests

sys.path.insert(0, "src")
from contextdag import ContextAgent, Session, parse_directives
from run_demo import api_generate, local_generate, server_alive, start_server, wait_until_ready


def scripted_chat(prompt: str) -> str:
    """Deterministic chat reply: references the last visible node."""
    ids = re.findall(r"<node=([0-9a-f]{16})>", prompt)
    ref = f"<ref={ids[-1]}>" if ids else ""
    return (
        f"{ref} 收到，已结合上下文处理。"
        "（脚本化模型：用于无 GPU 环境验证协议链路）"
    )


class ScriptedSummaryService:
    """Deterministic stand-in for the lazy summary service."""

    def summarize(self, content: str, node_id: str) -> str:
        head = " ".join(content.split())[:36]
        return f"{head}…（脚本摘要：需要相关细节时使用）"


def debug_view(agent: ContextAgent, raw: str) -> None:
    directives = parse_directives(raw)
    print("\n--- 模型看到的上下文（展开） ---")
    print(agent.last_context.text)
    print("\n--- 模型原始输出（含标签） ---")
    print(raw)
    print("\n--- 解析 ---")
    print(f"refs={directives.refs} requires={directives.requires}")
    print(
        f"agent_node={agent.last_agent_node.id if agent.last_agent_node else None} "
        f"refs={list(agent.last_agent_node.refs) if agent.last_agent_node else None} "
        f"page_faults={agent.session.page_faults}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "api", "scripted"], default="scripted")
    parser.add_argument("--verbose", action="store_true", help="show tagged internals")
    parser.add_argument("--once", help="single non-interactive turn")
    parser.add_argument("--instruction", default=None)
    parser.add_argument(
        "--catalog-size",
        type=int,
        default=1024,
        help="默认候选集 = 最近注册的 N 个节点",
    )
    parser.add_argument(
        "--summary-service",
        action="store_true",
        help="启用脚本化摘要服务（演示懒加载摘要生成）",
    )
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.backend == "local":
        if args.start_server and not server_alive():
            try:
                start_server()
            except FileNotFoundError as exc:
                print(f"[chat] {exc}")
                sys.exit(1)
            if not wait_until_ready():
                print("[chat] server failed to start; see tmp/server.log")
                sys.exit(1)
            print("[chat] server ready.")
        model = local_generate
    elif args.backend == "api":
        model = api_generate
    else:
        model = scripted_chat

    agent = ContextAgent(
        model=model,
        session=Session(
            catalog_size=args.catalog_size,
            summary_service=ScriptedSummaryService() if args.summary_service else None,
        ),
        instruction=args.instruction
        or "请先阅读上下文，再回答用户问题。需要引用上下文中的内容时，"
        "用 <ref=节点id> 标注；保持简短。",
    )

    def turn(message: str) -> None:
        try:
            raw = agent.say(message)
            if args.verbose:
                debug_view(agent, raw)
            else:
                print(f"\n{agent.name}: {raw}")
        except RuntimeError as exc:
            print(f"\n[chat] 请求失败: {exc}")
            if args.backend == "local" and not server_alive():
                print("[chat] 服务器已掉线，尝试重启 ...")
                try:
                    start_server()
                    if wait_until_ready():
                        print("[chat] 重启成功，请重发这条消息。")
                    else:
                        print("[chat] 重启失败；查看 tmp/server.log")
                except FileNotFoundError as exc2:
                    print(f"[chat] {exc2}")

    if args.once:
        turn(args.once)
        return

    print("ContextDAG chat（输入 /exit 退出）")
    while True:
        try:
            line = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("/exit", "/quit"):
            break
        turn(line)


if __name__ == "__main__":
    main()
