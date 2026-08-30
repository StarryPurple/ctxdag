#!/usr/bin/env bash
set -euo pipefail

# AutoDL-specific vLLM launcher for Qwen2.5-14B-Instruct-AWQ.
# Usage: scripts/start_autodl.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-14B-Instruct-AWQ}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

exec "$REPO_ROOT/scripts/start_vllm.sh"
