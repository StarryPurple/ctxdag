#!/usr/bin/env bash
set -euo pipefail

# Start a local OpenAI-compatible server with vLLM for ContextDAG eval.
# Usage: scripts/start_vllm.sh [--model PATH] [--port PORT] [--max-model-len N]
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}"
MODEL="${MODEL:-data/models/qwen3-4b-awq}"
PORT="${PORT:-30000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vllm not found at $VLLM_BIN; run 'uv sync --extra engine' first." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${EXTRA_ARGS[@]}"
