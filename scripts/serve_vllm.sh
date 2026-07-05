#!/usr/bin/env bash
# Serve a model behind an OpenAI-compatible endpoint with vLLM.
#
# vLLM lives in its own venv so its torch pin never touches the app venv
# (this is what makes the migration work on the cu130/torch-2.12 box — see
# docs/vllm_scoping.md). The same script works on the local 3090 or any CUDA
# VM; point the batch config's client.base_url at wherever this runs.
#
# Usage:
#   scripts/serve_vllm.sh [MODEL] [extra vllm serve args...]
#
# Examples:
#   scripts/serve_vllm.sh                          # Qwen3.5-9B thinking, NVFP4 4-bit
#   scripts/serve_vllm.sh Qwen/Qwen2.5-7B-Instruct # something else
set -euo pipefail

MODEL="${1:-AxionML/Qwen3.5-9B-NVFP4}"
shift || true

VENV="${VLLM_VENV:-$HOME/.venvs/vllm}"

if [ ! -x "$VENV/bin/vllm" ]; then
    echo "Installing vLLM into $VENV (one-time) ..."
    uv venv "$VENV"
    uv pip install -p "$VENV" vllm
fi

# vLLM JIT-compiles kernels at startup and needs nvcc. The box has no system
# CUDA toolkit; torch's cu13 wheels ship a full one inside the venv, so point
# CUDA_HOME there (fixes "Could not find nvcc and default cuda_home=
# '/usr/local/cuda' doesn't exist").
CUDA13="$("$VENV/bin/python" -c 'import nvidia, os; print(os.path.join(nvidia.__path__[0], "cu13"))')"
if [ -x "$CUDA13/bin/nvcc" ]; then
    export CUDA_HOME="$CUDA13"
    export PATH="$CUDA13/bin:$PATH"
fi

# Skip FlashInfer's JIT-compiled sampling kernels: compiling them needs the
# pip CUDA wheels to agree on a minor version (nvcc 13.2 vs runtime-headers
# 13.0 drift breaks CCCL's compat check), and on sm86 attention uses prebuilt
# FlashAttention anyway. vLLM's native sampler is equivalent for this workload.
export VLLM_USE_FLASHINFER_SAMPLER=0

# 3090 (24 GB): the default model is a pre-quantized NVFP4 checkpoint — vLLM
# auto-detects the quant method from its config (runs as weight-only W4A16
# Marlin on Ampere; native FP4 compute needs Blackwell). ~5.5 GB weights
# leaves KV room for 80k context and thinking traces. The reasoning parser
# strips <think> into the API's `reasoning` field; harmless for non-thinking
# models.
exec "$VENV/bin/vllm" serve "$MODEL" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 81920 \
    --reasoning-parser qwen3 \
    "$@"
