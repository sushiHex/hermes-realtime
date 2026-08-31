#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -W)"
TARGET="${1:-$ROOT/.hermes/runtime/kokoro-cuda}"
REQUIREMENTS="$ROOT/requirements/kokoro-cuda-worker-win-py311.txt"
KOKORO_REQUIREMENTS="$ROOT/requirements/kokoro-onnx-package-win-py311.txt"
PYTHON="$TARGET/Scripts/python.exe"

if [[ -e "$TARGET" ]]; then
  printf 'refusing to replace existing CUDA worker environment: %s\n' "$TARGET" >&2
  exit 2
fi

uv venv "$TARGET" --python 3.11
uv pip install --python "$PYTHON" --require-hashes --requirement "$REQUIREMENTS"
uv pip install --python "$PYTHON" --no-deps --require-hashes --requirement "$KOKORO_REQUIREMENTS"

"$PYTHON" -I -c 'import importlib.metadata as m; import kokoro_onnx; import onnxruntime as ort; distributions=m.packages_distributions().get("onnxruntime", []); assert distributions == ["onnxruntime-gpu"], distributions; providers=ort.get_available_providers(); assert "CUDAExecutionProvider" in providers, providers; print("python=" + __import__("sys").executable); print("onnxruntime-gpu=" + m.version("onnxruntime-gpu")); print("kokoro-onnx=" + m.version("kokoro-onnx")); print("providers=" + repr(providers))'
