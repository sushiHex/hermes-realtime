# Optional Windows speech dependencies

The host's CPU environment is resolved in [`uv.lock`](../uv.lock). The opt-in CUDA
worker uses its own Windows CPython 3.11 environment and two hashed manifests:

- [`kokoro-onnx-package-win-py311.txt`](kokoro-onnx-package-win-py311.txt) pins the
  Kokoro package installed with `--no-deps`.
- [`kokoro-cuda-worker-win-py311.txt`](kokoro-cuda-worker-win-py311.txt) pins its
  dependencies and the CUDA runtime.

Kokoro's metadata requires the CPU `onnxruntime` distribution, while the worker
uses `onnxruntime-gpu` as the sole owner of the same import namespace. Upstream
[requires one runtime package per environment](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime).
The [setup script](../scripts/setup-kokoro-cuda-worker.sh) installs the reviewed
closure first, then Kokoro without dependency resolution, and checks namespace
ownership and CUDA provider availability. A generic `pip check` still reports the
intentionally absent CPU distribution; it does not validate this substitution.
The production worker separately requires its pinned model's Conv/Gemm/LSTM
profile on CUDA before accepting synthesis.

## Updating the closure

Update the Kokoro pin in `pyproject.toml` and its package manifest together, then
refresh `uv.lock`. Regenerate the CUDA closure from its declared roots with the
repository's reviewed uv version, from the repository root:

```bash
uv pip compile requirements/kokoro-cuda-worker.in \
  --python-version 3.11 \
  --python-platform x86_64-pc-windows-msvc \
  --only-binary :all: --generate-hashes \
  --no-emit-package kokoro-onnx --no-emit-package onnxruntime \
  --no-header --no-annotate \
  --output-file requirements/kokoro-cuda-worker-win-py311.txt
```

The existing output preserves compatible pins. Use `--upgrade-package NAME==VERSION`
for a deliberate transitive update and review every changed artifact hash. Keep
CPU ONNX Runtime out of the worker manifest. Qualify both installed CPU synthesis
and the separate CUDA worker, including provider selection and owned cleanup, in
addition to the [committed-candidate release gate](../docs/release-gates.md).
Generated PCM checks do not establish physical audio quality or deployment latency.
