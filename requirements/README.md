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

## Installed Kokoro smoke test

On Windows, pre-provision the two pinned assets in one cache directory and create the
separate CUDA worker environment with the setup script above. Then run both installed
backends explicitly:

```powershell
$env:HERMES_REALTIME_INSTALLED_KOKORO = "1"
$env:HERMES_REALTIME_KOKORO_ASSET_CACHE = "C:\path\to\kokoro-cache"
$env:HERMES_REALTIME_KOKORO_CUDA_PYTHON = "C:\path\to\cuda-worker\Scripts\python.exe"
uv run --frozen --group dev --extra local pytest tests/integration/test_installed_kokoro.py -q
```

The test verifies the pinned assets before startup and does not download them. Once opted
in, missing assets, dependencies, CUDA support, or worker prerequisites fail the requested
case. For candidate-bound evidence, start from a clean checkout, record `git rev-parse HEAD`,
create the CPU environment with `uv sync --frozen --group dev --extra local`, and create the
CUDA environment from that checkout with `scripts/setup-kokoro-cuda-worker.sh`. Record the
commit, installed package versions, and test result without publishing private paths. The
smoke test does not qualify audio quality or latency.
