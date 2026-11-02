from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profile_summary_outputs_only_allowlisted_fields(tmp_path: Path) -> None:
    secret = "SYNTHETIC_SECRET_DO_NOT_PRINT_7d143a"
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "model": {
                    "provider": "custom",
                    "default": "example-model-v2",
                    "api_mode": "codex_responses",
                    "api_key": secret,
                },
                "plugins": {
                    "entries": {
                        "ordinary-lookup-experimental": {
                            "settings": {
                                "presentation": {
                                    "provider": "custom",
                                    "model": "example-model-v2",
                                    "api_mode": "codex_responses",
                                    "api_key": secret,
                                }
                            }
                        }
                    }
                },
                "private_arbitrary_map": {"nested_value": secret},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/profile_summary.py"), str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    assert summary == {
        "version": 1,
        "config_read": True,
        "native_model": "example-model-v2",
        "native_provider": "custom",
        "native_mode": "codex_responses",
        "presentation_model": "example-model-v2",
        "presentation_provider": "custom",
        "presentation_mode": "codex_responses",
        "same_model": True,
        "same_provider": True,
        "same_mode": True,
    }
    assert secret not in result.stdout
    assert str(config) not in result.stdout
    assert "api_key" not in result.stdout
    assert "private_arbitrary_map" not in result.stdout
