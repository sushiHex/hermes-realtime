from __future__ import annotations

import json
import runpy
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
        "version": 2,
        "config_read": True,
        "native_model_configured": True,
        "native_provider_configured": True,
        "native_mode_configured": True,
        "presentation_model_configured": True,
        "presentation_provider_configured": True,
        "presentation_mode_configured": True,
        "same_model": True,
        "same_provider": True,
        "same_mode": True,
    }
    assert secret not in result.stdout
    assert str(config) not in result.stdout
    assert "api_key" not in result.stdout
    assert "private_arbitrary_map" not in result.stdout


def test_profile_summary_never_echoes_routing_values() -> None:
    summarize = runpy.run_path(str(ROOT / "scripts/profile_summary.py"))["summarize_config"]
    for value in (
        "synthetic-token-abcd1234",
        "https://user:synthetic-password@host.invalid",
        "private-model-name",
    ):
        config = {
            "model": {"default": value, "provider": value, "api_mode": value},
            "plugins": {
                "entries": {
                    "ordinary-lookup-experimental": {
                        "settings": {
                            "presentation": {"model": value, "provider": value, "api_mode": value}
                        }
                    }
                }
            },
        }
        summary = summarize(config)
        assert value not in json.dumps(summary)
        assert summary["same_model"] is True
        assert summary["same_provider"] is True
        assert summary["same_mode"] is True
        assert summary["native_model_configured"] is True
    assert summarize({})["same_model"] is False
    assert summarize({"model": {"default": []}})["native_model_configured"] is False


def test_profile_summary_requires_present_values_for_equality() -> None:
    summarize = runpy.run_path(str(ROOT / "scripts/profile_summary.py"))["summarize_config"]
    for invalid in (None, "", [], {}, True, 1, "x" * 129):
        result = summarize({"model": {"default": invalid}})
        assert result["native_model_configured"] is False
        assert result["same_model"] is False
    result = summarize(
        {
            "model": {"default": "one"},
            "plugins": {
                "entries": {
                    "ordinary-lookup-experimental": {"settings": {"presentation": {"model": "two"}}}
                }
            },
        }
    )
    assert result["native_model_configured"] is True
    assert result["presentation_model_configured"] is True
    assert result["same_model"] is False
