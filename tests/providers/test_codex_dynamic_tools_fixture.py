from __future__ import annotations

import json
import re
from pathlib import Path


def test_codex_dynamic_work_tool_protocol_is_pinned() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "codex_app_server_dynamic_tools.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["codexVersion"] == "codex-cli 0.145.0"
    assert fixture["codexBinarySha256"] == (
        "83751f15cb6a0a7b97df67752c001e3fe1c20e18ffbfec3ff63567296205eb6c"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", fixture["codexBinarySha256"])
    assert fixture["serverRequestMethod"] == "item/tool/call"

    schemas = fixture["schemas"]
    assert set(schemas) == {
        "DynamicToolSpec",
        "DynamicToolCallParams",
        "DynamicToolCallResponse",
        "DynamicToolCallOutputContentItem",
    }

    assert schemas["DynamicToolSpec"] == {
        "properties": {
            "deferLoading": {"type": "boolean"},
            "description": {"type": "string"},
            "inputSchema": True,
            "name": {"type": "string"},
            "type": {"enum": ["function"], "type": "string"},
        },
        "required": ["description", "inputSchema", "name", "type"],
        "type": "object",
    }

    params = schemas["DynamicToolCallParams"]
    assert params["required"] == [
        "arguments",
        "callId",
        "threadId",
        "tool",
        "turnId",
    ]
    assert set(params["properties"]) == {
        "threadId",
        "turnId",
        "callId",
        "namespace",
        "tool",
        "arguments",
    }
    assert params["properties"]["namespace"] == {
        "type": ["string", "null"],
    }
    assert params["properties"]["arguments"] is True
    for identifier in ("callId", "threadId", "tool", "turnId"):
        assert params["properties"][identifier] == {"type": "string"}

    response = schemas["DynamicToolCallResponse"]
    assert response["required"] == ["contentItems", "success"]
    assert response["properties"] == {
        "contentItems": {
            "items": {
                "$ref": "#/schemas/DynamicToolCallOutputContentItem",
            },
            "type": "array",
        },
        "success": {"type": "boolean"},
    }

    output_content = schemas["DynamicToolCallOutputContentItem"]
    assert output_content == {
        "oneOf": [
            {
                "properties": {
                    "text": {"type": "string"},
                    "type": {"enum": ["inputText"], "type": "string"},
                },
                "required": ["text", "type"],
                "type": "object",
            },
            {
                "properties": {
                    "imageUrl": {"type": "string"},
                    "type": {"enum": ["inputImage"], "type": "string"},
                },
                "required": ["imageUrl", "type"],
                "type": "object",
            },
            {
                "properties": {
                    "audioUrl": {"type": "string"},
                    "type": {"enum": ["inputAudio"], "type": "string"},
                },
                "required": ["audioUrl", "type"],
                "type": "object",
            },
        ]
    }
