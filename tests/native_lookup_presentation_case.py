"""Explicit model-free pinned-Hermes streaming-to-presentation regression.

Run only in the supplied Hermes interpreter; network is denied and both SDK
transports receive local synthetic responses. No gateway or model is started.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

HELPER_SHA256 = "98693898ba6ca30c45fe27cc4edbc1784d67e4f264326a885110a3bd2388b7fb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hermes-source", "policy", "runtime", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(args.receipt.parent / "profile")
    helper_hash = hashlib.sha256((args.hermes_source /
        "agent/chat_completion_helpers.py").read_bytes()).hexdigest()
    assert helper_hash == HELPER_SHA256, "pinned-native-helper-source-mismatch"
    socketpair_code = getattr(socket.socketpair, "__code__", None)
    effects = {"socketpair_connects": 0, "denied_network": 0, "denied_process": 0}

    def audit(event, _):
        if event in {"subprocess.Popen", "os.system"}:
            effects["denied_process"] += 1
            raise PermissionError("native-response-proof-process-forbidden")
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            # Windows asyncio constructs a stdlib local socketpair. Only the
            # captured stdlib implementation itself may make that connection.
            if event == "socket.connect" and sys._getframe(1).f_code is socketpair_code:
                effects["socketpair_connects"] += 1
                return
            effects["denied_network"] += 1
            raise PermissionError("native-response-proof-network-forbidden")

    sys.addaudithook(audit)
    sys.path[:0] = [str(args.hermes_source), str(args.runtime)]
    import httpx
    from agent import chat_completion_helpers
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    from run_agent import AIAgent
    from tools.approval import reset_current_session_key, set_current_session_key

    assert Path(chat_completion_helpers.__file__).resolve() == (
        args.hermes_source / "agent/chat_completion_helpers.py").resolve()

    spec = importlib.util.spec_from_file_location("native_presentation_policy", args.policy)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    source_text = "Full source result."
    chunk = {"id": "synthetic-stream", "object": "chat.completion.chunk", "created": 0,
             "model": "test-model", "choices": [{"index": 0, "delta": {
                 "role": "assistant", "content": source_text,
                 "reasoning_content": "Synthetic reasoning."}, "finish_reason": "stop"}]}
    usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    tail = dict(chunk, choices=[], usage=usage)
    sse = ("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(tail)
           + "\n\ndata: [DONE]\n\n").encode()
    native_calls = []

    def native_http(request):
        native_calls.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=sse, request=request)

    client = OpenAI(api_key="synthetic", base_url="https://model.example.invalid/v1",
                    max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(native_http)))
    agent = AIAgent(api_key="synthetic", base_url="https://model.example.invalid/v1",
                    model="test-model", provider="custom", quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    with patch.object(agent, "_create_request_openai_client", return_value=client):
        original = agent._interruptible_streaming_api_call({
            "model": "test-model", "messages": [{"role": "user", "content": "synthetic"}],
            "reasoning_effort": "high"})
    assert len(native_calls) == 1
    assert native_calls[0]["reasoning_effort"] == "high"
    assert type(original) is SimpleNamespace
    assert not hasattr(original.choices[0].message, "refusal")
    assert not hasattr(original, "model_copy")
    assert original.choices[0].message.content == source_text

    sdk = ChatCompletion.model_validate({
        "id": "synthetic-sdk", "object": "chat.completion", "created": 42, "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": source_text, "reasoning_content": "SDK reasoning."}}],
        "usage": usage, "synthetic_metadata": {"retained": True},
    })
    real_http = httpx.AsyncClient
    config = {"provider": "custom", "base_url": "https://model.example.invalid/v1",
              "model": "test-model", "api_key_env": "LOOKUP_NATIVE_PROOF_KEY"}
    os.environ[config["api_key_env"]] = "synthetic"
    outcomes = []
    for name, source in (("native_stream", original), ("sdk", sdk)):
        forwarded = []

        def formatter_http(request, forwarded=forwarded):
            body = json.loads(request.content)
            forwarded.append(body)
            assert not any(key in body for key in ("tools", "tool_choice", "functions"))
            assert body["messages"][-1]["content"] == source_text
            assert body["reasoning_effort"] == "none"
            response = sdk.model_dump(mode="json")
            response["choices"][0]["message"]["content"] = json.dumps({
                "voice": "Short.", "detail": source_text})
            return httpx.Response(200, json=response, request=request)

        class IsolatedHttp(real_http):
            def __init__(self, **kwargs):
                super().__init__(transport=httpx.MockTransport(formatter_http), **kwargs)

        refusal = object()
        generation = policy.Generation(lambda refusal=refusal: refusal, policy.Presentation(config))
        generation.handles = [SimpleNamespace(active=True)]
        row = {"closed": False, "cancelled": False, "deadline": time.monotonic() + 10,
               "phase": "new", "settled": 0, "reserved": 0, "urls": set()}
        generation.budgets["synthetic-run"] = row
        generation.members = MappingProxyType({"synthetic-run": (policy.VERSION, "identity", row)})
        generation._membership_authority = generation.members
        token = set_current_session_key("synthetic-run")
        try:
            search_args = {"query": "synthetic lookup"}
            search = ChatCompletion.model_validate({
                "id": "synthetic-search", "object": "chat.completion", "created": 0,
                "model": "test-model", "choices": [{"index": 0, "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "synthetic-search-call", "type": "function", "function": {
                            "name": "web_search", "arguments": json.dumps(search_args)}}]}}],
            })
            assert generation.model_gate(
                api_mode="chat_completions", api_request_id="synthetic-search-request", request={},
                next_call=lambda _, search=search: search) is search
            source_url = "https://example.invalid/source"
            retrieved = generation.tool_gate(
                tool_name="web_search", tool_call_id="synthetic-search-call",
                api_request_id="synthetic-search-request", args=search_args,
                next_call=generation.wrap_search(lambda _, source_url=source_url: json.dumps({
                    "success": True, "data": {"web": [{"url": source_url}]}})),
            )
            assert json.loads(retrieved)["success"] is True and row["urls"] == {source_url}
            with patch("httpx.AsyncClient", IsolatedHttp):
                result = generation.model_gate(
                    api_mode="chat_completions", api_request_id="synthetic-request",
                    provider=config["provider"], base_url=config["base_url"], model=config["model"],
                    request={"model": config["model"]}, next_call=lambda _, source=source: source)
            assert result is not refusal, "native-source-to-publication-refused"
            assert type(result) is type(source) and result is not source
            assert result.choices[0] is not source.choices[0]
            assert result.choices[0].message is not source.choices[0].message
            assert source.choices[0].message.content == source_text
            assert result.choices[0].message.content == (
                "<voice>Short.</voice><detail>" + source_text + "</detail>"
            )
            assert result.id == source.id and result.model == source.model
            assert result.usage is source.usage
            for before, after, changed in (
                (source, result, "choices"),
                (source.choices[0], result.choices[0], "message"),
                (source.choices[0].message, result.choices[0].message, "content"),
            ):
                assert vars(after) | {changed: vars(before)[changed]} == vars(before)
            if name == "sdk":
                assert result.synthetic_metadata == source.synthetic_metadata
            assert row["original_response"] is source and len(forwarded) == 1
            assert row["presentation_evidence"]["local_http_settled"] is True
            assert row["presentation_evidence"]["wire"]["reasoning_effort_none"] is True
            outcomes.append({"shape": name, "passed": True, "formatter_calls": len(forwarded),
                             "reasoning_effort_none": True})
        finally:
            reset_current_session_key(token)
    result = {"ok": True, "native_helper_calls": len(native_calls), "cases": outcomes,
              "native_helper_sha256": helper_hash,
              "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
              "model_calls": 0, "task_reasoning_preserved": True, "effects": effects}
    args.receipt.write_text(json.dumps(result), encoding="utf-8")
    print("[native-presentation-response] " + json.dumps(result))


if __name__ == "__main__":
    main()
