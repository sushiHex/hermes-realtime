"""Review candidate entrypoint; explicit override and dedicated process lifetime required.

Only OpenAI chat-completions responses are supported. No user-profile install here.
"""

from openai.types.chat import ChatCompletion
from tools.registry import registry

from .policy import Generation


def refusal_result():
    return ChatCompletion.model_validate(
        {
            "id": "policy-refusal",
            "object": "chat.completion",
            "created": 0,
            "model": "policy-refusal",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "This lookup was refused by its execution policy.",
                    },
                }
            ],
        }
    )


def install(ctx, generation, wire=None):
    for name, asynchronous, wrap in (
        ("web_search", False, generation.wrap_search),
        ("web_extract", True, generation.wrap_extract),
    ):
        original = registry.get_entry(name)
        if original is None or original.is_async is not asynchronous:
            raise ValueError("unsupported-web-registration")
        handler = wrap(original.handler)
        generation.originals[name] = original.handler
        generation.handles.append(
            ctx.register_tool(
                name,
                original.toolset,
                original.schema,
                handler,
                check_fn=original.check_fn,
                requires_env=original.requires_env,
                is_async=asynchronous,
                description=original.description,
                emoji=original.emoji,
                override=True,
            )
        )
        generation.registrations.append((name, handler, asynchronous))
    generation.handles.extend(
        [
            ctx.register_middleware("llm_execution", generation.model_gate),
            ctx.register_middleware("tool_execution", generation.tool_gate),
        ]
    )
    ctx.register_platform_handler("api_server", wire or generation.wire)


def register(ctx):
    import tools.web_tools  # noqa: F401 -- shipped self-registration, no handler call.

    refusal = refusal_result()
    generation = Generation(lambda: refusal)
    install(ctx, generation)
