# Source-backed foreground routing and latency

The Codex full-host profile includes one deterministic, fail-closed knowledge path for
source-sensitive foreground turns. It supplies bounded web evidence before inference; it does not
grant the model unrestricted web, shell, browser, MCP, or background-task authority.

The runtime implementation is complete behind conservative controls. Final-transcript public-RSS lookup
is active for routed Codex turns. Partial-transcript overlap and recovery are disabled by default
and remain qualification features rather than product claims.

## Routed turns

The shared classifier routes four categories:

- `current_fact`: explicit fresh/current/latest questions;
- `explicit_source`: requests to search, verify, cite, or consult sources;
- `technical`: questions containing distinctive technical identifiers;
- `historical`: factual who/when/where questions that are not ordinary conversational prompts.

Current-fact cues include terms such as “current,” “latest,” “today,” and “now”; electrical-current
questions are excluded from that temporal cue. Technical routing preserves distinctive identifiers
rather than broadening them away. Personal plans and recommendations are also excluded from
deterministic prefetch, even when they contain a temporal cue; Codex may use the bounded
`search_knowledge` dynamic tool when external facts materially affect the answer. That tool is source
lookup only, not Hermes work authority.

Unrouted conversation creates no knowledge operation. Typed input uses the same final-transcript
lookup path but never speculative partial lookup. Once deterministic routing selects a turn,
dynamic `search_knowledge` is withheld for that turn even when lookup is empty, weak, timed out, or
failed. The assistant must state that it could not verify rather than filling an exact claim from
memory.

The only runtime backend is `PublicRssCurrentFactLookup`, which queries Bing RSS and, for
outcome-shaped queries, Google News RSS directly; see `THIRD_PARTY_NOTICES.md`. DDGS, Tavily, Exa,
and OpenAI hosted search are isolated benchmark arms;
selecting one in a benchmark does not alter host routing or authentication.

## Runtime path

1. `foreground_search_query()` and `foreground_search_route()` project one shared routing decision.
2. A routed final transcript enters one exact-turn knowledge deadline.
3. If stable-partial speculation produced an exact matching operation, the final turn reuses it;
   otherwise the final transcript starts one public-RSS lookup.
4. `KnowledgePrefetchCoordinator.consume_turn_result()` is the sole coordinator consumption path;
   it returns timing plus evidence while revoking stale or replaced authority.
5. Optional recovery may run once only for weak/empty evidence and only inside the same deadline.
6. The selected evidence is rendered into Codex `baseInstructions`; deterministic prefetch owns that
   decision, so routed failure does not trigger an additional model-selected search.
7. Host composition chooses either the coordinator lookup or direct public-RSS lookup, then constructs the
   Codex inference provider once with the selected callable.

## Evidence and authority

Every usable result is normalized into bounded immutable evidence with:

- a retrieval date and backend identity;
- unique source IDs, titles, and absolute public HTTP(S) URLs;
- sentence-aware attributed passages and optional source offsets;
- deterministic `usable`, `weak`, or `empty` quality;
- explicit conflict and recovery metadata.

Model context and dynamic-tool results render from the same passage structure. HTML enrichment keeps
at most two sources, 300 characters per selected passage, and the public-IP-only DNS and redirect
protections. Queries are capped at 512 characters. Recognized credentials, tokens, private paths,
and other private-looking material are rejected before external lookup. Enrichment accepts only
public HTTP(S) targets, validates every redirect, and rejects DNS resolution containing a non-public
address. Provider payloads, full pages, queries, and transcript text are not projected to the
browser. Source text is untrusted evidence, never instructions.

Speculation is bound to the exact `(session generation, media incarnation, utterance sequence)` and
the complete normalized transcript hash. A changed final, echo rejection, barge-in, reconnect,
media replacement, turn discard, or shutdown revokes result authority. Cancelling an asyncio task
does not pretend to stop an already-running public-RSS request: detached calls remain bounded,
counted, and unable to satisfy another utterance.

## Host controls

The two boolean controls require the Codex inference provider; `--knowledge-budget-seconds` is
accepted but unused without it. Partial speculation additionally requires Moonshine streaming STT.

```bash
env -u PYTHONPATH uv run --frozen --extra local hermes-realtime-host \
  --hermes-env-file C:/path/to/active/hermes/.env \
  --inference-provider codex \
  --stt-provider moonshine \
  --enable-public-search \
  --knowledge-budget-seconds 3.5 \
  --knowledge-speculation \
  --knowledge-recovery \
  --allow-unsandboxed-hermes-tasks
```

- `--knowledge-budget-seconds`: exact-turn deadline, bounded from `0.05` through `3.5` seconds;
- `--enable-public-search`: make the consent-bound public-RSS path available; default off;
- `--knowledge-speculation`: default-off stable-partial public-RSS prefetch;
- `--knowledge-recovery`: default-off single recovery attempt for weak or empty evidence.

Omit `--enable-public-search` for no transcript-derived public-search egress. If public search is
enabled and the browser binding consents, omitting the two knowledge flags retains final-transcript
public-RSS lookup only. Recovery shares the original exact-turn budget and cannot extend it.
Faster-whisper remains final-only. Ollama rejects public search, overlap, and recovery configuration
rather than silently ignoring it.

Knowledge controls are independent of `--natural-work-tools`: source-backed foreground lookup does
not authorize Hermes background work, and natural work does not bypass deterministic lookup policy.

## Telemetry

The browser receives a compact advisory `knowledge_timing` event containing route, backend,
`lastMs`, nearest-rank `p50Ms`/`p95Ms`, sample count, and lookup elapsed/blocking/overlap values.
Telemetry contains no query or page text and grants no transcript, source, task, tool, or playback
authority. Local coordinator counters retain speculation acceptance, rejection, revocation,
detached-call, and saturation evidence.

## Characterization

The checked-in corpus contains synthetic transcript progressions only. Reports contain case IDs,
categories, policy outcomes, timings, and environment fingerprints; they intentionally omit
transcript text. Private-looking unfinished speech is rejected before external speculation.

```bash
env -u PYTHONPATH uv run --frozen --group dev python \
  scripts/benchmark_foreground_knowledge.py \
  --backend ddgs \
  --mode characterize \
  --output artifacts/ddgs-characterization.json
```

`eligible` means the final changed Moonshine partial is exact after normalization, preserves the
configured stable word prefix, remains long enough to route, passes the private-text gate, and
leaves a positive debounce-adjusted window before endpoint finalization. Faster-whisper is the
final-only control because it emits no streaming partials.

## Experimental backend bakeoff

Credentialed arms require an explicit metered-API acknowledgement and the corresponding environment
variable. DDGS must remain one arm.

```bash
export TAVILY_API_KEY='<redacted>'
export EXA_API_KEY='<redacted>'
export OPENAI_API_KEY='<redacted>'

env -u PYTHONPATH uv run --frozen --group dev python \
  scripts/benchmark_foreground_knowledge.py \
  --mode bakeoff \
  --backend ddgs \
  --backend tavily-fast \
  --backend exa-instant \
  --backend openai-hosted-search \
  --allow-metered-api \
  --output artifacts/knowledge-bakeoff.json
```

Available experimental arms are `tavily-ultra-fast`, `tavily-fast`, `exa-instant`, and
`openai-hosted-search`. Missing credentials, duplicate/unknown arms, metered arms without
`--allow-metered-api`, malformed output, or omission of DDGS fail closed.

## Promotion gate

Do not promote speculation, recovery, or another backend from synthetic or vendor evidence alone.
Before changing defaults:

1. collect consented local STT progressions without committing audio or transcripts;
2. establish partial-to-final exact agreement and privacy-rejection rates;
3. run paired speculation-off/on end-to-end trials with route timing;
4. retain timeout, cancellation, detached-call, saturation, provider-error, and wasted-request
   outcomes in the denominator;
5. require non-inferior evidence support and the predeclared latency/p95 gates;
6. pass the Windows browser and iPhone physical matrix on one frozen candidate.

Generated files under `artifacts/` are qualification evidence and are not release inputs.
