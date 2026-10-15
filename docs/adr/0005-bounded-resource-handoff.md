# ADR 0005: Host carries explicit resource references across a work handoff

Status: Proposed for review. Work item:
[#263](https://github.com/sushiHex/hermes-realtime/issues/263); related qualification:
[#248](https://github.com/sushiHex/hermes-realtime/pull/248).
This proposal authorizes no implementation or protocol change and does not supersede
[ADR 0004](0004-model-action-authority.md).

## Context

An objective can depend on resources identified in the foreground conversation.
The reviewed dispatch path sends the objective to Hermes without binding that
conversation's resource context to the selected background profile. A successful
read through an isolated Hermes terminal/file profile proves access in that
profile; it does not prove that a real Runs handoff selects the same profile or
carries the resource binding. An implicit working directory is not such a binding.

A [bounded diagnostic](https://github.com/sushiHex/hermes-realtime/pull/248#issuecomment-6188694973)
retained all 16 decisions from eight paired cases. Among four intended starts per
arm, ordinary prior-user resource facts yielded sufficient resource anchors in
3/4 cases (A); moving the same facts to a labelled system reference section yielded
1/4 (B). This comparison changed role, position and framing together. It provides
no support for that placement change and does not isolate a cause. The diagnostic
was operationally incomplete despite retaining all decisions; it is not a
qualification pass. Earlier qualification failures remain failures.

## Proposal

The existing host would own a bounded set of explicit resource references bound
to the relevant session, selected backend profile and generation. The model would
propose the work objective. The host would attach the admitted references unchanged
alongside that objective through one work handoff, rather than depending on the
model to reproduce every locator in its prose.

Only explicitly admitted references would cross that boundary. The host would not
discover files automatically, forward the conversation history, infer access from
the current directory, or grant authority through a reference. A reference would
not establish that the selected backend can read its target. Backend binding and
access must be verified separately; unavailable, stale or mismatched bindings must
refuse the handoff rather than silently select another resource or profile.

This record deliberately leaves wire fields and numerical limits undecided. Those
belong to a reviewed, versioned contract covering size, count, identity, generation,
lifetime, invalidation, privacy and acknowledgment behavior before implementation.
The contract must distinguish resource data from instructions and permissions.

## Existing Hermes carriers and profile boundary

The pinned Hermes v0.21.0 Runs handler already accepts `input` as message content
and explicit `conversation_history` as role/content messages; no new Hermes API is
assumed here. These are model-visible content carriers, not typed resource bindings
or evidence that a tool can read a referenced target. The reviewed realtime adapter
currently sends the objective as `input` and does not send `conversation_history`
([adapter](../../src/hermes_realtime/integration/api.py)). Passing `session_id` is
not a resource binding: when explicit history is absent, Runs may load stored
messages for that session; when omitted, Hermes gives the run its own session ID.
([Runs request fields](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L480-L546),
[session behavior](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L600-L614)).

Hermes already selects multiplexed profiles through a `/p/<profile>/` URL prefix
and carries that request profile into the background Runs agent
([profile middleware](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L2100-L2206),
[Runs profile scope](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L719-L823)).
The current realtime API client accepts only a loopback origin, so it cannot
currently select that prefix
([client configuration](../../src/hermes_realtime/integration/api.py)). Hermes
already has file-reading and terminal tools, but their availability, backend and
working directory depend on the selected profile's tool configuration
([file tools](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/tools/file_tools.py),
[terminal tool](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/tools/terminal_tool.py)).
The terminal configuration also has a
[process-level environment bridge](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/tools/terminal_tool.py#L1712-L1761).
The witness below uses explicit locators; it does not establish per-profile working
directory propagation or exclusive filesystem permissions.
Review must choose and bound the existing content carrier and settle how the host
binds it to the correct local profile before deciding whether any new Hermes API
surface is needed.

## Limits and unchanged authority

Preserving references cannot guarantee correct model intent, a faithful objective,
or truthful lifecycle prose. It cannot resolve an unspecified target or make an
inaccessible resource available. A resource root plus an unambiguous, discoverable
resource index can be sufficient; a literal absolute filename is not universally
required. Sufficiency must be demonstrated for the selected backend contract.

Cancellation remains governed by ADR 0004: explicit user commands and trusted task
controls retain authority, with independent foreground and background scopes.
References confer no cancellation or approval authority. The host must not report
work started before a real accepted dispatch acknowledgment. This proposal does
not make arbitrary model replies reliable reports of task state.

## Gate before implementation

Review must first settle the wire contract and limits, including refusal behavior
for stale generations, profile mismatches and unavailable references. Deterministic
tests must cover those boundaries without expanding the model's authority.

A witness through the actual pinned Hermes Runs route and selected profile must
then demonstrate that admitted references arrive unchanged and identify resources
that the background executor can use. A direct terminal read alone is insufficient;
neither successful transport nor a working-directory setting proves this handoff.
The [existing-carrier witness](https://github.com/sushiHex/hermes-realtime/pull/269#issuecomment-6189897265)
now demonstrates this mechanism in two isolated profiles: two real Runs completed,
two registered `read_file` calls returned their known canaries, and four deterministic
stand-in completion requests used no external inference. Exact supplied reference
content reached the model; actual tool arguments, selected profile and local backend
matched. An unknown profile returned 404. Original handlers and dispatch were not
replaced; a passive observer retained tool-boundary metadata. The public owner
verified its Job empty, and both listeners were separately observed absent.

The initial fixture Run remains failed: its checker omitted a trailing empty line
from the expected pinned formatter output. A subsequent wrong-class formatter
preflight refused before any Run. Both are preserved separately from the successful
successor, which first checked the actual pinned formatter under the hidden owner.
These are fixture corrections, not a reinterpretation of earlier routing results.

This proves the existing explicit-history carrier for these configured profiles and
locators, not the realtime foreground handoff or approval to forward whole history.
The chosen host binding and wire contract still need their own reviewed witness.

Actual-adapter qualification must separately assess routing, objective fidelity,
resource sufficiency and truthful lifecycle claims. Real execution and voice gates
remain independent. Existing failed results and the untouched original corpus must
retain their identities; this design creates no retrospective pass.

## Alternatives

Relying solely on model-written locators leaves the observed omission boundary in
place; the tested placement change did not improve it. Forwarding full history
would enlarge the data and instruction boundary without establishing backend
access. Treating the current directory as an implicit resource contract would hide
the profile binding that needs to be proved. The proposed host attachment keeps
that binding explicit while leaving the objective and its semantic evaluation
separate.
