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
