# Proposed Hermes v0.21.0 source-materialization profile

[Qualification input authority](qualification-input-files.md) · [Execution protocol](qualification-execution.md) · [Issue #67](https://github.com/sushiHex/hermes-realtime/issues/67)

## Status and decision requested

This is a proposed source-only design for the owner-selected Hermes Agent
v0.21.0 baseline. It is not accepted implementation authority, a runtime-pin
migration, an installed-discovery result, or a supported-version claim.

The requested decision is whether a source-bound selected materialization may
be used for this exact upstream identity:

- repository: `NousResearch/hermes-agent`
- release tag: `v2026.8.31`
- annotated tag object: `6e8f8418e6378eb2617e4de074e13dedd091b8af`
- commit: `29112bef099274229cadff79cdff7bf7b99c4b77`
- tree: `daaffc303ae437041b7f76be17c5f61b14f2ce99`

The current complete-publisher-source policy cannot materialize that tree on a
case-insensitive Windows namespace because its known non-runtime collision is
not export-ignored. This proposal preserves the whole publisher archive as
provenance, but materializes a reviewed Windows-safe derivative. It must never
be described as materializing the complete upstream tree.

## Fixed-profile mechanism

Introduce one `HermesPublisherSelectionProfileV1`, valid for the identity above
only. It is a **complete-tree-minus-explicit-exclusions** profile, rather than a
runtime dependency allowlist. That keeps every regular member that is not in the
exact two-member reviewed exclusion set, without claiming an unproven minimal
import closure. Both excluded members are the known non-runtime regular-member
collision; their names stay out of public evidence.

Admission has two namespaces with separate purposes:

1. **Logical publisher inventory.** Authenticate the complete codeload bytes by
   reviewed digest and size before parsing. Check the decompressed bytes and
   enumerate one ordinary logical tar tree under its exact upstream prefix.
   Reject links, devices, sparse members, traversal, duplicate exact names,
   invalid root structure, invalid modes, changed member bytes, and inventory
   drift. This inventory is not a Windows materialization, so its inspection can
   record a case-fold collision without treating that fact as an overwrite.
2. **Windows selected materialization.** Require equality with the profile's
   full logical-inventory digest and its exact exclusion record. Derive the
   selected inventory as every logical member minus that record. Require the
   selected names, modes, payload digests, count, bytes, and case-folded
   namespace to match the profile. Serialize that selected inventory
   deterministically beneath a new fixed source root, retaining every selected
   relative spelling and mode. Submit this derived archive to the existing
   `SourceArchivePolicyV1` validator unchanged before any Windows extraction.

The profile stores the complete logical inventory digest, selected inventory
digest, and exclusion inventory digest. Its exclusion entries bind a path-name
digest, type, mode, content digest, and size; the implementation checks that
each digest resolves to exactly one logical member. Public metadata reports only
these digests and aggregate counts. It does not publish contributor identifiers
or raw upstream member names.

The selected archive is a derivative with its own reviewed digest and size. It
is bound to the authenticated complete archive and the exact profile; it is not
a replacement publisher archive. The retained authority carries both facts so a
consumer cannot attach the selected archive to another tag, tree, codeload, or
selection profile.

## Boundaries

The existing generic archive validator continues to reject unsafe components,
case aliases, duplicates, file/directory collisions, links, and oversized
trees. The existing candidate archive's `export-ignore` semantics are unchanged
and do not authorize this profile. No source member is renamed, and no member is
silently omitted: exclusion equality and the full-minus-exclusion relation are
both required before selection.

The final Hermes source role would bind the selected-archive identity and the
full-publisher provenance record. A later runtime consumer may only receive the
selected source tree. It must report that narrower fact; it cannot infer full
publisher materialization, installed discovery, dispatch, acknowledgement,
cancellation, cleanup, continuity, or complete qualification.

This design changes neither the current v0.20.0 source profile nor any existing
runtime pin, dependency recipe, parser, input schema, or acceptance rule.

### Refusal evidence

Every new logical-inventory or selection guard/bound introduced for this profile
must emit exactly one bounded JSON line from its `finally` path when it refuses.
The stable marker is `[hermes-source-selection-refused]`; its payload may carry
counts, kinds, and categories only. It must not carry paths, member names,
process IDs, handles, or error text. A successful neighboring case emits no
such marker.

## Alternatives considered

**Keep complete materialization.** Correctly refuses the selected tree on
Windows and remains the current behavior.

**Use `export-ignore`.** Not applicable to the known collision, and current
candidate archive handling rejects a tree collision before evaluating archive
attributes. It would also make upstream attribute changes an implicit source
selection rule.

**Use a runtime allowlist.** Rejected for this slice: it would require proving a
complete import/configuration closure before source preparation and would widen
the design beyond the materialization blocker.

**Unrecorded omission, renaming, or overwrite of colliding members.** Rejected.
The exact reviewed two-member exclusion above is the proposed mechanism; any
other omission loses provenance or weakens the Windows collision boundary.

**Use a case-sensitive host for the full tree.** It may support a separate
upstream inspection, but it does not create a Windows qualification input.

## Review and RED cases

Before implementation, the approval record must bind the codeload and
decompressed identities, the three inventory digests, counts/bytes, the exact
exclusion record, and the selected derivative identity. A separate review must
approve any new exact upstream identity or profile revision.

Each mutation fixture must rebind any enclosing reviewed identity that would
otherwise reject first, so it reaches its nominated guard. It must prove that
guard fails alone, rather than treating another refusal as coverage.

Tests must first fail for each of these mutations:

- change the tag, commit, tree, codeload, decompressed archive, root, member
  bytes, type, mode, or complete logical inventory;
- add, remove, alter, or make ambiguous an exclusion entry;
- leave a case collision in the selected inventory, or introduce any unsafe
  selected path or file/directory conflict;
- remove a non-excluded logical member from the selected derivative, add an
  excluded member, rename a selected member, or alter its mode or payload;
- substitute a selected archive or provenance record from another exact source;
- bypass the unchanged selected-archive validator or try to consume full-source
  materialization metadata as selected-source authority.
- omit, duplicate, or leak disallowed content through refusal evidence for a new
  logical/selection guard or bound, while its passing neighboring case remains
  marker-free.

The [qualification input authority](qualification-input-files.md) and
[execution protocol](qualification-execution.md) remain the governing plans for
their existing scopes. This proposal needs their reviewed update before it can
be implemented or used as any qualification input.
