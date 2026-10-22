"""Project committed blobs for the opt-in proof's source import boundary.

The caller verifies repository origin and pin and supplies a fresh owned directory.
The interpreter and prepared third-party dependencies remain a separate trust boundary.
No worktree overlay is imported. The one pinned attribution-metadata subtree is
explicitly omitted; receipts describe a committed-source projection, not a full export.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


class SourceExportError(RuntimeError):
    """Content-free failure of the disposable committed-source export."""

    @property
    def category(self) -> str:
        approved = {
            "source_export_git_failed",
            "source_export_path_invalid",
            "source_export_path_collision",
            "source_export_revision_invalid",
            "source_export_destination_not_fresh",
            "source_export_revision_mismatch",
            "source_export_tree_invalid",
            "source_export_entry_invalid",
            "source_export_blob_invalid",
            "source_export_metadata_projection_changed",
        }
        return str(self) if str(self) in approved else "source_export_refused"


def _git(source: Path, *arguments: str, payload: bytes | None = None) -> bytes:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(source), *arguments],
            input=payload,
            stdin=subprocess.DEVNULL if payload is None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise SourceExportError("source_export_git_failed") from error


def _paths(entries: list[tuple[str, str, str]]) -> None:
    aliases: dict[str, str] = {}
    files = {path for path, _, _ in entries}
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    for path, _, _ in entries:
        parts = path.split("/")
        if any(
            not part
            or part in {".", ".."}
            or part.casefold() == ".git"
            or part[-1:] in {".", " "}
            or part.split(".")[0].upper() in reserved
            or any(ord(char) < 32 or char in '\\:<>"|?*' for char in part)
            for part in parts
        ):
            raise SourceExportError("source_export_path_invalid")
        for size in range(1, len(parts) + 1):
            prefix = "/".join(parts[:size])
            alias = prefix.casefold()
            if (alias in aliases and aliases[alias] != prefix) or (
                size < len(parts) and prefix in files
            ):
                raise SourceExportError("source_export_path_collision")
            aliases[alias] = prefix


def export_tree(source: Path, revision: str, destination: Path) -> dict[str, str | int]:
    """Materialize a declared projection of regular Git blobs, never overlays.

    Git archive export-ignore attributes cannot omit tracked files because this reads
    the tree/blob objects directly. Symlinks and submodules refuse rather than escape
    the owned tree. A failed partial export stays in the caller's owned temp directory.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SourceExportError("source_export_revision_invalid")
    if destination.exists() or destination.is_symlink():
        raise SourceExportError("source_export_destination_not_fresh")
    commit = _git(source, "rev-parse", f"{revision}^{{commit}}").strip()
    if commit != revision.encode("ascii"):
        raise SourceExportError("source_export_revision_mismatch")
    tree = _git(source, "rev-parse", f"{revision}^{{tree}}").strip()
    if not re.fullmatch(rb"[0-9a-f]{40}", tree):
        raise SourceExportError("source_export_tree_invalid")
    listing = _git(source, "ls-tree", "-rz", "--full-tree", revision)
    entries = []
    try:
        for row in listing.split(b"\0")[:-1]:
            metadata, name = row.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ")
            if (
                mode not in {"100644", "100755"}
                or kind != "blob"
                or not re.fullmatch(r"[0-9a-f]{40}", oid)
            ):
                raise SourceExportError("source_export_entry_invalid")
            entries.append((name.decode("utf-8"), mode, oid))
    except (ValueError, UnicodeError) as error:
        raise SourceExportError("source_export_entry_invalid") from error
    if (
        not listing.endswith(b"\0")
        or not entries
        or len({p for p, _, _ in entries}) != len(entries)
    ):
        raise SourceExportError("source_export_entry_invalid")
    full_tree_files = len(entries)
    metadata_prefix = "contributors/emails/"
    expected_metadata_tree = b"04e3b76fdc46dc1c6b36fd6f4b94701bf975f51d"
    excluded = [entry for entry in entries if entry[0].startswith(metadata_prefix)]
    excluded_tree = b""
    if excluded:
        excluded_tree = _git(source, "rev-parse", f"{revision}:contributors/emails").strip()
        if excluded_tree != expected_metadata_tree or len(excluded) != 935:
            raise SourceExportError("source_export_metadata_projection_changed")
        entries = [entry for entry in entries if not entry[0].startswith(metadata_prefix)]
    _paths(entries)
    requests = "".join(f"{oid}\n" for _, _, oid in entries).encode("ascii")
    batch = _git(source, "cat-file", "--batch", payload=requests)
    offset = 0
    blobs = []
    try:
        for path, mode, oid in entries:
            end = batch.index(b"\n", offset)
            actual_oid, kind, size_text = batch[offset:end].decode("ascii").split(" ")
            size = int(size_text)
            offset = end + 1
            content = batch[offset : offset + size]
            if (
                size < 0
                or actual_oid != oid
                or kind != "blob"
                or len(content) != size
                or (batch[offset + size : offset + size + 1] != b"\n")
            ):
                raise SourceExportError("source_export_blob_invalid")
            blob_oid = hashlib.sha1(
                b"blob " + str(size).encode("ascii") + b"\0" + content
            ).hexdigest()
            if blob_oid != oid:
                raise SourceExportError("source_export_blob_invalid")
            blobs.append((path, mode, content))
            offset += size + 1
    except (ValueError, UnicodeError) as error:
        raise SourceExportError("source_export_blob_invalid") from error
    if offset != len(batch):
        raise SourceExportError("source_export_blob_invalid")
    destination.mkdir()
    manifest = []
    created_parents = {destination}
    for path, mode, content in blobs:
        target = destination.joinpath(*path.split("/"))
        if target.parent not in created_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
            created_parents.add(target.parent)
        with target.open("xb") as stream:
            stream.write(content)
        if os.name != "nt":
            target.chmod(0o755 if mode == "100755" else 0o644)
        manifest.append([path, mode, hashlib.sha256(content).hexdigest()])
    digest = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "source_export_tree": tree.decode("ascii"),
        "source_export_full_tree_files": full_tree_files,
        "source_export_excluded_metadata_tree": excluded_tree.decode("ascii"),
        "source_export_excluded_metadata_files": len(excluded),
        "source_export_projection": (
            "committed_without_pinned_attribution_metadata" if excluded else "committed_no_omission"
        ),
        "source_export_manifest_sha256": digest,
        "source_export_files": len(blobs),
        "source_export_bytes": sum(len(content) for _, _, content in blobs),
    }
