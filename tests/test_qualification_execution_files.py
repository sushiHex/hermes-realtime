"""Immutable snapshots must protect namespace entries as well as existing bytes."""

import hashlib
import os

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows execution snapshot")


def test_snapshot_denies_new_imports_and_byte_changes_then_removes_owned_root():
    from scripts.qualification_execution_files import (
        _execution_files_for_consumer,
        execution_file_metadata,
        owned_execution_files,
    )

    raw = b"synthetic_value = 7\n"
    with owned_execution_files({"Lib/synthetic.py": raw, "runner.py": b"pass\n"}) as receipt:
        root = _execution_files_for_consumer(receipt)
        assert (root / "Lib/synthetic.py").read_bytes() == raw
        assert (
            "Lib/synthetic.py",
            hashlib.sha256(raw).hexdigest(),
            len(raw),
        ) in execution_file_metadata(receipt)
        for path in (root / "shadow.py", root / "Lib/shadow.py", root / "Lib/synthetic.py"):
            with pytest.raises(OSError):
                path.write_bytes(b"changed\n")
        with pytest.raises(OSError):
            (root / "Lib/unbound").mkdir()
        with pytest.raises(OSError):
            (root / "Lib/synthetic.py").unlink()
        with pytest.raises(OSError):
            root.rename(root.with_name(root.name + "-replaced"))
    assert not root.exists()
    with pytest.raises(ValueError, match="closed"):
        execution_file_metadata(receipt)


def test_snapshot_removes_only_its_own_files_on_body_failure():
    from scripts.qualification_execution_files import (
        _execution_files_for_consumer,
        owned_execution_files,
    )

    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        owned_execution_files({"runner.py": b"pass\n"}) as receipt,
    ):
        root = _execution_files_for_consumer(receipt)
        raise RuntimeError("synthetic failure")
    assert not root.exists()


@pytest.mark.parametrize(
    "names",
    [
        {"../outside.py": b"pass\n"},
        {"Lib/A.py": b"", "lib/a.py": b""},
        {"Lib": b"", "Lib/a.py": b""},
        {"a.py:stream": b""},
        {},
    ],
)
def test_snapshot_refuses_unsafe_or_ambiguous_names_before_creating_a_tree(names, monkeypatch):
    from scripts import qualification_execution_files as owner

    def unexpected(*args, **kwargs):
        pytest.fail("invalid input allocated an execution root")

    monkeypatch.setattr(owner.tempfile, "mkdtemp", unexpected)
    with pytest.raises(ValueError), owner.owned_execution_files(names):
        pytest.fail("invalid snapshot was admitted")


def test_snapshot_receipt_cannot_be_constructed_or_forged():
    from scripts.qualification_execution_files import (
        ImmutableExecutionFilesV1,
        execution_file_metadata,
    )

    with pytest.raises(TypeError):
        ImmutableExecutionFilesV1()
    with pytest.raises(ValueError, match="unregistered"):
        execution_file_metadata(object.__new__(ImmutableExecutionFilesV1))


def test_snapshot_independently_verifies_written_bytes_before_minting(monkeypatch):
    from scripts import qualification_execution_files as owner

    original = owner._security

    def replace_written_bytes(path, *, directory, frozen):
        if not directory:
            path.write_bytes(b"synthetic altered copy\n")
        original(path, directory=directory, frozen=frozen)

    monkeypatch.setattr(owner, "_security", replace_written_bytes)
    with (
        pytest.raises(ValueError, match="copied bytes"),
        owner.owned_execution_files({"runner.py": b"pass\n"}),
    ):
        pytest.fail("a substituted file was admitted")


@pytest.mark.parametrize("member", ["launcher manifest.xml", "GMT+0", "script (dev).tmpl"])
def test_snapshot_preserves_ordinary_windows_resource_names(member):
    from scripts.qualification_execution_files import (
        _execution_files_for_consumer,
        owned_execution_files,
    )

    with owned_execution_files({member: b"synthetic resource"}) as receipt:
        root = _execution_files_for_consumer(receipt)
        assert (root / member).read_bytes() == b"synthetic resource"
        with pytest.raises(OSError):
            (root / member).write_bytes(b"changed")
    assert not root.exists()
