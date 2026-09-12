"""Immutable snapshots must protect namespace entries as well as existing bytes."""

import ctypes as c
import hashlib
import os

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows execution snapshot")


def _replace_named_dacl(path, sddl):
    from scripts.windows_storage_oracle import _api

    kernel, security = _api()
    pointer, word = c.c_void_p, c.c_uint32
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        c.c_wchar_p,
        word,
        pointer,
        pointer,
    ]
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = c.c_int
    security.GetSecurityDescriptorDacl.argtypes = [pointer, pointer, pointer, pointer]
    security.GetSecurityDescriptorDacl.restype = c.c_int
    security.SetNamedSecurityInfoW.argtypes = [
        c.c_wchar_p,
        c.c_int,
        word,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    security.SetNamedSecurityInfoW.restype = word
    descriptor, dacl = pointer(), pointer()
    present, defaulted = c.c_int(), c.c_int()
    converted = bool(
        security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, c.byref(descriptor), None
        )
    )
    assert converted
    try:
        available = bool(
            security.GetSecurityDescriptorDacl(
                descriptor, c.byref(present), c.byref(dacl), c.byref(defaulted)
            )
        )
        assert available and present.value and dacl.value
        return int(
            security.SetNamedSecurityInfoW(
                str(path), 1, 0x80000004, None, None, dacl, None
            )
        )
    finally:
        assert kernel.LocalFree(descriptor) is None


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


def test_snapshot_denies_ordinary_owner_dacl_rewrite_and_controller_cleanup_succeeds():
    from scripts.qualification_execution_files import (
        _execution_files_for_consumer,
        owned_execution_files,
    )
    from scripts.windows_storage_oracle import _user_sid

    with owned_execution_files({"runner.py": b"pass\n"}) as receipt:
        root = _execution_files_for_consumer(receipt)
        permissive = f"D:P(A;;FA;;;{_user_sid()})(A;;FA;;;SY)(A;;FA;;;BA)"
        root_result = _replace_named_dacl(root, permissive)
        file_result = _replace_named_dacl(root / "runner.py", permissive)
        assert root_result == 5
        assert file_result == 5
    assert not root.exists()


def test_snapshot_revalidates_exact_namespace_and_cleans_an_unexpected_entry():
    from scripts import qualification_execution_files as owner

    root = None
    with (
        pytest.raises(ValueError, match="namespace differs"),
        owner.owned_execution_files({"Lib/synthetic.py": b"value = 1\n"}) as receipt,
    ):
        root = owner._execution_files_for_consumer(receipt)
        state = owner._LIVE[receipt]
        control = state.controls[""]
        owner._set_security(
            control.handle,
            owner._security_sddl(directory=True, frozen=False),
        )
        (root / "shadow.py").write_bytes(b"pass\n")
        owner._set_security(
            control.handle,
            owner._security_sddl(directory=True, frozen=True),
        )
        with pytest.raises(ValueError, match="namespace differs"):
            owner.execution_file_metadata(receipt)
    assert root is not None and not root.exists()


def test_snapshot_revalidates_security_and_cleans_after_dacl_drift():
    from scripts import qualification_execution_files as owner

    root = None
    with (
        pytest.raises(ValueError, match="security differs"),
        owner.owned_execution_files({"runner.py": b"pass\n"}) as receipt,
    ):
        root = owner._execution_files_for_consumer(receipt)
        control = owner._LIVE[receipt].controls[""]
        owner._set_security(
            control.handle,
            owner._security_sddl(directory=True, frozen=False),
        )
        with pytest.raises(ValueError, match="security differs"):
            owner.execution_file_metadata(receipt)
    assert root is not None and not root.exists()


def test_snapshot_preserves_primary_and_security_restore_failures_while_cleaning(monkeypatch):
    from scripts import qualification_execution_files as owner

    root = None
    calls = 0
    original = owner._set_security

    with (
        pytest.raises(BaseExceptionGroup) as captured,
        owner.owned_execution_files(
            {"Lib/synthetic.py": b"value = 1\n", "runner.py": b"pass\n"}
        ) as receipt,
    ):
        root = owner._execution_files_for_consumer(receipt)

        def fail_one_restore(handle, sddl):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("synthetic security restoration failure")
            return original(handle, sddl)

        monkeypatch.setattr(owner, "_set_security", fail_one_restore)
        raise RuntimeError("synthetic body failure")

    flattened = []
    pending = [captured.value]
    while pending:
        error = pending.pop()
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
        else:
            flattened.append(error)
    assert any(type(error) is RuntimeError for error in flattened)
    assert any(type(error) is OSError for error in flattened)
    assert calls == 4
    assert root is not None and not root.exists()


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


def test_snapshot_refuses_an_excessive_directory_set_before_creating_a_tree(monkeypatch):
    from scripts import qualification_execution_files as owner

    def unexpected(*args, **kwargs):
        pytest.fail("oversized directory set allocated an execution root")

    monkeypatch.setattr(owner.tempfile, "mkdtemp", unexpected)
    monkeypatch.setattr(owner, "_MAX_DIRECTORIES", 2)
    name = "one/two/three/runner.py"
    with (
        pytest.raises(ValueError, match="directory set exceeds"),
        owner.owned_execution_files({name: b"pass\n"}),
    ):
        pytest.fail("oversized directory set was admitted")


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

    original = owner._open_security_control

    def replace_written_bytes(path, *, directory):
        if not directory:
            path.write_bytes(b"synthetic altered copy\n")
        return original(path, directory=directory)

    monkeypatch.setattr(owner, "_open_security_control", replace_written_bytes)
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
