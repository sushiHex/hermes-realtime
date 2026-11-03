from __future__ import annotations

import pytest

from scripts import preview_readiness as readiness


@pytest.fixture
def process_tree(monkeypatch):
    monkeypatch.setattr(readiness, "_role_members", lambda root: {root, root + 1})


@pytest.mark.usefixtures("process_tree")
@pytest.mark.parametrize("pid", [100, 101])
def test_gateway_health_accepts_owned_role(pid):
    assert readiness.require_gateway_health_owner(
        {"pid": pid, "status": "degraded"}, root_pid=100, members={100, 101, 200},
    ) == pid


@pytest.mark.usefixtures("process_tree")
@pytest.mark.parametrize("pid,members", [(True, {1, 2}), (200, {100, 101, 200}),
                                         (101, {100}), (101, {101}), (0, {0, 100})])
def test_gateway_health_refuses_wrong_role_or_containment(pid, members):
    with pytest.raises(readiness.PreviewReadinessRefusal):
        readiness.require_gateway_health_owner(
            {"pid": pid}, root_pid=1 if type(pid) is bool else 100, members=members,
        )


@pytest.mark.usefixtures("process_tree")
@pytest.mark.parametrize("receipt", [
    {"version": True, "ready": True, "pid": 101},
    {"version": 1, "ready": 1, "pid": 101},
    {"version": 1, "ready": False, "pid": 101},
    {"version": 1, "ready": True, "pid": True},
    {"version": 1, "ready": True, "pid": 101, "extra": True},
    {"version": 1, "ready": True},
])
def test_host_receipt_requires_exact_shape(receipt):
    with pytest.raises(readiness.PreviewReadinessRefusal):
        readiness.require_host_readiness(receipt, root_pid=100, members={100, 101})


def test_role_members_uses_existing_live_descendant_snapshot(monkeypatch):
    from scripts import preview_cutover, rehearse_desktop_mvp

    assert preview_cutover.ROOT.is_dir()
    monkeypatch.setattr(rehearse_desktop_mvp, "process_table", lambda: {
        100: (1, "owner"), 101: (100, "child"), 102: (101, "grandchild"), 200: (1, "foreign"),
    })
    assert readiness._role_members(100) == {100, 101, 102}
    assert readiness._role_members(999) == set()


def test_role_snapshot_failure_is_refused(monkeypatch):
    from scripts import rehearse_desktop_mvp

    def unavailable():
        raise OSError("synthetic private observation error")

    monkeypatch.setattr(rehearse_desktop_mvp, "process_table", unavailable)
    with pytest.raises(readiness.PreviewReadinessRefusal) as error:
        readiness._role_members(100)
    assert error.value.category == "process_observation_unavailable"


@pytest.mark.usefixtures("process_tree")
def test_host_receipt_binds_owned_descendant():
    receipt = {"version": 1, "ready": True, "pid": 101}
    assert readiness.require_host_readiness(receipt, root_pid=100, members={100, 101}) == 101
    with pytest.raises(readiness.PreviewReadinessRefusal):
        readiness.require_host_readiness(receipt, root_pid=200, members={100, 101, 200})


def test_tcp_accepting_closes_probe(monkeypatch):
    calls = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

    def connect(address, timeout):
        calls.append((address, timeout))
        return Connection()

    monkeypatch.setattr(readiness.socket, "create_connection", connect)
    assert readiness.tcp_accepting(12345) is True
    assert calls == [(("127.0.0.1", 12345), 3.0), "closed"]


@pytest.mark.parametrize("error,refused", [(ConnectionRefusedError(), False),
                                          (TimeoutError(), True), (OSError(), True)])
def test_tcp_observation_only_connection_refused_means_absent(monkeypatch, error, refused):
    def connect(*args, **kwargs):
        raise error

    monkeypatch.setattr(readiness.socket, "create_connection", connect)
    if refused:
        with pytest.raises(readiness.PreviewReadinessRefusal):
            readiness.tcp_accepting(12345)
    else:
        assert readiness.tcp_accepting(12345) is False


@pytest.mark.parametrize("port", [True, 0, 65536, "12345"])
def test_tcp_invalid_port_never_connects(monkeypatch, port):
    monkeypatch.setattr(
        readiness.socket, "create_connection", lambda *a, **k: pytest.fail("connected"),
    )
    with pytest.raises(readiness.PreviewReadinessRefusal):
        readiness.tcp_accepting(port)
