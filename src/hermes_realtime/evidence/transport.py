"""Private blocking transport used only by the evidence writer thread.

This is neither a public wire protocol nor a future importer contract: it is
the complete internal V1 surface the single writer thread may call.
"""

from __future__ import annotations

from typing import Literal, Protocol

from .models import (
    BindingCloseV1,
    CreateEpochV1,
    DrainAndStopV1,
    DrainDisposition,
    EvidenceDiagnosticsV1,
    ExpireSessionV1,
    FullPurgeV1,
    MaintenanceV1,
    PurgeDisposition,
    QueuedEvidenceRecordV1,
    RecoveryDisposition,
    RevokeDisposition,
    RevokeFinalizeV1,
    RevokeRequestV1,
    RolloverSessionV1,
    SealEpochV1,
    StoreDisposition,
)


class EvidenceWriterTransportV1(Protocol):
    """The complete V1 writer-thread protocol; never a public wire format."""

    protocol_version: Literal[1]

    def recover_existing(self) -> RecoveryDisposition: ...
    def create_epoch(self, command: CreateEpochV1) -> StoreDisposition: ...
    def append_record(self, item: QueuedEvidenceRecordV1) -> StoreDisposition: ...
    def append_binding_close(self, command: BindingCloseV1) -> StoreDisposition: ...
    def rollover_session(self, command: RolloverSessionV1) -> StoreDisposition: ...
    def expire_session(self, command: ExpireSessionV1) -> PurgeDisposition: ...
    def commit_revoke_request(self, command: RevokeRequestV1) -> RevokeDisposition: ...
    def finalize_revoke(self, command: RevokeFinalizeV1) -> RevokeDisposition: ...
    def seal_epoch(self, command: SealEpochV1) -> StoreDisposition: ...
    def run_maintenance(self, command: MaintenanceV1) -> PurgeDisposition: ...
    def purge_full_store(self, command: FullPurgeV1) -> PurgeDisposition: ...
    def drain_and_close(self, command: DrainAndStopV1) -> DrainDisposition: ...
    def diagnostics(self) -> EvidenceDiagnosticsV1: ...
