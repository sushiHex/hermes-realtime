"""Conservative accounting for speech confirmed as delivered."""

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from weakref import WeakSet

from .types import AudioFrame, SpeechChunk

_MAX_CLOSED_TURNS_LIMIT = 1024
_MAX_CLOSED_CHUNKS_LIMIT = 65_536
_MAX_CLOSED_TEXT_CHARS_LIMIT = 16_777_216
_MAX_LIVE_CHUNKS_LIMIT = 4096
_MAX_LIVE_PCM_BYTES_LIMIT = 268_435_456
_MAX_LIVE_TEXT_CHARS_LIMIT = 16_777_216


class SpeechDeliveryStage(StrEnum):
    """Conservative lifecycle stages for one synthesized speech chunk."""

    SYNTHESIZED = "synthesized"
    QUEUED = "queued"
    STARTED = "started"
    DELIVERED = "delivered"


class SpeechLedgerCapacityError(RuntimeError):
    """Raised before live speech state would exceed an explicit bound."""


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class SpeechDeliveryAdmission:
    """Opaque pre-playback capability bindable to one queued chunk."""


_BOUND_DELIVERY_ADMISSIONS: WeakSet[SpeechDeliveryAdmission] = WeakSet()
_BOUND_DELIVERY_ADMISSIONS_LOCK = Lock()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PlaybackReceipt:
    """Immutable proof that one specific chunk incarnation started playback."""

    receipt_id: int
    chunk: SpeechChunk

    def __post_init__(self) -> None:
        if type(self.receipt_id) is not int:
            raise TypeError("receipt_id must be an exact integer")
        if self.receipt_id <= 0:
            raise ValueError("receipt_id must be positive")
        if type(self.chunk) is not SpeechChunk:
            raise TypeError("receipt chunk must be an exact SpeechChunk")

    @property
    def turn_id(self) -> str:
        return self.chunk.turn_id

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass(frozen=True, slots=True)
class DeliveredSpeechConfirmation:
    """Ledger-issued capability proving complete delivery of one chunk."""


@dataclass(frozen=True, slots=True)
class SpeechTurnCounts:
    """Immutable, text-free terminal accounting for one live speech turn."""

    queued_chunk_count: int
    started_chunk_count: int
    transport_confirmed_full_count: int
    assistant_delivery_context_recorded: bool

    def __post_init__(self) -> None:
        counts = (
            self.queued_chunk_count,
            self.started_chunk_count,
            self.transport_confirmed_full_count,
        )
        if any(type(value) is not int for value in counts):
            raise TypeError("speech turn counts must be exact integers")
        if any(value < 0 for value in counts):
            raise ValueError("speech turn counts must be nonnegative")
        if type(self.assistant_delivery_context_recorded) is not bool:
            raise TypeError("delivery context flag must be an exact boolean")
        if not (
            self.transport_confirmed_full_count
            <= self.started_chunk_count
            <= self.queued_chunk_count
        ):
            raise ValueError("speech turn counts are inconsistent")


class DeliveredSpeechLedger:
    """Track queued chunks and retain only fully delivered text.

    Delivery is deliberately conservative: a chunk contributes text only after
    the playback transport confirms the complete chunk was delivered.
    """

    def __init__(
        self,
        *,
        max_closed_turns: int = 256,
        max_live_chunks: int = 256,
        max_live_pcm_bytes: int = 16_777_216,
        max_live_text_chars: int = 65_536,
        max_closed_chunks: int = 4096,
        max_closed_text_chars: int = 1_048_576,
    ) -> None:
        self._max_closed_turns = self._bounded_integer(
            max_closed_turns,
            "max_closed_turns",
            minimum=0,
            maximum=_MAX_CLOSED_TURNS_LIMIT,
        )
        self._max_live_chunks = self._bounded_integer(
            max_live_chunks,
            "max_live_chunks",
            minimum=1,
            maximum=_MAX_LIVE_CHUNKS_LIMIT,
        )
        self._max_live_pcm_bytes = self._bounded_integer(
            max_live_pcm_bytes,
            "max_live_pcm_bytes",
            minimum=1,
            maximum=_MAX_LIVE_PCM_BYTES_LIMIT,
        )
        self._max_live_text_chars = self._bounded_integer(
            max_live_text_chars,
            "max_live_text_chars",
            minimum=1,
            maximum=_MAX_LIVE_TEXT_CHARS_LIMIT,
        )
        self._max_closed_chunks = self._bounded_integer(
            max_closed_chunks,
            "max_closed_chunks",
            minimum=0,
            maximum=_MAX_CLOSED_CHUNKS_LIMIT,
        )
        self._max_closed_text_chars = self._bounded_integer(
            max_closed_text_chars,
            "max_closed_text_chars",
            minimum=0,
            maximum=_MAX_CLOSED_TEXT_CHARS_LIMIT,
        )
        self._live_pcm_bytes = 0
        self._live_text_chars = 0
        self._live_text_chars_by_key: dict[tuple[str, str], int] = {}
        self._chunks: dict[tuple[str, str], SpeechChunk] = {}
        self._pending_keys: list[tuple[str, str]] = []
        self._started_keys: set[tuple[str, str]] = set()
        self._next_receipt_id = 0
        self._started_receipt_keys: dict[int, tuple[str, str]] = {}
        # id(receipt) -> (receipt_id, receipt). The receipt is held *strongly* so
        # its address cannot be recycled by a later allocation while it is mapped;
        # a weak reference here let a collected receipt's address be reused and
        # cross-link two receipt IDs onto one object id. Entries are released by
        # mark_delivered and cancel_pending, exactly like _started_receipt_keys,
        # so retention stays bounded by the live started set.
        self._started_receipt_ids_by_object_id: dict[int, tuple[int, PlaybackReceipt]] = {}
        self._started_receipt_object_ids: dict[int, int] = {}
        self._seen_keys: set[tuple[str, str]] = set()
        self._delivered: dict[str, dict[int, str]] = defaultdict(dict)
        self._stage_history: dict[tuple[str, str], list[SpeechDeliveryStage]] = {}
        self._closed_turns: deque[str] = deque()
        self._closed_chunk_count = 0
        self._closed_text_chars = 0
        self._closed_chunk_counts_by_turn: dict[str, int] = {}
        self._closed_text_chars_by_turn: dict[str, int] = {}
        self._confirmations_by_object_id: dict[
            int,
            tuple[DeliveredSpeechConfirmation, SpeechDeliveryAdmission, str, str],
        ] = {}
        self._delivery_admissions_by_key: dict[
            tuple[str, str],
            SpeechDeliveryAdmission,
        ] = {}
        self._delivery_context_recorded_turns: set[str] = set()

    @property
    def retained_chunk_count(self) -> int:
        return len(self._chunks)

    def queue(
        self,
        chunk: SpeechChunk,
        *,
        admission: SpeechDeliveryAdmission | None = None,
    ) -> None:
        self._validate_chunk(chunk)
        if admission is not None and type(admission) is not SpeechDeliveryAdmission:
            raise TypeError("admission must be an exact SpeechDeliveryAdmission")
        chunk = self._copy_chunk(chunk)
        key = (chunk.turn_id, chunk.chunk_id)
        if key in self._seen_keys:
            raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
        if len(self._seen_keys) >= self._max_live_chunks:
            raise SpeechLedgerCapacityError("live speech chunk capacity exhausted")
        pcm_bytes = len(chunk.audio.pcm)
        if self._live_pcm_bytes + pcm_bytes > self._max_live_pcm_bytes:
            raise SpeechLedgerCapacityError("live speech PCM byte capacity exhausted")
        text_chars = len(chunk.text)
        if self._live_text_chars + text_chars > self._max_live_text_chars:
            raise SpeechLedgerCapacityError("live speech text capacity exhausted")
        if admission is not None:
            self._claim_delivery_admission(admission)
        self.begin_turn(chunk.turn_id)
        self._seen_keys.add(key)
        self._chunks[key] = chunk
        self._pending_keys.append(key)
        self._live_pcm_bytes += pcm_bytes
        self._live_text_chars += text_chars
        self._live_text_chars_by_key[key] = text_chars
        if admission is not None:
            self._delivery_admissions_by_key[key] = admission
        self._stage_history[key] = [
            SpeechDeliveryStage.SYNTHESIZED,
            SpeechDeliveryStage.QUEUED,
        ]

    def begin_turn(self, turn_id: str) -> None:
        """Clear retained history when a closed turn ID starts a new incarnation."""

        self._validate_lookup_identifier(turn_id, "turn_id")
        if turn_id in self._closed_turns:
            self._evict_closed_turn(turn_id)
        self._delivery_context_recorded_turns.discard(turn_id)

    def mark_delivered(self, receipt: PlaybackReceipt) -> SpeechChunk:
        receipt_id, key = self._authoritative_receipt_key(receipt)
        if key not in self._chunks or key not in self._pending_keys:
            raise KeyError(f"unknown chunk: {key[1]}")
        if key not in self._started_keys:
            raise RuntimeError(f"playback has not started: {key[1]}")
        self._pending_keys.remove(key)
        self._started_keys.discard(key)
        del self._started_receipt_keys[receipt_id]
        object_id = self._started_receipt_object_ids.pop(receipt_id)
        del self._started_receipt_ids_by_object_id[object_id]
        chunk = self._chunks.pop(key)
        self._delivery_admissions_by_key.pop(key, None)
        self._live_pcm_bytes -= len(chunk.audio.pcm)
        self._delivered[chunk.turn_id][receipt_id] = chunk.text
        self._stage_history[key].append(SpeechDeliveryStage.DELIVERED)
        return self._copy_chunk(chunk)

    def mark_delivered_confirmed(
        self,
        receipt: PlaybackReceipt,
    ) -> DeliveredSpeechConfirmation:
        _, key = self._authoritative_receipt_key(receipt)
        admission = self._delivery_admissions_by_key.get(key)
        if admission is None:
            raise RuntimeError("chunk has no pre-playback delivery admission")
        chunk = self.mark_delivered(receipt)
        confirmation = DeliveredSpeechConfirmation()
        self._confirmations_by_object_id[id(confirmation)] = (
            confirmation,
            admission,
            chunk.turn_id,
            chunk.text,
        )
        return confirmation

    def confirmed_text(
        self,
        confirmation: DeliveredSpeechConfirmation,
        admission: SpeechDeliveryAdmission,
    ) -> str:
        if type(confirmation) is not DeliveredSpeechConfirmation:
            raise TypeError("confirmation must be an exact DeliveredSpeechConfirmation")
        if type(admission) is not SpeechDeliveryAdmission:
            raise TypeError("admission must be an exact SpeechDeliveryAdmission")
        binding = self._confirmations_by_object_id.get(id(confirmation))
        if binding is None or binding[0] is not confirmation:
            raise KeyError("unknown delivery confirmation")
        if binding[1] is not admission:
            raise KeyError("delivery confirmation does not match admission")
        return binding[3]

    def consume_delivery_confirmation(
        self,
        confirmation: DeliveredSpeechConfirmation,
        admission: SpeechDeliveryAdmission,
    ) -> str:
        text = self.confirmed_text(confirmation, admission)
        binding = self._confirmations_by_object_id[id(confirmation)]
        self._delivery_context_recorded_turns.add(binding[2])
        del self._confirmations_by_object_id[id(confirmation)]
        return text

    def snapshot_turn_counts(self, turn_id: str) -> SpeechTurnCounts:
        """Freeze bounded live-turn counts before destructive close cleanup."""

        self._validate_lookup_identifier(turn_id, "turn_id")
        if turn_id in self._closed_turns:
            raise RuntimeError("cannot snapshot a closed speech turn")
        histories = tuple(
            history for key, history in self._stage_history.items() if key[0] == turn_id
        )
        return SpeechTurnCounts(
            queued_chunk_count=len(histories),
            started_chunk_count=sum(
                SpeechDeliveryStage.STARTED in history for history in histories
            ),
            transport_confirmed_full_count=sum(
                SpeechDeliveryStage.DELIVERED in history for history in histories
            ),
            assistant_delivery_context_recorded=(
                turn_id in self._delivery_context_recorded_turns
            ),
        )

    def mark_started(self, turn_id: str, chunk_id: str) -> PlaybackReceipt:
        self._validate_lookup_identifier(turn_id, "turn_id")
        self._validate_lookup_identifier(chunk_id, "chunk_id")
        key = (turn_id, chunk_id)
        if key not in self._chunks or key not in self._pending_keys:
            raise KeyError(f"unknown chunk: {chunk_id}")
        if key in self._started_keys:
            raise ValueError(f"chunk already started: {chunk_id}")
        self._started_keys.add(key)
        self._stage_history[key].append(SpeechDeliveryStage.STARTED)
        self._next_receipt_id += 1
        receipt = PlaybackReceipt(
            self._next_receipt_id,
            self._copy_chunk(self._chunks[key]),
        )
        self._started_receipt_keys[receipt.receipt_id] = key
        self._started_receipt_ids_by_object_id[id(receipt)] = (receipt.receipt_id, receipt)
        self._started_receipt_object_ids[receipt.receipt_id] = id(receipt)
        return receipt

    def cancel_pending(self, turn_id: str) -> tuple[SpeechChunk, ...]:
        self._validate_lookup_identifier(turn_id, "turn_id")
        cancelled_keys = [key for key in self._pending_keys if key[0] == turn_id]
        self._live_pcm_bytes -= sum(
            len(self._chunks[key].audio.pcm) for key in cancelled_keys
        )
        for key in cancelled_keys:
            self._live_text_chars -= self._live_text_chars_by_key.pop(key)
            self._delivery_admissions_by_key.pop(key, None)
        cancelled = tuple(
            self._copy_chunk(self._chunks.pop(key)) for key in cancelled_keys
        )
        self._pending_keys = [key for key in self._pending_keys if key[0] != turn_id]
        self._started_keys.difference_update(cancelled_keys)
        cancelled_key_set = set(cancelled_keys)
        cancelled_receipt_ids = {
            receipt_id
            for receipt_id, key in self._started_receipt_keys.items()
            if key in cancelled_key_set
        }
        for receipt_id in cancelled_receipt_ids:
            del self._started_receipt_keys[receipt_id]
            object_id = self._started_receipt_object_ids.pop(receipt_id)
            del self._started_receipt_ids_by_object_id[object_id]
        return cancelled

    def close_turn(self, turn_id: str) -> None:
        self._validate_lookup_identifier(turn_id, "turn_id")
        if any(key[0] == turn_id for key in self._pending_keys):
            raise RuntimeError("cannot close a turn with pending speech")
        if turn_id in self._closed_turns:
            return
        closed_keys = {key for key in self._seen_keys if key[0] == turn_id}
        for key in closed_keys:
            retained_chars = self._live_text_chars_by_key.pop(key, 0)
            self._live_text_chars -= retained_chars
        self._seen_keys.difference_update(closed_keys)
        self._remove_confirmations(turn_id)
        if turn_id in self._delivered or any(
            key[0] == turn_id for key in self._stage_history
        ):
            self._closed_turns.append(turn_id)
            chunk_count = sum(
                key[0] == turn_id for key in self._stage_history
            )
            text_chars = sum(
                len(text) for text in self._delivered.get(turn_id, {}).values()
            )
            self._closed_chunk_counts_by_turn[turn_id] = chunk_count
            self._closed_text_chars_by_turn[turn_id] = text_chars
            self._closed_chunk_count += chunk_count
            self._closed_text_chars += text_chars
        while (
            len(self._closed_turns) > self._max_closed_turns
            or self._closed_chunk_count > self._max_closed_chunks
            or self._closed_text_chars > self._max_closed_text_chars
        ):
            self._evict_closed_turn(self._closed_turns[0])

    def _authoritative_receipt_key(
        self,
        receipt: PlaybackReceipt,
    ) -> tuple[int, tuple[str, str]]:
        if type(receipt) is not PlaybackReceipt:
            raise TypeError("receipt must be an exact PlaybackReceipt")
        # Resolve by object identity, never by receipt.receipt_id: a caller can
        # mutate that field on an issued receipt, and delivery must stay bound to
        # the exact object the ledger issued.
        entry = self._started_receipt_ids_by_object_id.get(id(receipt))
        if entry is None:
            raise KeyError("unknown playback receipt")
        receipt_id, issued_receipt = entry
        if issued_receipt is not receipt:
            raise KeyError("unknown playback receipt")
        key = self._started_receipt_keys.get(receipt_id)
        if key is None:
            raise KeyError("unknown playback receipt")
        return receipt_id, key

    @staticmethod
    def _claim_delivery_admission(admission: SpeechDeliveryAdmission) -> None:
        with _BOUND_DELIVERY_ADMISSIONS_LOCK:
            if admission in _BOUND_DELIVERY_ADMISSIONS:
                raise ValueError("delivery admission is already bound")
            _BOUND_DELIVERY_ADMISSIONS.add(admission)

    @staticmethod
    def _bounded_integer(
        value: int,
        name: str,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        if type(value) is not int:
            raise TypeError(f"{name} must be an exact integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        if value > maximum:
            raise ValueError(f"{name} exceeds supported maximum {maximum}")
        return value

    @staticmethod
    def _validate_chunk(chunk: SpeechChunk) -> None:
        if type(chunk) is not SpeechChunk:
            raise TypeError("chunk must be an exact SpeechChunk")
        if type(chunk.turn_id) is not str:
            raise TypeError("chunk turn_id must be an exact built-in string")
        if type(chunk.chunk_id) is not str:
            raise TypeError("chunk chunk_id must be an exact built-in string")
        if type(chunk.text) is not str:
            raise TypeError("chunk text must be an exact built-in string")
        if type(chunk.audio) is not AudioFrame:
            raise TypeError("chunk audio must be an exact AudioFrame")
        if type(chunk.audio.pcm) is not bytes:
            raise TypeError("audio pcm must be exact bytes")
        if type(chunk.audio.sample_rate_hz) is not int:
            raise TypeError("audio sample_rate_hz must be an exact integer")
        if type(chunk.audio.channels) is not int:
            raise TypeError("audio channels must be an exact integer")
        if not chunk.audio.pcm:
            raise ValueError("audio pcm must not be empty")
        if chunk.audio.sample_rate_hz <= 0:
            raise ValueError("audio sample_rate_hz must be positive")
        if chunk.audio.channels <= 0:
            raise ValueError("audio channels must be positive")
        if len(chunk.audio.pcm) % (2 * chunk.audio.channels):
            raise ValueError("audio pcm contains incomplete sample groups")
        if not chunk.turn_id.strip():
            raise ValueError("chunk turn_id must not be blank")
        if not chunk.chunk_id.strip():
            raise ValueError("chunk chunk_id must not be blank")
        if len(chunk.turn_id) > 128 or len(chunk.chunk_id) > 128:
            raise ValueError("chunk identifier length exceeds 128 characters")
        if not chunk.text.strip():
            raise ValueError("chunk text must not be blank")

    @staticmethod
    def _validate_lookup_identifier(value: str, name: str) -> None:
        if type(value) is not str:
            raise TypeError(f"{name} must be an exact built-in string")
        if not value.strip() or len(value) > 128:
            raise ValueError(f"{name} is not a valid identifier")

    @staticmethod
    def _copy_chunk(chunk: SpeechChunk) -> SpeechChunk:
        return SpeechChunk(
            turn_id=chunk.turn_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            audio=AudioFrame(
                pcm=chunk.audio.pcm,
                sample_rate_hz=chunk.audio.sample_rate_hz,
                channels=chunk.audio.channels,
            ),
            word_timings=chunk.word_timings,
            timing_source=chunk.timing_source,
        )

    def _remove_confirmations(self, turn_id: str) -> None:
        for object_id, binding in tuple(self._confirmations_by_object_id.items()):
            if binding[2] == turn_id:
                del self._confirmations_by_object_id[object_id]

    def _evict_closed_turn(self, turn_id: str) -> None:
        self._closed_turns.remove(turn_id)
        self._closed_chunk_count -= self._closed_chunk_counts_by_turn.pop(turn_id, 0)
        self._closed_text_chars -= self._closed_text_chars_by_turn.pop(turn_id, 0)
        self._delivered.pop(turn_id, None)
        self._remove_stage_history(turn_id)
        self._remove_confirmations(turn_id)
        self._delivery_context_recorded_turns.discard(turn_id)

    def _remove_stage_history(self, turn_id: str) -> None:
        for key in tuple(self._stage_history):
            if key[0] == turn_id:
                del self._stage_history[key]

    def pending(self, turn_id: str | None = None) -> tuple[SpeechChunk, ...]:
        if turn_id is not None:
            self._validate_lookup_identifier(turn_id, "turn_id")
        chunks = (self._copy_chunk(self._chunks[key]) for key in self._pending_keys)
        if turn_id is None:
            return tuple(chunks)
        return tuple(chunk for chunk in chunks if chunk.turn_id == turn_id)

    def queued(self, turn_id: str) -> tuple[SpeechChunk, ...]:
        self._validate_lookup_identifier(turn_id, "turn_id")
        return tuple(
            self._copy_chunk(self._chunks[key])
            for key in self._pending_keys
            if key[0] == turn_id and key not in self._started_keys
        )

    def started(self, turn_id: str) -> tuple[SpeechChunk, ...]:
        self._validate_lookup_identifier(turn_id, "turn_id")
        return tuple(
            self._copy_chunk(self._chunks[key])
            for key in self._pending_keys
            if key[0] == turn_id and key in self._started_keys
        )

    def delivered_text(self, turn_id: str) -> str:
        self._validate_lookup_identifier(turn_id, "turn_id")
        delivered = self._delivered.get(turn_id, {})
        return "".join(text for _, text in sorted(delivered.items()))

    def stage_history(
        self, turn_id: str, chunk_id: str
    ) -> tuple[SpeechDeliveryStage, ...]:
        self._validate_lookup_identifier(turn_id, "turn_id")
        self._validate_lookup_identifier(chunk_id, "chunk_id")
        return tuple(self._stage_history.get((turn_id, chunk_id), ()))
