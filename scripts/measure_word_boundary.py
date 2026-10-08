"""Measure the word-boundary stop (#211) on generated Kokoro speech; print numbers only.

Needs the local profile (``uv sync --extra local``) and the pinned Kokoro assets,
which the provider downloads and verifies on first use. No audio is written.

    uv run --extra local python scripts/measure_word_boundary.py

prints, for today's cut, the shipped gap detector and the Kokoro-hinted variant:
how often a gap is found, the tail each adds, and a click rate from a model of
LiveKit's native source (one 10 ms block per timer tick, unknown phase, 0-16 ms
wake overshoot). It also times ``LiveKitRoomPeer.cancel_speech_chunk`` with real
asyncio timers and a fake ``AudioSource``.

Where a stop lands relative to words needs the model's own phoneme durations,
which the pinned ONNX computes but does not output. To get them, derive a copy
that also outputs them (it needs ``onnx``, which is not a project dependency),
then pass it in. The script checks that its audio is byte-identical to the
pinned model's before it labels anything:

    uv run --extra local --with onnx python scripts/measure_word_boundary.py \
        --derive-timed-model <dir>/kokoro-timed.onnx
    uv run --extra local python scripts/measure_word_boundary.py \
        --timed-model <dir>/kokoro-timed.onnx
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from hermes_realtime.livekit import LiveKitConnection, LiveKitRoomPeer
from hermes_realtime.providers.kokoro import _MODEL_ASSET, _VOICES_ASSET, KokoroSynthesizer
from hermes_realtime.speech import AudioFrame, SpeechChunk
from hermes_realtime.speech.word_boundary import (
    FADE_OUT_SECONDS,
    MAX_WORD_TAIL_SECONDS,
    fade_out,
    word_boundary_stop,
)

RATE = 48_000
BLOCK = RATE // 100
FRAME = RATE // 200
CAP = round(MAX_WORD_TAIL_SECONDS * RATE)
FADE = round(FADE_OUT_SECONDS * RATE)
HINT_SNAP = RATE * 40 // 1000
DURATION_TENSOR = "/encoder/Clip_output_0"
VOICES = ("bf_isabella", "af_heart", "am_michael")
BOUNDARY = set(" .,!?;:—-")
VOWELS = set("aeiouyɑɒɐəɛɪʊʌæɜɔᵻɚɝøœɨʉɵɘɤɯ")
CORPUS = (
    "Sure, I can help with that. Let me check the latest build status for you.",
    "The task finished successfully, and the report is ready whenever you want it.",
    "I started a background job to compare the two branches; it should take a minute.",
    "That depends on the configuration you chose earlier, so let me look it up.",
    "Here is a quick summary of what changed since yesterday's meeting.",
    "Stopping the deployment now would leave the database in a partial state.",
    "Okay, I'll keep listening. Tell me when you're ready to continue.",
    "Practically speaking, the expected throughput is about twelve thousand requests per second.",
    "Interrupt me any time if something sounds wrong or you want more detail.",
    "Unfortunately the upstream package hasn't published a compatible release yet.",
    "Absolutely, scheduling that for tomorrow afternoon at three thirty.",
    "The quick brown fox jumps over the lazy dog while the kettle boils.",
)


def derive_timed_model(source: Path, target: Path) -> None:
    onnx = importlib.import_module("onnx")
    model = onnx.load(str(source))
    graph = model.graph
    graph.node.append(
        onnx.helper.make_node(
            "Cast", [DURATION_TENSOR], ["duration"], to=onnx.TensorProto.INT64
        )
    )
    graph.output.append(
        onnx.helper.make_tensor_value_info("duration", onnx.TensorProto.INT64, None)
    )
    onnx.save(model, str(target))


def engine(model: Path, voices: Path) -> Any:
    runtime = importlib.import_module("onnxruntime")
    kokoro = importlib.import_module("kokoro_onnx")
    options = runtime.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = runtime.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return kokoro.Kokoro.from_session(session, str(voices))


def transport(samples: Any) -> np.ndarray:
    """The provider's own int16 conversion and 48 kHz upsampling."""

    clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = KokoroSynthesizer._upsample_pcm_for_transport((clipped * 32767.0).astype("<i2").tobytes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


def frame_rms(x: np.ndarray) -> np.ndarray:
    usable = len(x) // FRAME * FRAME
    return np.sqrt((x[:usable].reshape(-1, FRAME) ** 2).mean(1))


def audio_of(x: np.ndarray) -> AudioFrame:
    pcm = np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()
    return AudioFrame(pcm=pcm, sample_rate_hz=RATE, channels=1)


def timing_lag_frames(records: list[dict[str, Any]]) -> int:
    """The shift that best separates space tokens from vowels in energy."""

    best, best_contrast = 0, -np.inf
    for lag in range(-30, 31):
        space, vowel = [], []
        for record in records:
            db = 20 * np.log10(np.maximum(record["rms"], 1e-9) / np.percentile(record["rms"], 90))
            space.append(db[np.roll(record["space"], lag)])
            vowel.append(db[np.roll(record["vowel"], lag)])
        contrast = np.median(np.concatenate(vowel)) - np.median(np.concatenate(space))
        if contrast > best_contrast:
            best, best_contrast = lag, contrast
    return best


def word_labels(spoken: list[Any], frames: int, shift: int) -> np.ndarray:
    """B between words (+-2 frames), F after a word's last vowel, M before one."""

    labels = np.full(frames, "M", dtype="<U1")
    words: list[list[Any]] = [[]]
    gaps: list[tuple[int, int]] = []
    for token in spoken:
        start = max(0, round(token.start * RATE) - shift)
        end = max(0, round(token.end * RATE) - shift)
        if token.phoneme in BOUNDARY:
            gaps.append((start // FRAME, max(start // FRAME + 1, end // FRAME)))
            words.append([])
        else:
            words[-1].append((token.phoneme, start, end))
    for word in words:
        if not word:
            continue
        first, last = word[0][1] // FRAME, max(word[0][1] // FRAME + 1, word[-1][2] // FRAME)
        vowel_ends = [end for phoneme, _start, end in word if phoneme in VOWELS]
        final = max(first, vowel_ends[-1] // FRAME) if vowel_ends else first
        labels[first:last] = "M"
        labels[final:last] = "F"
    for start, end in gaps:
        labels[max(0, start - 2) : min(frames, end + 2)] = "B"
    return labels


def quietest(rms: np.ndarray, lo: int, hi: int) -> int:
    lo_frame, hi_frame = -(-lo // FRAME), hi // FRAME
    if hi_frame <= lo_frame:
        return lo
    return (lo_frame + int(np.argmin(rms[lo_frame:hi_frame]))) * FRAME


def synthesize(
    production: Any, timed: Any | None
) -> list[dict[str, Any]]:
    records = []
    for voice in VOICES:
        for text in CORPUS:
            audio, _rate = production.create(text, voice, speed=1.0, lang="en-gb")
            spoken: list[Any] = []
            if timed is not None:
                # create() minus its timing-driven pause insertion, which the
                # pinned model (no duration output) never performs.
                style, _phonemes, batches = timed._prepare(
                    text, voice, 1.0, "en-gb", False, 0.25, 0.1
                )
                timed_audio, spoken = timed._create_batches(batches, style, 1.0, True)
                if not np.array_equal(np.asarray(timed_audio), np.asarray(audio)):
                    raise SystemExit("timed model audio differs from the pinned model")
            x = transport(audio)
            rms = frame_rms(x)
            space = np.zeros(len(rms), dtype=bool)
            vowel = np.zeros(len(rms), dtype=bool)
            for token in spoken:
                a, b = int(token.start * RATE) // FRAME, int(token.end * RATE) // FRAME
                if token.phoneme in BOUNDARY:
                    space[a : max(a + 1, b)] = True
                elif token.phoneme in VOWELS:
                    vowel[a : max(a + 1, b)] = True
            padded = len(x) + (-len(x) % BLOCK)
            hints = KokoroSynthesizer._estimate_word_timings(text, padded)
            records.append(
                {
                    "x": x,
                    "rms": rms,
                    "spoken": spoken,
                    "space": space,
                    "vowel": vowel,
                    "ends": np.array([hint.end_sample for hint in hints[:-1]]),
                }
            )
    return records


def compare(records: list[dict[str, Any]], labelled: bool, points: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows: dict[str, dict[str, list[Any]]] = {
        name: {"label": [], "tail": [], "source": [], "click": []}
        for name in ("today", "gap", "hinted")
    }
    for record in records:
        x, rms = record["x"], record["rms"]
        audio = audio_of(x)
        reference = np.percentile(rms, 90)
        natural = np.percentile(np.abs(np.diff(x)), 99)
        total = len(x)
        for _ in range(points):
            p = int(rng.integers(0, total - RATE // 2)) // BLOCK * BLOCK
            if rms[p // FRAME] <= reference * 0.1:
                continue  # interruptions land during speech
            cap = min(total, p + CAP)
            gap = word_boundary_stop(audio, p)
            gap_source = "gap" if gap < cap or cap == total else "cap"
            ahead = record["ends"][(record["ends"] > p) & (record["ends"] < cap)]
            if len(ahead):
                end = int(ahead[0])
                hinted = quietest(rms, max(p, end - HINT_SNAP), min(cap, end + HINT_SNAP))
                hinted_source = "hint"
            else:
                hinted, hinted_source = gap, gap_source
            phase = rng.uniform(0, 0.01)
            overshoot = rng.uniform(0, 0.016)
            call = p / RATE + rng.uniform(-0.005, 0.005)
            for name, stop, source in (
                ("today", p, "cut"),
                ("gap", gap, gap_source),
                ("hinted", hinted, hinted_source),
            ):
                row = rows[name]
                row["tail"].append((stop - p) * 1000 / RATE)
                row["source"].append(source)
                if labelled:
                    labels = record["labels"]
                    row["label"].append(labels[min(len(labels) - 1, stop // FRAME)])
                wake = call + (stop - p) / RATE + (overshoot if stop > p else 0.0)
                ticks = int(np.floor((wake - phase) / 0.01)) + 1
                played = min(total - FADE - 1, max(1, ticks * BLOCK))
                estimate = min(total - FADE - 1, round(wake * RATE / BLOCK) * BLOCK)
                if name == "today":
                    step = abs(x[played - 1])
                else:
                    faded = fade_out(audio, estimate)
                    assert faded is not None
                    first = int.from_bytes(faded.pcm[:2], "little", signed=True)
                    step = abs(first - x[played - 1])
                row["click"].append(step / natural > 1)
    for name, row in rows.items():
        tail = np.array(row["tail"])
        source = np.array(row["source"])
        summary: dict[str, object] = {
            "method": name,
            "interruptions": len(tail),
            "tail_ms_p50": float(np.percentile(tail, 50)),
            "tail_ms_p90": float(np.percentile(tail, 90)),
            "tail_ms_mean": round(float(tail.mean()), 1),
            "click_pct": round(100 * float(np.mean(row["click"])), 1),
            "from": {k: round(100 * float(np.mean(source == k)), 1) for k in set(row["source"])},
        }
        if labelled:
            label = np.array(row["label"])
            for key, name_ in (("B", "between_words_pct"), ("F", "after_last_vowel_pct")):
                summary[name_] = round(100 * float(np.mean(label == key)), 1)
            summary["syllable_clipped_pct"] = round(100 * float(np.mean(label == "M")), 1)
        print(json.dumps(summary, sort_keys=True))


class _FakeSource:
    def clear_queue(self) -> None:
        return None

    async def capture_frame(self, frame: object) -> None:
        del frame

    async def wait_for_playout(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


async def peer_latency(records: list[dict[str, Any]], samples: int, seed: int) -> None:
    rng = random.Random(seed)
    elapsed = []
    for _ in range(samples):
        x = rng.choice(records)["x"]
        peer = LiveKitRoomPeer(
            LiveKitConnection("ws://127.0.0.1:7880", "measure", "synthetic-measurement-secret-32b"),
            identity="measure",
        )
        peer._connected = True
        peer._audio_source = cast(Any, _FakeSource())
        peer._publication_sid = "TR_measure"
        peer._active_speech_track_name = "measure"
        chunk = SpeechChunk(turn_id="t", chunk_id="c", text="measure", audio=audio_of(x))
        await peer.prepare_speech_chunk(chunk)
        await peer.publish_speech_chunk(chunk)
        await asyncio.sleep(rng.uniform(0.05, max(0.06, len(x) / RATE - 0.35)))
        started = time.perf_counter()
        await peer.cancel_speech_chunk(chunk, finish_word=True)
        elapsed.append((time.perf_counter() - started) * 1000)
    values = np.array(elapsed)
    print(
        json.dumps(
            {
                "peer_cancel_finish_word_ms_p50": round(float(np.median(values)), 1),
                "peer_cancel_finish_word_ms_p90": round(float(np.percentile(values, 90)), 1),
                "peer_cancel_finish_word_ms_max": round(float(values.max()), 1),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timed-model", type=Path)
    parser.add_argument("--derive-timed-model", type=Path)
    parser.add_argument("--points", type=int, default=120, help="interruptions per sentence")
    parser.add_argument("--latency-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=211)
    arguments = parser.parse_args()
    # The provider downloads and verifies the pinned assets on first use.
    synthesizer = KokoroSynthesizer()
    model = synthesizer._ensure_asset(_MODEL_ASSET)
    voices = synthesizer._ensure_asset(_VOICES_ASSET)
    if arguments.derive_timed_model is not None:
        derive_timed_model(model, arguments.derive_timed_model)
        return
    timed = engine(arguments.timed_model, voices) if arguments.timed_model else None
    if timed is not None and not timed.has_timings:
        raise SystemExit("the timed model does not output durations")
    records = synthesize(engine(model, voices), timed)
    print(json.dumps({"sentences": len(records), "audio_seconds": round(
        sum(len(r["x"]) for r in records) / RATE, 1)}))
    if timed is not None:
        lag = timing_lag_frames(records)
        print(json.dumps({"timing_lag_ms": lag * 1000 * FRAME // RATE}))
        for record in records:
            record["labels"] = word_labels(record["spoken"], len(record["rms"]), -lag * FRAME)
    compare(records, timed is not None, arguments.points, arguments.seed)
    asyncio.run(peer_latency(records, arguments.latency_samples, arguments.seed))


if __name__ == "__main__":
    main()
