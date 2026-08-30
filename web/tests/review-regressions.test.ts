import { describe, expect, it } from "vitest";

import {
  ApprovalDecisionController,
  AssistantGenerationAuthority,
  parseSpeechTiming,
  SpeechMediaStartRegistry,
  SpeechTimingRegistry,
} from "../src/controller";

describe("approval decision concurrency", () => {
  it("refuses a concurrent decision instead of reusing an unsent sequence", async () => {
    // Regression: submit() computed next = sequence + 1 before awaiting and
    // committed only afterwards, so a double-click, or Approve immediately
    // followed by Reject, sent two decisions carrying the same number and the
    // server rejected the user's actual final choice as a duplicate.
    const attempts: Array<[number, string, string]> = [];
    let release!: () => void;
    const inFlight = new Promise<void>((resolve) => {
      release = resolve;
    });
    const approvals = new ApprovalDecisionController(
      async (sequence, approvalId, decision) => {
        attempts.push([sequence, approvalId, decision]);
        await inFlight;
      },
    );

    const first = approvals.submit("approval_0123456789abcdef", "approve");
    await Promise.resolve();
    expect(approvals.isSubmitting).toBe(true);

    await expect(
      approvals.submit("approval_0123456789abcdef", "reject"),
    ).rejects.toThrow("already in flight");

    release();
    await first;

    expect(attempts).toEqual([[1, "approval_0123456789abcdef", "approve"]]);
    expect(approvals.lastSequence).toBe(1);
    expect(approvals.isSubmitting).toBe(false);
  });

  it("decides two concurrent approvals on distinct serialised sequences", async () => {
    // Codex P1: concurrent background runs each hold a pending approval and the
    // server resolves them by exact id in any order
    // (test_api_session_resolves_concurrent_run_approvals_by_exact_id). A single
    // in-flight guard would block the second run's decision entirely.
    const attempts: Array<[number, string, string]> = [];
    let releaseFirst!: () => void;
    const firstSent = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const approvals = new ApprovalDecisionController(
      async (sequence, approvalId, decision) => {
        attempts.push([sequence, approvalId, decision]);
        if (approvalId === "approval_1111111111111111") await firstSent;
      },
    );

    const first = approvals.submit("approval_1111111111111111", "approve");
    await Promise.resolve();
    const second = approvals.submit("approval_2222222222222222", "reject");

    releaseFirst();
    await Promise.all([first, second]);

    expect(attempts).toEqual([
      [1, "approval_1111111111111111", "approve"],
      [2, "approval_2222222222222222", "reject"],
    ]);
    expect(approvals.lastSequence).toBe(2);
    expect(approvals.isSubmitting).toBe(false);
  });

  it("tracks in-flight state per approval rather than globally", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const approvals = new ApprovalDecisionController(async () => {
      await gate;
    });

    const pending = approvals.submit("approval_1111111111111111", "approve");
    await Promise.resolve();

    expect(approvals.isSubmittingApproval("approval_1111111111111111")).toBe(true);
    expect(approvals.isSubmittingApproval("approval_2222222222222222")).toBe(false);

    release();
    await pending;
    expect(approvals.isSubmittingApproval("approval_1111111111111111")).toBe(false);
  });

  it("still allows a retry after a failed decision without consuming a sequence", async () => {
    const attempts: Array<[number, string, string]> = [];
    let fail = true;
    const approvals = new ApprovalDecisionController(
      async (sequence, approvalId, decision) => {
        attempts.push([sequence, approvalId, decision]);
        if (fail) throw new Error("temporary failure");
      },
    );

    await expect(
      approvals.submit("approval_0123456789abcdef", "approve"),
    ).rejects.toThrow("temporary failure");
    expect(approvals.isSubmitting).toBe(false);

    fail = false;
    await approvals.submit("approval_0123456789abcdef", "reject");

    expect(attempts).toEqual([
      [1, "approval_0123456789abcdef", "approve"],
      [1, "approval_0123456789abcdef", "reject"],
    ]);
  });
});

describe("assistant generation authority retention", () => {
  it("keeps a streaming turn resident against LRU eviction", () => {
    // Regression: Map insertion order is the LRU key but was refreshed only on
    // create/replace. A turn still receiving chunks aged out as the "oldest"
    // entry, after which a replayed lower generation read as "create" instead
    // of "reject" and the stale generation rendered as a fresh turn.
    const authority = new AssistantGenerationAuthority(2);

    expect(authority.disposition("turn_live", 4)).toBe("create");
    expect(authority.disposition("turn_other", 1)).toBe("create");
    expect(authority.disposition("turn_live", 4)).toBe("current");
    expect(authority.disposition("turn_third", 1)).toBe("create");

    expect(authority.disposition("turn_live", 2)).toBe("reject");
  });
});

describe("speech timing stream isolation", () => {
  it("keeps per-stream timing separate when one chunk is re-admitted", () => {
    // Regression: the composite key omitted streamId, so re-admitting the same
    // turn/generation/chunk under a resumed stream rewrote the entry that
    // #latestByStream still pointed at for the previous stream.
    const base = {
      turnId: "resume_1",
      presentationTurnId: "turn_001",
      turnGeneration: 7,
      chunkId: "speech_chunk_3",
      segmentId: "chunk_3",
      sampleRate: 48000,
      timingSource: "estimated",
      timings: "0,5,0,12000;6,11,12000,24000",
    };
    const first = parseSpeechTiming({ ...base, streamId: "speech_a" });
    const second = parseSpeechTiming({ ...base, streamId: "speech_b" });
    if (first === null || second === null) {
      throw new Error("timing unexpectedly absent");
    }
    const registry = new SpeechTimingRegistry(32);

    registry.admit(first);
    registry.admit(second);

    expect(registry.get("speech_a")).toBe(first);
    expect(registry.get("speech_b")).toBe(second);
  });
});

describe("speech media-start namespaces", () => {
  it("bounds stream attachments and passage rebases independently", () => {
    // Regression: both namespaces shared one map and one eviction budget, so a
    // turn with more passages than the bound evicted the stream attachment that
    // resolvePlaying and observe still needed.
    const starts = new SpeechMediaStartRegistry(2);

    starts.attach("stream_1", 0);
    starts.beginPassage("passage_1", 1);
    starts.beginPassage("passage_2", 2);
    starts.beginPassage("passage_3", 3);

    expect(starts.resolvePlaying("stream_1", 0.5)).toBe(0.5);
    expect(starts.getPassage("passage_1")).toBeUndefined();
    expect(starts.getPassage("passage_3")).toBeUndefined();
  });

  it("does not let a passage identifier clobber a stream attachment", () => {
    const starts = new SpeechMediaStartRegistry(8);

    starts.attach("shared_id", 4);
    starts.beginPassage("shared_id", 9);

    expect(starts.resolvePlaying("shared_id", 4.25)).toBe(4.25);
    expect(starts.observePassage("shared_id", 9.5)).toBe(9.5);
  });
});
