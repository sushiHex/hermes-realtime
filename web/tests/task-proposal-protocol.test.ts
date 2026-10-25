import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { parseEventBatch, parseTaskProposal, TASK_PROPOSAL_OBJECTIVE_MAX_CHARS } from "../src/protocol";

const data = {proposalId:"proposal_12345678", objective:"Exact objective", generation:1,
  status:"pending", reason:null};

describe("task proposal public contract", () => {
  it("matches the authoritative host objective cap exactly", () => {
    const source = readFileSync(new URL("../../src/hermes_realtime/conversation/task_proposals.py", import.meta.url), "utf8");
    const bound = source.match(/^MAX_TASK_PROPOSAL_OBJECTIVE_CHARS = (\d+)$/m);
    expect(bound).not.toBeNull();
    expect(TASK_PROPOSAL_OBJECTIVE_MAX_CHARS).toBe(Number(bound![1]));
  });
  it("accepts the Unicode boundary and strict public event", () => {
    const boundary = {...data, objective:"🧪".repeat(1024)};
    expect(parseTaskProposal(boundary)).toEqual(boundary);
    const batch = {version:1, events:[{sequence:1, kind:"task_proposal", monotonicMs:0, data:boundary}]};
    expect(parseEventBatch(batch)).toEqual(batch);
  });
  it.each([
    {extra:true}, {proposalId:"invalid"}, {proposalId:1}, {objective:1}, {objective:" "},
    {objective:"x".repeat(1025)}, {generation:true}, {generation:0}, {generation:2**53},
    {status:"active"}, {reason:"arbitrary"},
  ])("refuses malformed field %j", change => {
    expect(() => parseTaskProposal({...data, ...change})).toThrow();
    expect(parseTaskProposal(data)).toEqual(data);
  });
});
