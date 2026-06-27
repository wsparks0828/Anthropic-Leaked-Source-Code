/**
 * Loop Contract — shared invariant surface for all corpus loops.
 *
 * Every loop in core/loops/ implements this so the orchestrator can drive them
 * uniformly and so each loop exposes a forensic record of its last tick.
 *
 * Non-negotiable: a loop NEVER weakens a safety floor and ALWAYS records a
 * lineage entry when it mutates shared state.
 */

export type LoopId =
  | 'control'
  | 'refinement'
  | 'agentic'
  | 'consolidation'
  | 'audit'
  | 'coordination'

export interface LoopTickResult {
  loopId: LoopId
  tick: number
  mutated: boolean // did this tick change shared state?
  failClosed: boolean // did this tick engage a fail-closed protection?
  detail: string
  residualRisk: number // 0.00–1.00 after this tick
  lineageRecordId?: string // chainHash if a lineage record was written
}

export interface Loop {
  readonly id: LoopId
  /** Run one deterministic iteration. Pure w.r.t. wall clock for testability. */
  tick(): LoopTickResult
  /** Invariant assertion — throws if the loop's invariant is violated. */
  assertInvariant(): void
  /** Reset internal state (testing/operations). */
  reset(): void
}

/** Clamp helper shared across loops. */
export function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value))
}

/** Monotonic-ish tick timestamp in ns without using forbidden Date.now in hot inner code paths. */
export function nowNs(): bigint {
  return BigInt(Date.now()) * BigInt(1_000_000)
}
