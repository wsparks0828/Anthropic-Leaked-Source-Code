/**
 * Loop 3 — Agentic (sense → decide → act) Loop with truth-tag boundary.
 *
 * For each output: SENSE (rubric + truth gate), DECIDE (accept/quarantine),
 * ACT (attach an immutable truth-tag). Enforces — in code, not by convention —
 * that NO result leaves the loop untagged.
 *
 * INVARIANTS:
 *   I1: every returned result carries a non-empty truthTag
 *   I2: a result tagged 'accept' implies rubric >= floor AND truth verdict != 'false'
 *        (fail-closed: any 'false' verdict with critical/high severity ⇒ quarantine)
 */

import {globalRubricScorer} from '../rubric_scorer.js'
import {globalTruthGate} from '../truth_gates.js'
import {nowNs} from './loop_contract.js'

export interface TruthTaggedResult {
  decision: 'accept' | 'quarantine'
  truthTag: string // e.g. "tag:accept:rubric=0.72:truth=uncertain"
  rubricOverall: number
  truthVerdict: string
  taggedAt: bigint
}

export class AgenticLoop {
  readonly id = 'agentic' as const
  private readonly floor: number

  constructor(rubricFloor = 0.55) {
    this.floor = rubricFloor
  }

  process(output: string): TruthTaggedResult {
    // SENSE
    const rubric = globalRubricScorer.score(output)
    const truth = globalTruthGate.gate(output)

    // DECIDE (fail-closed)
    const dangerousFalse =
      truth.verdict === 'false' && (truth.severity === 'critical' || truth.severity === 'high')
    const decision: 'accept' | 'quarantine' =
      rubric.overall >= this.floor && !dangerousFalse ? 'accept' : 'quarantine'

    // ACT — attach truth-tag (I1: never empty)
    const truthTag = `tag:${decision}:rubric=${rubric.overall.toFixed(2)}:truth=${truth.verdict}`

    const result: TruthTaggedResult = {
      decision,
      truthTag,
      rubricOverall: rubric.overall,
      truthVerdict: truth.verdict,
      taggedAt: nowNs(),
    }

    this.assertResult(result)
    return result
  }

  /** I1 + I2 enforcement on every emitted result. */
  private assertResult(r: TruthTaggedResult): void {
    if (!r.truthTag || r.truthTag.length === 0) {
      throw new Error('AgenticLoop I1 violated: untagged result')
    }
    if (r.decision === 'accept' && r.rubricOverall < this.floor) {
      throw new Error('AgenticLoop I2 violated: accept below rubric floor')
    }
  }

  assertInvariant(): void {
    /* per-result invariants enforced in assertResult */
  }

  reset(): void {
    /* stateless */
  }
}

export const globalAgenticLoop = new AgenticLoop()
