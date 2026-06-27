/**
 * Loop Subsystem — barrel export + orchestrator.
 *
 * Six concurrent loop topologies engineered into the guardrail corpus:
 *   1 control       — feedback threshold controller (bounded, fail-floored)
 *   2 refinement    — recursive proposal refinement (monotone risk, fail-closed)
 *   3 agentic       — sense/decide/act with enforced truth-tag boundary
 *   4 consolidation — atomic episodic→semantic memory consolidation
 *   5 audit         — continuous lineage chain re-verification (sticky fail-closed)
 *   6 coordination  — verifier barrier join (no partial-quorum accept)
 */

export * from './loop_contract.js'
export {ControlLoop, globalControlLoop} from './control_loop.js'
export {RefinementLoop, globalRefinementLoop, type RefinementOutcome} from './refinement_loop.js'
export {AgenticLoop, globalAgenticLoop, type TruthTaggedResult} from './agentic_loop.js'
export {ConsolidationLoop, globalConsolidationLoop} from './consolidation_loop.js'
export {AuditLoop, globalAuditLoop} from './audit_loop.js'
export {
  CoordinationLoop,
  globalCoordinationLoop,
  type VerifierFn,
  type VerifierVote,
  type BarrierVerdict,
  type BarrierOutcome,
} from './coordination_loop.js'

import {globalControlLoop} from './control_loop.js'
import {globalConsolidationLoop} from './consolidation_loop.js'
import {globalAuditLoop} from './audit_loop.js'
import {type LoopTickResult} from './loop_contract.js'

/**
 * Drive all tick-based loops once and return their forensic results.
 * (refinement/agentic/coordination are request-driven, not tick-driven, so they
 *  are exercised at their call sites rather than here.)
 */
export function tickAllStatefulLoops(): LoopTickResult[] {
  const results: LoopTickResult[] = []
  results.push(globalControlLoop.tick())
  results.push(globalConsolidationLoop.tick())
  results.push(globalAuditLoop.tick())

  // Assert invariants after every coordinated tick — fail loud, never silent.
  globalControlLoop.assertInvariant()
  globalConsolidationLoop.assertInvariant()
  globalAuditLoop.assertInvariant()

  return results
}
