/**
 * Loop 1 — Control (feedback) Loop.
 *
 * Closed-loop controller for the rubric acceptance threshold.
 * Measures acceptance-rate error against a target band and nudges the threshold
 * by a BOUNDED step, never crossing the safety floor.
 *
 * INVARIANTS (assertInvariant enforces):
 *   I1: SAFETY_FLOOR <= threshold <= CEILING at all times
 *   I2: |Δthreshold| per tick <= MAX_STEP
 *   I3: every threshold mutation writes a lineage record (who=control_loop)
 *
 * Blast radius: a single scalar threshold in [floor, ceiling]. Cannot disable
 * the gate (floor > 0) and cannot accept-all (ceiling < 1).
 */

import {globalLineageAuditor} from '../lineage_auditor.js'
import {type Loop, type LoopTickResult, clamp, nowNs} from './loop_contract.js'

export class ControlLoop implements Loop {
  readonly id = 'control' as const

  private threshold: number
  private readonly target: number // desired acceptance rate (e.g. 0.90)
  private readonly band: number // tolerance around target before acting
  private readonly maxStep = 0.05
  private readonly floor = 0.45 // safety floor — NEVER below
  private readonly ceiling = 0.85
  private accepts = 0
  private quarantines = 0
  private tickCount = 0

  constructor(initialThreshold = 0.55, target = 0.9, band = 0.05) {
    this.threshold = clamp(initialThreshold, this.floor, this.ceiling)
    this.target = target
    this.band = band
  }

  /** Feed an observed decision into the controller's window. */
  observe(decision: 'accept' | 'quarantine'): void {
    if (decision === 'accept') this.accepts++
    else this.quarantines++
  }

  getThreshold(): number {
    return this.threshold
  }

  tick(): LoopTickResult {
    this.tickCount++
    const total = this.accepts + this.quarantines
    const before = this.threshold

    if (total === 0) {
      return this.result(false, false, 'no observations; threshold held', 0)
    }

    const acceptanceRate = this.accepts / total
    const error = this.target - acceptanceRate

    // Inside the dead-band: do nothing (prevents oscillation).
    if (Math.abs(error) <= this.band) {
      this.windowReset()
      return this.result(false, false, `within band (acc=${acceptanceRate.toFixed(2)})`, 0)
    }

    // error > 0 → accepting too little → LOWER threshold (toward floor).
    // error < 0 → accepting too much → RAISE threshold (toward ceiling).
    const rawStep = clamp(error, -this.maxStep, this.maxStep)
    const next = clamp(before - rawStep, this.floor, this.ceiling)
    this.threshold = next
    this.windowReset()

    const delta = next - before
    // I2 guard (defensive; clamp already guarantees it)
    if (Math.abs(delta) > this.maxStep + 1e-9) {
      throw new Error(`ControlLoop I2 violated: |Δ|=${Math.abs(delta)} > ${this.maxStep}`)
    }

    if (delta === 0) {
      return this.result(false, false, `at bound, no change (acc=${acceptanceRate.toFixed(2)})`, 0)
    }

    const rec = globalLineageAuditor.addRecord({
      verificationId: `control_${this.tickCount}_${Date.now()}`,
      timestamp: nowNs(),
      decision: 'accept',
      rubricScore: next,
      truthVerdict: 'control_adjustment',
      lineage: {
        who: 'control_loop',
        what: {before: {threshold: before}, after: {threshold: next, acceptanceRate}},
        when: nowNs(),
        auth: 'control_signal',
      },
    })

    return this.result(true, false, `threshold ${before.toFixed(3)}→${next.toFixed(3)}`, 0.05, rec.chainHash)
  }

  assertInvariant(): void {
    if (this.threshold < this.floor || this.threshold > this.ceiling) {
      throw new Error(`ControlLoop I1 violated: threshold ${this.threshold} outside [${this.floor},${this.ceiling}]`)
    }
  }

  reset(): void {
    this.threshold = clamp(0.55, this.floor, this.ceiling)
    this.accepts = 0
    this.quarantines = 0
    this.tickCount = 0
  }

  private windowReset(): void {
    this.accepts = 0
    this.quarantines = 0
  }

  private result(
    mutated: boolean,
    failClosed: boolean,
    detail: string,
    residualRisk: number,
    lineageRecordId?: string,
  ): LoopTickResult {
    return {loopId: this.id, tick: this.tickCount, mutated, failClosed, detail, residualRisk, lineageRecordId}
  }
}

export const globalControlLoop = new ControlLoop()
