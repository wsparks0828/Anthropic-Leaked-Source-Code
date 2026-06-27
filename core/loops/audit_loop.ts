/**
 * Loop 5 — Audit / Verification Loop (continuous, fail-closed).
 *
 * Re-walks the lineage chain every tick and verifies cryptographic integrity.
 * On ANY broken link it engages quarantine mode (fail-closed) and records an
 * anomaly — it never silently accepts a broken chain.
 *
 * INVARIANTS:
 *   I1: chain invalid ⇒ quarantineMode == true AND an anomaly was recorded
 *   I2: quarantineMode is sticky until an explicit operator reset()
 *        (a single good tick must not silently clear a prior detected breach)
 *
 * Runtime: start(intervalMs) drives tick() on a timer; tick() is also callable
 * directly for deterministic tests (no wall-clock dependence in the assertion).
 */

import {globalLineageAuditor} from '../lineage_auditor.js'
import {globalHealthMonitor} from '../guardrail_health.js'
import {type Loop, type LoopTickResult} from './loop_contract.js'

export class AuditLoop implements Loop {
  readonly id = 'audit' as const

  private quarantineMode = false
  private tickCount = 0
  private timer: ReturnType<typeof setInterval> | null = null
  private lastBrokenAt?: string

  isQuarantined(): boolean {
    return this.quarantineMode
  }

  getLastBreak(): string | undefined {
    return this.lastBrokenAt
  }

  tick(): LoopTickResult {
    this.tickCount++
    const verification = globalLineageAuditor.verifyLineageChain()

    if (!verification.valid) {
      // FAIL-CLOSED. Sticky quarantine + anomaly.
      this.quarantineMode = true
      this.lastBrokenAt = verification.brokenAt
      globalHealthMonitor.recordAnomaly()
      return this.result(
        true,
        true,
        `chain INVALID at ${verification.brokenAt ?? 'unknown'} (${verification.violations?.length ?? 0} violations)`,
        1.0,
      )
    }

    // Valid this tick — but quarantine is sticky (I2): do NOT auto-clear.
    return this.result(
      false,
      this.quarantineMode,
      this.quarantineMode ? 'chain valid but prior breach not yet cleared' : 'chain valid',
      this.quarantineMode ? 1.0 : 0.0,
    )
  }

  /** Drive the loop on an interval (production). Returns a stop function. */
  start(intervalMs = 30_000): () => void {
    if (this.timer) return () => this.stop()
    this.timer = setInterval(() => {
      this.tick()
    }, intervalMs)
    return () => this.stop()
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer)
      this.timer = null
    }
  }

  assertInvariant(): void {
    // If we ever recorded a break, quarantine must still be engaged (I2 stickiness).
    if (this.lastBrokenAt !== undefined && !this.quarantineMode) {
      throw new Error('AuditLoop I2 violated: detected breach but quarantine not engaged')
    }
  }

  reset(): void {
    this.stop()
    this.quarantineMode = false
    this.lastBrokenAt = undefined
    this.tickCount = 0
  }

  private result(mutated: boolean, failClosed: boolean, detail: string, residualRisk: number): LoopTickResult {
    return {loopId: this.id, tick: this.tickCount, mutated, failClosed, detail, residualRisk}
  }
}

export const globalAuditLoop = new AuditLoop()
