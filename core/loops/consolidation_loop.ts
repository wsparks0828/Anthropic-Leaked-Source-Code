/**
 * Loop 4 — Memory Consolidation Loop.
 *
 * Periodically replays accumulated episodic updates into the semantic layer
 * under a SINGLE atomic transaction, then records one lineage entry for the
 * whole consolidation. All-or-nothing: a failure rolls back every update.
 *
 * INVARIANTS:
 *   I1: a consolidation either applies ALL staged updates or NONE (atomic)
 *   I2: every successful consolidation writes exactly one lineage record
 *   I3: the episodic stage is cleared only on successful commit
 */

import {createMemoryTransaction, globalMemoryTransactionManager} from '../memory_transaction.js'
import {globalLineageAuditor} from '../lineage_auditor.js'
import {type Loop, type LoopTickResult, nowNs} from './loop_contract.js'

interface EpisodicUpdate {
  key: string
  value: unknown
}

export class ConsolidationLoop implements Loop {
  readonly id = 'consolidation' as const

  // Minimal in-memory semantic/episodic/graph layers (Map-backed) for the loop.
  private readonly layers: {
    semantic: Map<string, unknown>
    episodic: Map<string, unknown>
    graph: Map<string, unknown>
  } = {semantic: new Map(), episodic: new Map(), graph: new Map()}

  private staged: EpisodicUpdate[] = []
  private tickCount = 0

  /** Stage an episodic observation for the next consolidation. */
  stage(key: string, value: unknown): void {
    this.staged.push({key, value})
  }

  semanticSize(): number {
    return this.layers.semantic.size
  }

  tick(): LoopTickResult {
    this.tickCount++
    if (this.staged.length === 0) {
      return this.result(false, false, 'nothing staged', 0)
    }

    const toApply = this.staged
    const txn = createMemoryTransaction()
    for (const u of toApply) {
      txn.addUpdate('semantic', u.key, u.value)
    }

    const commit = globalMemoryTransactionManager.commitTransaction(txn, this.layers)
    if (!commit) {
      // I1: rollback already performed inside transaction; episodic NOT cleared (I3).
      return this.result(false, true, `consolidation rolled back (${toApply.length} updates)`, 0.2)
    }

    // I2: one lineage record for the whole consolidation.
    const rec = globalLineageAuditor.addRecord({
      verificationId: `consolidate_${this.tickCount}_${Date.now()}`,
      timestamp: nowNs(),
      decision: 'accept',
      rubricScore: 1,
      truthVerdict: 'consolidation',
      lineage: {
        who: 'consolidation_loop',
        what: {before: {staged: toApply.length}, after: {semantic: this.layers.semantic.size}},
        when: nowNs(),
        auth: 'memory_consolidation',
      },
    })

    // I3: clear episodic stage only after successful commit.
    this.staged = []

    return this.result(true, false, `consolidated ${toApply.length} updates`, 0.05, rec.chainHash)
  }

  assertInvariant(): void {
    // Duplicate keys within a single staged batch would violate transaction atomicity guarantees.
    const seen = new Set<string>()
    for (const u of this.staged) {
      if (seen.has(u.key)) {
        throw new Error(`ConsolidationLoop: duplicate staged key ${u.key} (would break atomic commit)`)
      }
      seen.add(u.key)
    }
  }

  reset(): void {
    this.layers.semantic.clear()
    this.layers.episodic.clear()
    this.layers.graph.clear()
    this.staged = []
    this.tickCount = 0
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

export const globalConsolidationLoop = new ConsolidationLoop()
