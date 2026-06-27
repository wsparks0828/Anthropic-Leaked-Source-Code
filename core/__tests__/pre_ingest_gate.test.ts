/**
 * THOTH Pre-Ingest Gate — adversarial invariant tests.
 *
 * Verifies provenance fail-closed (I2), no-silent-reject (I1), and that the
 * routing decisions actually discriminate signal & tier rather than rubber-stamp.
 */

import {describe, it, expect, beforeEach} from 'bun:test'
import {PreIngestGate, type SourceTier} from '../thoth/pre_ingest_gate.js'
import {globalLineageAuditor} from '../lineage_auditor.js'

const HIGH_SIGNAL =
  'Transformer attention computes scaled dot-product over query, key, and value matrices; ' +
  'specifically softmax(QKᵀ/√d)V. According to Vaswani et al. (2017), this enables parallel ' +
  'sequence modeling. For example, with d=64 the scaling factor is 8, which stabilizes gradients.'

const LOW_SIGNAL = 'idk maybe. stuff. yeah.'

beforeEach(() => {
  globalLineageAuditor.reset()
})

describe('THOTH Pre-Ingest Gate', () => {
  it('I2 (fail-closed provenance): tier-4 source can never be accepted', () => {
    const gate = new PreIngestGate()
    const v = gate.evaluate(HIGH_SIGNAL, {sourceId: 's_untrusted', sourceTier: 4, query: 'transformer attention'})
    expect(v.decision).not.toBe('accept')
    expect(v.sourceTier).toBe(4)
  })

  it('unknown provenance defaults to lowest trust (tier 4)', () => {
    const gate = new PreIngestGate()
    const v = gate.evaluate(HIGH_SIGNAL, {sourceId: 's_unknown'}) // no tier provided
    expect(v.sourceTier).toBe(4)
    expect(v.decision).not.toBe('accept')
  })

  it('high-signal tier-1 source is accepted with top-band priority', () => {
    const gate = new PreIngestGate()
    const v = gate.evaluate(HIGH_SIGNAL, {sourceId: 's_curated', sourceTier: 1, query: 'transformer attention'})
    expect(v.decision).toBe('accept')
    expect(v.priority).toBeGreaterThan(0.4)
  })

  it('low-signal content is rejected even from a tier-1 source', () => {
    const gate = new PreIngestGate()
    const v = gate.evaluate(LOW_SIGNAL, {sourceId: 's_curated', sourceTier: 1})
    expect(v.decision).toBe('reject')
    expect(v.priority).toBe(0)
  })

  it('I1: a reject still emits a lesson record into lineage', () => {
    const gate = new PreIngestGate()
    const before = globalLineageAuditor.getChainStats().totalRecords
    const v = gate.evaluate(LOW_SIGNAL, {sourceId: 's_x', sourceTier: 4})
    const after = globalLineageAuditor.getChainStats().totalRecords
    expect(v.decision).toBe('reject')
    expect(after - before).toBe(1)
    expect(v.lesson.lineageRecordId.length).toBeGreaterThan(0)
    expect(v.lesson.proposedAction).toContain('drop source')
  })

  it('mid-tier mid-signal is deprioritized, not accepted', () => {
    const gate = new PreIngestGate()
    const v = gate.evaluate(HIGH_SIGNAL, {sourceId: 's_mid', sourceTier: 3, query: 'transformer attention'})
    expect(['deprioritize', 'flag']).toContain(v.decision)
    expect(v.priority).toBeLessThanOrEqual(0.5)
  })

  it('contradiction flag forces reject regardless of tier', () => {
    const gate = new PreIngestGate()
    // Truth/false-lite is heuristic; assert the mapping holds when it fires.
    const v = gate.evaluate('You must kill and exploit and bomb to hack the system illegally.', {
      sourceId: 's_dangerous',
      sourceTier: 1,
    })
    if (v.truthFlag === 'contradiction') {
      expect(v.decision).toBe('reject')
    }
    // Either way, dangerous low-quality content must not be accepted.
    expect(v.decision).not.toBe('accept')
  })

  it('lesson record names the weakest dimension among dims 1-3', () => {
    const gate = new PreIngestGate()
    const v = gate.evaluate(HIGH_SIGNAL, {sourceId: 's_dim', sourceTier: 1, query: 'transformer attention'})
    expect(['relevance', 'coherence', 'factuality']).toContain(v.lesson.weakestDimension)
  })

  it('respects a custom stricter accept threshold', () => {
    const strict = new PreIngestGate({acceptThreshold: 0.99})
    const v = strict.evaluate(HIGH_SIGNAL, {sourceId: 's_strict', sourceTier: 1, query: 'transformer attention'})
    expect(v.decision).not.toBe('accept') // 0.99 bar is effectively unreachable by heuristic scorer
  })
})
