/**
 * THOTH Pre-Ingest Gate — adversarial invariant tests.
 *
 * Verifies provenance fail-closed (I2), no-silent-reject (I1), and that the
 * routing decisions actually discriminate signal & tier rather than rubber-stamp.
 */

import {describe, it, expect, beforeEach} from 'bun:test'
import {PreIngestGate, computeInformativeness, type SourceTier} from '../thoth/pre_ingest_gate.js'
import {globalLineageAuditor} from '../lineage_auditor.js'

const WORTHLESS_DENSE =
  'Basically this is just a thing about stuff and really it is very much the kind of thing that ' +
  'things are, and honestly it is just kind of whatever you might think it is, more or less, in a ' +
  'general sense you know, it really just depends and stuff like that.'

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

  it('SIGNAL-QUALITY GAP: dense-but-worthless content is rejected on low informativeness', () => {
    const gate = new PreIngestGate()
    const v = gate.evaluate(WORTHLESS_DENSE, {sourceId: 's_filler', sourceTier: 1})
    // Passes density (it is long & lexically varied) but must still be rejected.
    expect(v.density).toBeGreaterThanOrEqual(8)
    expect(v.decision).toBe('reject')
    expect(v.informativeness).toBeLessThan(0.15)
    expect(v.reasons.some((r) => r.includes('informativeness'))).toBe(true)
  })

  it('SIGNAL-QUALITY GAP: substantive concrete content scores high informativeness and is accepted', () => {
    const gate = new PreIngestGate()
    const v = gate.evaluate(
      'Transformer attention computes softmax(QKᵀ/√d)V. Vaswani et al. (2017) showed d=64 yields factor 8, stabilizing gradients across 12 layers.',
      {sourceId: 's_substantive', sourceTier: 1, query: 'transformer attention'},
    )
    expect(v.informativeness).toBeGreaterThan(0.5)
    expect(v.decision).toBe('accept')
  })

  it('computeInformativeness separates filler from concrete content', () => {
    const filler = computeInformativeness(WORTHLESS_DENSE)
    const concrete = computeInformativeness('Vaswani et al. (2017) reported d=64, factor 8, across 12 transformer layers.')
    expect(filler).toBeLessThan(0.2)
    expect(concrete).toBeGreaterThan(0.6)
    expect(concrete - filler).toBeGreaterThan(0.4) // strong separation
  })

  it('respects a custom stricter accept threshold', () => {
    const strict = new PreIngestGate({acceptThreshold: 0.99})
    const v = strict.evaluate(HIGH_SIGNAL, {sourceId: 's_strict', sourceTier: 1, query: 'transformer attention'})
    expect(v.decision).not.toBe('accept') // 0.99 bar is effectively unreachable by heuristic scorer
  })
})
