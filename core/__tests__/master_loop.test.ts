/**
 * THOTH Master Loop — adversarial state-machine invariant tests.
 */

import {describe, it, expect, beforeEach} from 'bun:test'
import {MasterLoop} from '../thoth/master_loop.js'
import {globalLineageAuditor} from '../lineage_auditor.js'

const GOOD =
  'Transformer attention computes softmax(QKᵀ/√d)V over query/key/value matrices. ' +
  'According to Vaswani et al. (2017), with d=64 the scaling factor is 8, stabilizing gradients. ' +
  'Therefore parallel sequence modeling becomes tractable, as a result cutting training time.'

beforeEach(() => {
  globalLineageAuditor.reset()
})

// Ambiguous-composite content (mid-band 0.48–0.62) so the heuristic short-circuit
// falls through to the injected LLM. Tier-2 ensures it is reasoned (not rejected).
const WEAK = 'A short but ingestible sentence about a topic with a little detail included here now.'

describe('THOTH Master Loop — reasoning token optimization', () => {
  it('default loop (no LLM injected) keeps REASONING zero-token', () => {
    const loop = new MasterLoop()
    const r = loop.runCycle({content: GOOD, sourceId: 's_zt', sourceTier: 1, intent: 'x', query: 'transformer'})
    expect(r.reasoningSource).toBe('none')
    expect(r.reasoningTokens).toBe(0)
    expect(loop.getReasoningStats()).toBeNull()
  })

  it('decisive content heuristic-short-circuits — injected LLM is never called', () => {
    let llmCalls = 0
    const loop = new MasterLoop({reasoningLlm: () => { llmCalls++; return {text: 'r', tokensUsed: 500} }})
    const r = loop.runCycle({content: GOOD, sourceId: 's_dec', sourceTier: 1, intent: 'x', query: 'transformer'})
    expect(r.reasoningSource).toBe('heuristic')
    expect(r.reasoningTokens).toBe(0)
    expect(llmCalls).toBe(0)
  })

  it('ambiguous content reaches the LLM once, then caches (zero tokens on repeat)', () => {
    let llmCalls = 0
    const loop = new MasterLoop({reasoningLlm: () => { llmCalls++; return {text: 'reasoned', tokensUsed: 500} }})
    const r1 = loop.runCycle({content: WEAK, sourceId: 's_amb1', sourceTier: 2})
    expect(r1.reasoningSource).toBe('llm')
    expect(r1.reasoningTokens).toBe(500)
    expect(llmCalls).toBe(1)

    const r2 = loop.runCycle({content: WEAK, sourceId: 's_amb2', sourceTier: 2})
    expect(r2.reasoningSource).toBe('cache') // identical content → cache hit
    expect(r2.reasoningTokens).toBe(0)
    expect(llmCalls).toBe(1) // still 1: the model was not called again

    const stats = loop.getReasoningStats()
    expect(stats?.llmCalls).toBe(1)
    expect(stats?.cacheHits).toBe(1)
  })
})

describe('THOTH Master Loop', () => {
  it('I1: a pre-ingest reject never reaches REASONING', () => {
    const loop = new MasterLoop()
    const r = loop.runCycle({content: 'idk maybe. stuff.', sourceId: 's_thin', sourceTier: 1})
    expect(r.preIngestDecision).toBe('reject')
    expect(r.reasoned).toBe(false)
    expect(r.path).not.toContain('REASONING')
    expect(r.path).not.toContain('VERIFYING')
  })

  it('I2: every cycle ends in IDLE and emits one cycle lineage record', () => {
    const loop = new MasterLoop()
    const before = globalLineageAuditor.getChainStats().totalRecords
    const r = loop.runCycle({content: GOOD, sourceId: 's_ok', sourceTier: 1, intent: 'explain', query: 'transformer attention'})
    expect(loop.getState()).toBe('IDLE')
    // pre-ingest writes 1 record + cycle summary writes 1 record => +2; cycle summary is the last.
    const after = globalLineageAuditor.getChainStats().totalRecords
    expect(after).toBeGreaterThan(before)
    expect(r.lineageRecordId.length).toBeGreaterThan(0)
  })

  it('I3: the recorded state path is in canonical order', () => {
    const loop = new MasterLoop()
    const r = loop.runCycle({content: GOOD, sourceId: 's_order', sourceTier: 1, intent: 'x', query: 'transformer attention'})
    const order = ['IDLE', 'PRE_INGEST', 'REASONING', 'VERIFYING', 'REFLECT_HEAL', 'IMPROVING', 'DRAINAGE', 'WRITING_STATE']
    const idxs = r.path.map((s) => order.indexOf(s))
    for (let i = 1; i < idxs.length; i++) {
      // IDLE(0) only legal at the very start
      if (r.path[i] === 'IDLE') continue
      expect(idxs[i]).toBeGreaterThanOrEqual(idxs[i - 1])
    }
  })

  it('a good tier-1 input is reasoned and runs the full path', () => {
    const loop = new MasterLoop()
    const r = loop.runCycle({content: GOOD, sourceId: 's_full', sourceTier: 1, intent: 'explain', query: 'transformer attention'})
    expect(r.reasoned).toBe(true)
    for (const s of ['PRE_INGEST', 'REASONING', 'VERIFYING', 'REFLECT_HEAL', 'IMPROVING', 'DRAINAGE', 'WRITING_STATE']) {
      expect(r.path as string[]).toContain(s)
    }
  })

  it('threshold stays within control-loop safety bounds across many cycles', () => {
    const loop = new MasterLoop()
    for (let i = 0; i < 30; i++) {
      loop.runCycle({content: GOOD, sourceId: `s_${i}`, sourceTier: 1, intent: 'x', query: 'transformer attention'})
    }
    expect(loop['control'].getThreshold()).toBeGreaterThanOrEqual(0.45)
    expect(loop['control'].getThreshold()).toBeLessThanOrEqual(0.85)
  })

  it('drainage archives rejected/quarantined cycles', () => {
    const loop = new MasterLoop()
    loop.runCycle({content: 'idk maybe. stuff.', sourceId: 's_rej', sourceTier: 1}) // reject → drained
    const r = loop.runCycle({content: 'idk. no.', sourceId: 's_rej2', sourceTier: 1}) // reject → drained
    expect(r.drained).toBeGreaterThanOrEqual(2)
  })

  it('integrity wiring: a clean cycle reports chainIntact=true', () => {
    const loop = new MasterLoop()
    const r = loop.runCycle({content: GOOD, sourceId: 's_intact', sourceTier: 1, intent: 'x', query: 'transformer attention'})
    expect(r.chainIntact).toBe(true)
    expect(r.accepted).toBe(true)
  })

  it('integrity wiring: a broken lineage chain fails the cycle closed (never accept)', () => {
    const loop = new MasterLoop()
    const original = globalLineageAuditor.verifyLineageChain.bind(globalLineageAuditor)
    // Simulate tamper detection during the audit-loop tick.
    ;(globalLineageAuditor as any).verifyLineageChain = () => ({
      valid: false,
      brokenAt: 'x',
      violations: [{verificationId: 'x', type: 'hash', description: 'mismatch'}],
    })
    const r = loop.runCycle({content: GOOD, sourceId: 's_tamper', sourceTier: 1, intent: 'x', query: 'transformer attention'})
    ;(globalLineageAuditor as any).verifyLineageChain = original

    expect(r.chainIntact).toBe(false)
    expect(r.accepted).toBe(false) // fail-closed regardless of lifecycle verdict
  })

  it('logger wiring: healing actions + lessons are recorded when provided', () => {
    const healing: any[] = []
    const lessons: any[] = []
    const scores: any[] = []
    const logger = {
      logHealingAction: (l: any) => healing.push(l),
      logLesson: (l: any) => lessons.push(l),
      logRubricScore: (l: any) => scores.push(l),
    }
    const loop = new MasterLoop({logger})
    // A weak-but-not-rejected input drives REFLECT_HEAL.
    loop.runCycle({content: 'A short but ingestible sentence about a topic with a little detail here.', sourceId: 's_heal', sourceTier: 2})
    // No-duplication: exactly one lesson + one rubric_score per cycle (gate owns both).
    expect(lessons.length).toBe(1)
    expect(scores.length).toBe(1)
    // healing only logged if reasoning happened and was sub-floor; assert structure if present
    for (const h of healing) {
      expect(h).toHaveProperty('target')
      expect(h).toHaveProperty('applied')
    }
  })
})
