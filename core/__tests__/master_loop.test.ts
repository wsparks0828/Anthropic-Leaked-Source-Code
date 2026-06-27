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
      expect(r.path).toContain(s)
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
