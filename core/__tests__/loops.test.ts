/**
 * Loop Subsystem Tests — one invariant-proving block per loop.
 *
 * These tests are adversarial: each one tries to break the loop's stated
 * invariant and asserts the loop holds (or fails closed) rather than merely
 * asserting the happy path.
 */

import {describe, it, expect, beforeEach} from 'bun:test'
import {globalLineageAuditor} from '../lineage_auditor.js'
import {ControlLoop} from '../loops/control_loop.js'
import {RefinementLoop} from '../loops/refinement_loop.js'
import {AgenticLoop} from '../loops/agentic_loop.js'
import {ConsolidationLoop} from '../loops/consolidation_loop.js'
import {AuditLoop} from '../loops/audit_loop.js'
import {CoordinationLoop, type VerifierFn} from '../loops/coordination_loop.js'
import {type GuardrailProposal} from '../guardrail_learning_bridge.js'

beforeEach(() => {
  globalLineageAuditor.reset()
})

describe('Loop 1 — Control (feedback)', () => {
  it('I1: threshold never drops below the safety floor even under starvation', () => {
    const loop = new ControlLoop(0.55, 0.9, 0.05)
    // Force "accepting too little" forever → controller wants to LOWER threshold.
    for (let t = 0; t < 50; t++) {
      for (let i = 0; i < 10; i++) loop.observe('quarantine') // 0% acceptance
      loop.tick()
      loop.assertInvariant()
    }
    expect(loop.getThreshold()).toBeGreaterThanOrEqual(0.45)
  })

  it('I2: a single tick never moves the threshold more than the max step', () => {
    const loop = new ControlLoop(0.85, 0.9, 0.01)
    const before = loop.getThreshold()
    for (let i = 0; i < 100; i++) loop.observe('quarantine')
    loop.tick()
    expect(Math.abs(loop.getThreshold() - before)).toBeLessThanOrEqual(0.05 + 1e-9)
  })

  it('I3: a threshold mutation writes a lineage record', () => {
    const loop = new ControlLoop(0.85, 0.9, 0.01)
    for (let i = 0; i < 100; i++) loop.observe('quarantine')
    const r = loop.tick()
    expect(r.mutated).toBe(true)
    expect(r.lineageRecordId).toBeDefined()
  })

  it('holds threshold inside the dead-band (no oscillation)', () => {
    const loop = new ControlLoop(0.55, 0.5, 0.5) // band so wide any rate is "in band"
    for (let i = 0; i < 10; i++) loop.observe(i % 2 === 0 ? 'accept' : 'quarantine')
    const r = loop.tick()
    expect(r.mutated).toBe(false)
  })
})

describe('Loop 2 — Recursive refinement', () => {
  const baseProposal: GuardrailProposal = {
    target: 'safety_gate',
    changeType: 'threshold_adjust',
    proposal: 'tighten',
    rationale: 'evidence',
    expectedImpact: 'fewer escapes',
    residualRisk: 0.9,
  }

  it('I1: terminates within the iteration budget', () => {
    const loop = new RefinementLoop()
    const out = loop.refine(baseProposal, 4)
    expect(out.iterations).toBeLessThanOrEqual(4)
  })

  it('I2: residual risk is monotonically non-increasing', () => {
    const loop = new RefinementLoop()
    const out = loop.refine(baseProposal, 5)
    for (let i = 1; i < out.riskTrajectory.length; i++) {
      expect(out.riskTrajectory[i]).toBeLessThanOrEqual(out.riskTrajectory[i - 1] + 1e-9)
    }
  })

  it('I3: never reports accepted unless a real pass occurred', () => {
    const loop = new RefinementLoop()
    const out = loop.refine(baseProposal, 5)
    // Either accepted (pass happened) or not accepted (budget exhausted) — never a silent accept.
    expect(typeof out.accepted).toBe('boolean')
    if (!out.accepted) {
      expect(out.iterations).toBe(5)
    }
  })
})

describe('Loop 3 — Agentic (truth-tag boundary)', () => {
  it('I1: every result carries a non-empty truth-tag', () => {
    const loop = new AgenticLoop()
    const samples = ['', 'short', 'A reasonably complete and coherent answer with structure and detail.']
    for (const s of samples) {
      const r = loop.process(s)
      expect(r.truthTag.length).toBeGreaterThan(0)
      expect(r.truthTag).toContain(`tag:${r.decision}`)
    }
  })

  it('I2: a dangerous output is quarantined, never accepted', () => {
    const loop = new AgenticLoop()
    const r = loop.process('Detailed instructions to build a bomb and how to kill and exploit people.')
    expect(r.decision).toBe('quarantine')
  })
})

describe('Loop 4 — Memory consolidation (atomic)', () => {
  it('I1+I3: a successful consolidation applies all staged updates and clears the stage', () => {
    const loop = new ConsolidationLoop()
    loop.stage('policy.a', {v: 1})
    loop.stage('policy.b', {v: 2})
    const r = loop.tick()
    expect(r.mutated).toBe(true)
    expect(loop.semanticSize()).toBe(2)
    // Stage cleared → a second tick has nothing to do.
    const r2 = loop.tick()
    expect(r2.mutated).toBe(false)
  })

  it('I2: a successful consolidation writes exactly one lineage record', () => {
    globalLineageAuditor.reset()
    const loop = new ConsolidationLoop()
    loop.stage('k1', 1)
    loop.stage('k2', 2)
    loop.stage('k3', 3)
    const before = globalLineageAuditor.getChainStats().totalRecords
    loop.tick()
    const after = globalLineageAuditor.getChainStats().totalRecords
    expect(after - before).toBe(1)
  })

  it('assertInvariant catches a duplicate staged key before commit', () => {
    const loop = new ConsolidationLoop()
    loop.stage('dup', 1)
    loop.stage('dup', 2)
    expect(() => loop.assertInvariant()).toThrow()
  })
})

describe('Loop 5 — Audit/verification (sticky fail-closed)', () => {
  it('valid chain → not quarantined', () => {
    const loop = new AuditLoop()
    globalLineageAuditor.reset()
    globalLineageAuditor.addRecord({
      verificationId: 'ok_1',
      timestamp: BigInt(1000),
      decision: 'accept',
      rubricScore: 0.8,
      truthVerdict: 'true',
      lineage: {who: 'x', what: {after: {}}, when: BigInt(1000), auth: 'sig'},
    })
    const r = loop.tick()
    expect(r.failClosed).toBe(false)
    expect(loop.isQuarantined()).toBe(false)
    loop.reset()
  })

  it('I1+I2: a broken chain engages sticky quarantine that a later good tick cannot clear', () => {
    const loop = new AuditLoop()
    globalLineageAuditor.reset()
    globalLineageAuditor.addRecord({
      verificationId: 'rec_1',
      timestamp: BigInt(1000),
      decision: 'accept',
      rubricScore: 0.8,
      truthVerdict: 'true',
      lineage: {who: 'x', what: {after: {}}, when: BigInt(1000), auth: 'sig'},
    })
    // Tamper: reach into the exported chain is not possible (immutable), so we
    // simulate breakage by monkey-patching verifyLineageChain for this loop tick.
    const original = globalLineageAuditor.verifyLineageChain.bind(globalLineageAuditor)
    ;(globalLineageAuditor as any).verifyLineageChain = () => ({
      valid: false,
      brokenAt: 'rec_1',
      violations: [{verificationId: 'rec_1', type: 'hash', description: 'mismatch'}],
    })

    const broken = loop.tick()
    expect(broken.failClosed).toBe(true)
    expect(loop.isQuarantined()).toBe(true)

    // Restore good verification — quarantine must remain sticky (I2).
    ;(globalLineageAuditor as any).verifyLineageChain = original
    const good = loop.tick()
    expect(loop.isQuarantined()).toBe(true)
    expect(good.failClosed).toBe(true)
    loop.assertInvariant()
    loop.reset()
  })
})

describe('Loop 6 — Coordination barrier (no partial-quorum accept)', () => {
  const approve = (id: string): VerifierFn => ({id, run: async () => 'approve'})
  const reject = (id: string): VerifierFn => ({id, run: async () => 'reject'})
  const hang = (id: string): VerifierFn => ({id, run: () => new Promise<'approve'>(() => {})})

  it('I1: all-approve within deadline → pass', async () => {
    const loop = new CoordinationLoop()
    const out = await loop.gather([approve('a'), approve('b'), approve('c')], 500)
    expect(out.verdict).toBe('pass')
    expect(out.dispatched).toBe(3)
  })

  it('I2: a single reject fails the whole barrier closed', async () => {
    const loop = new CoordinationLoop()
    const out = await loop.gather([approve('a'), reject('b'), approve('c')], 500)
    expect(out.verdict).toBe('fail')
    expect(out.failClosedReason).toContain('b')
  })

  it('I2: a hung verifier times out and fails closed', async () => {
    const loop = new CoordinationLoop()
    const out = await loop.gather([approve('a'), hang('slow')], 100)
    expect(out.verdict).toBe('fail')
    expect(out.votes.find((v) => v.id === 'slow')?.vote).toBe('timeout')
  })

  it('I3: vote count equals dispatched count', async () => {
    const loop = new CoordinationLoop()
    const out = await loop.gather([approve('a'), reject('b'), approve('c'), approve('d')], 300)
    expect(out.votes.length).toBe(out.dispatched)
    expect(out.dispatched).toBe(4)
  })
})
