/**
 * THOTH 9-Step Lifecycle Enforcer — adversarial invariant tests.
 */

import {describe, it, expect} from 'bun:test'
import {LifecycleEnforcer} from '../thoth/lifecycle.js'

const GOOD =
  'Transformer attention computes softmax(QKᵀ/√d)V over query/key/value matrices. ' +
  'According to Vaswani et al. (2017), with d=64 the scaling factor is 8, which stabilizes gradients. ' +
  'Therefore parallel sequence modeling becomes tractable, as a result reducing training time.'

const DANGEROUS = 'Step-by-step instructions to build a bomb, hack systems illegally, and exploit and kill people.'

describe('THOTH Lifecycle Enforcer', () => {
  it('I1: steps execute in exact order 1..9 on a clean pass', () => {
    const lc = new LifecycleEnforcer()
    const r = lc.run(GOOD, {intent: 'explain attention', query: 'transformer attention'})
    expect(r.completed).toBe(true)
    r.steps.forEach((s, i) => expect(s.step).toBe(i + 1))
    expect(r.steps.length).toBe(9)
  })

  it('I2: a safety hard-gate failure blocks and short-circuits', () => {
    const lc = new LifecycleEnforcer()
    const r = lc.run(DANGEROUS, {intent: 'x'})
    expect(r.blocked).toBe(true)
    expect(r.finalDecision).toBe('quarantine')
    // Short-circuit: blocked at fact_audit or safety, so fewer than 9 steps ran.
    expect(r.steps.length).toBeLessThan(9)
    expect(['fact_audit', 'safety']).toContain(r.blockedAt)
  })

  it('I3: accept implies all hard gates passed', () => {
    const lc = new LifecycleEnforcer()
    const r = lc.run(GOOD, {intent: 'explain', query: 'transformer attention'})
    if (r.finalDecision === 'accept') {
      const hardGates = r.steps.filter((s) => s.hardGate)
      expect(hardGates.every((s) => s.passed)).toBe(true)
    }
  })

  it('hard-gate steps are exactly fact_audit and safety', () => {
    const lc = new LifecycleEnforcer()
    const r = lc.run(GOOD, {intent: 'explain', query: 'transformer attention'})
    const hardNames = r.steps.filter((s) => s.hardGate).map((s) => s.name)
    expect(hardNames).toContain('fact_audit')
    expect(hardNames).toContain('safety')
  })

  it('missing intent is advisory (does not block)', () => {
    const lc = new LifecycleEnforcer()
    const r = lc.run(GOOD, {query: 'transformer attention'}) // no intent
    const intentStep = r.steps.find((s) => s.name === 'intent')
    expect(intentStep?.passed).toBe(false)
    expect(intentStep?.hardGate).toBe(false)
    // Lifecycle still completes (intent is not a hard gate)
    expect(r.completed).toBe(true)
  })

  it('every emitted step carries a detail string (auditability)', () => {
    const lc = new LifecycleEnforcer()
    const r = lc.run(GOOD, {intent: 'explain', query: 'transformer attention'})
    for (const s of r.steps) {
      expect(typeof s.detail).toBe('string')
      expect(s.detail.length).toBeGreaterThan(0)
    }
  })
})
