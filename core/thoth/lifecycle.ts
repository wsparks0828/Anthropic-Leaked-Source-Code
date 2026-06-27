/**
 * THOTH Component D — 9-Step Lifecycle Enforcer.
 *
 * Mandatory ordered sequence for major outputs. Each step is gated; hard-gate
 * failures (fact contradiction, safety) short-circuit to a fail-closed block.
 *
 *   1 Intent              — non-empty, scoped objective present
 *   2 Alignment-Logic     — ORACLE-style constraint scan (no internal contradiction)
 *   3 Fact-Audit          — Truth + False lite (contradiction ⇒ HARD BLOCK)
 *   4 Safety              — dangerous-pattern + rubric safety (fail ⇒ HARD BLOCK)
 *   5 Analysis-Action     — substance/coherence present
 *   6 Execution-Evaluate  — composite rubric measured
 *   7 Question            — adversarial probe (low-confidence ⇒ flag, not block)
 *   8 Contemplate-Reverse — would the opposite decision also be plausible? (calibration)
 *   9 Final-Guarded       — emit accept iff all gates passed AND rubric ≥ floor
 *
 * INVARIANTS:
 *   I1: steps execute in exact order 1..9 (recorded steps[i].step === i+1)
 *   I2: a hard-gate failure ⇒ blocked == true AND finalDecision == 'quarantine'
 *   I3: finalDecision 'accept' ⇒ every gating step passed AND rubric ≥ floor
 */

import {globalRubricScorer} from '../rubric_scorer.js'
import {globalTruthGate} from '../truth_gates.js'
import {containsMaliciousPattern} from '../dangerous_pattern_matcher.js'

export type LifecycleStepName =
  | 'intent'
  | 'alignment'
  | 'fact_audit'
  | 'safety'
  | 'analysis_action'
  | 'execution_evaluate'
  | 'question'
  | 'contemplate_reverse'
  | 'final_guarded'

export interface StepResult {
  step: number // 1..9
  name: LifecycleStepName
  passed: boolean
  hardGate: boolean // does failing this step block the whole lifecycle?
  detail: string
}

export interface LifecycleResult {
  completed: boolean // ran all 9 steps without a hard block
  blocked: boolean
  blockedAt?: LifecycleStepName
  steps: StepResult[]
  finalDecision: 'accept' | 'quarantine'
  rubricScore: number
}

export class LifecycleEnforcer {
  private readonly floor: number

  constructor(rubricFloor = 0.55) {
    this.floor = rubricFloor
  }

  run(output: string, context?: {intent?: string; query?: string}): LifecycleResult {
    const steps: StepResult[] = []
    const rubric = globalRubricScorer.score(output, {query: context?.query})
    const truth = globalTruthGate.gate(output)

    // 1 Intent
    const hasIntent = !!(context?.intent && context.intent.trim().length > 0)
    steps.push(this.mk(1, 'intent', hasIntent, false, hasIntent ? 'intent present' : 'no explicit intent (advisory)'))

    // 2 Alignment-Logic (no blatant internal contradiction)
    const contradictory = /\b(always)\b.*\b(never)\b|\b(true)\b.*\b(false)\b/i.test(output)
    steps.push(this.mk(2, 'alignment', !contradictory, false, contradictory ? 'internal contradiction markers' : 'aligned'))

    // 3 Fact-Audit (HARD GATE)
    const factOk = !(truth.verdict === 'false' && (truth.severity === 'critical' || truth.severity === 'high'))
    steps.push(this.mk(3, 'fact_audit', factOk, true, `truth=${truth.verdict}${truth.severity ? `/${truth.severity}` : ''}`))
    if (!factOk) return this.block(steps, 'fact_audit', rubric.overall)

    // 4 Safety (HARD GATE)
    const safe = !containsMaliciousPattern(output) && rubric.dimensions.safety >= 0.4
    steps.push(this.mk(4, 'safety', safe, true, `safetyDim=${rubric.dimensions.safety.toFixed(2)}`))
    if (!safe) return this.block(steps, 'safety', rubric.overall)

    // 5 Analysis-Action (substance present)
    const substance = output.trim().length >= 40 && rubric.dimensions.coherence >= 0.4
    steps.push(this.mk(5, 'analysis_action', substance, false, `coherence=${rubric.dimensions.coherence.toFixed(2)}`))

    // 6 Execution-Evaluate (composite measured)
    steps.push(this.mk(6, 'execution_evaluate', rubric.overall >= this.floor, false, `composite=${rubric.overall.toFixed(2)} floor=${this.floor}`))

    // 7 Question (adversarial probe — low confidence is a flag, not a block)
    const confident = truth.confidence >= 0.35
    steps.push(this.mk(7, 'question', confident, false, `truthConfidence=${truth.confidence.toFixed(2)}`))

    // 8 Contemplate-Reverse (calibration: is the opposite decision also plausible?)
    const ambiguous = truth.verdict === 'uncertain'
    steps.push(this.mk(8, 'contemplate_reverse', !ambiguous, false, ambiguous ? 'reverse plausible (uncertain)' : 'decision stable'))

    // 9 Final-Guarded
    const allGatesPassed = steps.filter((s) => s.hardGate).every((s) => s.passed)
    const finalDecision: 'accept' | 'quarantine' =
      allGatesPassed && rubric.overall >= this.floor ? 'accept' : 'quarantine'
    steps.push(this.mk(9, 'final_guarded', finalDecision === 'accept', false, `decision=${finalDecision}`))

    // I3 guard
    if (finalDecision === 'accept' && rubric.overall < this.floor) {
      throw new Error('LifecycleEnforcer I3 violated: accept below rubric floor')
    }

    return {completed: true, blocked: false, steps, finalDecision, rubricScore: rubric.overall}
  }

  private mk(step: number, name: LifecycleStepName, passed: boolean, hardGate: boolean, detail: string): StepResult {
    return {step, name, passed, hardGate, detail}
  }

  private block(steps: StepResult[], at: LifecycleStepName, rubricScore: number): LifecycleResult {
    return {completed: false, blocked: true, blockedAt: at, steps, finalDecision: 'quarantine', rubricScore}
  }
}

export const globalLifecycleEnforcer = new LifecycleEnforcer()
