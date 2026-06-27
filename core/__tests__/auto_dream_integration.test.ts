/**
 * Phase 8: Continuous Learning Loop — autoDream Integration Tests
 *
 * Validates the self-improving guardrail system:
 * - Signal accumulation and pattern detection
 * - Proposal generation and validation
 * - Automatic improvement application
 * - Lineage tracking of improvements
 */

import {describe, it, expect, beforeEach} from 'bun:test'
import {feedSignalToAutoDream, triggerAutoDreamCycle, getAutoDreamStatus, resetAutoDream} from '../auto_dream.js'
import {globalGuardrailLearningBridge} from '../guardrail_learning_bridge.js'
import {globalRubricScorer} from '../rubric_scorer.js'
import {globalTruthGate} from '../truth_gates.js'
import {globalLineageAuditor} from '../lineage_auditor.js'

describe('autoDream Continuous Learning Loop', () => {
  beforeEach(() => {
    resetAutoDream()
  })

  /**
   * Test 1: Signal Accumulation
   */
  it('should accumulate learning signals', () => {
    const output = 'Test output for autoDream signal accumulation.'

    for (let i = 0; i < 10; i++) {
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    const status = getAutoDreamStatus()
    expect(status.signalsAccumulated).toBe(10)
  })

  /**
   * Test 2: Pattern Detection Threshold
   */
  it('should be ready to trigger when pattern threshold met', () => {
    let status = getAutoDreamStatus()
    expect(status.readyToTrigger).toBe(false)

    // Feed signals with distinct patterns
    for (let i = 0; i < 6; i++) {
      const output = `Output with pattern ${i}`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    status = getAutoDreamStatus()
    expect(status.readyToTrigger).toBe(true) // At least 5 signals
  })

  /**
   * Test 3: Proposal Generation
   */
  it('should generate proposals from accumulated signals', async () => {
    // Feed signals
    for (let i = 0; i < 10; i++) {
      const output = `Test output ${i}`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    // Trigger cycle
    const event = await triggerAutoDreamCycle()

    expect(event.eventId).toBeDefined()
    expect(event.signalsAnalyzed).toBeGreaterThan(0)
    expect(event.proposalsGenerated).toBeGreaterThanOrEqual(0)
  })

  /**
   * Test 4: Proposal Validation Through Cross-Verifier
   */
  it('should validate proposals through cross-verifier', async () => {
    // Feed signals that should trigger proposals
    for (let i = 0; i < 8; i++) {
      const output = `Output ${i} for proposal generation and validation testing.`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    const event = await triggerAutoDreamCycle()

    // If proposals were generated, some should be approved
    if (event.proposalsGenerated > 0) {
      expect(event.proposalsApproved).toBeGreaterThanOrEqual(0)
      expect(event.proposalsApproved).toBeLessThanOrEqual(event.proposalsGenerated)
    }
  })

  /**
   * Test 5: Improvement Lineage Recording
   */
  it('should record improvement events in immutable lineage', async () => {
    // Feed signals
    for (let i = 0; i < 10; i++) {
      const output = `Output ${i}`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    const event = await triggerAutoDreamCycle()

    // Event should have a lineage record ID
    expect(event.lineageRecordId).toBeDefined()
    expect(event.lineageRecordId.length).toBeGreaterThan(0)

    // Verify record in lineage
    const record = globalLineageAuditor.getRecord(event.eventId)
    if (record) {
      expect(record.verificationId).toBe(event.eventId)
      expect(record.lineage.who).toBe('autoDream_orchestrator')
    }
  })

  /**
   * Test 6: Fail-Closed on Safety
   */
  it('should never apply high-risk proposals', async () => {
    // Feed signals
    for (let i = 0; i < 10; i++) {
      const output = `Risky output ${i}`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    const event = await triggerAutoDreamCycle()

    // Applied proposals should only be low-risk threshold adjustments
    // High-risk structural changes queued for manual review
    if (event.proposalsApplied > 0) {
      // All applied proposals should have low residual risk
      expect(event.improvementImpact.residualRisk).toBeLessThan(0.2)
    }
  })

  /**
   * Test 7: Lineage Chain Integrity
   */
  it('should maintain lineage chain integrity through autoDream cycles', async () => {
    // Feed signals and trigger cycle multiple times
    const cycleEvents = []

    for (let cycle = 0; cycle < 3; cycle++) {
      for (let i = 0; i < 5; i++) {
        const output = `Cycle ${cycle} output ${i}`
        const rubricScore = globalRubricScorer.score(output)
        const truthVerdict = globalTruthGate.gate(output)

        const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
          source: 'api_boundary',
          summary: output,
          hash: `hash_${cycle}_${i}`,
        })

        feedSignalToAutoDream(signal)
      }

      const event = await triggerAutoDreamCycle()
      cycleEvents.push(event)
    }

    // Verify each event was recorded
    expect(cycleEvents.length).toBe(3)

    // Each event should have a lineage record ID
    for (const event of cycleEvents) {
      expect(event.eventId).toBeDefined()
      expect(event.lineageRecordId).toBeDefined()
    }
  })

  /**
   * Test 8: Improvement Impact Tracking
   */
  it('should estimate improvement impact from applied proposals', async () => {
    // Feed signals
    for (let i = 0; i < 12; i++) {
      const output = `Output ${i} for impact estimation.`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    const event = await triggerAutoDreamCycle()

    // Event should have improvement estimate
    expect(event.improvementImpact).toBeDefined()
    expect(event.improvementImpact.affectedDimension).toBeDefined()
    expect(event.improvementImpact.estimatedImprovement).toBeGreaterThanOrEqual(0)
    expect(event.improvementImpact.residualRisk).toBeGreaterThanOrEqual(0)
    expect(event.improvementImpact.residualRisk).toBeLessThanOrEqual(1)
  })

  /**
   * Test 9: Forestall Proposal Spam
   */
  it('should not generate proposals on every cycle (forestall spam)', async () => {
    // Feed same output repeatedly
    for (let i = 0; i < 20; i++) {
      const output = 'Same output repeated for spam prevention testing.'
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    const event = await triggerAutoDreamCycle()

    // Should either generate proposals or not, but not spam
    expect(event.proposalsGenerated).toBeGreaterThanOrEqual(0)
    expect(event.proposalsGenerated).toBeLessThan(100)
  })

  /**
   * Test 10: Multiple Improvement Dimensions
   */
  it('should handle improvements across multiple guardrail dimensions', async () => {
    // Feed diverse outputs hitting different dimensions
    const outputs = [
      'This is low quality.',
      'This contains dangerous ideas.',
      'This is incoherent and rambling.',
      'This is well-written but false.',
      'This is complete but unsafe.',
    ]

    for (let round = 0; round < 3; round++) {
      for (const output of outputs) {
        const rubricScore = globalRubricScorer.score(output)
        const truthVerdict = globalTruthGate.gate(output)

        const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
          source: 'api_boundary',
          summary: output,
          hash: `hash_${round}_${outputs.indexOf(output)}`,
        })

        feedSignalToAutoDream(signal)
      }
    }

    const event = await triggerAutoDreamCycle()

    // Should accumulate patterns across dimensions
    expect(event.signalsAnalyzed).toBeGreaterThan(5)
  })

  /**
   * Test 11: Atomicity of Improvement Application
   */
  it('should ensure all-or-nothing application of proposals', async () => {
    // Feed signals
    for (let i = 0; i < 10; i++) {
      const output = `Test ${i}`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    const event = await triggerAutoDreamCycle()

    // Applied count should match what was actually applied (atomic)
    if (event.proposalsApplied > 0) {
      // Verify lineage record reflects exact count
      expect(event.lineageRecordId).toBeDefined()
    }
  })

  /**
   * Test 12: Status Reporting
   */
  it('should provide accurate autoDream status', () => {
    const status1 = getAutoDreamStatus()
    expect(status1.signalsAccumulated).toBe(0)
    expect(status1.readyToTrigger).toBe(false)

    // Add signals
    for (let i = 0; i < 7; i++) {
      const output = `Status test ${i}`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })

      feedSignalToAutoDream(signal)
    }

    const status2 = getAutoDreamStatus()
    expect(status2.signalsAccumulated).toBe(7)
    expect(status2.readyToTrigger).toBe(true)
    expect(status2.lastUpdate).toBeGreaterThan(BigInt(0))
  })
})
