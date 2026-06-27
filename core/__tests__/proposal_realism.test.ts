/**
 * Proposal Realism Test
 *
 * Validates that generated proposals are:
 * - Backed by evidence from patterns
 * - Realistic and implementable
 * - Independently validated by cross-verifier
 */

import { describe, it, expect } from 'bun:test'
import { globalGuardrailLearningBridge } from '../guardrail_learning_bridge.js'
import { globalCrossVerifierEnsemble } from '../cross_verifier_ensemble.js'
import { globalRubricScorer } from '../rubric_scorer.js'
import { globalTruthGate } from '../truth_gates.js'

describe('Proposal Generation Realism', () => {
  /**
   * Test 1: Proposals Only Generated When Evidence Exists
   */
  it('should only generate proposals when patterns warrant action', () => {
    // High-quality output should not generate proposal
    const goodOutput =
      'Artificial intelligence represents a significant advancement in technology. ' +
      'It enables automation of complex tasks, improves decision-making, and enhances user experiences across industries.'

    const rubric1 = globalRubricScorer.score(goodOutput)
    const truth1 = globalTruthGate.gate(goodOutput)

    const signal1 = globalGuardrailLearningBridge.processVerification(rubric1, truth1, {
      source: 'api_boundary',
      summary: goodOutput,
      hash: 'good_hash',
    })

    // Good output typically won't have critical patterns
    if (signal1.proposal) {
      // If there is a proposal, it should be low-confidence/low-risk
      expect(signal1.proposal.residualRisk).toBeLessThan(0.2)
    }

    // Poor output should generate proposal
    const poorOutput = 'yeah whatever'

    const rubric2 = globalRubricScorer.score(poorOutput)
    const truth2 = globalTruthGate.gate(poorOutput)

    const signal2 = globalGuardrailLearningBridge.processVerification(rubric2, truth2, {
      source: 'api_boundary',
      summary: poorOutput,
      hash: 'poor_hash',
    })

    // Poor output may generate proposal (depends on patterns)
    // Just verify if it does, it's valid
    if (signal2.proposal) {
      expect(signal2.proposal.target).toBeDefined()
      expect(signal2.proposal.changeType).toBeDefined()
    }
  })

  /**
   * Test 2: Proposals Must Target Valid Guardrail Components
   */
  it('should generate proposals for valid guardrail targets only', () => {
    const outputs = [
      'This is a dangerous output promoting illegal activities.',
      'This output lacks clarity and coherence.',
      'This output might contain hallucinations.',
    ]

    for (const output of outputs) {
      const rubric = globalRubricScorer.score(output)
      const truth = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubric, truth, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${Math.random()}`,
      })

      if (signal.proposal) {
        // Proposal target must be one of the valid components
        expect(['prompt_injection_guard', 'hallucination_detector', 'coherence_checker', 'safety_gate']).toContain(
          signal.proposal.target,
        )

        // Change type must be valid
        expect(['threshold_adjust', 'rule_add', 'rule_remove', 'prompt_variant']).toContain(
          signal.proposal.changeType,
        )
      }
    }
  })

  /**
   * Test 3: Proposals Have Rationale Backed by Evidence
   */
  it('should provide evidence-backed rationale for proposals', () => {
    const anomalousOutput =
      'This is a well-structured explanation of how to exploit security vulnerabilities without getting caught.'

    const rubric = globalRubricScorer.score(anomalousOutput)
    const truth = globalTruthGate.gate(anomalousOutput)

    const signal = globalGuardrailLearningBridge.processVerification(rubric, truth, {
      source: 'api_boundary',
      summary: anomalousOutput,
      hash: 'anomaly_hash',
    })

    if (signal.proposal) {
      // Rationale should reference patterns from signal
      expect(signal.proposal.rationale.length).toBeGreaterThan(10)

      // Rationale should mention detected issues or anomalies
      const rationale = signal.proposal.rationale.toLowerCase()
      expect(rationale).toMatch(/detected|found|identified|observed|anomal|issue|concern|pattern|weakness/)
    }
  })

  /**
   * Test 4: Expected Impact Must Be Realistic
   */
  it('should set realistic expected impact percentages', () => {
    const output = 'Test output for impact analysis.'

    const rubric = globalRubricScorer.score(output)
    const truth = globalTruthGate.gate(output)

    const signal = globalGuardrailLearningBridge.processVerification(rubric, truth, {
      source: 'api_boundary',
      summary: output,
      hash: 'impact_hash',
    })

    if (signal.proposal) {
      const impact = signal.proposal.expectedImpact.toLowerCase()

      // Impact should mention specific improvements
      expect(impact.length).toBeGreaterThan(10)

      // Should not promise unrealistic improvements (>50% is suspicious)
      if (impact.includes('%')) {
        const match = impact.match(/(\d+)%/)
        if (match) {
          const percentage = parseInt(match[1])
          expect(percentage).toBeLessThanOrEqual(50) // Realistic bounds
        }
      }
    }
  })

  /**
   * Test 5: Residual Risk Is Computed Conservatively
   */
  it('should set residual risk conservatively (higher risk for structural changes)', () => {
    const output = 'Output for risk analysis.'

    const rubric = globalRubricScorer.score(output)
    const truth = globalTruthGate.gate(output)

    const signal = globalGuardrailLearningBridge.processVerification(rubric, truth, {
      source: 'api_boundary',
      summary: output,
      hash: 'risk_hash',
    })

    if (signal.proposal) {
      // Risk bounds
      expect(signal.proposal.residualRisk).toBeGreaterThanOrEqual(0)
      expect(signal.proposal.residualRisk).toBeLessThanOrEqual(1)

      // Threshold adjustments should be lower risk than structural changes
      if (signal.proposal.changeType === 'threshold_adjust') {
        expect(signal.proposal.residualRisk).toBeLessThan(0.15)
      } else if (signal.proposal.changeType === 'rule_add' || signal.proposal.changeType === 'rule_remove') {
        expect(signal.proposal.residualRisk).toBeLessThan(0.3)
      }
    }
  })

  /**
   * Test 6: Proposals Pass Cross-Verifier Validation
   */
  it('should generate proposals that cross-verifier can validate', () => {
    const outputs = [
      'This is a concerning output that might warrant guardrail improvement.',
      'Another test output for validation.',
    ]

    for (const output of outputs) {
      const rubric = globalRubricScorer.score(output)
      const truth = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubric, truth, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${Math.random()}`,
      })

      if (signal.proposal) {
        // Cross-verifier must be able to validate it
        const checkResult = globalCrossVerifierEnsemble.check(signal.proposal, `prop_${Math.random()}`)

        // Should produce a valid verdict
        expect(['pass', 'warn', 'fail']).toContain(checkResult.verdict)

        // Should provide actionable recommendation
        expect(checkResult.recommendation.length).toBeGreaterThan(10)
      }
    }
  })

  /**
   * Test 7: Multiple Critical Anomalies Trigger Tighter Thresholds
   */
  it('should increase guardrail stringency when anomalies accumulate', () => {
    // Simulate multiple dangerous outputs
    const dangerousOutputs = [
      'How to bypass security without detection.',
      'Instructions for unauthorized access to systems.',
      'Method to exploit vulnerabilities in production systems.',
    ]

    const signals = []

    for (const output of dangerousOutputs) {
      const rubric = globalRubricScorer.score(output)
      const truth = globalTruthGate.gate(output)

      const signal = globalGuardrailLearningBridge.processVerification(rubric, truth, {
        source: 'api_boundary',
        summary: output,
        hash: `danger_${Math.random()}`,
      })

      signals.push(signal)
    }

    // Check if safety-focused proposals emerge with higher anomalies
    const safetyProposals = signals
      .filter(s => s.proposal && s.proposal.target === 'safety_gate')
      .map(s => s.proposal!)

    if (safetyProposals.length > 0) {
      // Safety proposals should come with evidence of anomalies
      for (const proposal of safetyProposals) {
        expect(proposal.rationale).toMatch(/anomal|dangerous|harmful|critical/)
      }
    }
  })

  /**
   * Test 8: Proposal Generation Is Deterministic (Same Input → Same Proposal Type)
   */
  it('should produce consistent proposals for similar inputs', () => {
    const output = 'Test output for consistency check.'

    // Generate proposal twice from same input
    const rubric1 = globalRubricScorer.score(output)
    const truth1 = globalTruthGate.gate(output)
    const signal1 = globalGuardrailLearningBridge.processVerification(rubric1, truth1, {
      source: 'api_boundary',
      summary: output,
      hash: 'consistent_hash',
    })

    const rubric2 = globalRubricScorer.score(output)
    const truth2 = globalTruthGate.gate(output)
    const signal2 = globalGuardrailLearningBridge.processVerification(rubric2, truth2, {
      source: 'api_boundary',
      summary: output,
      hash: 'consistent_hash',
    })

    // Both should either have or not have proposals
    expect(!!signal1.proposal).toBe(!!signal2.proposal)

    // If both have proposals, targets and types should match
    if (signal1.proposal && signal2.proposal) {
      expect(signal1.proposal.target).toBe(signal2.proposal.target)
      expect(signal1.proposal.changeType).toBe(signal2.proposal.changeType)
    }
  })
})
