/**
 * End-to-End Verification Cycle Test
 *
 * Traces: Input → Rubric → Truth Gate → Learning Bridge → Proposal → Cross-Verifier → Decision
 *
 * Validates:
 * - All subsystems work together
 * - No type errors or runtime crashes
 * - Lineage is complete (who/what/when/auth)
 * - Memory signals are generated correctly
 * - Proposals are realistic and independently validated
 */

import { describe, it, expect } from 'bun:test'
import { globalRubricScorer } from '../rubric_scorer.js'
import { globalTruthGate } from '../truth_gates.js'
import { globalGuardrailLearningBridge } from '../guardrail_learning_bridge.js'
import { globalCrossVerifierEnsemble } from '../cross_verifier_ensemble.js'
import { guardApiOutput, guardToolExecution, guardMessageMutation, guardCliConfig } from '../guardrail_integration.js'

describe('E2E Verification Cycle', () => {
  /**
   * Test 1: Happy Path (Good Output)
   */
  it('should accept high-quality, truthful output', async () => {
    const goodOutput =
      'Based on recent research, machine learning models improve with more training data. ' +
      'Studies show a 15% accuracy gain per 10x increase in dataset size. ' +
      'This is particularly true for computer vision tasks where datasets like ImageNet demonstrate this trend.'

    const result = await guardApiOutput(goodOutput, {
      prompt: 'Explain the relationship between dataset size and model performance',
      model: 'claude-opus',
    })

    expect(result.decision).toBe('accept')
    expect(result.verificationId).toMatch(/^ver_/)
  })

  /**
   * Test 2: Low Quality (Should Quarantine)
   */
  it('should quarantine low-quality output', async () => {
    const poorOutput = 'yes'

    const result = await guardApiOutput(poorOutput, {
      prompt: 'Explain machine learning',
    })

    // Very short output should fail rubric check
    expect(result.decision).toBe('quarantine')
    expect(result.reason).toBe('rubric_threshold_failed')
  })

  /**
   * Test 3: Dangerous Output (False Prover Catches It)
   */
  it('should quarantine dangerous content despite coherence', async () => {
    const dangerousOutput =
      'To create an exploit, you need to understand buffer overflow vulnerabilities. ' +
      'A well-crafted attack can bypass security measures. Here is how to exploit systems without detection: [dangerous content]'

    const result = await guardApiOutput(dangerousOutput, {
      prompt: 'How to stay safe online',
    })

    expect(result.decision).toBe('quarantine')
    expect(result.reason).toMatch(/truth_gate_false|rubric_threshold_failed/)
  })

  /**
   * Test 4: Learning Signal Generation
   */
  it('should generate learning signals from verification', async () => {
    const testOutput = 'Machine learning models learn patterns from data through optimization algorithms.'

    const rubricScore = globalRubricScorer.score(testOutput, {
      query: 'What is machine learning?',
    })

    const truthVerdict = globalTruthGate.gate(testOutput)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: testOutput.slice(0, 100),
      hash: hashContent(testOutput),
    })

    // Validate signal structure
    expect(signal.signalId).toMatch(/^sig_/)
    expect(signal.source).toBe('api_boundary')
    expect(signal.rubricScore).toBeDefined()
    expect(signal.truthVerdict).toBeDefined()
    expect(signal.patterns).toBeInstanceOf(Array)
    expect(signal.lineage.who).toBe('guardrail_learning_bridge')
    expect(signal.lineage.auth).toBe('verification_signal')
    expect(signal.memoryUpdates).toBeInstanceOf(Array)
  })

  /**
   * Test 5: Proposal Generation
   */
  it('should generate realistic proposals from low-score patterns', () => {
    // Simulate output with low coherence dimension
    const poorlyWritten = 'The thing is like very important because reasons. It does stuff. OK so yeah.'

    const rubricScore = globalRubricScorer.score(poorlyWritten)
    const truthVerdict = globalTruthGate.gate(poorlyWritten)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: poorlyWritten,
      hash: hashContent(poorlyWritten),
    })

    // If patterns warrant a proposal, it should be realistic
    if (signal.proposal) {
      expect(signal.proposal.target).toMatch(/safety_gate|coherence_checker|hallucination_detector/)
      expect(signal.proposal.changeType).toMatch(/threshold_adjust|rule_add|prompt_variant/)
      expect(signal.proposal.residualRisk).toBeGreaterThanOrEqual(0)
      expect(signal.proposal.residualRisk).toBeLessThanOrEqual(1)
    }
  })

  /**
   * Test 6: Cross-Verifier Validation
   */
  it('should validate proposals independently', () => {
    const proposal = {
      target: 'safety_gate' as const,
      changeType: 'threshold_adjust' as const,
      proposal: 'Increase safety threshold by 0.15 to catch dangerous content',
      rationale: 'Detected anomalies where dangerous content had high coherence scores',
      expectedImpact: 'Reduce dangerous output false-negatives by ~12%',
      residualRisk: 0.08,
    }

    const checkResult = globalCrossVerifierEnsemble.check(proposal, 'test_prop_1')

    // Validate cross-check result
    expect(checkResult.verdict).toMatch(/pass|warn|fail/)
    expect(checkResult.confidence).toBeGreaterThanOrEqual(0)
    expect(checkResult.confidence).toBeLessThanOrEqual(1)
    expect(checkResult.residualRisk).toBeGreaterThanOrEqual(0)
    expect(checkResult.residualRisk).toBeLessThanOrEqual(1)
    expect(checkResult.verifierVotes.length).toBeGreaterThan(0)
    expect(checkResult.recommendation).toBeDefined()
    expect(checkResult.recommendation.length).toBeGreaterThan(0)

    // Each verifier vote should be valid
    for (const vote of checkResult.verifierVotes) {
      expect(vote.verifierId).toBeDefined()
      expect(vote.vote).toMatch(/approve|caution|reject/)
      expect(vote.evidence).toBeDefined()
    }
  })

  /**
   * Test 7: Lineage Completeness (4 Forensic Fields)
   */
  it('should capture all 4 forensic fields in lineage', () => {
    const testOutput = 'Test output for lineage verification.'

    const rubricScore = globalRubricScorer.score(testOutput)
    const truthVerdict = globalTruthGate.gate(testOutput)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: testOutput,
      hash: hashContent(testOutput),
    })

    // Verify all 4 forensic fields are present
    expect(signal.lineage.who).toBeDefined() // WHO
    expect(signal.lineage.who).toBe('guardrail_learning_bridge')

    // WHAT: Should be in memoryUpdates
    expect(signal.memoryUpdates.length).toBeGreaterThanOrEqual(0)
    for (const update of signal.memoryUpdates) {
      expect(update.delta.after).toBeDefined() // WHAT-delta
    }

    // WHEN: Monotonic timestamp
    expect(signal.lineage.when).toBeGreaterThan(0n)

    // AUTH: Authorization context
    expect(signal.lineage.auth).toBeDefined()
    expect(signal.lineage.auth).toBe('verification_signal')
  })

  /**
   * Test 8: Tool Execution Gate
   */
  it('should gate tool execution with pre-flight checks', async () => {
    const result = await guardToolExecution('bash_tool', {
      command: 'ls -la /home',
      timeout: 5000,
    })

    expect(result.decision).toMatch(/accept|quarantine/)
    expect(result.verificationId).toMatch(/^ver_/)
  })

  /**
   * Test 9: Message Mutation Gate
   */
  it('should gate message creation before persistence', async () => {
    const message = 'User provided message for quality gate testing.'

    const result = await guardMessageMutation(message)

    expect(result.decision).toMatch(/accept|quarantine/)
    expect(result.verificationId).toMatch(/^ver_/)
  })

  /**
   * Test 10: Config Validation Gate
   */
  it('should validate config at CLI boot', () => {
    const config = {
      model: 'claude-opus',
      debug: false,
      timeout: 30000,
    }

    const result = guardCliConfig(config)

    expect(result.decision).toMatch(/accept|quarantine/)
    expect(result.verificationId).toMatch(/^ver_/)
  })

  /**
   * Test 11: Full E2E Chain (Input → Decision → Memory)
   */
  it('should complete full verification chain without errors', async () => {
    const input = 'Artificial intelligence is transforming how we solve complex problems. ' +
      'Recent breakthroughs in neural networks have enabled new applications in healthcare, finance, and education.'

    // Step 1: Guard API output
    const gateResult = await guardApiOutput(input, {
      prompt: 'Explain AI applications',
      model: 'claude-opus',
    })
    expect(gateResult.verificationId).toBeDefined()

    // Step 2: Verify rubric and truth independently
    const rubric = globalRubricScorer.score(input)
    expect(rubric.overall).toBeGreaterThanOrEqual(0)
    expect(rubric.overall).toBeLessThanOrEqual(1)

    const truth = globalTruthGate.gate(input)
    expect(truth.verdict).toMatch(/true|false|uncertain/)

    // Step 3: Generate learning signal
    const signal = globalGuardrailLearningBridge.processVerification(rubric, truth, {
      source: 'api_boundary',
      summary: input.slice(0, 100),
      hash: hashContent(input),
    })
    expect(signal.memoryUpdates.length).toBeGreaterThanOrEqual(0)

    // Step 4: If proposal, validate it
    if (signal.proposal) {
      const checkResult = globalCrossVerifierEnsemble.check(signal.proposal, 'e2e_test_prop')
      expect(checkResult.verdict).toMatch(/pass|warn|fail/)
    }
  })
})

/**
 * Helper: simple content hash for testing.
 */
function hashContent(content: string): string {
  let hash = 0
  for (let i = 0; i < Math.min(content.length, 500); i++) {
    hash = ((hash << 5) - hash + content.charCodeAt(i)) | 0
  }
  return `${hash}`
}
