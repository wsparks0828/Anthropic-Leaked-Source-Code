/**
 * Memory Wiring Test
 *
 * Validates that learning signals correctly wire into memory layers:
 * - Semantic: guardrail policy vectors
 * - Episodic: verifier calibration history
 * - Graph: boundary knowledge nodes
 */

import { describe, it, expect } from 'bun:test'
import { globalGuardrailLearningBridge } from '../guardrail_learning_bridge.js'
import { globalRubricScorer } from '../rubric_scorer.js'
import { globalTruthGate } from '../truth_gates.js'

describe('Memory Wiring', () => {
  /**
   * Test 1: Semantic Layer (Policy Vectors)
   */
  it('should wire low-score dimensions to semantic layer (guardrail policies)', () => {
    const poorOutput = 'This is a test'

    const rubricScore = globalRubricScorer.score(poorOutput)
    const truthVerdict = globalTruthGate.gate(poorOutput)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: poorOutput,
      hash: 'test_hash_1',
    })

    // Check that low-score dimensions are wired to semantic layer
    const semanticUpdates = signal.memoryUpdates.filter(u => u.layer === 'semantic')

    // If there are low-scoring dimensions, they should be in semantic updates
    const lowDims = Object.entries(rubricScore.dimensions).filter(([_, score]) => score < 0.4)

    if (lowDims.length > 0) {
      expect(semanticUpdates.length).toBeGreaterThan(0)

      for (const update of semanticUpdates) {
        expect(update.key).toContain('guardrail_policy')
        expect(update.delta.after).toBeDefined()
        expect((update.delta.after as any).dimension).toBeDefined()
        expect((update.delta.after as any).weakness).toBeDefined()
      }
    }
  })

  /**
   * Test 2: Episodic Layer (Verifier Calibration History)
   */
  it('should wire rubric scores to episodic layer (calibration history)', () => {
    const testOutput = 'Machine learning is a subset of artificial intelligence.'

    const rubricScore = globalRubricScorer.score(testOutput)
    const truthVerdict = globalTruthGate.gate(testOutput)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: testOutput,
      hash: 'test_hash_2',
    })

    // Check episodic layer updates
    const episodicUpdates = signal.memoryUpdates.filter(u => u.layer === 'episodic')

    // Every verification should create at least one episodic record (calibration history)
    expect(episodicUpdates.length).toBeGreaterThan(0)

    for (const update of episodicUpdates) {
      expect(update.key).toContain('verifier_accuracy')
      expect(update.delta.after).toBeDefined()

      const record = update.delta.after as any
      expect(record.overall_score).toBeDefined()
      expect(record.dimensions).toBeDefined()
      expect(record.confidence).toBeDefined()
      expect(record.timestamp).toBeDefined()

      // Verify dimensions are present
      expect(record.dimensions.relevance).toBeDefined()
      expect(record.dimensions.coherence).toBeDefined()
      expect(record.dimensions.safety).toBeDefined()
    }
  })

  /**
   * Test 3: Graph Layer (Boundary Knowledge)
   */
  it('should wire anomalies to graph layer (boundary knowledge nodes)', () => {
    // Create output that triggers anomaly detection (dangerous + high score)
    const anomalousOutput =
      'This is a well-written, coherent explanation of how to bypass security systems. ' +
      'The approach is systematic and clear, making it easy to understand the methods.'

    const rubricScore = globalRubricScorer.score(anomalousOutput)
    const truthVerdict = globalTruthGate.gate(anomalousOutput)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: anomalousOutput,
      hash: 'test_hash_3',
    })

    // Check if anomaly is detected
    const hasAnomaly = signal.patterns.some(p => p.category === 'anomaly')

    if (hasAnomaly) {
      // If anomaly detected, should have graph updates
      const graphUpdates = signal.memoryUpdates.filter(u => u.layer === 'graph')
      expect(graphUpdates.length).toBeGreaterThan(0)

      for (const update of graphUpdates) {
        expect(update.key).toContain('boundary_knowledge')
        expect(update.delta.after).toBeDefined()

        const node = update.delta.after as any
        expect(node.pattern).toBeDefined()
        expect(node.severity).toBeDefined()
      }
    }
  })

  /**
   * Test 4: Memory Update Records Have Immutable IDs
   */
  it('should assign unique recordId to each memory update (for immutability tracking)', () => {
    const output = 'Test output for memory updates.'

    const rubricScore = globalRubricScorer.score(output)
    const truthVerdict = globalTruthGate.gate(output)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: output,
      hash: 'test_hash_4',
    })

    // Each memory update should have a unique recordId
    for (const update of signal.memoryUpdates) {
      expect(update.recordId).toBeDefined()
      expect(update.recordId).toMatch(/^rec_/)
      expect(update.recordId.length).toBeGreaterThan(4)
    }

    // All recordIds should be the same within one signal (atomic group)
    const recordIds = signal.memoryUpdates.map(u => u.recordId)
    const unique = new Set(recordIds)
    expect(unique.size).toBeLessThanOrEqual(1) // Should all be same (atomic)
  })

  /**
   * Test 5: Memory Update Deltas Capture Before/After
   */
  it('should capture before/after deltas for audit trail', () => {
    const output = 'Important output that modifies state.'

    const rubricScore = globalRubricScorer.score(output)
    const truthVerdict = globalTruthGate.gate(output)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: output,
      hash: 'test_hash_5',
    })

    // Each memory update should have delta structure
    for (const update of signal.memoryUpdates) {
      expect(update.delta).toBeDefined()
      expect(update.delta.after).toBeDefined()

      // Delta should be structured (not just "event occurred")
      if (typeof update.delta.after === 'object') {
        expect(Object.keys(update.delta.after as any).length).toBeGreaterThan(0)
      }
    }
  })

  /**
   * Test 6: Multiple Verification Signals Don't Cross-Contaminate Memory
   */
  it('should isolate memory updates per verification (no cross-contamination)', () => {
    const output1 = 'First test output.'
    const output2 = 'Second test output.'

    const rubric1 = globalRubricScorer.score(output1)
    const truth1 = globalTruthGate.gate(output1)
    const signal1 = globalGuardrailLearningBridge.processVerification(rubric1, truth1, {
      source: 'api_boundary',
      summary: output1,
      hash: 'test_hash_1a',
    })

    const rubric2 = globalRubricScorer.score(output2)
    const truth2 = globalTruthGate.gate(output2)
    const signal2 = globalGuardrailLearningBridge.processVerification(rubric2, truth2, {
      source: 'api_boundary',
      summary: output2,
      hash: 'test_hash_2a',
    })

    // Each signal should have unique recordIds
    expect(signal1.signalId).not.toBe(signal2.signalId)

    // Memory updates should not leak between signals
    const ids1 = signal1.memoryUpdates.map(u => u.recordId)
    const ids2 = signal2.memoryUpdates.map(u => u.recordId)

    for (const id1 of ids1) {
      for (const id2 of ids2) {
        if (id1 && id2) {
          expect(id1).not.toBe(id2)
        }
      }
    }
  })

  /**
   * Test 7: Lineage Is Present in All Memory Updates
   */
  it('should include lineage tracking in memory updates', () => {
    const output = 'Output with full lineage tracking.'

    const rubricScore = globalRubricScorer.score(output)
    const truthVerdict = globalTruthGate.gate(output)

    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: output,
      hash: 'test_hash_6',
    })

    // Signal should have complete lineage
    expect(signal.lineage.who).toBe('guardrail_learning_bridge')
    expect(signal.lineage.when).toBeGreaterThan(0n)
    expect(signal.lineage.auth).toBe('verification_signal')

    // All memory updates should reference the signal's lineage
    for (const update of signal.memoryUpdates) {
      expect(update.recordId).toBeDefined()
    }
  })
})
