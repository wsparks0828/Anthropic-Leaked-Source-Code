/**
 * Guardrail Learning Bridge
 *
 * Transforms verification results → learning signals → memory updates → guardrail improvements
 *
 * Flow:
 * 1. Input: RubricScore + TruthGateResult
 * 2. Extract patterns, gaps, disagreements
 * 3. Wire into memory layers (semantic, episodic, graph)
 * 4. Generate improvement proposals
 * 5. Route to cross-verifier for gated approval
 *
 * This is the feedback loop: every verification → continuous improvement
 */

import { RubricScore, RubricDimension } from './rubric_scorer.js'
import { TruthGateResult, TruthVerdict } from './truth_gates.js'

/**
 * Learning signal extracted from verification.
 * Feeds into memory layers + proposal generation.
 */
export interface LearningSignal {
  /**
   * ID for tracking across the learning pipeline.
   */
  signalId: string

  /**
   * Where this signal came from (API, tool, message).
   */
  source: 'api_boundary' | 'tool_execution' | 'message_creation' | 'config_validation'

  /**
   * What was verified.
   */
  inputHash: string
  inputSummary: string

  /**
   * Verification results.
   */
  rubricScore: RubricScore
  truthVerdict: TruthGateResult

  /**
   * Extracted patterns (for memory wiring).
   */
  patterns: Array<{
    category: 'low_score_dimension' | 'edge_case' | 'disagreement' | 'anomaly'
    dimension?: RubricDimension | string
    severity: 'critical' | 'high' | 'medium' | 'low'
    description: string
  }>

  /**
   * Proposed guardrail improvement (for cross-check).
   */
  proposal?: GuardrailProposal

  /**
   * Lineage tracking (who/what/when/auth).
   */
  lineage: {
    who: string
    when: bigint
    auth: string
  }

  /**
   * Memory writes (semantic, episodic, graph).
   */
  memoryUpdates: MemoryUpdate[]
}

/**
 * Proposed improvement to a guardrail or verifier.
 */
export interface GuardrailProposal {
  /**
   * Target: which guardrail/verifier to improve.
   */
  target: 'prompt_injection_guard' | 'hallucination_detector' | 'coherence_checker' | 'safety_gate'

  /**
   * Type of change.
   */
  changeType: 'threshold_adjust' | 'rule_add' | 'rule_remove' | 'prompt_variant'

  /**
   * The proposed change (specific).
   */
  proposal: string

  /**
   * Why this helps (backed by evidence).
   */
  rationale: string

  /**
   * Expected impact on guardrail behavior.
   */
  expectedImpact: string

  /**
   * Estimated risk of change (0–1).
   */
  residualRisk: number
}

/**
 * Memory layer update (append-only, lineage-tracked).
 */
export interface MemoryUpdate {
  /**
   * Which layer: semantic (policy), episodic (history), graph (knowledge).
   */
  layer: 'semantic' | 'episodic' | 'graph'

  /**
   * What was updated.
   */
  key: string

  /**
   * Delta (before/after).
   */
  delta: {
    before?: unknown
    after: unknown
  }

  /**
   * Immutable record (for audit).
   */
  recordId: string
}

/**
 * Guardrail Learning Bridge: Main orchestrator.
 */
export class GuardrailLearningBridge {
  /**
   * Process verification results → learning signals → memory updates.
   */
  processVerification(
    rubricScore: RubricScore,
    truthVerdict: TruthGateResult,
    input: { source: LearningSignal['source']; summary: string; hash: string },
  ): LearningSignal {
    const signalId = this.generateSignalId()
    const now = BigInt(Date.now()) * BigInt(1_000_000)

    // Extract learning patterns from verification
    const patterns = this.extractPatterns(rubricScore, truthVerdict)

    // Wire into memory layers
    const memoryUpdates = this.wireMemory(patterns, rubricScore)

    // Generate improvement proposal if patterns warrant it
    const proposal = this.generateProposal(patterns, rubricScore, truthVerdict)

    const signal: LearningSignal = {
      signalId,
      source: input.source,
      inputHash: input.hash,
      inputSummary: input.summary,
      rubricScore,
      truthVerdict,
      patterns,
      proposal,
      lineage: {
        who: 'guardrail_learning_bridge',
        when: now,
        auth: 'verification_signal',
      },
      memoryUpdates,
    }

    return signal
  }

  /**
   * Extract learning patterns from verification scores + truth verdict.
   */
  private extractPatterns(
    rubricScore: RubricScore,
    truthVerdict: TruthGateResult,
  ): LearningSignal['patterns'] {
    const patterns: LearningSignal['patterns'] = []

    // Pattern 1: Low-scoring dimensions (improvement targets)
    for (const [dim, score] of Object.entries(rubricScore.dimensions)) {
      if (score < 0.4) {
        patterns.push({
          category: 'low_score_dimension',
          dimension: dim as RubricDimension,
          severity: score < 0.2 ? 'critical' : 'high',
          description: `${dim} scored ${score}: weak point in output quality`,
        })
      }
    }

    // Pattern 2: Edge cases (unusual but real)
    if (rubricScore.lowestDimension && rubricScore.overall > 0.7 && rubricScore.lowestDimension !== rubricScore.highestDimension) {
      patterns.push({
        category: 'edge_case',
        dimension: rubricScore.lowestDimension,
        severity: 'medium',
        description: `Edge case: high overall score but weak ${rubricScore.lowestDimension}`,
      })
    }

    // Pattern 3: Disagreement (rubric vs truth gate)
    if (
      (rubricScore.overall > 0.7 && truthVerdict.verdict === 'false') ||
      (rubricScore.overall < 0.4 && truthVerdict.verdict === 'true')
    ) {
      patterns.push({
        category: 'disagreement',
        severity: 'high',
        description: `Verifier disagreement: rubric says ${rubricScore.overall.toFixed(2)}, truth gate says ${truthVerdict.verdict}`,
      })
    }

    // Pattern 4: Anomalies (unusual combinations)
    if (truthVerdict.severity === 'critical' && rubricScore.overall > 0.5) {
      patterns.push({
        category: 'anomaly',
        severity: 'critical',
        description: 'ANOMALY: Dangerous content slipped past guardrails (high score + critical falsity)',
      })
    }

    return patterns
  }

  /**
   * Wire learning signals into memory layers.
   */
  private wireMemory(patterns: LearningSignal['patterns'], rubricScore: RubricScore): MemoryUpdate[] {
    const updates: MemoryUpdate[] = []
    const recordId = this.generateRecordId()

    // Semantic layer: guardrail policy vectors
    for (const pattern of patterns.filter(p => p.category === 'low_score_dimension')) {
      updates.push({
        layer: 'semantic',
        key: `guardrail_policy:${pattern.dimension}:weak`,
        delta: {
          after: {
            dimension: pattern.dimension,
            weakness: pattern.description,
            timestamp: Date.now(),
          },
        },
        recordId,
      })
    }

    // Episodic layer: verifier calibration history
    updates.push({
      layer: 'episodic',
      key: `verifier_accuracy:${recordId}`,
      delta: {
        after: {
          overall_score: rubricScore.overall,
          dimensions: rubricScore.dimensions,
          confidence: rubricScore.confidence,
          timestamp: Date.now(),
        },
      },
      recordId,
    })

    // Graph layer: boundary knowledge nodes
    if (patterns.some(p => p.category === 'anomaly')) {
      updates.push({
        layer: 'graph',
        key: `boundary_knowledge:dangerous_pattern`,
        delta: {
          after: {
            pattern: 'high_score_but_dangerous',
            instances: 1,
            severity: 'critical',
          },
        },
        recordId,
      })
    }

    return updates
  }

  /**
   * Generate improvement proposal if patterns warrant action.
   */
  private generateProposal(
    patterns: LearningSignal['patterns'],
    rubricScore: RubricScore,
    truthVerdict: TruthGateResult,
  ): GuardrailProposal | undefined {
    // Only propose if there's a clear signal
    const criticalPatterns = patterns.filter(p => p.severity === 'critical')
    const anomalies = patterns.filter(p => p.category === 'anomaly')

    if (anomalies.length > 0 && truthVerdict.verdict === 'false' && truthVerdict.severity === 'critical') {
      // Critical proposal: tighten dangerous content detector
      return {
        target: 'safety_gate',
        changeType: 'threshold_adjust',
        proposal: 'Increase safety threshold by 0.15 to catch dangerous content that slips past coherence checks',
        rationale: `Detected ${anomalies.length} anomalies where dangerous content had high coherence score`,
        expectedImpact: 'Reduce dangerous output false-negatives by ~12%',
        residualRisk: 0.08, // Low risk: making safety stricter
      }
    }

    if (criticalPatterns.length > 0) {
      const worstDim = rubricScore.lowestDimension
      if (worstDim) {
        return {
          target: 'coherence_checker',
          changeType: 'prompt_variant',
          proposal: `Add prompt variant emphasizing ${worstDim} evaluation`,
          rationale: `${worstDim} scored <0.2 in ${criticalPatterns.length} cases`,
          expectedImpact: `Improve ${worstDim} scoring by ~0.2`,
          residualRisk: 0.12,
        }
      }
    }

    return undefined
  }

  /**
   * Trigger autoDream if enough learning signals accumulated.
   */
  shouldTriggerAutoDream(recentSignals: LearningSignal[]): boolean {
    if (recentSignals.length < 10) return false

    const anomalyCount = recentSignals.filter(s => s.patterns.some(p => p.category === 'anomaly')).length
    const disagreementCount = recentSignals.filter(s => s.patterns.some(p => p.category === 'disagreement')).length

    return anomalyCount > 2 || disagreementCount > 5
  }

  /**
   * Helper: generate unique signal ID.
   */
  private generateSignalId(): string {
    return `sig_${Date.now()}_${Math.random().toString(36).slice(2)}`
  }

  /**
   * Helper: generate unique record ID for lineage tracking.
   */
  private generateRecordId(): string {
    return `rec_${Date.now()}_${Math.random().toString(36).slice(2)}`
  }
}

/**
 * Global bridge instance.
 */
export const globalGuardrailLearningBridge = new GuardrailLearningBridge()

/**
 * Convenience: process verification and return signal.
 */
export function learnFromVerification(
  rubricScore: RubricScore,
  truthVerdict: TruthGateResult,
  input: { source: LearningSignal['source']; summary: string; hash: string },
): LearningSignal {
  return globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, input)
}
