/**
 * Comprehensive Type Definitions for Guardrail Meta-Learning System
 *
 * Exports all types used across:
 * - rubric_scorer.ts
 * - truth_gates.ts
 * - guardrail_learning_bridge.ts
 * - cross_verifier_ensemble.ts
 *
 * Central location for schema validation, type consistency, and documentation.
 */

import type { RubricScore, RubricDimension } from './rubric_scorer.js'
import type { TruthGateResult, TruthVerdict, FalseSeverity } from './truth_gates.js'
import type { LearningSignal, GuardrailProposal, MemoryUpdate } from './guardrail_learning_bridge.js'
import type { CrossCheckResult, CrossCheckVerdict, CrossCheckReason } from './cross_verifier_ensemble.js'

/**
 * Composite result of guardrail verification + learning pipeline.
 */
export interface GuardrailVerificationResult {
  /**
   * Input that was verified.
   */
  input: {
    content: string
    source: 'api' | 'tool' | 'message' | 'config'
    metadata?: Record<string, unknown>
  }

  /**
   * Rubric scoring results.
   */
  rubric: RubricScore

  /**
   * Truth/false verification.
   */
  truth: TruthGateResult

  /**
   * Learning signal generated from verification.
   */
  learning: LearningSignal

  /**
   * If a proposal was generated, its cross-check result.
   */
  proposalCrossCheck?: CrossCheckResult

  /**
   * Overall decision: ACCEPT / QUARANTINE.
   */
  decision: 'accept' | 'quarantine'

  /**
   * Confidence in the decision (0–1).
   */
  confidence: number

  /**
   * Lineage for audit trail.
   */
  lineage: {
    verificationId: string
    timestamp: bigint
    path: string // "api→rubric→truth→learn→proposal→crosscheck"
  }
}

/**
 * Configuration for guardrail thresholds.
 * Can be updated by learning loop proposals.
 */
export interface GuardrailConfig {
  /**
   * Rubric composite score threshold for acceptance (0–1).
   * Default: 0.65
   */
  rubricThreshold: number

  /**
   * Truth verdict threshold.
   * 'true' = accept, 'uncertain' = depends on rubric, 'false' = quarantine
   */
  truthThreshold: TruthVerdict

  /**
   * Cross-check requirement for proposals.
   * 'pass' = apply, 'warn' = apply cautiously, 'fail' = quarantine
   */
  crossCheckThreshold: CrossCheckVerdict

  /**
   * Max residual risk allowed for automatic application.
   * Default: 0.2 (20%)
   */
  maxResidualRisk: number

  /**
   * Feature flags for specific guardrails.
   */
  features: {
    enableRubricScoring: boolean
    enableTruthGating: boolean
    enableLearningBridge: boolean
    enableCrossVerifier: boolean
  }

  /**
   * Versioning for config changes.
   */
  version: string
  updatedAt: bigint
}

/**
 * Status of guardrail system health.
 */
export interface GuardrailHealthStatus {
  /**
   * Overall health: healthy / degraded / critical
   */
  status: 'healthy' | 'degraded' | 'critical'

  /**
   * Metrics for monitoring.
   */
  metrics: {
    /**
     * % of outputs accepted (trend).
     */
    acceptanceRate: number

    /**
     * % of proposals approved by cross-check (trend).
     */
    proposalApprovalRate: number

    /**
     * Avg rubric score (trend).
     */
    avgRubricScore: number

    /**
     * Avg truth verdict confidence (trend).
     */
    truthConfidence: number

    /**
     * Recent anomalies detected.
     */
    anomalyCount: number

    /**
     * Verifier disagreement rate.
     */
    disagreementRate: number
  }

  /**
   * Last update.
   */
  lastUpdate: bigint

  /**
   * Alerts (if any).
   */
  alerts: Array<{
    level: 'info' | 'warning' | 'critical'
    message: string
  }>
}

/**
 * Audit record for a single verification + learning cycle.
 */
export interface GuardrailAuditRecord {
  /**
   * Unique ID.
   */
  recordId: string

  /**
   * Verification result.
   */
  verificationResult: GuardrailVerificationResult

  /**
   * Decision rationale.
   */
  rationale: string

  /**
   * Memory updates committed.
   */
  memoryUpdatesApplied: MemoryUpdate[]

  /**
   * Proposal (if any) + decision.
   */
  proposalDecision?: {
    proposal: GuardrailProposal
    crossCheckResult: CrossCheckResult
    applied: boolean
  }

  /**
   * Immutable chain hash (for tamper detection).
   */
  chainHash: string

  /**
   * Previous record ID (for chain linking).
   */
  prevRecordId: string | null

  /**
   * Timestamp.
   */
  timestamp: bigint
}

/**
 * Export all types for external use.
 */
export type {
  RubricScore,
  RubricDimension,
  TruthGateResult,
  TruthVerdict,
  FalseSeverity,
  LearningSignal,
  GuardrailProposal,
  MemoryUpdate,
  CrossCheckResult,
  CrossCheckVerdict,
  CrossCheckReason,
}

/**
 * Default config values.
 */
export const DEFAULT_GUARDRAIL_CONFIG: GuardrailConfig = {
  rubricThreshold: 0.65,
  truthThreshold: 'true',
  crossCheckThreshold: 'pass',
  maxResidualRisk: 0.2,
  features: {
    enableRubricScoring: true,
    enableTruthGating: true,
    enableLearningBridge: true,
    enableCrossVerifier: true,
  },
  version: '1.0.0',
  updatedAt: BigInt(Date.now()) * BigInt(1_000_000),
}
