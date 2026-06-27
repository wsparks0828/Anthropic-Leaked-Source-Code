/**
 * Guardrail Health Status Reporting
 *
 * Tracks and reports:
 * - Acceptance rate (% of outputs accepted)
 * - Proposal approval rate (% of proposals approved)
 * - Rubric score trends
 * - Truth verdict confidence
 * - Anomalies detected
 * - Verifier disagreement rate
 * - Alerts for concerning patterns
 */

import { GuardrailHealthStatus } from './schemas.js'

/**
 * Metrics accumulator (in-memory, circular buffer for recent data).
 */
interface MetricsBuffer {
  acceptanceRates: number[] // Last N decisions (1 = accept, 0 = quarantine)
  proposalApprovalRates: number[] // Last N proposal votes
  rubricScores: number[] // Last N rubric composite scores
  truthConfidences: number[] // Last N truth verdict confidences
  anomalyCount: number // Total anomalies this session
  disagreementCount: number // Times rubric ≠ truth verdict
}

/**
 * Guardrail Health Monitor
 */
export class GuardrailHealthMonitor {
  private metrics: MetricsBuffer = {
    acceptanceRates: [],
    proposalApprovalRates: [],
    rubricScores: [],
    truthConfidences: [],
    anomalyCount: 0,
    disagreementCount: 0,
  }

  private maxBufferSize = 100 // Keep last 100 decisions
  private lastUpdate = BigInt(Date.now()) * BigInt(1_000_000)

  /**
   * Record a verification decision (accept/quarantine).
   */
  recordDecision(decision: 'accept' | 'quarantine'): void {
    const accepted = decision === 'accept' ? 1 : 0
    this.metrics.acceptanceRates.push(accepted)

    if (this.metrics.acceptanceRates.length > this.maxBufferSize) {
      this.metrics.acceptanceRates.shift()
    }

    this.lastUpdate = BigInt(Date.now()) * BigInt(1_000_000)
  }

  /**
   * Record a proposal approval (pass/warn/fail).
   */
  recordProposalVote(verdict: 'pass' | 'warn' | 'fail'): void {
    const approved = verdict === 'pass' ? 1 : verdict === 'warn' ? 0.5 : 0
    this.metrics.proposalApprovalRates.push(approved)

    if (this.metrics.proposalApprovalRates.length > this.maxBufferSize) {
      this.metrics.proposalApprovalRates.shift()
    }
  }

  /**
   * Record rubric score and truth confidence.
   */
  recordVerification(rubricScore: number, truthConfidence: number, disagreement: boolean): void {
    this.metrics.rubricScores.push(rubricScore)
    this.metrics.truthConfidences.push(truthConfidence)

    if (this.metrics.rubricScores.length > this.maxBufferSize) {
      this.metrics.rubricScores.shift()
      this.metrics.truthConfidences.shift()
    }

    if (disagreement) {
      this.metrics.disagreementCount += 1
    }
  }

  /**
   * Record anomaly detection.
   */
  recordAnomaly(): void {
    this.metrics.anomalyCount += 1
  }

  /**
   * Get current health status.
   */
  getStatus(): GuardrailHealthStatus {
    const acceptanceRate = this.computeAcceptanceRate()
    const proposalApprovalRate = this.computeProposalApprovalRate()
    const avgRubricScore = this.computeAvgRubricScore()
    const truthConfidence = this.computeAvgTruthConfidence()

    const status = this.determineHealthStatus(acceptanceRate, proposalApprovalRate, avgRubricScore)
    const alerts = this.generateAlerts(acceptanceRate, proposalApprovalRate, avgRubricScore)

    return {
      status,
      metrics: {
        acceptanceRate,
        proposalApprovalRate,
        avgRubricScore,
        truthConfidence,
        anomalyCount: this.metrics.anomalyCount,
        disagreementRate: this.computeDisagreementRate(),
      },
      lastUpdate: this.lastUpdate,
      alerts,
    }
  }

  /**
   * Compute acceptance rate (% accepted).
   */
  private computeAcceptanceRate(): number {
    if (this.metrics.acceptanceRates.length === 0) return 0

    const accepted = this.metrics.acceptanceRates.filter(r => r === 1).length
    return Math.round((accepted / this.metrics.acceptanceRates.length) * 100)
  }

  /**
   * Compute proposal approval rate.
   */
  private computeProposalApprovalRate(): number {
    if (this.metrics.proposalApprovalRates.length === 0) return 0

    const avg = this.metrics.proposalApprovalRates.reduce((a, b) => a + b, 0) / this.metrics.proposalApprovalRates.length

    return Math.round(avg * 100)
  }

  /**
   * Compute average rubric score.
   */
  private computeAvgRubricScore(): number {
    if (this.metrics.rubricScores.length === 0) return 0

    const avg = this.metrics.rubricScores.reduce((a, b) => a + b, 0) / this.metrics.rubricScores.length

    return Math.round(avg * 100) / 100
  }

  /**
   * Compute average truth confidence.
   */
  private computeAvgTruthConfidence(): number {
    if (this.metrics.truthConfidences.length === 0) return 0

    const avg = this.metrics.truthConfidences.reduce((a, b) => a + b, 0) / this.metrics.truthConfidences.length

    return Math.round(avg * 100) / 100
  }

  /**
   * Compute verifier disagreement rate.
   */
  private computeDisagreementRate(): number {
    if (this.metrics.rubricScores.length === 0) return 0

    return Math.round((this.metrics.disagreementCount / this.metrics.rubricScores.length) * 100)
  }

  /**
   * Determine overall health status.
   */
  private determineHealthStatus(
    acceptanceRate: number,
    proposalApprovalRate: number,
    avgRubricScore: number,
  ): 'healthy' | 'degraded' | 'critical' {
    // Critical: acceptance too low or too high, or rubric score failing
    if (acceptanceRate < 10 || acceptanceRate > 95 || avgRubricScore < 0.45) {
      return 'critical'
    }

    // Degraded: slightly off thresholds
    if (acceptanceRate < 25 || acceptanceRate > 85 || avgRubricScore < 0.55 || proposalApprovalRate < 30) {
      return 'degraded'
    }

    return 'healthy'
  }

  /**
   * Generate alerts based on trends.
   */
  private generateAlerts(
    acceptanceRate: number,
    proposalApprovalRate: number,
    avgRubricScore: number,
  ): GuardrailHealthStatus['alerts'] {
    const alerts: GuardrailHealthStatus['alerts'] = []

    // Don't alert if no data yet
    if (this.metrics.acceptanceRates.length === 0) {
      return alerts
    }

    // Alert 1: Acceptance rate anomaly
    if (acceptanceRate < 10) {
      alerts.push({
        level: 'critical',
        message: `Critical: Acceptance rate is ${acceptanceRate}%. Almost all outputs are being quarantined.`,
      })
    } else if (acceptanceRate > 95) {
      alerts.push({
        level: 'warning',
        message: `Warning: Acceptance rate is ${acceptanceRate}%. Guardrails may be too lenient.`,
      })
    }

    // Alert 2: Rubric score drop
    if (avgRubricScore < 0.5) {
      alerts.push({
        level: 'critical',
        message: `Critical: Average rubric score is ${avgRubricScore}. Output quality is degraded.`,
      })
    } else if (avgRubricScore < 0.6) {
      alerts.push({
        level: 'warning',
        message: `Warning: Average rubric score is ${avgRubricScore}. Monitor output quality.`,
      })
    }

    // Alert 3: Proposal approval collapse
    if (proposalApprovalRate < 20) {
      alerts.push({
        level: 'warning',
        message: `Warning: Only ${proposalApprovalRate}% of proposals are approved. Cross-verifier may be too strict.`,
      })
    }

    // Alert 4: Anomaly surge
    if (this.metrics.anomalyCount > 5) {
      alerts.push({
        level: 'warning',
        message: `Warning: ${this.metrics.anomalyCount} anomalies detected. Review guardrail effectiveness.`,
      })
    }

    // Alert 5: High disagreement
    const disagreementRate = this.computeDisagreementRate()
    if (disagreementRate > 30) {
      alerts.push({
        level: 'warning',
        message: `Warning: Rubric and truth-gate disagree ${disagreementRate}% of the time. May need recalibration.`,
      })
    }

    return alerts
  }

  /**
   * Reset metrics (e.g., at session end or for new baseline).
   */
  reset(): void {
    this.metrics = {
      acceptanceRates: [],
      proposalApprovalRates: [],
      rubricScores: [],
      truthConfidences: [],
      anomalyCount: 0,
      disagreementCount: 0,
    }

    this.lastUpdate = BigInt(Date.now()) * BigInt(1_000_000)
  }

  /**
   * Get raw metrics for detailed inspection.
   */
  getRawMetrics() {
    return {
      recentAcceptanceRates: this.metrics.acceptanceRates,
      recentRubricScores: this.metrics.rubricScores,
      recentTruthConfidences: this.metrics.truthConfidences,
      totalAnomalies: this.metrics.anomalyCount,
      totalDisagreements: this.metrics.disagreementCount,
    }
  }
}

/**
 * Global health monitor instance.
 */
export const globalHealthMonitor = new GuardrailHealthMonitor()

/**
 * Convenience: get current health status.
 */
export function getGuardrailHealthStatus(): GuardrailHealthStatus {
  return globalHealthMonitor.getStatus()
}
