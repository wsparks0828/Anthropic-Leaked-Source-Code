/**
 * Guardrail Monitoring and Alerting Configuration
 *
 * Defines alert thresholds, severity levels, and escalation paths
 * for operational monitoring of the guardrail meta-learning system.
 */

/**
 * Alert Severity Levels
 */
export type AlertSeverity = 'critical' | 'warning' | 'info'

/**
 * Alert Trigger Event
 */
export interface AlertEvent {
  severity: AlertSeverity
  metric: string
  threshold: number
  actual: number
  timestamp: bigint
  message: string
  action: string
  escalateTo?: string[]
}

/**
 * Alert Rule Definition
 */
export interface AlertRule {
  name: string
  metric: string
  threshold: number
  operator: '>' | '<' | '==' | '!='
  duration: number // milliseconds
  severity: AlertSeverity
  message: (actual: number) => string
  action: (event: AlertEvent) => void
  escalateToSlack?: string
  escalateToPagerDuty?: boolean
  escalateToEmail?: string[]
}

/**
 * Critical Alert Rules (Page On-Call)
 */
export const criticalAlerts: AlertRule[] = [
  {
    name: 'GuardrailAcceptanceRateCritical',
    metric: 'guardrail.acceptance_rate',
    threshold: 10,
    operator: '<',
    duration: 60000, // 1 minute
    severity: 'critical',
    message: (actual) =>
      `Critical: Acceptance rate is ${actual}%. Almost all outputs are being quarantined. System may be failing.`,
    action: (event) => {
      console.error(`[CRITICAL] ${event.message}`)
      // Trigger PagerDuty page
    },
    escalateToPagerDuty: true,
    escalateToSlack: '#ops-critical',
  },

  {
    name: 'GuardrailAcceptanceRateTooHigh',
    metric: 'guardrail.acceptance_rate',
    threshold: 95,
    operator: '>',
    duration: 300000, // 5 minutes
    severity: 'critical',
    message: (actual) =>
      `Critical: Acceptance rate is ${actual}%. Guardrails may be too permissive. Safety risk.`,
    action: (event) => {
      console.error(`[CRITICAL] ${event.message}`)
    },
    escalateToPagerDuty: true,
    escalateToSlack: '#ops-critical',
  },

  {
    name: 'GuardrailRubricScoreLow',
    metric: 'guardrail.rubric_score.avg',
    threshold: 0.45,
    operator: '<',
    duration: 300000, // 5 minutes
    severity: 'critical',
    message: (actual) =>
      `Critical: Average rubric score is ${actual.toFixed(2)}. Output quality degraded.`,
    action: (event) => {
      console.error(`[CRITICAL] ${event.message}`)
    },
    escalateToPagerDuty: true,
    escalateToSlack: '#ops-critical',
  },

  {
    name: 'GuardrailAnomalySurge',
    metric: 'guardrail.anomalies.count',
    threshold: 10,
    operator: '>',
    duration: 600000, // 10 minutes (per hour window)
    severity: 'critical',
    message: (actual) =>
      `Critical: ${actual} anomalies detected in last hour. Possible attack or system failure.`,
    action: (event) => {
      console.error(`[CRITICAL] ${event.message}`)
    },
    escalateToPagerDuty: true,
    escalateToSlack: '#ops-critical',
  },

  {
    name: 'GuardrailLineageBroken',
    metric: 'guardrail.lineage.chain_valid',
    threshold: 1, // expect true (1)
    operator: '!=',
    duration: 0, // immediate
    severity: 'critical',
    message: () => 'CRITICAL: Lineage chain integrity compromised. Immutability guarantee violated.',
    action: (event) => {
      console.error(`[CRITICAL] ${event.message}`)
      // Escalate to compliance immediately
    },
    escalateToPagerDuty: true,
    escalateToSlack: '#ops-critical',
    escalateToEmail: ['compliance-team@anthropic.com'],
  },
]

/**
 * Warning Alert Rules (Email + Slack, no page)
 */
export const warningAlerts: AlertRule[] = [
  {
    name: 'GuardrailAcceptanceRateDegraded',
    metric: 'guardrail.acceptance_rate',
    threshold: 25,
    operator: '<',
    duration: 300000, // 5 minutes
    severity: 'warning',
    message: (actual) =>
      `Warning: Acceptance rate is ${actual}%. Approaching critical threshold. Monitor closely.`,
    action: (event) => {
      console.warn(`[WARNING] ${event.message}`)
    },
    escalateToSlack: '#ops-guardrails',
  },

  {
    name: 'GuardrailAcceptanceRateHigh',
    metric: 'guardrail.acceptance_rate',
    threshold: 85,
    operator: '>',
    duration: 300000, // 5 minutes
    severity: 'warning',
    message: (actual) =>
      `Warning: Acceptance rate is ${actual}%. Guardrails may be too lenient.`,
    action: (event) => {
      console.warn(`[WARNING] ${event.message}`)
    },
    escalateToSlack: '#ops-guardrails',
  },

  {
    name: 'GuardrailRubricScoreDegraded',
    metric: 'guardrail.rubric_score.avg',
    threshold: 0.55,
    operator: '<',
    duration: 600000, // 10 minutes
    severity: 'warning',
    message: (actual) =>
      `Warning: Average rubric score is ${actual.toFixed(2)}. Output quality trending low.`,
    action: (event) => {
      console.warn(`[WARNING] ${event.message}`)
    },
    escalateToSlack: '#ops-guardrails',
  },

  {
    name: 'GuardrailProposalApprovalsLow',
    metric: 'guardrail.proposal_approval_rate',
    threshold: 30,
    operator: '<',
    duration: 1800000, // 30 minutes
    severity: 'warning',
    message: (actual) =>
      `Warning: Only ${actual}% of proposals approved. Cross-verifier may be too strict.`,
    action: (event) => {
      console.warn(`[WARNING] ${event.message}`)
    },
    escalateToSlack: '#ops-guardrails',
  },

  {
    name: 'GuardrailVerifierDisagreement',
    metric: 'guardrail.verifier_disagreement_rate',
    threshold: 25,
    operator: '>',
    duration: 1800000, // 30 minutes
    severity: 'warning',
    message: (actual) =>
      `Warning: Rubric and truth-gate disagree ${actual}% of the time. Calibration drift detected.`,
    action: (event) => {
      console.warn(`[WARNING] ${event.message}`)
    },
    escalateToSlack: '#ops-guardrails',
  },

  {
    name: 'GuardrailHealthDegraded',
    metric: 'guardrail.health.status',
    threshold: 0, // degraded = 0, healthy = 1
    operator: '==',
    duration: 900000, // 15 minutes
    severity: 'warning',
    message: () =>
      'Warning: Guardrail health is degraded. Monitor for escalation to critical.',
    action: (event) => {
      console.warn(`[WARNING] ${event.message}`)
    },
    escalateToSlack: '#ops-guardrails',
  },

  {
    name: 'GuardrailAnomaliesElevated',
    metric: 'guardrail.anomalies.count',
    threshold: 5,
    operator: '>',
    duration: 3600000, // 1 hour
    severity: 'warning',
    message: (actual) =>
      `Warning: ${actual} anomalies detected in last hour. Review guardrail effectiveness.`,
    action: (event) => {
      console.warn(`[WARNING] ${event.message}`)
    },
    escalateToSlack: '#ops-guardrails',
  },
]

/**
 * Info Alert Rules (Logging, no escalation)
 */
export const infoAlerts: AlertRule[] = [
  {
    name: 'GuardrailProposalGenerated',
    metric: 'guardrail.proposals.generated',
    threshold: 0,
    operator: '!=',
    duration: 0,
    severity: 'info',
    message: () => 'Info: Guardrail proposal generated. Will be validated by cross-verifier.',
    action: (event) => {
      console.log(`[INFO] ${event.message}`)
    },
  },

  {
    name: 'GuardrailLearningSignal',
    metric: 'guardrail.learning_signals.generated',
    threshold: 0,
    operator: '!=',
    duration: 0,
    severity: 'info',
    message: () => 'Info: Learning signal extracted and wired to memory layers.',
    action: (event) => {
      console.log(`[INFO] ${event.message}`)
    },
  },
]

/**
 * Alert Manager
 */
export class AlertManager {
  private activeAlerts: Map<string, AlertEvent> = new Map()
  private alertHistory: AlertEvent[] = []

  /**
   * Check a metric against all applicable rules
   */
  checkMetric(metric: string, value: number, duration: number = 0): void {
    const allRules = [...criticalAlerts, ...warningAlerts, ...infoAlerts]
    const applicableRules = allRules.filter((rule) => rule.metric === metric)

    for (const rule of applicableRules) {
      const isBreach = this.evaluateRule(rule, value)

      if (isBreach && duration >= rule.duration) {
        this.triggerAlert(rule, value)
      }
    }
  }

  /**
   * Evaluate if a rule condition is met
   */
  private evaluateRule(rule: AlertRule, value: number): boolean {
    switch (rule.operator) {
      case '>':
        return value > rule.threshold
      case '<':
        return value < rule.threshold
      case '==':
        return value === rule.threshold
      case '!=':
        return value !== rule.threshold
      default:
        return false
    }
  }

  /**
   * Trigger an alert
   */
  private triggerAlert(rule: AlertRule, actual: number): void {
    const event: AlertEvent = {
      severity: rule.severity,
      metric: rule.metric,
      threshold: rule.threshold,
      actual,
      timestamp: BigInt(Date.now()),
      message: rule.message(actual),
      action: rule.action.name,
      escalateTo: [],
    }

    // Add escalation targets
    if (rule.escalateToSlack) {
      event.escalateTo?.push(`slack:${rule.escalateToSlack}`)
    }
    if (rule.escalateToPagerDuty) {
      event.escalateTo?.push('pagerduty:on-call-security')
    }
    if (rule.escalateToEmail) {
      event.escalateTo?.push(...rule.escalateToEmail.map((e) => `email:${e}`))
    }

    // Trigger action
    rule.action(event)

    // Track alert
    this.activeAlerts.set(rule.name, event)
    this.alertHistory.push(event)

    // Keep history bounded
    if (this.alertHistory.length > 10000) {
      this.alertHistory.shift()
    }
  }

  /**
   * Get current active alerts
   */
  getActiveAlerts(): AlertEvent[] {
    return Array.from(this.activeAlerts.values())
  }

  /**
   * Clear a resolved alert
   */
  clearAlert(ruleName: string): void {
    this.activeAlerts.delete(ruleName)
  }

  /**
   * Get alert history
   */
  getHistory(limit: number = 100): AlertEvent[] {
    return this.alertHistory.slice(-limit)
  }

  /**
   * Export alert history for persistence (called during graceful shutdown)
   */
  exportAlertHistory(): AlertEvent[] {
    return [...this.alertHistory]
  }

  /**
   * Reset alert manager (for testing)
   */
  reset(): void {
    this.activeAlerts.clear()
    this.alertHistory = []
  }
}

/**
 * Global alert manager instance
 */
export const globalAlertManager = new AlertManager()

/**
 * Convenience function: check metric and trigger alerts if needed
 */
export function checkMetricAndAlert(metric: string, value: number, duration: number = 0): void {
  globalAlertManager.checkMetric(metric, value, duration)
}
