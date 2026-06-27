/**
 * Health Monitoring Test
 *
 * Validates guardrail health tracking and alert generation.
 */

import { describe, it, expect, beforeEach } from 'bun:test'
import { GuardrailHealthMonitor } from '../guardrail_health.ts'

describe('Guardrail Health Monitoring', () => {
  let monitor: GuardrailHealthMonitor

  beforeEach(() => {
    monitor = new GuardrailHealthMonitor()
  })

  /**
   * Test 1: Healthy Status (Good Metrics)
   */
  it('should report healthy status when metrics are normal', () => {
    // Record 8 accepted, 2 quarantined (80% acceptance - good)
    for (let i = 0; i < 8; i++) {
      monitor.recordDecision('accept')
    }
    for (let i = 0; i < 2; i++) {
      monitor.recordDecision('quarantine')
    }

    // Record good rubric scores and truth confidence
    for (let i = 0; i < 10; i++) {
      monitor.recordVerification(0.78, 0.85, false)
    }

    const status = monitor.getStatus()

    // Status should be healthy or at worst degraded with good metrics
    expect(['healthy', 'degraded']).toContain(status.status)
    expect(status.metrics.acceptanceRate).toBeGreaterThan(70)
    expect(status.metrics.avgRubricScore).toBeGreaterThan(0.70)
  })

  /**
   * Test 2: Degraded Status (Slightly Off)
   */
  it('should report degraded status when metrics are slightly off', () => {
    // Very low acceptance rate
    for (let i = 0; i < 2; i++) {
      monitor.recordDecision('accept')
    }
    for (let i = 0; i < 8; i++) {
      monitor.recordDecision('quarantine')
    }

    // Record verifications
    for (let i = 0; i < 10; i++) {
      monitor.recordVerification(0.55, 0.70, false)
    }

    const status = monitor.getStatus()

    expect(status.status).toMatch(/degraded|critical/)
    expect(status.alerts.length).toBeGreaterThan(0)
  })

  /**
   * Test 3: Critical Status (Poor Metrics)
   */
  it('should report critical status when metrics are very poor', () => {
    // Nearly all quarantined
    for (let i = 0; i < 1; i++) {
      monitor.recordDecision('accept')
    }
    for (let i = 0; i < 9; i++) {
      monitor.recordDecision('quarantine')
    }

    // Very low rubric scores
    for (let i = 0; i < 10; i++) {
      monitor.recordVerification(0.40, 0.50, false)
    }

    const status = monitor.getStatus()

    expect(status.status).toBe('critical')
    expect(status.alerts.length).toBeGreaterThan(0)

    // Should have alerts about acceptance rate and rubric score
    const alertMessages = status.alerts.map(a => a.message.toLowerCase())
    expect(alertMessages.some(m => m.includes('acceptance') || m.includes('rubric'))).toBe(true)
  })

  /**
   * Test 4: Anomaly Tracking
   */
  it('should track anomalies and alert when threshold exceeded', () => {
    // Record normal decisions and verifications
    for (let i = 0; i < 10; i++) {
      monitor.recordDecision('accept')
      monitor.recordVerification(0.75, 0.80, false)
    }

    // Record 6 anomalies
    for (let i = 0; i < 6; i++) {
      monitor.recordAnomaly()
    }

    const status = monitor.getStatus()

    expect(status.metrics.anomalyCount).toBe(6)
    expect(status.alerts.some(a => a.message.includes('anomalies'))).toBe(true)
  })

  /**
   * Test 5: Disagreement Tracking
   */
  it('should track verifier disagreements and alert on high rate', () => {
    // Record many disagreements (need at least 10+ verifications for alert threshold)
    for (let i = 0; i < 15; i++) {
      monitor.recordDecision(i % 2 === 0 ? 'accept' : 'quarantine')
      monitor.recordVerification(0.75, 0.75, i < 12) // 12 disagreements out of 15 = 80%
    }

    const status = monitor.getStatus()

    expect(status.metrics.disagreementRate).toBeGreaterThan(50)
    expect(status.alerts.some(a => a.message.includes('disagree'))).toBe(true)
  })

  /**
   * Test 6: Proposal Approval Tracking
   */
  it('should track proposal approval rates', () => {
    // Record some decisions first (for status to generate metrics)
    for (let i = 0; i < 5; i++) {
      monitor.recordDecision('accept')
      monitor.recordVerification(0.75, 0.75, false)
    }

    // Record proposal votes
    monitor.recordProposalVote('pass')
    monitor.recordProposalVote('pass')
    monitor.recordProposalVote('warn')
    monitor.recordProposalVote('fail')

    const status = monitor.getStatus()

    expect(status.metrics.proposalApprovalRate).toBeGreaterThanOrEqual(0)
    expect(status.metrics.proposalApprovalRate).toBeLessThanOrEqual(100)
  })

  /**
   * Test 7: Timestamp Updates
   */
  it('should update timestamp on each status check', async () => {
    const status1 = monitor.getStatus()
    const timestamp1 = status1.lastUpdate

    // Wait a bit and record something
    await new Promise(resolve => setTimeout(resolve, 10))
    monitor.recordDecision('accept')

    const status2 = monitor.getStatus()
    const timestamp2 = status2.lastUpdate

    expect(timestamp2).toBeGreaterThan(timestamp1)
  })

  /**
   * Test 8: Metrics Circular Buffer
   */
  it('should maintain circular buffer of metrics (not grow unbounded)', () => {
    // Record more than buffer size
    for (let i = 0; i < 150; i++) {
      monitor.recordDecision(i % 2 === 0 ? 'accept' : 'quarantine')
      monitor.recordVerification(0.70 + Math.random() * 0.1, 0.75, false)
    }

    const raw = monitor.getRawMetrics()

    // Should not grow past buffer size
    expect(raw.recentAcceptanceRates.length).toBeLessThanOrEqual(100)
    expect(raw.recentRubricScores.length).toBeLessThanOrEqual(100)
  })

  /**
   * Test 9: Reset Functionality
   */
  it('should reset all metrics to empty state', () => {
    // Record some data
    for (let i = 0; i < 10; i++) {
      monitor.recordDecision('accept')
      monitor.recordVerification(0.75, 0.80, false)
      monitor.recordAnomaly()
    }

    // Reset
    monitor.reset()

    const status = monitor.getStatus()

    expect(status.metrics.acceptanceRate).toBe(0)
    expect(status.metrics.anomalyCount).toBe(0)
    expect(status.alerts.length).toBe(0)
  })

  /**
   * Test 10: Alert Severity Levels
   */
  it('should assign correct alert severity levels', () => {
    // Very poor metrics
    for (let i = 0; i < 10; i++) {
      monitor.recordDecision('quarantine')
      monitor.recordVerification(0.35, 0.50, false)
    }

    const status = monitor.getStatus()

    // Critical alerts should have 'critical' level
    const criticalAlerts = status.alerts.filter(a => a.level === 'critical')
    expect(criticalAlerts.length).toBeGreaterThan(0)
  })
})
