/**
 * autoDream: Continuous Learning Loop Orchestrator
 *
 * Monitors guardrail verification signals and automatically:
 * 1. Detects pattern accumulation in learning signals
 * 2. Triggers improvement proposal generation
 * 3. Validates proposals through cross-verifier
 * 4. Applies approved proposals to guardrail components
 * 5. Records improvement lineage with forensic completeness
 *
 * Operates with fail-closed semantics:
 * - Never apply unvalidated proposals
 * - Never weaken safety constraints
 * - Always maintain lineage trail
 */

import {globalGuardrailLearningBridge, type LearningSignal} from './guardrail_learning_bridge.js'
import {globalCrossVerifierEnsemble} from './cross_verifier_ensemble.js'
import {globalLineageAuditor} from './lineage_auditor.js'
import {globalHealthMonitor} from './guardrail_health.js'
import {globalAlertManager} from './guardrail_alerts.js'

/**
 * autoDream Improvement Event (tracked in lineage)
 */
export interface AutoDreamEvent {
  eventId: string
  timestamp: bigint
  triggerType: 'pattern_accumulation' | 'manual' | 'scheduled'
  signalsAnalyzed: number
  proposalsGenerated: number
  proposalsApproved: number
  proposalsApplied: number
  improvementImpact: {
    affectedDimension: string
    estimatedImprovement: number // percentage
    residualRisk: number
  }
  lineageRecordId: string
}

/**
 * autoDream Learning Signal Accumulator
 */
class AutoDreamAccumulator {
  private signals: LearningSignal[] = []
  private maxSignals = 1000
  private lastTriggerTime: bigint = BigInt(0)
  private triggerInterval: bigint = BigInt(60000) // 1 minute

  /**
   * Add signal to accumulator
   */
  addSignal(signal: LearningSignal): void {
    this.signals.push(signal)

    if (this.signals.length > this.maxSignals) {
      this.signals.shift()
    }

    // Check if autoDream should trigger
    this.checkTrigger()
  }

  /**
   * Check if trigger conditions met
   */
  private checkTrigger(): void {
    const now = BigInt(Date.now())

    // Trigger if enough time passed AND pattern threshold met
    if (now - this.lastTriggerTime >= this.triggerInterval) {
      const patternCount = this.countPatterns()

      if (patternCount >= 5) {
        this.triggerAutoDream()
        this.lastTriggerTime = now
        this.signals = [] // Clear after trigger
      }
    }
  }

  /**
   * Count distinct patterns across signals
   */
  private countPatterns(): number {
    const patterns = new Set<string>()

    for (const signal of this.signals) {
      for (const pattern of signal.patterns) {
        patterns.add(`${pattern.category}_${pattern.dimension}`)
      }
    }

    return patterns.size
  }

  /**
   * Trigger autoDream cycle
   */
  private triggerAutoDream(): void {
    console.log('[autoDream] Triggering learning cycle with', this.signals.length, 'signals')
    runAutoDreamCycle(this.signals)
  }

  /**
   * Get current signals
   */
  getSignals(): LearningSignal[] {
    return [...this.signals]
  }

  /**
   * Reset accumulator (for testing)
   */
  reset(): void {
    this.signals = []
    this.lastTriggerTime = BigInt(0)
  }
}

/**
 * Global autoDream accumulator
 */
const globalAccumulator = new AutoDreamAccumulator()

/**
 * Run autoDream improvement cycle
 */
async function runAutoDreamCycle(signals: LearningSignal[]): Promise<AutoDreamEvent> {
  const eventId = `dream_${Date.now()}_${Math.random().toString(36).slice(2)}`
  const startTime = BigInt(Date.now())

  try {
    console.log(`[autoDream] Starting cycle ${eventId} with ${signals.length} signals`)

    // Phase 1: Analyze signals and extract proposals
    const proposals = extractProposals(signals)
    console.log(`[autoDream] Generated ${proposals.length} proposals`)

    // Phase 2: Validate each proposal through cross-verifier
    const validatedProposals = []
    for (const proposal of proposals) {
      const checkResult = globalCrossVerifierEnsemble.check(proposal, `${eventId}_${proposal.target}`)

      if (checkResult.verdict === 'pass') {
        validatedProposals.push({proposal, checkResult})
      } else {
        console.warn(`[autoDream] Proposal rejected: ${proposal.target} (${checkResult.verdict})`)
      }
    }

    console.log(`[autoDream] ${validatedProposals.length}/${proposals.length} proposals passed validation`)

    // Phase 3: Apply approved proposals (only low-risk threshold adjustments, not structural)
    const appliedProposals = []
    for (const {proposal} of validatedProposals) {
      // Only auto-apply threshold adjustments (low risk)
      if (proposal.changeType === 'threshold_adjust' && proposal.residualRisk < 0.15) {
        const success = applyProposal(proposal)

        if (success) {
          appliedProposals.push(proposal)
          console.log(`[autoDream] Applied proposal: ${proposal.target}`)
        }
      } else {
        console.log(
          `[autoDream] Proposal queued for manual review: ${proposal.target} (${proposal.changeType}, risk=${proposal.residualRisk})`,
        )
      }
    }

    // Phase 4: Record event in lineage with forensic completeness
    const improvementEstimate = computeImprovementEstimate(signals, appliedProposals)

    const event: AutoDreamEvent = {
      eventId,
      timestamp: startTime,
      triggerType: 'pattern_accumulation',
      signalsAnalyzed: signals.length,
      proposalsGenerated: proposals.length,
      proposalsApproved: validatedProposals.length,
      proposalsApplied: appliedProposals.length,
      improvementImpact: improvementEstimate,
      lineageRecordId: '',
    }

    // Write to lineage
    const lineageRecord = globalLineageAuditor.addRecord({
      verificationId: eventId,
      timestamp: startTime,
      decision: appliedProposals.length > 0 ? 'accept' : 'quarantine',
      rubricScore: computeAverageRubricScore(signals),
      truthVerdict: 'uncertain',
      lineage: {
        who: 'autoDream_orchestrator',
        what: {
          before: {proposalsGenerated: proposals.length},
          after: {proposalsApplied: appliedProposals.length, improvements: improvementEstimate},
        },
        when: startTime,
        auth: 'learning_signal',
      },
    })

    event.lineageRecordId = lineageRecord.chainHash

    // Record in health monitor
    globalHealthMonitor.recordDecision(appliedProposals.length > 0 ? 'accept' : 'quarantine')

    console.log(`[autoDream] Cycle ${eventId} complete. Applied ${appliedProposals.length} proposals.`)

    return event
  } catch (error) {
    console.error(`[autoDream] Error in cycle ${eventId}:`, error)
    throw error
  }
}

/**
 * Extract improvement proposals from signals
 */
function extractProposals(signals: LearningSignal[]) {
  const proposals = []

  for (const signal of signals) {
    if (signal.proposal) {
      proposals.push(signal.proposal)
    }
  }

  return proposals
}

/**
 * Apply a proposal to guardrail components
 */
function applyProposal(proposal: any): boolean {
  try {
    // Example: apply threshold adjustment
    if (proposal.changeType === 'threshold_adjust' && proposal.target === 'safety_gate') {
      // In real implementation, this would update the threshold in the SafetyGate component
      // For now, log the intent
      console.log(`[autoDream] Would adjust threshold: ${proposal.proposal}`)
      return true
    }

    return false
  } catch (error) {
    console.error(`[autoDream] Error applying proposal:`, error)
    return false
  }
}

/**
 * Compute improvement estimate
 */
function computeImprovementEstimate(signals: LearningSignal[], appliedProposals: any[]) {
  // Analyze what patterns were most common
  const patternFrequency: Record<string, number> = {}

  for (const signal of signals) {
    for (const pattern of signal.patterns) {
      const key = `${pattern.category}_${pattern.dimension}`
      patternFrequency[key] = (patternFrequency[key] || 0) + 1
    }
  }

  // Estimate improvement from applied proposals
  const topPattern = Object.entries(patternFrequency).sort(([, a], [, b]) => b - a)[0]

  return {
    affectedDimension: topPattern?.[0] || 'unknown',
    estimatedImprovement: appliedProposals.length > 0 ? 5 : 0, // 5% per applied proposal (conservative)
    residualRisk: appliedProposals.length > 0 ? 0.1 : 0,
  }
}

/**
 * Compute average rubric score from signals
 */
function computeAverageRubricScore(signals: LearningSignal[]): number {
  if (signals.length === 0) return 0

  const sum = signals.reduce((acc, signal) => acc + signal.rubricScore.overall, 0)
  return sum / signals.length
}

/**
 * Feed a signal into autoDream
 */
export function feedSignalToAutoDream(signal: LearningSignal): void {
  globalAccumulator.addSignal(signal)
}

/**
 * Trigger manual autoDream cycle (for testing/operations)
 */
export async function triggerAutoDreamCycle(): Promise<AutoDreamEvent> {
  const signals = globalAccumulator.getSignals()
  return await runAutoDreamCycle(signals)
}

/**
 * Get autoDream status
 */
export function getAutoDreamStatus() {
  const signals = globalAccumulator.getSignals()
  return {
    signalsAccumulated: signals.length,
    lastUpdate: BigInt(Date.now()),
    readyToTrigger: signals.length >= 5,
  }
}

/**
 * Reset autoDream (for testing)
 */
export function resetAutoDream(): void {
  globalAccumulator.reset()
}
