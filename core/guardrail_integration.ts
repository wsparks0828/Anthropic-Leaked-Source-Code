/**
 * Guardrail Integration Layer
 *
 * Bridges guardrail subsystems into Jarvis critical paths.
 * Minimal, surgical wraps around existing code.
 *
 * Used by:
 * - services/api/claude.ts (API boundary)
 * - services/tools/toolExecution.ts (tool execution)
 * - utils/messages.ts (message mutation)
 * - cli/print.ts (CLI boot)
 */

import { globalRubricScorer } from './rubric_scorer.js'
import { globalTruthGate } from './truth_gates.js'
import { globalGuardrailLearningBridge, type LearningSignal } from './guardrail_learning_bridge.js'
import { globalCrossVerifierEnsemble, shouldApplyProposal } from './cross_verifier_ensemble.js'
import { DEFAULT_GUARDRAIL_CONFIG, type GuardrailVerificationResult } from './schemas.js'

/**
 * Quarantine reason (why output was rejected).
 */
export type QuarantineReason =
  | 'rubric_threshold_failed'
  | 'truth_gate_false'
  | 'residual_risk_high'
  | 'cross_check_failed'
  | 'manual_escalation'

/**
 * Guardrail gate result (pass or quarantine).
 */
export interface GuardrailGateResult {
  decision: 'accept' | 'quarantine'
  reason?: QuarantineReason
  verificationId: string
  quarantineDetails?: {
    rubricScore: number
    truthVerdict: string
    residualRisk: number
  }
}

/**
 * Recent learning signals (for autoDream triggering).
 */
const recentSignals: LearningSignal[] = []
const MAX_RECENT_SIGNALS = 50

/**
 * API Boundary Gate
 *
 * Place in: services/api/claude.ts before/after Claude API call
 *
 * ```typescript
 * const output = await apiCall(...)  // existing Claude API call
 * const gateResult = await guardApiOutput(output, prompt)
 * if (gateResult.decision === 'quarantine') {
 *   throw new Error(`Output quarantined: ${gateResult.reason}`)
 * }
 * return output
 * ```
 */
export function guardApiOutput(
  output: string,
  context?: { prompt?: string; model?: string },
): GuardrailGateResult {
  const verificationId = generateVerificationId()

  try {
    // OUTPUT SIZE LIMIT CHECK: Prevent DoS via huge outputs
    const MAX_OUTPUT_SIZE = 100_000 // 100KB limit
    if (output.length > MAX_OUTPUT_SIZE) {
      return {
        decision: 'quarantine',
        reason: 'rubric_threshold_failed', // Use existing reason type
        verificationId,
        quarantineDetails: {
          rubricScore: 0,
          truthVerdict: 'size_limit_exceeded',
          residualRisk: 1.0,
        },
      }
    }

    // Score output
    const rubricScore = globalRubricScorer.score(output, {
      query: context?.prompt,
      intent: context?.model,
    })

    // Check truth
    const truthVerdict = globalTruthGate.gate(output)

    // Decision logic (fail-closed on safety, more permissive on quality)
    if (rubricScore.overall < DEFAULT_GUARDRAIL_CONFIG.rubricThreshold) {
      return {
        decision: 'quarantine',
        reason: 'rubric_threshold_failed',
        verificationId,
        quarantineDetails: {
          rubricScore: rubricScore.overall,
          truthVerdict: truthVerdict.verdict,
          residualRisk: 0,
        },
      }
    }

    // Quarantine if definitely false AND dangerous
    if (
      truthVerdict.verdict === 'false' &&
      truthVerdict.severity &&
      ['critical', 'high'].includes(truthVerdict.severity)
    ) {
      return {
        decision: 'quarantine',
        reason: 'truth_gate_false',
        verificationId,
        quarantineDetails: {
          rubricScore: rubricScore.overall,
          truthVerdict: truthVerdict.verdict,
          residualRisk: truthVerdict.confidence,
        },
      }
    }

    // Generate learning signal
    const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
      source: 'api_boundary',
      summary: output.slice(0, 100),
      hash: hashContent(output),
    })

    trackSignal(signal)

    // Check if proposal was generated, if so, validate it
    if (signal.proposal) {
      const proposalId = `prop_${verificationId}`
      const crossCheckResult = globalCrossVerifierEnsemble.check(signal.proposal, proposalId)

      if (crossCheckResult.verdict === 'fail') {
        return {
          decision: 'quarantine',
          reason: 'cross_check_failed',
          verificationId,
          quarantineDetails: {
            rubricScore: rubricScore.overall,
            truthVerdict: truthVerdict.verdict,
            residualRisk: crossCheckResult.residualRisk,
          },
        }
      }
    }

    // Check for autoDream trigger
    if (globalGuardrailLearningBridge.shouldTriggerAutoDream(recentSignals)) {
      // Signal autoDream to run (implementation in services/autoDream)
      emitAutoDreamSignal()
    }

    return { decision: 'accept', verificationId }
  } catch (error) {
    // On error, default to accept (fail-open for transient issues, but log)
    logGuardrailError('API boundary gate error', error)
    return { decision: 'accept', verificationId }
  }
}

/**
 * Tool Execution Gate
 *
 * Place in: services/tools/toolExecution.ts before tool invocation
 *
 * ```typescript
 * const gateResult = await guardToolExecution(tool, args)
 * if (gateResult.decision === 'quarantine') {
 *   return { error: `Tool quarantined: ${gateResult.reason}` }
 * }
 * const result = await tool.execute(args)
 * ```
 */
export function guardToolExecution(toolName: string, args: unknown): GuardrailGateResult {
  const verificationId = generateVerificationId()

  try {
    // Dangerous tools: always quarantine
    const dangerousTools = ['system_command', 'execute_python_code', 'shell_exec', 'os_call', 'exec_raw']
    if (dangerousTools.includes(toolName)) {
      return {
        decision: 'quarantine',
        reason: 'rubric_threshold_failed',
        verificationId,
      }
    }

    // Check arguments for dangerous patterns
    const argsStr = JSON.stringify(args).toLowerCase()
    const dangerousPatterns = [
      'rm -rf',
      '/etc/passwd',
      'system(',
      'os.system',
      'subprocess',
      'exec(',
      'eval(',
      'import os',
      'import sys',
      'malicious',
      '/root',
      '/bin/bash',
    ]

    if (dangerousPatterns.some((pattern) => argsStr.includes(pattern))) {
      return {
        decision: 'quarantine',
        reason: 'truth_gate_false',
        verificationId,
      }
    }

    // Score the tool invocation
    const argsStrForScoring = JSON.stringify(args).slice(0, 500)
    const rubricScore = globalRubricScorer.score(
      `Tool: ${toolName}, Args: ${argsStrForScoring}`,
      { intent: 'tool_execution' },
    )

    // Tool gates are strict: any low dimension = quarantine
    if (rubricScore.overall < 0.5) {
      return {
        decision: 'quarantine',
        reason: 'rubric_threshold_failed',
        verificationId,
      }
    }

    return { decision: 'accept', verificationId }
  } catch (error) {
    logGuardrailError('Tool execution gate error', error)
    return { decision: 'accept', verificationId }
  }
}

/**
 * Message Mutation Gate
 *
 * Place in: utils/messages.ts after message creation, before storage
 *
 * ```typescript
 * const message = createMessage(content)
 * const gateResult = guardMessageMutation(oldMessage, newMessage)
 * if (gateResult.decision === 'quarantine') {
 *   // Flag message as quarantined in lineage
 * }
 * ```
 */
export function guardMessageMutation(oldMsg: any, newMsg: any): GuardrailGateResult {
  const verificationId = generateVerificationId()

  try {
    // Extract content from message objects
    const oldContent = typeof oldMsg === 'string' ? oldMsg : oldMsg?.content || ''
    const newContent = typeof newMsg === 'string' ? newMsg : newMsg?.content || ''

    const rubricScore = globalRubricScorer.score(newContent)

    // Messages should maintain quality
    if (rubricScore.overall < 0.6) {
      return {
        decision: 'quarantine',
        reason: 'rubric_threshold_failed',
        verificationId,
      }
    }

    // Check for dangerous mutations (injecting harmful content)
    if (newContent.length > oldContent.length) {
      const addedContent = newContent.slice(oldContent.length)
      const addedScore = globalRubricScorer.score(addedContent)
      if (addedScore.overall < 0.5) {
        return {
          decision: 'quarantine',
          reason: 'truth_gate_false',
          verificationId,
        }
      }
    }

    // Generate signal for memory wiring
    const signal = globalGuardrailLearningBridge.processVerification(
      rubricScore,
      { verdict: 'uncertain', confidence: 0.5, evidenceFor: [], evidenceAgainst: [], reasoning: '' },
      {
        source: 'message_creation',
        summary: newContent.slice(0, 100),
        hash: hashContent(newContent),
      },
    )

    trackSignal(signal)

    return { decision: 'accept', verificationId }
  } catch (error) {
    logGuardrailError('Message mutation gate error', error)
    return { decision: 'accept', verificationId }
  }
}

/**
 * CLI Boot Gate
 *
 * Place in: cli/print.ts at startup, before command dispatch
 *
 * ```typescript
 * const configGateResult = guardCliConfig(config)
 * if (configGateResult.decision === 'quarantine') {
 *   process.exit(1)
 * }
 * ```
 */
export function guardCliConfig(config: unknown): GuardrailGateResult {
  const verificationId = generateVerificationId()

  try {
    // Check for bypass attempts in config
    const configObj = config as Record<string, any>
    const bypassPatterns = [
      'skip_safety_checks',
      'disable_lineage_tracking',
      'allow_dangerous_tools',
      'disable_guardrails',
      'bypass_verification',
      'unsafe_mode',
    ]

    for (const pattern of bypassPatterns) {
      if (configObj[pattern] === true) {
        return {
          decision: 'quarantine',
          reason: 'truth_gate_false',
          verificationId,
          quarantineDetails: {
            rubricScore: 0,
            truthVerdict: 'false',
            residualRisk: 1,
          },
        }
      }
    }

    const configStr = JSON.stringify(config).slice(0, 500)

    // Config must be coherent and safe
    const rubricScore = globalRubricScorer.score(configStr, { intent: 'config_validation' })

    if (rubricScore.dimensions.safety < 0.8) {
      return {
        decision: 'quarantine',
        reason: 'rubric_threshold_failed',
        verificationId,
        quarantineDetails: {
          rubricScore: rubricScore.overall,
          truthVerdict: 'false',
          residualRisk: 1 - rubricScore.dimensions.safety,
        },
      }
    }

    return { decision: 'accept', verificationId }
  } catch (error) {
    logGuardrailError('Config gate error', error)
    // Config errors are critical: fail-closed
    return { decision: 'quarantine', reason: 'manual_escalation', verificationId }
  }
}

/**
 * Helper: track learning signal for autoDream trigger.
 */
function trackSignal(signal: LearningSignal): void {
  recentSignals.push(signal)
  if (recentSignals.length > MAX_RECENT_SIGNALS) {
    recentSignals.shift()
  }
}

/**
 * Helper: emit autoDream trigger signal.
 */
function emitAutoDreamSignal(): void {
  // In real implementation, this would call services/autoDream/autoDream.ts
  // For now, just log
  console.log('[guardrail] autoDream trigger signal emitted')
}

/**
 * Helper: simple content hash.
 */
function hashContent(content: string): string {
  let hash = 0
  for (let i = 0; i < Math.min(content.length, 500); i++) {
    hash = ((hash << 5) - hash + content.charCodeAt(i)) | 0
  }
  return `${hash}`
}

/**
 * Helper: generate verification ID.
 */
function generateVerificationId(): string {
  return `ver_${Date.now()}_${Math.random().toString(36).slice(2)}`
}

/**
 * Helper: log guardrail errors (structured).
 */
function logGuardrailError(context: string, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error)
  console.error(`[guardrail] ${context}: ${message}`)
}
