/**
 * Error Lineage Handler
 *
 * Creates immutable lineage records for component failures.
 * Ensures all exceptions during verification are auditable and traceable.
 */

import {globalLineageAuditor} from './lineage_auditor.js'
import {globalAlertManager} from './guardrail_alerts.js'

export interface ErrorRecord {
  errorId: string
  component: string
  errorType: string
  errorMessage: string
  stackTrace?: string
  failureSeverity: 'critical' | 'high' | 'medium' | 'low'
}

/**
 * Record component failure in lineage
 */
export function recordComponentFailure(
  component: string,
  error: Error,
  failureSeverity: 'critical' | 'high' | 'medium' | 'low' = 'high',
): void {
  const errorId = `error_${component}_${Date.now()}_${Math.random().toString(36).slice(2)}`
  const timestamp = BigInt(Date.now() * 1_000_000)

  try {
    // Add error record to lineage
    globalLineageAuditor.addRecord({
      verificationId: errorId,
      timestamp,
      decision: 'error',
      rubricScore: 0, // Error state
      truthVerdict: 'error',
      lineage: {
        who: `${component}_error_handler`,
        what: {
          before: {component, status: 'active'},
          after: {
            component,
            status: 'failed',
            error: error.message,
            type: error.constructor.name,
            severity: failureSeverity,
          },
        },
        when: timestamp,
        auth: 'error_signal',
      },
    })

    console.error(`[error-lineage] Component failure recorded: ${component} (${failureSeverity})`)
    console.error(`[error-lineage] Error: ${error.message}`)
    if (error.stack) {
      console.error(`[error-lineage] Stack: ${error.stack.split('\n').slice(0, 3).join(' | ')}`)
    }

    // Trigger alert if critical
    if (failureSeverity === 'critical') {
      globalAlertManager.checkMetric('guardrail.component_failure', 1)
    }
  } catch (recordingError) {
    console.error(`[error-lineage] Failed to record error in lineage:`, recordingError)
  }
}

/**
 * Record verification pipeline failure
 */
export function recordVerificationFailure(
  verificationId: string,
  stage: string,
  error: Error,
  input?: any,
): void {
  const errorId = `verify_error_${verificationId}_${stage}`
  const timestamp = BigInt(Date.now() * 1_000_000)

  try {
    globalLineageAuditor.addRecord({
      verificationId: errorId,
      timestamp,
      decision: 'error',
      rubricScore: 0,
      truthVerdict: 'verification_error',
      lineage: {
        who: 'verification_pipeline',
        what: {
          before: {stage, verification: verificationId, status: 'in_progress'},
          after: {
            stage,
            verification: verificationId,
            status: 'failed',
            failurePoint: stage,
            error: error.message,
            errorType: error.constructor.name,
          },
        },
        when: timestamp,
        auth: 'verification_error',
      },
    })

    console.error(
      `[error-lineage] Verification failure: ${verificationId} failed at ${stage}: ${error.message}`,
    )
  } catch (recordingError) {
    console.error(`[error-lineage] Failed to record verification error:`, recordingError)
  }
}

/**
 * Wrap async function with error lineage recording
 */
export async function withErrorLineage<T>(
  component: string,
  fn: () => Promise<T>,
): Promise<{success: boolean; result?: T; error?: Error}> {
  try {
    const result = await fn()
    return {success: true, result}
  } catch (error) {
    recordComponentFailure(component, error as Error, 'high')
    return {success: false, error: error as Error}
  }
}

/**
 * Wrap sync function with error lineage recording
 */
export function withErrorLineageSync<T>(
  component: string,
  fn: () => T,
): {success: boolean; result?: T; error?: Error} {
  try {
    const result = fn()
    return {success: true, result}
  } catch (error) {
    recordComponentFailure(component, error as Error, 'high')
    return {success: false, error: error as Error}
  }
}
