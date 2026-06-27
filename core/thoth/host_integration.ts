/**
 * Host Integration Shim.
 *
 * Thin, DEFENSIVE adapter between the guardrail corpus and the host application's
 * runtime (services/api/claude.ts, services/tools/toolExecution.ts, utils/messages.ts,
 * cli/print.ts).
 *
 * Design guarantees (so wiring the guard into live CLI paths is safe):
 *   - FAIL-OPEN: a guard exception is caught and treated as "allow" — a bug in the
 *     guardrail must NEVER break the host or deny-of-service the user.
 *   - OBSERVE BY DEFAULT: mode starts at 'observe', where allow is always true and the
 *     guard only records/scores. Switch to 'enforce' to let quarantine actually block.
 *   - SIDE-EFFECT-LIGHT: each call returns a verdict; the host decides what to do.
 *
 * This keeps all real logic in tested corpus code and makes the host edits minimal.
 */

import {
  guardApiOutput,
  guardToolExecution,
  guardMessageMutation,
  guardCliConfig,
  type GuardrailGateResult,
} from '../guardrail_integration.js'
import {recordComponentFailure} from '../error_lineage_handler.js'

export type HostGuardMode = 'observe' | 'enforce'

let mode: HostGuardMode = 'observe'

/** Switch global host-guard mode. 'enforce' lets quarantine actually block. */
export function setHostGuardMode(m: HostGuardMode): void {
  mode = m
}

export function getHostGuardMode(): HostGuardMode {
  return mode
}

export interface HostGuardDecision {
  allow: boolean
  decision: 'accept' | 'quarantine' | 'error'
  reason?: string
}

/** Run a guard fail-open: any throw → allow, with a recorded component failure. */
function evaluate(fn: () => GuardrailGateResult, component: string): HostGuardDecision {
  try {
    const r = fn()
    const allow = mode === 'observe' ? true : r.decision === 'accept'
    return {allow, decision: r.decision, reason: r.reason}
  } catch (e) {
    // FAIL-OPEN — never let a guard bug break the host.
    try {
      recordComponentFailure(component, e as Error, 'medium')
    } catch {
      /* even failure recording must not throw into the host */
    }
    return {allow: true, decision: 'error', reason: 'guard_error_fail_open'}
  }
}

/** API boundary: gate model output text. */
export function guardHostApiOutput(output: string, ctx?: {prompt?: string; model?: string}): HostGuardDecision {
  if (!output) return {allow: true, decision: 'accept'}
  return evaluate(() => guardApiOutput(output, ctx), 'host_api_boundary')
}

/** Tool boundary: gate a tool invocation before execution. */
export function guardHostToolExecution(toolName: string, input: unknown): HostGuardDecision {
  return evaluate(() => guardToolExecution(toolName, input), 'host_tool_execution')
}

/** Message boundary: gate a message creation/mutation. */
export function guardHostMessage(oldMsg: unknown, newMsg: unknown): HostGuardDecision {
  return evaluate(() => guardMessageMutation(oldMsg, newMsg), 'host_message_mutation')
}

/** CLI boundary: gate config at boot. */
export function guardHostCliConfig(config: unknown): HostGuardDecision {
  return evaluate(() => guardCliConfig(config), 'host_cli_config')
}
