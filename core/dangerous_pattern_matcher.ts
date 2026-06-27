/**
 * Dangerous Pattern Matcher
 *
 * Centralized detection of malicious patterns and dangerous operations.
 * Single source of truth for threat detection across guardrail components.
 */

/**
 * Dangerous pattern categories
 */
export type PatternCategory =
  | 'dangerous_tool'
  | 'malicious_argument'
  | 'bypass_attempt'
  | 'injection_attack'
  | 'privilege_escalation'

/**
 * Pattern match result
 */
export interface PatternMatch {
  matched: boolean
  category?: PatternCategory
  pattern?: string
  severity: 'critical' | 'high' | 'medium' | 'low'
}

/**
 * Dangerous pattern matcher
 */
export class DangerousPatternMatcher {
  // Tools that should never be allowed
  private dangerousTools = [
    'system_command',
    'execute_python_code',
    'shell_exec',
    'os_call',
    'exec_raw',
    'eval_code',
    'system',
    'subprocess',
  ]

  // Malicious argument patterns
  private maliciousArguments = [
    /rm\s+-rf/i, // File deletion
    /\/etc\/passwd/i, // System file access
    /os\.system/i, // OS execution
    /subprocess\./i, // Process execution
    /eval\(/i, // Code evaluation
    /import\s+os/i, // OS module import
    /chmod\s+777/i, // Permission escalation
    /sudo/i, // Privilege escalation
    /`.*`/i, // Command substitution
    /\$\(/i, // Command substitution
  ]

  // Bypass attempt patterns
  private bypassPatterns = [
    /skip_safety_checks/i,
    /disable_lineage_tracking/i,
    /allow_dangerous_tools/i,
    /bypass_guardrails/i,
    /ignore_safety/i,
    /disable_verification/i,
    /unsafe_mode/i,
  ]

  /**
   * Check if tool is dangerous
   */
  isDangerousTool(toolName: string): PatternMatch {
    if (this.dangerousTools.some((tool) => toolName.toLowerCase().includes(tool))) {
      return {
        matched: true,
        category: 'dangerous_tool',
        pattern: toolName,
        severity: 'critical',
      }
    }

    return {matched: false, severity: 'low'}
  }

  /**
   * Check for malicious arguments
   */
  hasMaliciousArguments(input: string): PatternMatch {
    for (const pattern of this.maliciousArguments) {
      if (pattern.test(input)) {
        return {
          matched: true,
          category: 'malicious_argument',
          pattern: pattern.source,
          severity: 'high',
        }
      }
    }

    return {matched: false, severity: 'low'}
  }

  /**
   * Check for bypass attempts
   */
  hasBypassAttempt(input: string): PatternMatch {
    for (const pattern of this.bypassPatterns) {
      if (pattern.test(input)) {
        return {
          matched: true,
          category: 'bypass_attempt',
          pattern: pattern.source,
          severity: 'critical',
        }
      }
    }

    return {matched: false, severity: 'low'}
  }

  /**
   * Check for injection attacks
   */
  hasInjectionPattern(input: string): PatternMatch {
    // Common injection patterns
    const injectionPatterns = [
      /['"].*['"];.*['"].*['"]/, // Quote escaping
      /\bOR\b.*\d+.*\d+/i, // SQL injection
      /\$\{.*\}/, // Template injection
      /\{\{.*\}\}/, // Expression injection
    ]

    for (const pattern of injectionPatterns) {
      if (pattern.test(input)) {
        return {
          matched: true,
          category: 'injection_attack',
          pattern: pattern.source,
          severity: 'high',
        }
      }
    }

    return {matched: false, severity: 'low'}
  }

  /**
   * Comprehensive threat check
   */
  checkThreat(input: string, inputType: 'tool' | 'argument' | 'config' = 'argument'): PatternMatch {
    if (inputType === 'tool') {
      return this.isDangerousTool(input)
    }

    if (inputType === 'config') {
      return this.hasBypassAttempt(input)
    }

    // For arguments, check all patterns
    const maliciousCheck = this.hasMaliciousArguments(input)
    if (maliciousCheck.matched) return maliciousCheck

    const injectionCheck = this.hasInjectionPattern(input)
    if (injectionCheck.matched) return injectionCheck

    const bypassCheck = this.hasBypassAttempt(input)
    if (bypassCheck.matched) return bypassCheck

    return {matched: false, severity: 'low'}
  }

  /**
   * Get all dangerous tools
   */
  getDangerousTools(): string[] {
    return [...this.dangerousTools]
  }

  /**
   * Add custom dangerous pattern
   */
  addCustomPattern(category: PatternCategory, pattern: RegExp): void {
    if (category === 'malicious_argument') {
      this.maliciousArguments.push(pattern)
    } else if (category === 'bypass_attempt') {
      this.bypassPatterns.push(pattern)
    }
  }
}

/**
 * Global pattern matcher
 */
export const globalPatternMatcher = new DangerousPatternMatcher()

/**
 * Check if tool is dangerous
 */
export function isDangerousTool(toolName: string): boolean {
  return globalPatternMatcher.isDangerousTool(toolName).matched
}

/**
 * Check if input contains malicious patterns
 */
export function containsMaliciousPattern(input: string): boolean {
  return (
    globalPatternMatcher.hasMaliciousArguments(input).matched ||
    globalPatternMatcher.hasInjectionPattern(input).matched
  )
}

/**
 * Check if input contains bypass attempts
 */
export function containsBypassAttempt(input: string): boolean {
  return globalPatternMatcher.hasBypassAttempt(input).matched
}
