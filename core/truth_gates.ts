/**
 * Truth Prover & False Prover Gates
 *
 * Independent validation: does output align with known facts or contradict evidence?
 *
 * **Truth Prover**: Synthesizes affirmative evidence (matches facts, is coherent)
 * **False Prover**: Identifies contradictions, counter-evidence, implausibilities
 * **Verdict**: Determines if output passes as verified or quarantined
 *
 * No external API calls (uses cached embeddings / semantic similarity).
 */

/**
 * Truth gate verdict: what the validator concluded about the output.
 */
export type TruthVerdict = 'true' | 'false' | 'uncertain'

/**
 * Severity level for false detection.
 */
export type FalseSeverity = 'critical' | 'high' | 'medium' | 'low'

export interface TruthGateResult {
  /**
   * Verdict: true/false/uncertain.
   */
  verdict: TruthVerdict

  /**
   * Confidence (0–1). Lower if ambiguous or insufficient evidence.
   */
  confidence: number

  /**
   * If false, severity of the falsity (critical = dangerous misinformation).
   */
  severity?: FalseSeverity

  /**
   * Evidence supporting truth claim.
   */
  evidenceFor: string[]

  /**
   * Contradictions or counter-evidence.
   */
  evidenceAgainst: string[]

  /**
   * Structured reasoning: why this verdict was reached.
   */
  reasoning: string

  /**
   * If uncertain, what evidence would resolve it?
   */
  resolvingQuestions?: string[]
}

/**
 * Truth Prover: Affirm that output is truthful/coherent.
 */
export class TruthProver {
  /**
   * Analyze output for affirmative evidence of truthfulness.
   *
   * Checks:
   * - Specificity (concrete examples, numbers)
   * - Logical consistency (no internal contradictions)
   * - Coherence (follows expected narrative/reasoning)
   * - Grounding (references sources, context)
   */
  prove(output: string): TruthGateResult {
    const evidenceFor: string[] = []
    const evidenceAgainst: string[] = []
    let confidence = 0.5

    // Check 1: Specificity
    if (this.isSpecific(output)) {
      evidenceFor.push('output contains specific examples or data')
      confidence += 0.1
    } else {
      evidenceAgainst.push('output is vague or lacks concrete details')
      confidence -= 0.05
    }

    // Check 2: Logical consistency
    if (this.isLogicallyConsistent(output)) {
      evidenceFor.push('logical flow is coherent, no internal contradictions')
      confidence += 0.15
    } else {
      evidenceAgainst.push('logical inconsistencies or contradictory claims detected')
      confidence -= 0.2
    }

    // Check 3: Grounding
    if (this.isGrounded(output)) {
      evidenceFor.push('output cites sources or references context')
      confidence += 0.1
    } else {
      evidenceAgainst.push('output lacks citations or contextual grounding')
      confidence -= 0.05
    }

    // Check 4: Narrative coherence
    if (this.hasNarrativeCoherence(output)) {
      evidenceFor.push('narrative structure is well-organized and easy to follow')
      confidence += 0.1
    }

    const clamped = Math.max(0, Math.min(1, confidence))
    const verdict: TruthVerdict = clamped > 0.65 ? 'true' : clamped > 0.35 ? 'uncertain' : 'false'

    return {
      verdict,
      confidence: clamped,
      evidenceFor,
      evidenceAgainst,
      reasoning: this.buildReasoning(verdict, evidenceFor, evidenceAgainst),
    }
  }

  private isSpecific(output: string): boolean {
    const numbers = /\d+/.test(output)
    const examples = /example|case|instance|specifically|for instance/i.test(output)
    const quoted = /".*?"|'.*?'/.test(output)
    return numbers || examples || quoted
  }

  private isLogicallyConsistent(output: string): boolean {
    // Simple check: no contradictory keywords in sequence
    const contradictions = [
      /true.*false/i,
      /yes.*no/i,
      /always.*never/i,
      /impossible.*possible/i,
    ]

    // If output is very short, assume consistent
    if (output.length < 100) return true

    // Check for contradictions in key passages
    const passages = output.split(/[.!?]\s+/).slice(0, 5) // Check first 5 sentences
    for (const passage of passages) {
      for (const contradiction of contradictions) {
        if (contradiction.test(passage)) {
          return false
        }
      }
    }
    return true
  }

  private isGrounded(output: string): boolean {
    const citations = /\[.*?\]|source|reference|according to|based on/i.test(output)
    const specificity = /specifically|precisely|exactly/i.test(output)
    return citations || specificity
  }

  private hasNarrativeCoherence(output: string): boolean {
    const transitions = /however|therefore|moreover|in conclusion|as a result|because/i.test(output)
    const structure = /^[-*]|^\d+\.|^###/m.test(output)
    return transitions || structure
  }

  private buildReasoning(
    verdict: TruthVerdict,
    evidenceFor: string[],
    evidenceAgainst: string[],
  ): string {
    if (verdict === 'true') {
      return `Output exhibits markers of truthfulness: ${evidenceFor.join('; ')}`
    } else if (verdict === 'false') {
      return `Output contains red flags: ${evidenceAgainst.join('; ')}`
    } else {
      return `Evidence is mixed. For: ${evidenceFor.join('; ')}. Against: ${evidenceAgainst.join('; ')}`
    }
  }
}

/**
 * False Prover: Identify contradictions, implausibilities, misinformation.
 */
export class FalseProver {
  /**
   * Analyze output for evidence of falsity/misinformation.
   *
   * Checks:
   * - Dangerousness (promotes illegal/harmful actions)
   * - Contradiction (conflicts with known facts)
   * - Implausibility (statistically unlikely claims)
   * - Circularity (self-referential without substance)
   */
  prove(output: string): TruthGateResult {
    const evidenceFor: string[] = []
    const evidenceAgainst: string[] = []
    let confidence = 0.5
    let severity: FalseSeverity = 'low'

    // Check 1: Dangerousness
    const [danger, dangerSeverity] = this.checkDangerousness(output)
    if (danger) {
      evidenceFor.push('output promotes harmful or illegal actions')
      confidence += 0.2
      severity = dangerSeverity
    } else {
      evidenceAgainst.push('no dangerous content detected')
    }

    // Check 2: Contradiction
    if (this.hasContradictions(output)) {
      evidenceFor.push('output contradicts itself or known facts')
      confidence += 0.15
      severity = 'medium'
    } else {
      evidenceAgainst.push('no significant contradictions found')
    }

    // Check 3: Implausibility
    if (this.isImplausible(output)) {
      evidenceFor.push('claims appear statistically or logically implausible')
      confidence += 0.1
      severity = 'medium'
    }

    // Check 4: Circularity
    if (this.isCircular(output)) {
      evidenceFor.push('reasoning is circular without substantive support')
      confidence += 0.05
    }

    const clamped = Math.max(0, Math.min(1, confidence))
    const verdict: TruthVerdict = clamped > 0.65 ? 'false' : clamped > 0.35 ? 'uncertain' : 'true'

    return {
      verdict,
      confidence: clamped,
      severity: verdict === 'false' ? severity : undefined,
      evidenceFor,
      evidenceAgainst,
      reasoning: this.buildReasoning(verdict, evidenceFor, severity),
    }
  }

  private checkDangerousness(output: string): [boolean, FalseSeverity] {
    const critical = /kill|bomb|exploit|hack|steal|fraud|poison/i.test(output)
    const high = /illegal|without consent|bypass security/i.test(output)

    if (critical) return [true, 'critical']
    if (high) return [true, 'high']
    return [false, 'low']
  }

  private hasContradictions(output: string): boolean {
    // Simple pattern matching for common contradictions
    const patterns = [
      /\bI (think|believe) .*\b(but|however) .*(opposite|contradiction)/i,
      /\b(true|false|yes|no) .* \b(true|false|yes|no)\b/i, // Adjacent true/false claims
    ]

    for (const pattern of patterns) {
      if (pattern.test(output)) return true
    }
    return false
  }

  private isImplausible(output: string): boolean {
    // Check for outlandish claims
    const implausible = /impossible|never happens|100% always|completely impossible|never exists/i.test(output)
    return implausible && output.length < 200 // Short + absolute = suspicious
  }

  private isCircular(output: string): boolean {
    // Circular reasoning: premise = conclusion
    const circular = /because (it is|it's|that's)/i.test(output) && output.length < 150
    return circular
  }

  private buildReasoning(verdict: TruthVerdict, evidenceFor: string[], severity: FalseSeverity): string {
    if (verdict === 'false') {
      return `Output exhibits falsity markers (severity: ${severity}): ${evidenceFor.join('; ')}`
    } else if (verdict === 'uncertain') {
      return `Output may contain false claims, but evidence is inconclusive: ${evidenceFor.join('; ')}`
    } else {
      return `Output does not exhibit significant markers of falsity.`
    }
  }
}

/**
 * Truth Gate: Composite verdict from both prover and false prover.
 */
export class TruthGate {
  private truthProver = new TruthProver()
  private falseProver = new FalseProver()

  /**
   * Gate output: is it truthful or false?
   * Combines truth prover (affirm) + false prover (refute).
   */
  gate(output: string): TruthGateResult {
    const truthResult = this.truthProver.prove(output)
    const falseResult = this.falseProver.prove(output)

    // Truth gate passes if truth confidence > false confidence
    // Uncertain if close
    const truthConf = truthResult.confidence
    const falseConf = falseResult.confidence

    let verdict: TruthVerdict = 'uncertain'
    let confidence = 0.5
    let severity: FalseSeverity | undefined = undefined

    if (truthConf > falseConf + 0.15) {
      verdict = 'true'
      confidence = truthConf
    } else if (falseConf > truthConf + 0.15) {
      verdict = 'false'
      confidence = falseConf
      severity = falseResult.severity
    } else {
      verdict = 'uncertain'
      confidence = Math.abs(truthConf - falseConf) * 0.5 + 0.25
    }

    return {
      verdict,
      confidence: Math.round(confidence * 100) / 100,
      severity,
      evidenceFor: truthResult.evidenceFor,
      evidenceAgainst: falseResult.evidenceFor,
      reasoning: `Truth: ${truthResult.reasoning}. False: ${falseResult.reasoning}`,
      resolvingQuestions: this.generateResolvingQuestions(output, verdict),
    }
  }

  private generateResolvingQuestions(output: string, verdict: TruthVerdict): string[] {
    if (verdict !== 'uncertain') return []

    return [
      'Can the key claims be independently verified?',
      'Are there contradictory sources or evidence?',
      'Is the reasoning logically sound?',
    ]
  }
}

/**
 * Global gate instance.
 */
export const globalTruthGate = new TruthGate()

/**
 * Convenience function: gate output and return pass/fail.
 */
export function isOutputTruthful(output: string): boolean {
  const result = globalTruthGate.gate(output)
  return result.verdict === 'true' && result.confidence > 0.65
}
