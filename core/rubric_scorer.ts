/**
 * 8-Dimensional Rubric Scorer
 *
 * Lightweight, cached scoring system for evaluating output quality across:
 * 1. Relevance: Does output address the query/context?
 * 2. Coherence: Is reasoning logically structured and clear?
 * 3. Factuality: Is information accurate and verifiable?
 * 4. Completeness: Does it cover required scope?
 * 5. Safety: Does it avoid harmful/risky outputs?
 * 6. Attribution: Are sources/evidence cited?
 * 7. Originality: Is it derivative or novel?
 * 8. Utility: Is output actionable/useful to user?
 *
 * Used by: Guardrail layer, truth gates, learning signals
 */

import memoize from 'lodash-es/memoize.js'

export type RubricDimension =
  | 'relevance'
  | 'coherence'
  | 'factuality'
  | 'completeness'
  | 'safety'
  | 'attribution'
  | 'originality'
  | 'utility'

export interface RubricScore {
  /**
   * Composite score (0–1). Calculated as weighted average of dimensions.
   * Default weights: all equal. Can be overridden per use case.
   */
  overall: number

  /**
   * Per-dimension scores (0–1).
   */
  dimensions: Record<RubricDimension, number>

  /**
   * Evidence snippets supporting score. Used for healing proposals + debugging.
   */
  evidence: Array<{
    dimension: RubricDimension
    signal: 'positive' | 'negative' | 'neutral'
    excerpt: string
  }>

  /**
   * Dimension with lowest score (learning signal: where to improve).
   */
  lowestDimension: RubricDimension | null

  /**
   * Dimension with highest score (learning signal: what's working).
   */
  highestDimension: RubricDimension | null

  /**
   * Confidence in this score (0–1). Lower if ambiguous signals.
   */
  confidence: number

  /**
   * Timestamp when score was computed (for trend analysis).
   */
  timestamp: bigint
}

/**
 * Lightweight rubric scorer using text analysis + heuristics.
 * No external API calls (stays sub-50ms per call).
 *
 * **Caching**: Scores are memoized by content hash to avoid redundant scoring.
 */
export class RubricScorer {
  private cache: Map<string, RubricScore> = new Map()
  private maxCacheSize = 500

  /**
   * Score output across all 8 dimensions.
   *
   * @param output - The text to score
   * @param context - Optional context (query, user intent) for relevance scoring
   * @returns RubricScore with dimension breakdowns + evidence
   */
  score(output: string, context?: { query?: string; intent?: string }): RubricScore {
    const hash = this.hashContent(output)
    if (this.cache.has(hash)) {
      return this.cache.get(hash)!
    }

    const now = BigInt(Date.now()) * BigInt(1_000_000) // Convert to nanoseconds for monotonicity

    const dimensions: Record<RubricDimension, number> = {
      relevance: this.scoreRelevance(output, context),
      coherence: this.scoreCoherence(output),
      factuality: this.scoreFactuality(output),
      completeness: this.scoreCompleteness(output),
      safety: this.scoreSafety(output),
      attribution: this.scoreAttribution(output),
      originality: this.scoreOriginality(output),
      utility: this.scoreUtility(output),
    }

    const scores = Object.values(dimensions)
    const overall = scores.reduce((a, b) => a + b, 0) / scores.length
    const lowestScore = Math.min(...scores)
    const highestScore = Math.max(...scores)

    const lowestDimension = (Object.entries(dimensions).find(([_, v]) => v === lowestScore)?.[0] ||
      null) as RubricDimension | null
    const highestDimension = (Object.entries(dimensions).find(([_, v]) => v === highestScore)?.[0] ||
      null) as RubricDimension | null

    const evidence = this.extractEvidence(output, dimensions, context)
    const confidence = this.computeConfidence(scores, evidence)

    const rubricScore: RubricScore = {
      overall: Math.round(overall * 100) / 100,
      dimensions: Object.fromEntries(
        Object.entries(dimensions).map(([k, v]) => [k, Math.round(v * 100) / 100]),
      ) as Record<RubricDimension, number>,
      evidence,
      lowestDimension,
      highestDimension,
      confidence: Math.round(confidence * 100) / 100,
      timestamp: now,
    }

    // Maintain cache size
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value
      this.cache.delete(firstKey)
    }
    this.cache.set(hash, rubricScore)

    return rubricScore
  }

  /**
   * Score relevance: Does output address query/context?
   */
  private scoreRelevance(output: string, context?: { query?: string; intent?: string }): number {
    if (!context?.query && !context?.intent) return 0.7 // Neutral if no context

    const query = (context?.query || '').toLowerCase()
    const output_lower = output.toLowerCase()

    if (!query) return 0.7
    const keywords = query.split(/\s+/).filter(w => w.length > 3)
    const matches = keywords.filter(kw => output_lower.includes(kw)).length
    const ratio = matches / keywords.length
    return Math.min(1, 0.3 + ratio * 0.7)
  }

  /**
   * Score coherence: Is reasoning structured and clear?
   */
  private scoreCoherence(output: string): number {
    const lines = output.split('\n').filter(l => l.trim())
    if (lines.length === 0) return 0.1

    // Heuristics: structured lists, transitions, logical flow
    const hasStructure =
      /^[-*]\s|^\d+\.\s|^###|^##/m.test(output) || // Lists, headers
      /therefore|however|moreover|in conclusion|as a result/i.test(output) // Transitions

    const avgLineLength = output.split('\n').reduce((sum, l) => sum + l.length, 0) / lines.length
    const wellFormatted = avgLineLength > 30 && avgLineLength < 200 // Not too sparse, not too dense

    return hasStructure && wellFormatted ? 0.75 : 0.5
  }

  /**
   * Score factuality: Is information plausible and verifiable?
   */
  private scoreFactuality(output: string): number {
    // Heuristics: citations, evidence, specific claims vs vague
    const hasCitations = /\[.*?\]|citation|reference|source|evidence/i.test(output)
    const hasNumbers = /\d+/.test(output)
    const hasQualifiers = /approximately|likely|probably|may|could|seems/i.test(output)

    // Presence of qualifiers (honest uncertainty) is good; vague absolutes are risky
    let score = 0.5
    if (hasNumbers) score += 0.15
    if (hasCitations) score += 0.2
    if (hasQualifiers && output.length > 50) score += 0.15 // Qualifiers + substance

    return Math.min(1, score)
  }

  /**
   * Score completeness: Does it cover required scope?
   */
  private scoreCompleteness(output: string): number {
    const length = output.length
    const lines = output.split('\n').filter(l => l.trim()).length

    if (length < 20 || lines < 2) return 0.2 // Too short
    if (length > 10000) return 0.7 // Reasonable depth
    return 0.5 + (Math.min(length, 2000) / 2000) * 0.3 // Scale from 0.5 to 0.8
  }

  /**
   * Score safety: Avoid harmful/dangerous outputs?
   */
  private scoreSafety(output: string): number {
    const dangerous = /kill|bomb|exploit|hack|illegal|steal|fraud/i.test(output)
    const unsafe = /without consent|without permission|bypass security/i.test(output)

    if (dangerous || unsafe) return 0.2
    return 0.85 // Default to safe
  }

  /**
   * Score attribution: Are sources/evidence cited?
   */
  private scoreAttribution(output: string): number {
    const citations = (output.match(/\[.*?\]|source:|reference:/gi) || []).length
    const reasoning = /because|due to|based on|according to/i.test(output)

    return citations > 0 ? 0.8 : reasoning ? 0.6 : 0.3
  }

  /**
   * Score originality: Is it derivative or novel?
   */
  private scoreOriginality(output: string): number {
    const clichés = /as I mentioned|as stated|clearly|obviously|everyone knows/i.test(output)
    const specific = /specific|concrete|example|novel|unique|distinct/i.test(output)

    let score = 0.5
    if (specific) score += 0.3
    if (!clichés) score += 0.2
    return Math.min(1, score)
  }

  /**
   * Score utility: Is output actionable/useful?
   */
  private scoreUtility(output: string): number {
    const actionable =
      /step|procedure|how to|guide|recommendation|suggestion|next|action|do this/i.test(output)
    const concrete = /example|code|command|specific/i.test(output)
    const organized = /^[-*]|^\d+\.|^###/m.test(output)

    let score = 0.4
    if (actionable) score += 0.3
    if (concrete) score += 0.15
    if (organized) score += 0.15
    return Math.min(1, score)
  }

  /**
   * Extract evidence supporting the score.
   */
  private extractEvidence(
    output: string,
    dimensions: Record<RubricDimension, number>,
    context?: { query?: string; intent?: string },
  ): Array<{ dimension: RubricDimension; signal: 'positive' | 'negative' | 'neutral'; excerpt: string }> {
    const evidence: Array<{ dimension: RubricDimension; signal: 'positive' | 'negative' | 'neutral'; excerpt: string }> = []

    // Extract snippets where dimensions are strong/weak
    const lines = output.split('\n').filter(l => l.trim())
    const sampleLines = lines.slice(0, Math.ceil(lines.length / 3)) // Sample first third

    for (const [dim, score] of Object.entries(dimensions)) {
      if (score > 0.7) {
        const line = sampleLines[0]
        if (line) {
          evidence.push({
            dimension: dim as RubricDimension,
            signal: 'positive',
            excerpt: line.slice(0, 80),
          })
        }
      } else if (score < 0.4) {
        const line = sampleLines[sampleLines.length - 1]
        if (line) {
          evidence.push({
            dimension: dim as RubricDimension,
            signal: 'negative',
            excerpt: line.slice(0, 80),
          })
        }
      }
    }

    return evidence.slice(0, 5) // Limit to 5 evidence items
  }

  /**
   * Compute confidence in the score (0–1).
   */
  private computeConfidence(scores: number[], evidence: Array<any>): number {
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length
    const variance = scores.reduce((sum, s) => sum + Math.pow(s - avgScore, 2), 0) / scores.length
    const stdDev = Math.sqrt(variance)

    // High agreement between dimensions = high confidence
    const dimensionalConfidence = 1 - Math.min(stdDev, 1)
    const evidenceConfidence = Math.min(evidence.length / 3, 1)

    return (dimensionalConfidence * 0.7 + evidenceConfidence * 0.3)
  }

  /**
   * Simple content hash for caching.
   */
  private hashContent(content: string): string {
    let hash = 0
    for (let i = 0; i < Math.min(content.length, 1000); i++) {
      hash = ((hash << 5) - hash + content.charCodeAt(i)) | 0
    }
    return `${hash}`
  }

  /**
   * Clear cache (e.g., on session change).
   */
  clearCache(): void {
    this.cache.clear()
  }
}

/**
 * Global singleton scorer instance.
 */
export const globalRubricScorer = new RubricScorer()

/**
 * Convenience function: score output and return composite.
 */
export function scoreOutput(
  output: string,
  context?: { query?: string; intent?: string },
): number {
  return globalRubricScorer.score(output, context).overall
}
