/**
 * THOTH Component A — Pre-Ingest Gate (Layer 0).
 *
 * Runs BEFORE content is accepted into the corpus (before chunking/embedding).
 * A lightweight, gated, rubric-scored pre-filter so low-signal / low-tier material
 * is deprioritized, flagged, or rejected rather than depleting the corpus.
 *
 * Faithful to the THOTH spec (TypeScript-integrated, not Python):
 *   - Source-tier check (Tier 1 = highest trust)
 *   - Basic rubric scan on dimensions 1,2,3 (relevance, coherence, factuality)
 *   - Early Truth/False "lite" (contradiction / low-evidence flag)
 *   - Emits a structured lesson record (dimension, score, evidence, action, source)
 *     into the immutable lineage store. JSONL serialization is a separate deliverable.
 *
 * INVARIANTS:
 *   I1: a 'reject' decision is NEVER silently dropped — it emits a lesson record
 *   I2: tier-4 (untrusted) source can never be 'accept' (fail-closed on provenance)
 *   I3: the gate is read-only w.r.t. content — it scores and routes, never mutates input
 */

import {globalRubricScorer} from '../rubric_scorer.js'
import {globalTruthGate} from '../truth_gates.js'
import {globalLineageAuditor} from '../lineage_auditor.js'
import type {RubricScoreLine, LessonLine} from './jsonl_logger.js'

/** 1 = highest-trust curated source … 4 = unknown/untrusted scrape. */
export type SourceTier = 1 | 2 | 3 | 4

export type PreIngestDecision = 'accept' | 'deprioritize' | 'flag' | 'reject'

export interface PreIngestConfig {
  /** Composite (dims 1-3) at/above which content is accept-eligible. */
  acceptThreshold: number
  /** Below this composite → reject outright (very low signal). */
  rejectThreshold: number
  /** Tiers strictly above this are deprioritized regardless of score. */
  deprioritizeAboveTier: SourceTier
  /** Tiers strictly above this can never be accepted (provenance floor). */
  rejectAboveTier: SourceTier
  /**
   * Density floor (THOTH component G). The heuristic rubric is non-discriminating
   * on dims 1-3 when no query is supplied (relevance defaults to neutral 0.7,
   * coherence/factuality are near-constant) — so it CANNOT reject thin junk on
   * signal alone. Density scoring gives the gate a query-independent discriminator.
   */
  minChars: number
  minDistinctTokens: number
  /**
   * Informativeness floors (closes the "dense-but-worthless" gap the query-blind
   * rubric + density cannot catch). Concrete-signal ratio minus filler ratio.
   */
  minInformativeness: number // below → at best deprioritized
  rejectInformativeness: number // below → rejected outright
}

export const DEFAULT_PRE_INGEST_CONFIG: PreIngestConfig = {
  acceptThreshold: 0.55,
  rejectThreshold: 0.35,
  deprioritizeAboveTier: 2, // tier 3+ → at best deprioritized
  rejectAboveTier: 3, // tier 4 → never accepted (I2)
  minChars: 40,
  minDistinctTokens: 8,
  minInformativeness: 0.35,
  rejectInformativeness: 0.15,
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** Filler / hedge tokens that carry little information. */
const FILLER_TOKENS = new Set([
  'basically', 'just', 'thing', 'things', 'stuff', 'really', 'very', 'kind', 'honestly',
  'whatever', 'more', 'less', 'general', 'sense', 'know', 'depends', 'like', 'maybe', 'idk',
  'yeah', 'ok', 'okay', 'sorta', 'kinda', 'actually', 'literally', 'somewhat', 'anyway',
  'somehow', 'sort', 'lot', 'lots', 'much', 'many', 'some', 'good', 'nice', 'pretty',
])

/**
 * Query-independent informativeness in [0,1].
 * Rewards concrete signals (numbers, proper nouns, long/technical words, math/citation
 * symbols); penalizes filler density. Operates on the ORIGINAL (cased) content.
 */
export function computeInformativeness(content: string): number {
  const words = content.toLowerCase().split(/\W+/).filter((t) => t.length > 0)
  const n = words.length
  if (n === 0) return 0

  const fillerCount = words.filter((w) => FILLER_TOKENS.has(w)).length
  const fillerRatio = fillerCount / n

  const numbers = (content.match(/\d+/g) || []).length
  const properNouns = (content.match(/\b[A-Z][a-z]{2,}\b/g) || []).length
  const longWords = words.filter((w) => w.length >= 9).length
  const symbols = (content.match(/[()[\]/√ᵀ=+×∑]/g) || []).length
  const concreteCount = numbers + properNouns + longWords + Math.min(symbols, n)
  const concreteRatio = concreteCount / n

  const info = 0.5 + 2.0 * concreteRatio - 1.2 * fillerRatio
  return clamp(info, 0, 1)
}

export interface LessonRecord {
  sourceId: string
  weakestDimension: string
  composite: number
  decision: PreIngestDecision
  evidence: string[]
  proposedAction: string
  lineageRecordId: string
}

export interface PreIngestVerdict {
  decision: PreIngestDecision
  composite: number // dims 1-3 composite (0-1)
  density: number // distinct-token count (query-independent signal)
  informativeness: number // concrete-signal vs filler ratio (0-1, query-independent)
  dimensionScores: {relevance: number; coherence: number; factuality: number}
  truthFlag: 'ok' | 'contradiction' | 'low_evidence'
  sourceTier: SourceTier
  priority: number // 0 (drop) … 1 (top); informs queue ordering
  reasons: string[]
  lesson: LessonRecord
}

export class PreIngestGate {
  private readonly cfg: PreIngestConfig

  private readonly logger?: {
    logRubricScore: (l: RubricScoreLine) => void
    logLesson: (l: LessonLine) => void
  }

  constructor(
    cfg: Partial<PreIngestConfig> = {},
    logger?: {logRubricScore: (l: RubricScoreLine) => void; logLesson: (l: LessonLine) => void},
  ) {
    this.cfg = {...DEFAULT_PRE_INGEST_CONFIG, ...cfg}
    this.logger = logger
  }

  evaluate(
    content: string,
    meta: {sourceId: string; sourceTier?: SourceTier; query?: string},
  ): PreIngestVerdict {
    const sourceTier: SourceTier = meta.sourceTier ?? 4 // unknown provenance ⇒ lowest trust
    const reasons: string[] = []

    // --- Density scan (query-independent; THOTH component G) ---
    const tokens = content.toLowerCase().split(/\W+/).filter((t) => t.length > 0)
    const density = new Set(tokens).size
    const belowDensity = content.trim().length < this.cfg.minChars || density < this.cfg.minDistinctTokens

    // --- Informativeness scan (query-independent; closes dense-but-worthless gap) ---
    const informativeness = computeInformativeness(content)

    // --- Basic rubric scan: dimensions 1,2,3 only (THOTH "lite") ---
    const rubric = globalRubricScorer.score(content, {query: meta.query})
    const d = {
      relevance: rubric.dimensions.relevance,
      coherence: rubric.dimensions.coherence,
      factuality: rubric.dimensions.factuality,
    }
    const composite = (d.relevance + d.coherence + d.factuality) / 3

    // --- Truth/False lite ---
    const truth = globalTruthGate.gate(content)
    let truthFlag: PreIngestVerdict['truthFlag'] = 'ok'
    if (truth.verdict === 'false') {
      truthFlag = 'contradiction'
      reasons.push(`truth-lite: contradiction (severity=${truth.severity ?? 'n/a'})`)
    } else if (truth.verdict === 'uncertain' && truth.confidence < 0.35) {
      truthFlag = 'low_evidence'
      reasons.push('truth-lite: low-evidence')
    }

    // --- Decision logic (provenance + signal, fail-closed on tier) ---
    let decision: PreIngestDecision

    if (
      belowDensity ||
      informativeness < this.cfg.rejectInformativeness ||
      sourceTier > this.cfg.rejectAboveTier ||
      composite < this.cfg.rejectThreshold ||
      truthFlag === 'contradiction'
    ) {
      decision = 'reject'
      if (belowDensity) reasons.push(`thin content (density=${density}, len=${content.trim().length})`)
      if (informativeness < this.cfg.rejectInformativeness) reasons.push(`low informativeness ${informativeness.toFixed(2)} < ${this.cfg.rejectInformativeness}`)
      if (sourceTier > this.cfg.rejectAboveTier) reasons.push(`tier ${sourceTier} below provenance floor`)
      if (composite < this.cfg.rejectThreshold) reasons.push(`composite ${composite.toFixed(2)} < reject ${this.cfg.rejectThreshold}`)
    } else if (
      sourceTier > this.cfg.deprioritizeAboveTier ||
      composite < this.cfg.acceptThreshold ||
      informativeness < this.cfg.minInformativeness ||
      truthFlag === 'low_evidence'
    ) {
      decision = composite < this.cfg.acceptThreshold && truthFlag === 'ok' ? 'flag' : 'deprioritize'
      if (informativeness < this.cfg.minInformativeness) reasons.push(`informativeness ${informativeness.toFixed(2)} < ${this.cfg.minInformativeness}`)
      reasons.push(`tier ${sourceTier} / composite ${composite.toFixed(2)} below accept bar`)
    } else {
      decision = 'accept'
      reasons.push(`tier ${sourceTier}, composite ${composite.toFixed(2)} ≥ ${this.cfg.acceptThreshold}`)
    }

    // I2 hard guard: tier above provenance floor can NEVER be accept.
    if (decision === 'accept' && sourceTier > this.cfg.rejectAboveTier) {
      throw new Error('PreIngestGate I2 violated: accepted content above provenance floor')
    }

    const weakestDimension = (Object.entries(d).sort(([, a], [, b]) => a - b)[0]?.[0]) ?? 'relevance'
    const priority = this.computePriority(decision, composite, sourceTier)
    const proposedAction = this.proposedAction(decision, weakestDimension)

    // --- Lesson record into immutable lineage (I1) ---
    const rec = globalLineageAuditor.addRecord({
      verificationId: `preingest_${meta.sourceId}_${Date.now()}`,
      timestamp: BigInt(Date.now()) * BigInt(1_000_000),
      decision: decision === 'accept' ? 'accept' : 'quarantine',
      rubricScore: composite,
      truthVerdict: `preingest:${truthFlag}`,
      lineage: {
        who: 'pre_ingest_gate',
        what: {
          before: {sourceId: meta.sourceId, sourceTier},
          after: {decision, composite, weakestDimension, truthFlag, proposedAction},
        },
        when: BigInt(Date.now()) * BigInt(1_000_000),
        auth: 'pre_ingest_signal',
      },
    })

    const lesson: LessonRecord = {
      sourceId: meta.sourceId,
      weakestDimension,
      composite,
      decision,
      evidence: rubric.evidence.slice(0, 3).map((e) => `${e.dimension}:${e.signal}`),
      proposedAction,
      lineageRecordId: rec.chainHash,
    }

    // Optional durable JSONL trail (off unless a logger is injected → tests stay hermetic).
    if (this.logger) {
      this.logger.logRubricScore({
        sourceId: meta.sourceId,
        composite,
        dimensions: d,
        decision,
        lineageRecordId: rec.chainHash,
      })
      this.logger.logLesson(lesson)
    }

    return {decision, composite, density, informativeness, dimensionScores: d, truthFlag, sourceTier, priority, reasons, lesson}
  }

  private computePriority(decision: PreIngestDecision, composite: number, tier: SourceTier): number {
    if (decision === 'reject') return 0
    const tierWeight = (5 - tier) / 4 // tier1→1.0, tier4→0.25
    const base = composite * tierWeight
    if (decision === 'deprioritize' || decision === 'flag') return Math.min(base, 0.5)
    return base
  }

  private proposedAction(decision: PreIngestDecision, weakest: string): string {
    switch (decision) {
      case 'reject':
        return `drop source; do not ingest (weakest: ${weakest})`
      case 'deprioritize':
        return `queue at low priority; prefer higher-tier sources (weakest: ${weakest})`
      case 'flag':
        return `ingest but flag for review on dimension '${weakest}'`
      case 'accept':
        return 'ingest at computed priority'
    }
  }
}

export const globalPreIngestGate = new PreIngestGate()

/** Convenience: evaluate a candidate source chunk. */
export function preIngestEvaluate(
  content: string,
  meta: {sourceId: string; sourceTier?: SourceTier; query?: string},
): PreIngestVerdict {
  return globalPreIngestGate.evaluate(content, meta)
}
