/**
 * THOTH Component C/Master Flow — Master Loop State Machine.
 *
 * Wires the THOTH corpus flow end-to-end over the existing TS corpus:
 *
 *   IDLE → PRE_INGEST → REASONING → VERIFYING → REFLECT_HEAL → IMPROVING
 *        → DRAINAGE → WRITING_STATE → IDLE
 *
 * Integrates: PreIngestGate (Layer 0), LifecycleEnforcer (9-step), RefinementLoop
 * (healing), ControlLoop (bounded threshold improvement), LineageAuditor (provenance),
 * and the optional JsonlLogger (durable lesson/healing trail).
 *
 * INVARIANTS:
 *   I1: a PRE_INGEST 'reject' never reaches REASONING (fail-closed ingestion);
 *       it routes straight to DRAINAGE → WRITING_STATE → IDLE
 *   I2: every completed cycle ends in IDLE and emits one cycle lineage record
 *   I3: the recorded state path is a legal prefix-ordered subsequence of the
 *       canonical flow (no state appears before its predecessor)
 */

import {PreIngestGate, type SourceTier, type PreIngestDecision} from './pre_ingest_gate.js'
import {LifecycleEnforcer} from './lifecycle.js'
import {RefinementLoop} from '../loops/refinement_loop.js'
import {ControlLoop} from '../loops/control_loop.js'
import {AuditLoop} from '../loops/audit_loop.js'
import {globalLineageAuditor} from '../lineage_auditor.js'
import {type GuardrailProposal} from '../guardrail_learning_bridge.js'
import {globalJsonlLogger, type JsonlLogger} from './jsonl_logger.js'

export type MasterState =
  | 'IDLE'
  | 'PRE_INGEST'
  | 'REASONING'
  | 'VERIFYING'
  | 'REFLECT_HEAL'
  | 'IMPROVING'
  | 'DRAINAGE'
  | 'WRITING_STATE'

const CANONICAL_ORDER: MasterState[] = [
  'IDLE',
  'PRE_INGEST',
  'REASONING',
  'VERIFYING',
  'REFLECT_HEAL',
  'IMPROVING',
  'DRAINAGE',
  'WRITING_STATE',
]

export interface CycleInput {
  content: string
  sourceId: string
  sourceTier?: SourceTier
  intent?: string
  query?: string
}

export interface CycleResult {
  sourceId: string
  path: MasterState[]
  preIngestDecision: PreIngestDecision
  reasoned: boolean
  lifecycleDecision?: 'accept' | 'quarantine'
  lifecycleBlocked?: boolean
  healingProposed: boolean
  healingAccepted: boolean
  thresholdAfter: number
  drained: number
  chainIntact: boolean
  accepted: boolean
  lineageRecordId: string
}

export class MasterLoop {
  private state: MasterState = 'IDLE'
  private readonly gate: PreIngestGate
  private readonly lifecycle: LifecycleEnforcer
  private readonly refinement: RefinementLoop
  private readonly control: ControlLoop
  private readonly audit: AuditLoop
  private readonly logger?: Pick<JsonlLogger, 'logHealingAction' | 'logLesson' | 'logRubricScore'>
  private drainArchive = 0

  constructor(opts?: {
    gate?: PreIngestGate
    lifecycle?: LifecycleEnforcer
    control?: ControlLoop
    logger?: Pick<JsonlLogger, 'logHealingAction' | 'logLesson' | 'logRubricScore'>
  }) {
    // Forward the logger into the gate so rubric_scores are emitted on the default
    // path (unless an explicit gate is supplied, which takes precedence).
    this.gate = opts?.gate ?? new PreIngestGate({}, opts?.logger)
    this.lifecycle = opts?.lifecycle ?? new LifecycleEnforcer()
    this.refinement = new RefinementLoop()
    this.control = opts?.control ?? new ControlLoop()
    this.audit = new AuditLoop()
    this.logger = opts?.logger
  }

  getState(): MasterState {
    return this.state
  }

  /** Run one full corpus cycle. */
  runCycle(input: CycleInput): CycleResult {
    const path: MasterState[] = ['IDLE']
    const go = (s: MasterState) => {
      this.state = s
      path.push(s)
    }

    // PRE_INGEST (Layer 0)
    go('PRE_INGEST')
    const pre = this.gate.evaluate(input.content, {
      sourceId: input.sourceId,
      sourceTier: input.sourceTier,
      query: input.query,
    })

    let reasoned = false
    let lifecycleDecision: 'accept' | 'quarantine' | undefined
    let lifecycleBlocked: boolean | undefined
    let healingProposed = false
    let healingAccepted = false

    // I1: reject ⇒ skip REASONING, go straight to drainage.
    if (pre.decision !== 'reject') {
      // REASONING (corpus-centric: the accepted content is the candidate under scrutiny)
      go('REASONING')
      reasoned = true

      // VERIFYING (9-step lifecycle)
      go('VERIFYING')
      const lc = this.lifecycle.run(input.content, {intent: input.intent, query: input.query})
      lifecycleDecision = lc.finalDecision
      lifecycleBlocked = lc.blocked

      // REFLECT_HEAL — if sub-floor, propose + refine a healing change
      go('REFLECT_HEAL')
      if (lc.finalDecision === 'quarantine' || lc.rubricScore < 0.55) {
        healingProposed = true
        const proposal: GuardrailProposal = {
          target: 'safety_gate',
          changeType: 'threshold_adjust',
          proposal: `tune for weak cycle on ${input.sourceId}`,
          rationale: `lifecycle ${lc.finalDecision}, composite ${lc.rubricScore.toFixed(2)}`,
          expectedImpact: 'raise composite on similar inputs',
          residualRisk: 0.3,
        }
        const refined = this.refinement.refine(proposal, 4)
        healingAccepted = refined.accepted
        this.logger?.logHealingAction({
          target: proposal.target,
          changeType: proposal.changeType,
          rationale: proposal.rationale,
          residualRisk: refined.finalProposal.residualRisk,
          applied: refined.accepted,
        })
      }

      // IMPROVING — feed the bounded control loop and tick.
      go('IMPROVING')
      this.control.observe(lifecycleDecision === 'accept' ? 'accept' : 'quarantine')
      this.control.tick()
      this.control.assertInvariant()
    }

    // DRAINAGE — archive low-value cycles. (Lesson logging is owned by the gate,
    // which already emitted this cycle's lesson; re-logging here would duplicate it.)
    go('DRAINAGE')
    if (pre.decision === 'reject' || lifecycleDecision === 'quarantine') {
      this.drainArchive++
    }

    // WRITING_STATE — verify lineage integrity (Audit Loop, fail-closed) then persist.
    go('WRITING_STATE')
    this.audit.tick()
    const chainIntact = !this.audit.isQuarantined()
    const accepted = reasoned && lifecycleDecision === 'accept' && chainIntact
    const rec = globalLineageAuditor.addRecord({
      verificationId: `cycle_${input.sourceId}_${Date.now()}`,
      timestamp: BigInt(Date.now()) * BigInt(1_000_000),
      decision: accepted ? 'accept' : 'quarantine',
      rubricScore: pre.composite,
      truthVerdict: `cycle:${pre.decision}`,
      lineage: {
        who: 'thoth_master_loop',
        what: {
          before: {sourceId: input.sourceId, tier: pre.sourceTier},
          after: {
            preIngest: pre.decision,
            reasoned,
            lifecycle: lifecycleDecision ?? 'n/a',
            healingAccepted,
            threshold: this.control.getThreshold(),
            chainIntact,
            accepted,
          },
        },
        when: BigInt(Date.now()) * BigInt(1_000_000),
        auth: 'master_loop_cycle',
      },
    })

    // back to IDLE
    this.state = 'IDLE'
    this.assertPathLegal(path)

    return {
      sourceId: input.sourceId,
      path,
      preIngestDecision: pre.decision,
      reasoned,
      lifecycleDecision,
      lifecycleBlocked,
      healingProposed,
      healingAccepted,
      thresholdAfter: this.control.getThreshold(),
      drained: this.drainArchive,
      chainIntact,
      accepted,
      lineageRecordId: rec.chainHash,
    }
  }

  /** I3: each state in the path must not precede its canonical predecessor. */
  private assertPathLegal(path: MasterState[]): void {
    let lastIdx = -1
    for (const s of path) {
      const idx = CANONICAL_ORDER.indexOf(s)
      if (s === 'IDLE') {
        lastIdx = 0
        continue
      }
      if (idx < lastIdx) {
        throw new Error(`MasterLoop I3 violated: ${s} appeared out of canonical order`)
      }
      lastIdx = idx
    }
  }

  reset(): void {
    this.state = 'IDLE'
    this.drainArchive = 0
    this.control.reset()
    this.audit.reset()
  }
}

/**
 * Default global master loop — wired to the durable global JSONL logger so the
 * default path emits rubric_scores / lessons_learned / healing_actions trails.
 */
export const globalMasterLoop = new MasterLoop({logger: globalJsonlLogger})
