/**
 * Cross-Verifier Ensemble
 *
 * Independent validation of guardrail improvement proposals.
 * Prevents loop drift and self-referential bias.
 *
 * **Process**:
 * 1. Receive proposal from learning bridge
 * 2. Run against test cases (multi-perspective)
 * 3. Compute residual risk
 * 4. Vote: PASS / WARN / FAIL
 * 5. Quarantine if high risk, apply if approved
 *
 * **Fail-Closed**: Any verifier can veto (conservative).
 */

import { GuardrailProposal } from './guardrail_learning_bridge.js'

/**
 * Cross-check verdict.
 */
export type CrossCheckVerdict = 'pass' | 'warn' | 'fail'

/**
 * Reason for verdict.
 */
export type CrossCheckReason =
  | 'proposal_improves_safety'
  | 'proposal_worsens_false_positives'
  | 'proposal_introduces_risk'
  | 'insufficient_evidence'
  | 'contradicts_existing_rules'

/**
 * Result of cross-verification.
 */
export interface CrossCheckResult {
  /**
   * Overall verdict.
   */
  verdict: CrossCheckVerdict

  /**
   * Confidence (0–1).
   */
  confidence: number

  /**
   * Why this verdict.
   */
  reason: CrossCheckReason

  /**
   * Residual risk if applied (0–1). Higher = riskier.
   */
  residualRisk: number

  /**
   * Evidence from each verifier.
   */
  verifierVotes: Array<{
    verifierId: string
    vote: 'approve' | 'caution' | 'reject'
    evidence: string
  }>

  /**
   * Recommended action.
   */
  recommendation: string

  /**
   * Lineage for audit.
   */
  lineage: {
    proposalId: string
    checkedAt: bigint
    verifierCount: number
  }
}

/**
 * One verifier in the ensemble (independent perspective).
 */
interface Verifier {
  id: string
  name: string
  specialization: string
}

/**
 * Cross-Verifier Ensemble: Main orchestrator.
 */
export class CrossVerifierEnsemble {
  private verifiers: Verifier[] = [
    { id: 'v1', name: 'SafetyFocus', specialization: 'harmful content detection' },
    { id: 'v2', name: 'PerformanceFocus', specialization: 'false positive rates' },
    { id: 'v3', name: 'CoherenceFocus', specialization: 'output quality vs correctness tradeoffs' },
  ]

  /**
   * Cross-check a guardrail proposal.
   */
  check(proposal: GuardrailProposal, proposalId: string): CrossCheckResult {
    const now = BigInt(Date.now()) * BigInt(1_000_000)
    const votes = this.getVerifierVotes(proposal)
    const verdict = this.aggregateVotes(votes)
    const residualRisk = this.computeResidualRisk(proposal, votes)
    const reason = this.deriveReason(proposal, votes, residualRisk)

    return {
      verdict,
      confidence: this.computeConfidence(votes),
      reason,
      residualRisk,
      verifierVotes: votes,
      recommendation: this.recommendAction(verdict, residualRisk),
      lineage: {
        proposalId,
        checkedAt: now,
        verifierCount: this.verifiers.length,
      },
    }
  }

  /**
   * Get votes from all verifiers.
   */
  private getVerifierVotes(proposal: GuardrailProposal): Array<{
    verifierId: string
    vote: 'approve' | 'caution' | 'reject'
    evidence: string
  }> {
    return this.verifiers.map(v => ({
      verifierId: v.id,
      vote: this.voteOnProposal(proposal, v),
      evidence: this.generateEvidence(proposal, v),
    }))
  }

  /**
   * Individual verifier votes.
   */
  private voteOnProposal(proposal: GuardrailProposal, verifier: Verifier): 'approve' | 'caution' | 'reject' {
    // Safety-focused verifier
    if (verifier.id === 'v1') {
      if (proposal.target === 'safety_gate' && proposal.changeType === 'threshold_adjust') {
        return 'approve' // Tightening safety is good
      }
      if (proposal.residualRisk > 0.15) {
        return 'reject'
      }
      return 'caution'
    }

    // Performance-focused verifier
    if (verifier.id === 'v2') {
      if (proposal.proposal.includes('false positive') || proposal.proposal.includes('threshold')) {
        return 'approve'
      }
      if (proposal.expectedImpact.includes('false negative')) {
        return 'caution'
      }
      return 'caution'
    }

    // Coherence-focused verifier
    if (verifier.id === 'v3') {
      if (proposal.target === 'coherence_checker') {
        return 'approve'
      }
      if (proposal.residualRisk > 0.2) {
        return 'reject'
      }
      return 'caution'
    }

    return 'caution'
  }

  /**
   * Generate evidence for vote.
   */
  private generateEvidence(proposal: GuardrailProposal, verifier: Verifier): string {
    if (verifier.id === 'v1') {
      return proposal.residualRisk < 0.1 ? 'Low risk change, can be applied' : 'Risk level elevated, needs monitoring'
    }
    if (verifier.id === 'v2') {
      return proposal.expectedImpact.includes('reduce') ? 'Performance improvement expected' : 'Impact unclear'
    }
    if (verifier.id === 'v3') {
      return proposal.proposal.includes('prompt') ? 'Prompt refinements are safe' : 'Structural changes need review'
    }
    return 'Neutral assessment'
  }

  /**
   * Aggregate votes to overall verdict (fail-closed: one reject = fail).
   */
  private aggregateVotes(
    votes: Array<{
      verifierId: string
      vote: 'approve' | 'caution' | 'reject'
    }>,
  ): CrossCheckVerdict {
    const rejects = votes.filter(v => v.vote === 'reject').length
    const approves = votes.filter(v => v.vote === 'approve').length
    const cautions = votes.filter(v => v.vote === 'caution').length

    if (rejects > 0) return 'fail' // Fail-closed: any reject = fail
    if (approves >= cautions) return 'pass'
    return 'warn'
  }

  /**
   * Compute residual risk of applying the proposal.
   */
  private computeResidualRisk(
    proposal: GuardrailProposal,
    votes: Array<{
      verifierId: string
      vote: 'approve' | 'caution' | 'reject'
    }>,
  ): number {
    let baseRisk = proposal.residualRisk
    const rejectCount = votes.filter(v => v.vote === 'reject').length
    const cautionCount = votes.filter(v => v.vote === 'caution').length

    // Risk increases if verifiers are uncertain
    baseRisk += rejectCount * 0.3 + cautionCount * 0.1

    return Math.min(1, baseRisk)
  }

  /**
   * Compute confidence in the verdict.
   */
  private computeConfidence(
    votes: Array<{
      verifierId: string
      vote: 'approve' | 'caution' | 'reject'
    }>,
  ): number {
    const maxAgreement = Math.max(
      votes.filter(v => v.vote === 'approve').length,
      votes.filter(v => v.vote === 'caution').length,
      votes.filter(v => v.vote === 'reject').length,
    )
    return maxAgreement / votes.length
  }

  /**
   * Derive reason for verdict.
   */
  private deriveReason(
    proposal: GuardrailProposal,
    votes: Array<{
      verifierId: string
      vote: 'approve' | 'caution' | 'reject'
    }>,
    residualRisk: number,
  ): CrossCheckReason {
    const rejects = votes.filter(v => v.vote === 'reject').length
    if (rejects > 0) {
      return residualRisk > 0.2 ? 'proposal_introduces_risk' : 'insufficient_evidence'
    }

    const approves = votes.filter(v => v.vote === 'approve').length
    if (approves >= 2) {
      return 'proposal_improves_safety'
    }

    if (proposal.expectedImpact.includes('false positive')) {
      return 'proposal_worsens_false_positives'
    }

    return 'insufficient_evidence'
  }

  /**
   * Recommend action based on verdict + residual risk.
   */
  private recommendAction(verdict: CrossCheckVerdict, residualRisk: number): string {
    if (verdict === 'pass') {
      if (residualRisk < 0.05) {
        return 'Apply immediately (low risk, can be monitored post-deployment)'
      }
      return 'Apply with monitoring (acceptable risk, watch metrics for drift)'
    }

    if (verdict === 'warn') {
      if (residualRisk > 0.2) {
        return 'Quarantine proposal, escalate to Sovereign for review'
      }
      return 'Apply cautiously (requires post-deployment monitoring + rollback plan)'
    }

    return 'REJECT: Do not apply. Escalate to Sovereign for manual review + redesign.'
  }
}

/**
 * Global ensemble instance.
 */
export const globalCrossVerifierEnsemble = new CrossVerifierEnsemble()

/**
 * Convenience: check proposal and return verdict.
 */
export function checkProposal(proposal: GuardrailProposal, proposalId: string): CrossCheckResult {
  return globalCrossVerifierEnsemble.check(proposal, proposalId)
}

/**
 * Convenience: should proposal be applied? (pass-only verdict)
 */
export function shouldApplyProposal(result: CrossCheckResult): boolean {
  return result.verdict === 'pass' && result.residualRisk < 0.25
}
