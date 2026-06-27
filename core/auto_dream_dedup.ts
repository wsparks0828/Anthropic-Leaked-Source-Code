/**
 * autoDream Proposal Deduplicator
 *
 * Ensures idempotency: prevents the same proposal from being applied multiple times
 * even if autoDream cycles repeat or are triggered with overlapping signals.
 *
 * Maintains hash of all applied proposals with timestamps.
 */

import {createHash} from 'crypto'

export interface AppliedProposal {
  hash: string
  proposalId: string
  appliedAt: bigint
  lineageId: string
}

/**
 * Deduplicator for autoDream proposals
 */
class ProposalDeduplicator {
  private appliedProposals: Map<string, AppliedProposal> = new Map()
  private maxHistoryAge: bigint = BigInt(7 * 24 * 60 * 60 * 1000) // 7 days

  /**
   * Check if proposal is duplicate (already applied)
   */
  isDuplicate(proposal: any): boolean {
    const hash = this.hashProposal(proposal)
    return this.appliedProposals.has(hash)
  }

  /**
   * Record proposal as applied
   */
  markApplied(proposal: any, proposalId: string, lineageId: string): void {
    const hash = this.hashProposal(proposal)

    this.appliedProposals.set(hash, {
      hash,
      proposalId,
      appliedAt: BigInt(Date.now()),
      lineageId,
    })

    console.log(`[autoDream-dedup] Marked proposal as applied: ${proposalId} (hash=${hash.substring(0, 8)}...)`)

    // Cleanup old entries
    this.cleanupStaleEntries()
  }

  /**
   * Hash a proposal for deduplication
   */
  private hashProposal(proposal: any): string {
    // Include key fields in hash: target, changeType, and proposal text
    const keyFields = {
      target: proposal.target,
      changeType: proposal.changeType,
      proposal: proposal.proposal,
    }

    const serialized = JSON.stringify(keyFields)
    return createHash('sha256').update(serialized).digest('hex')
  }

  /**
   * Clean up old entries outside retention window
   */
  private cleanupStaleEntries(): void {
    const now = BigInt(Date.now())
    const entriesToDelete = []

    for (const [hash, record] of this.appliedProposals.entries()) {
      if (now - record.appliedAt > this.maxHistoryAge) {
        entriesToDelete.push(hash)
      }
    }

    for (const hash of entriesToDelete) {
      this.appliedProposals.delete(hash)
    }

    if (entriesToDelete.length > 0) {
      console.log(`[autoDream-dedup] Cleaned up ${entriesToDelete.length} stale proposal records`)
    }
  }

  /**
   * Get deduplication stats
   */
  getStats(): {
    totalTracked: number
    oldestProposal?: {proposalId: string; age: bigint}
    newestProposal?: {proposalId: string; age: bigint}
  } {
    if (this.appliedProposals.size === 0) {
      return {totalTracked: 0}
    }

    const now = BigInt(Date.now())
    const proposals = Array.from(this.appliedProposals.values())
    const oldest = proposals.reduce((a, b) => (a.appliedAt < b.appliedAt ? a : b))
    const newest = proposals.reduce((a, b) => (a.appliedAt > b.appliedAt ? a : b))

    return {
      totalTracked: this.appliedProposals.size,
      oldestProposal: {proposalId: oldest.proposalId, age: now - oldest.appliedAt},
      newestProposal: {proposalId: newest.proposalId, age: now - newest.appliedAt},
    }
  }

  /**
   * Reset deduplicator (for testing)
   */
  reset(): void {
    this.appliedProposals.clear()
  }
}

/**
 * Global proposal deduplicator
 */
export const globalProposalDeduplicator = new ProposalDeduplicator()

/**
 * Check if proposal is duplicate
 */
export function isProposalDuplicate(proposal: any): boolean {
  return globalProposalDeduplicator.isDuplicate(proposal)
}

/**
 * Mark proposal as applied
 */
export function markProposalAsApplied(proposal: any, proposalId: string, lineageId: string): void {
  globalProposalDeduplicator.markApplied(proposal, proposalId, lineageId)
}
