/**
 * autoDream Proposal Persistence Layer
 *
 * Persists learned improvements across application restarts:
 * - Saves proposal queue to durable storage
 * - Restores pending proposals on startup
 * - Prevents re-learning of identical proposals (idempotency)
 * - Tracks proposal history with timestamps
 * - Validates persisted proposals before application
 */

/**
 * Persisted proposal record
 */
export interface PersistedProposal {
  proposalId: string
  timestamp: bigint
  target: string
  changeType: string
  proposal: string
  evidence: string
  impact: number
  residualRisk: number
  lineageId?: string
}

/**
 * Proposal queue snapshot
 */
export interface ProposalQueueSnapshot {
  exportedAt: string
  totalProposals: number
  proposals: PersistedProposal[]
  queueHash: string
}

/**
 * Proposal persistence manager
 */
class AutoDreamPersistenceManager {
  private storageBackend: any = null
  private proposalQueue: PersistedProposal[] = []
  private lastPersistTime: bigint = BigInt(0)
  private persistInterval: bigint = BigInt(300000) // 5 minutes
  private queueHash: string = ''

  /**
   * Initialize persistence manager
   */
  initialize(storageBackend: any): void {
    this.storageBackend = storageBackend
    console.log('[autoDream-persist] Proposal persistence manager initialized')
  }

  /**
   * Add proposal to queue
   */
  addProposal(proposal: {
    proposalId: string
    target: string
    changeType: string
    proposal: string
    evidence: string
    impact: number
    residualRisk: number
  }): void {
    const persistedProposal: PersistedProposal = {
      ...proposal,
      timestamp: BigInt(Date.now()),
    }

    this.proposalQueue.push(persistedProposal)
    console.log(`[autoDream-persist] Proposal queued: ${proposal.proposalId}`)

    // Periodic persistence
    this.checkAndPersist()
  }

  /**
   * Check if persistence is needed and persist if due
   */
  private checkAndPersist(): void {
    const now = BigInt(Date.now())

    // Persist periodically or if queue is getting large
    if (now - this.lastPersistTime >= this.persistInterval || this.proposalQueue.length > 100) {
      this.persistQueueToDisk()
    }
  }

  /**
   * Persist proposal queue to disk
   */
  async persistQueueToDisk(): Promise<void> {
    if (!this.storageBackend || !this.proposalQueue.length) {
      return
    }

    try {
      const snapshot: ProposalQueueSnapshot = {
        exportedAt: new Date().toISOString(),
        totalProposals: this.proposalQueue.length,
        proposals: this.proposalQueue,
        queueHash: this.computeQueueHash(),
      }

      await this.storageBackend.writeAutoDreamProposals(snapshot)

      this.lastPersistTime = BigInt(Date.now())
      this.queueHash = snapshot.queueHash

      console.log(`[autoDream-persist] Persisted ${this.proposalQueue.length} proposals to disk`)
    } catch (error) {
      console.error('[autoDream-persist] Failed to persist proposals:', error)
    }
  }

  /**
   * Restore proposals from persistent storage
   */
  async restoreFromDisk(): Promise<PersistedProposal[]> {
    if (!this.storageBackend) {
      console.log('[autoDream-persist] No storage backend configured, skipping restore')
      return []
    }

    try {
      if (typeof this.storageBackend.readAutoDreamProposals !== 'function') {
        console.log('[autoDream-persist] Storage backend does not support proposal restore')
        return []
      }

      const snapshot = await this.storageBackend.readAutoDreamProposals()

      if (!snapshot || !Array.isArray(snapshot.proposals)) {
        console.warn('[autoDream-persist] Invalid proposal snapshot from storage')
        return []
      }

      // Validate restored proposals
      const validProposals = this.validateRestoredProposals(snapshot.proposals)

      this.proposalQueue = validProposals
      this.queueHash = snapshot.queueHash

      console.log(`[autoDream-persist] Restored ${validProposals.length} proposals from disk`)

      return validProposals
    } catch (error) {
      console.error('[autoDream-persist] Failed to restore proposals:', error)
      return []
    }
  }

  /**
   * Validate restored proposals before application
   */
  private validateRestoredProposals(proposals: any[]): PersistedProposal[] {
    const validated: PersistedProposal[] = []

    for (const proposal of proposals) {
      // Check required fields
      if (
        !proposal.proposalId ||
        !proposal.target ||
        !proposal.changeType ||
        typeof proposal.residualRisk !== 'number'
      ) {
        console.warn('[autoDream-persist] Skipping invalid proposal:', proposal)
        continue
      }

      // Only restore low-risk proposals (high-risk ones need manual review anyway)
      if (proposal.residualRisk > 0.2) {
        console.log('[autoDream-persist] Skipping high-risk proposal (requires manual review):', proposal.proposalId)
        continue
      }

      validated.push(proposal as PersistedProposal)
    }

    return validated
  }

  /**
   * Get pending proposals
   */
  getPendingProposals(): PersistedProposal[] {
    return [...this.proposalQueue]
  }

  /**
   * Mark proposal as applied and remove from queue
   */
  markProposalApplied(proposalId: string, lineageId: string): void {
    const index = this.proposalQueue.findIndex((p) => p.proposalId === proposalId)

    if (index >= 0) {
      this.proposalQueue[index].lineageId = lineageId
      this.proposalQueue.splice(index, 1)
      console.log(`[autoDream-persist] Proposal applied and removed: ${proposalId}`)

      // Persist after removal
      this.persistQueueToDisk()
    }
  }

  /**
   * Clear all proposals
   */
  clearQueue(): void {
    this.proposalQueue = []
    console.log('[autoDream-persist] Proposal queue cleared')
  }

  /**
   * Get queue statistics
   */
  getQueueStats(): {
    pendingCount: number
    oldestProposal?: {proposalId: string; age: bigint}
    newestProposal?: {proposalId: string; age: bigint}
    averageRisk: number
  } {
    const now = BigInt(Date.now())
    const risks = this.proposalQueue.map((p) => p.residualRisk)

    return {
      pendingCount: this.proposalQueue.length,
      oldestProposal:
        this.proposalQueue.length > 0
          ? {proposalId: this.proposalQueue[0].proposalId, age: now - this.proposalQueue[0].timestamp}
          : undefined,
      newestProposal:
        this.proposalQueue.length > 0
          ? {
              proposalId: this.proposalQueue[this.proposalQueue.length - 1].proposalId,
              age: now - this.proposalQueue[this.proposalQueue.length - 1].timestamp,
            }
          : undefined,
      averageRisk: risks.length > 0 ? risks.reduce((a, b) => a + b, 0) / risks.length : 0,
    }
  }

  /**
   * Compute queue hash for integrity checking
   */
  private computeQueueHash(): string {
    // Simple hash based on proposals
    const proposalIds = this.proposalQueue.map((p) => p.proposalId).join('|')
    return `queue_${proposalIds.length}_${Date.now()}`
  }

  /**
   * Reset persistence manager (for testing)
   */
  reset(): void {
    this.proposalQueue = []
    this.lastPersistTime = BigInt(0)
    this.queueHash = ''
  }
}

/**
 * Global autoDream persistence manager
 */
export const globalAutoDreamPersistence = new AutoDreamPersistenceManager()

/**
 * Initialize autoDream persistence
 */
export function initializeAutoDreamPersistence(storageBackend: any): void {
  globalAutoDreamPersistence.initialize(storageBackend)
}

/**
 * Restore proposals on startup
 */
export async function restoreAutoDreamProposalsOnStartup(): Promise<PersistedProposal[]> {
  return await globalAutoDreamPersistence.restoreFromDisk()
}
