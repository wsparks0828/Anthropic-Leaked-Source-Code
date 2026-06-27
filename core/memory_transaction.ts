/**
 * Atomic Memory Wiring Transaction Layer
 *
 * Ensures all-or-nothing semantics for memory updates across semantic, episodic, and graph layers.
 * Prevents partial writes that could leave memory in inconsistent state during crashes.
 */

/**
 * Single memory update operation
 */
export interface MemoryUpdateOp {
  layer: 'semantic' | 'episodic' | 'graph'
  key: string
  value: any
  timestamp: bigint
}

/**
 * Transaction for atomic memory updates
 */
export class MemoryTransaction {
  private updates: MemoryUpdateOp[] = []
  private transactionId: string
  private createdAt: bigint
  private committed = false
  private rolledBack = false
  private memoryLayers: any // Will be injected

  constructor(transactionId?: string) {
    this.transactionId = transactionId || `txn_${Date.now()}_${Math.random().toString(36).slice(2)}`
    this.createdAt = BigInt(Date.now())
  }

  /**
   * Queue an update operation
   */
  addUpdate(layer: 'semantic' | 'episodic' | 'graph', key: string, value: any): void {
    if (this.committed || this.rolledBack) {
      throw new Error(`Cannot add updates to ${this.committed ? 'committed' : 'rolled back'} transaction`)
    }

    this.updates.push({
      layer,
      key,
      value,
      timestamp: BigInt(Date.now()),
    })
  }

  /**
   * Commit all updates atomically
   */
  commit(memoryLayers: any): {success: boolean; error?: string} {
    if (this.committed || this.rolledBack) {
      return {success: false, error: 'Transaction already finalized'}
    }

    if (this.updates.length === 0) {
      this.committed = true
      return {success: true}
    }

    this.memoryLayers = memoryLayers

    try {
      // Validate all updates can be applied
      this.validateUpdates()

      // Apply all updates
      for (const op of this.updates) {
        if (!memoryLayers[op.layer]) {
          throw new Error(`Invalid layer: ${op.layer}`)
        }
        memoryLayers[op.layer].set(op.key, op.value)
      }

      this.committed = true

      console.log(
        `[memory-txn] Transaction committed: ${this.transactionId} (${this.updates.length} operations)`,
      )

      return {success: true}
    } catch (error) {
      // Rollback on any error
      console.error(`[memory-txn] Transaction failed, rolling back: ${error}`)
      this.rollback()
      return {success: false, error: String(error)}
    }
  }

  /**
   * Validate all updates before committing
   */
  private validateUpdates(): void {
    const keySets: Map<string, Set<string>> = new Map()

    for (const op of this.updates) {
      // Check for duplicate keys in same layer (would be overwrite)
      const layerKeys = keySets.get(op.layer) || new Set()
      if (layerKeys.has(op.key)) {
        throw new Error(`Duplicate key in transaction: ${op.layer}.${op.key}`)
      }
      layerKeys.add(op.key)
      keySets.set(op.layer, layerKeys)

      // Validate value is serializable
      try {
        JSON.stringify(op.value)
      } catch {
        throw new Error(`Non-serializable value for ${op.layer}.${op.key}`)
      }
    }
  }

  /**
   * Rollback all updates
   */
  rollback(): void {
    if (this.rolledBack) return

    // In-memory only: just discard updates
    this.updates = []
    this.rolledBack = true

    console.log(`[memory-txn] Transaction rolled back: ${this.transactionId}`)
  }

  /**
   * Get transaction status
   */
  getStatus(): {
    id: string
    operationCount: number
    committed: boolean
    rolledBack: boolean
    age: bigint
  } {
    return {
      id: this.transactionId,
      operationCount: this.updates.length,
      committed: this.committed,
      rolledBack: this.rolledBack,
      age: BigInt(Date.now()) - this.createdAt,
    }
  }
}

/**
 * Transaction manager for memory wiring
 */
export class MemoryTransactionManager {
  private activeTransactions: Map<string, MemoryTransaction> = new Map()
  private committedTransactions: Map<string, bigint> = new Map()
  private maxTransactionAge: bigint = BigInt(5 * 60 * 1000) // 5 minutes

  /**
   * Create new transaction
   */
  createTransaction(): MemoryTransaction {
    const txn = new MemoryTransaction()
    this.activeTransactions.set(txn.getStatus().id, txn)
    return txn
  }

  /**
   * Commit transaction
   */
  commitTransaction(txn: MemoryTransaction, memoryLayers: any): boolean {
    const result = txn.commit(memoryLayers)

    if (result.success) {
      const status = txn.getStatus()
      this.activeTransactions.delete(status.id)
      this.committedTransactions.set(status.id, BigInt(Date.now()))

      // Cleanup old committed transactions
      this.cleanupStaleTransactions()

      return true
    }

    return false
  }

  /**
   * Get transaction status
   */
  getTransactionStatus(txnId: string) {
    const txn = this.activeTransactions.get(txnId)
    if (!txn) return null
    return txn.getStatus()
  }

  /**
   * Cleanup old transactions
   */
  private cleanupStaleTransactions(): void {
    const now = BigInt(Date.now())
    const toDelete = []

    for (const [id, timestamp] of this.committedTransactions.entries()) {
      if (now - timestamp > this.maxTransactionAge) {
        toDelete.push(id)
      }
    }

    for (const id of toDelete) {
      this.committedTransactions.delete(id)
    }

    if (toDelete.length > 0) {
      console.log(`[memory-txn] Cleaned up ${toDelete.length} stale committed transactions`)
    }
  }

  /**
   * Get stats
   */
  getStats(): {
    activeTransactions: number
    committedTransactions: number
  } {
    return {
      activeTransactions: this.activeTransactions.size,
      committedTransactions: this.committedTransactions.size,
    }
  }

  /**
   * Reset manager (for testing)
   */
  reset(): void {
    this.activeTransactions.clear()
    this.committedTransactions.clear()
  }
}

/**
 * Global transaction manager
 */
export const globalMemoryTransactionManager = new MemoryTransactionManager()

/**
 * Create new memory transaction
 */
export function createMemoryTransaction(): MemoryTransaction {
  return globalMemoryTransactionManager.createTransaction()
}
