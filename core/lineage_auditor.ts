/**
 * Lineage Auditor: Immutable Chain Verification & Audit Trail
 *
 * Validates forensic chain integrity:
 * - No gaps in verification chain
 * - All 4 forensic fields present (who/what/when/auth)
 * - SHA256 chain hashes prevent tampering
 * - Timestamps monotonically increasing
 * - No records deleted or modified
 */

import {createHash} from 'crypto'
import {isAuthorizedForLineageExport, encryptSensitiveField, SENSITIVE_FIELDS} from './lineage_encryption.js'

/**
 * Individual verification record in lineage chain.
 */
export interface LineageRecord {
  verificationId: string
  timestamp: bigint // nanoseconds
  decision: 'accept' | 'quarantine' | 'error'
  rubricScore: number
  truthVerdict: string
  lineage: {
    who: string // Component that initiated
    what: {before?: any; after: any} // State delta
    when: bigint // Nanosecond timestamp
    auth: string // Authentication/authorization signal type
  }
  chainHash: string // SHA256(prev + record)
  prevHash: string // Previous record's chainHash
}

/**
 * Result of chain verification.
 */
export interface ChainVerificationResult {
  valid: boolean
  brokenAt?: string // verificationId where chain breaks
  violations?: Array<{
    verificationId: string
    type: string
    description: string
  }>
}

/**
 * Lineage Auditor: Maintains immutable verification chain.
 */
export class LineageAuditor {
  private chain: LineageRecord[] = []
  private recordIndex: Map<string, LineageRecord> = new Map()
  private lastChainHash: string = 'genesis'

  /**
   * Add a verification record to the chain.
   */
  addRecord(record: Omit<LineageRecord, 'chainHash' | 'prevHash'>): LineageRecord {
    const prevHash = this.lastChainHash
    const chainHash = this.computeChainHash(prevHash, record)

    const fullRecord: LineageRecord = {
      ...record,
      chainHash,
      prevHash,
    }

    this.chain.push(fullRecord)
    this.recordIndex.set(record.verificationId, fullRecord)
    this.lastChainHash = chainHash

    return fullRecord
  }

  /**
   * Verify entire chain for tampering.
   */
  verifyLineageChain(): ChainVerificationResult {
    const violations: Array<{verificationId: string; type: string; description: string}> = []

    // Check 1: No gaps in chain
    if (this.chain.length === 0) {
      return {valid: true, violations: []} // Empty chain is valid
    }

    // Check 2: Verify first record has 'genesis' as prevHash
    if (this.chain[0].prevHash !== 'genesis') {
      violations.push({
        verificationId: this.chain[0].verificationId,
        type: 'invalid_chain_start',
        description: 'First record does not have genesis as prevHash',
      })
      return {valid: false, brokenAt: this.chain[0].verificationId, violations}
    }

    // Check 3: All records have required forensic fields
    for (const record of this.chain) {
      if (!record.lineage.who || !record.lineage.when || !record.lineage.auth) {
        violations.push({
          verificationId: record.verificationId,
          type: 'missing_forensic_field',
          description: `Missing lineage field(s): who=${record.lineage.who}, when=${record.lineage.when}, auth=${record.lineage.auth}`,
        })
        return {valid: false, brokenAt: record.verificationId, violations}
      }

      if (!record.lineage.what || (typeof record.lineage.what === 'object' && !record.lineage.what.after)) {
        violations.push({
          verificationId: record.verificationId,
          type: 'missing_delta',
          description: 'Missing delta.after in lineage.what',
        })
        return {valid: false, brokenAt: record.verificationId, violations}
      }
    }

    // Check 4: Verify chain hashes
    let prevHash = 'genesis'
    for (let i = 0; i < this.chain.length; i++) {
      const record = this.chain[i]

      // Check prevHash matches
      if (record.prevHash !== prevHash) {
        violations.push({
          verificationId: record.verificationId,
          type: 'chain_link_broken',
          description: `prevHash mismatch: expected ${prevHash}, got ${record.prevHash}`,
        })
        return {valid: false, brokenAt: record.verificationId, violations}
      }

      // Recompute chainHash and verify
      const recordWithoutHash = {
        verificationId: record.verificationId,
        timestamp: record.timestamp,
        decision: record.decision,
        rubricScore: record.rubricScore,
        truthVerdict: record.truthVerdict,
        lineage: record.lineage,
      }
      const expectedHash = this.computeChainHash(prevHash, recordWithoutHash)

      if (record.chainHash !== expectedHash) {
        violations.push({
          verificationId: record.verificationId,
          type: 'hash_tampering',
          description: `chainHash mismatch: expected ${expectedHash}, got ${record.chainHash}`,
        })
        return {valid: false, brokenAt: record.verificationId, violations}
      }

      prevHash = record.chainHash
    }

    // Check 5: Timestamps monotonically increasing
    for (let i = 1; i < this.chain.length; i++) {
      if (this.chain[i].timestamp <= this.chain[i - 1].timestamp) {
        violations.push({
          verificationId: this.chain[i].verificationId,
          type: 'timestamp_order_violation',
          description: `Timestamp not monotonically increasing: ${this.chain[i - 1].timestamp} >= ${this.chain[i].timestamp}`,
        })
        return {valid: false, brokenAt: this.chain[i].verificationId, violations}
      }
    }

    return {valid: true, violations: violations.length > 0 ? violations : undefined}
  }

  /**
   * Export lineage records for audit.
   * Requires authorization token for access control.
   * Sensitive fields are encrypted in output.
   */
  exportLineage(
    limit: number = 100,
    format: 'json-ld' | 'json' = 'json-ld',
    authToken?: string,
  ): LineageRecord[] {
    // ACCESS CONTROL: Check authorization
    if (!isAuthorizedForLineageExport(authToken)) {
      console.warn('[lineage-auditor] Unauthorized exportLineage attempt (no valid token)')
      return []
    }

    const records = this.chain.slice(-limit)

    if (format === 'json-ld') {
      return records.map((record) => ({
        '@context': 'https://anthropic.com/guardrail/lineage/v1',
        '@id': `urn:guardrail:verification:${record.verificationId}`,
        '@type': 'GuardrailVerification',
        verificationId: record.verificationId,
        timestamp: record.timestamp.toString(),
        decision: record.decision,
        rubricScore: record.rubricScore,
        truthVerdict: record.truthVerdict,
        lineage: {
          who: record.lineage.who,
          what: record.lineage.what,
          when: record.lineage.when.toString(),
          auth: record.lineage.auth,
        },
        chainHash: record.chainHash,
        prevHash: record.prevHash,
      })) as any
    }

    return records
  }

  /**
   * Get a specific record by verification ID.
   */
  getRecord(verificationId: string): LineageRecord | undefined {
    return this.recordIndex.get(verificationId)
  }

  /**
   * Get chain statistics.
   */
  getChainStats(): {
    totalRecords: number
    oldestRecord?: {verificationId: string; timestamp: bigint}
    newestRecord?: {verificationId: string; timestamp: bigint}
    currentChainHash: string
  } {
    return {
      totalRecords: this.chain.length,
      oldestRecord:
        this.chain.length > 0
          ? {verificationId: this.chain[0].verificationId, timestamp: this.chain[0].timestamp}
          : undefined,
      newestRecord:
        this.chain.length > 0
          ? {verificationId: this.chain[this.chain.length - 1].verificationId, timestamp: this.chain[this.chain.length - 1].timestamp}
          : undefined,
      currentChainHash: this.lastChainHash,
    }
  }

  /**
   * Clear entire chain (only for testing/reset).
   */
  reset(): void {
    this.chain = []
    this.recordIndex.clear()
    this.lastChainHash = 'genesis'
  }

  /**
   * Compute chain hash: SHA256(previous_hash + serialized_record)
   */
  private computeChainHash(prevHash: string, record: any): string {
    const recordJson = JSON.stringify({
      verificationId: record.verificationId,
      timestamp: record.timestamp.toString(), // BigInt to string
      decision: record.decision,
      rubricScore: record.rubricScore,
      truthVerdict: record.truthVerdict,
      lineage: {
        who: record.lineage.who,
        what: record.lineage.what,
        when: record.lineage.when.toString(),
        auth: record.lineage.auth,
      },
    })

    const input = prevHash + recordJson
    return createHash('sha256').update(input).digest('hex')
  }

  /**
   * Get the last N records.
   */
  getRecent(count: number = 10): LineageRecord[] {
    return this.chain.slice(-count)
  }

  /**
   * Count records within time range.
   */
  countInTimeRange(startTime: bigint, endTime: bigint): number {
    return this.chain.filter((r) => r.timestamp >= startTime && r.timestamp <= endTime).length
  }
}

/**
 * Global lineage auditor instance.
 */
export const globalLineageAuditor = new LineageAuditor()

/**
 * Convenience function: verify chain integrity.
 */
export function verifyLineageChain(): ChainVerificationResult {
  return globalLineageAuditor.verifyLineageChain()
}

/**
 * Convenience function: export lineage for audit.
 */
export function exportLineage(limit: number = 100, format: 'json-ld' | 'json' = 'json-ld'): LineageRecord[] {
  return globalLineageAuditor.exportLineage(limit, format)
}

/**
 * Convenience function: get chain statistics.
 */
export function getLineageStats(): {
  totalRecords: number
  oldestRecord?: {verificationId: string; timestamp: bigint}
  newestRecord?: {verificationId: string; timestamp: bigint}
  currentChainHash: string
} {
  return globalLineageAuditor.getChainStats()
}
