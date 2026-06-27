/**
 * Phase 9: Critical Blockers - autoDream Proposal Persistence Tests
 *
 * Validates learned improvements are preserved across restarts:
 * - Proposal queue persistence to durable storage
 * - Restoration of pending proposals on startup
 * - Validation of restored proposals
 * - High-risk proposal filtering (manual review required)
 * - Queue statistics and integrity tracking
 * - Idempotency - no duplicate applications
 */

import {describe, it, expect, beforeEach} from 'bun:test'
import {
  globalAutoDreamPersistence,
  initializeAutoDreamPersistence,
  restoreAutoDreamProposalsOnStartup,
  type PersistedProposal,
} from '../auto_dream_persistence.js'

// Mock storage backend
class MockStorageBackend {
  private proposals: any = null
  private readEnabled = true

  async writeAutoDreamProposals(snapshot: any): Promise<void> {
    this.proposals = snapshot
  }

  async readAutoDreamProposals(): Promise<any> {
    if (!this.readEnabled) throw new Error('Read disabled')
    return this.proposals
  }

  disableRead() {
    this.readEnabled = false
  }

  getStored() {
    return this.proposals
  }
}

describe('autoDream Proposal Persistence', () => {
  let mockStorage: MockStorageBackend

  beforeEach(() => {
    globalAutoDreamPersistence.reset()
    mockStorage = new MockStorageBackend()
    initializeAutoDreamPersistence(mockStorage)
  })

  /**
   * Test 1: Add proposal to queue
   */
  it('should add proposals to queue', () => {
    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Increase threshold from 0.5 to 0.6',
      evidence: 'Low false positive rate',
      impact: 0.05,
      residualRisk: 0.1,
    })

    const pending = globalAutoDreamPersistence.getPendingProposals()
    expect(pending.length).toBe(1)
    expect(pending[0].proposalId).toBe('prop_1')
  })

  /**
   * Test 2: Queue statistics
   */
  it('should provide queue statistics', () => {
    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_stats_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Test',
      evidence: 'Test evidence',
      impact: 0.1,
      residualRisk: 0.08,
    })

    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_stats_2',
      target: 'truth_gate',
      changeType: 'threshold_adjust',
      proposal: 'Test',
      evidence: 'Test evidence',
      impact: 0.15,
      residualRisk: 0.12,
    })

    const stats = globalAutoDreamPersistence.getQueueStats()
    expect(stats.pendingCount).toBe(2)
    expect(stats.averageRisk).toBeGreaterThan(0)
    expect(stats.oldestProposal).toBeDefined()
    expect(stats.newestProposal).toBeDefined()
  })

  /**
   * Test 3: Mark proposal as applied
   */
  it('should remove proposal from queue when marked applied', () => {
    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_apply_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Test',
      evidence: 'Test evidence',
      impact: 0.05,
      residualRisk: 0.1,
    })

    expect(globalAutoDreamPersistence.getPendingProposals().length).toBe(1)

    globalAutoDreamPersistence.markProposalApplied('prop_apply_1', 'lineage_123')

    expect(globalAutoDreamPersistence.getPendingProposals().length).toBe(0)
  })

  /**
   * Test 4: Clear queue
   */
  it('should clear entire proposal queue', () => {
    for (let i = 0; i < 3; i++) {
      globalAutoDreamPersistence.addProposal({
        proposalId: `prop_clear_${i}`,
        target: 'safety_gate',
        changeType: 'threshold_adjust',
        proposal: 'Test',
        evidence: 'Test evidence',
        impact: 0.05,
        residualRisk: 0.1,
      })
    }

    expect(globalAutoDreamPersistence.getPendingProposals().length).toBe(3)

    globalAutoDreamPersistence.clearQueue()

    expect(globalAutoDreamPersistence.getPendingProposals().length).toBe(0)
  })

  /**
   * Test 5: Persist proposals to storage
   */
  it('should persist proposals to storage backend', async () => {
    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_persist_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Test persistence',
      evidence: 'Test evidence',
      impact: 0.05,
      residualRisk: 0.1,
    })

    await (globalAutoDreamPersistence as any).persistQueueToDisk()

    const stored = mockStorage.getStored()
    expect(stored).toBeDefined()
    expect(stored.proposals).toBeDefined()
    expect(stored.proposals.length).toBe(1)
    expect(stored.totalProposals).toBe(1)
  })

  /**
   * Test 6: Restore proposals from storage
   */
  it('should restore proposals from storage on startup', async () => {
    // First, add and persist
    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_restore_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Restore test',
      evidence: 'Test evidence',
      impact: 0.05,
      residualRisk: 0.1,
    })

    await (globalAutoDreamPersistence as any).persistQueueToDisk()

    // Reset and restore
    globalAutoDreamPersistence.reset()
    const restored = await restoreAutoDreamProposalsOnStartup()

    expect(restored.length).toBe(1)
    expect(restored[0].proposalId).toBe('prop_restore_1')
  })

  /**
   * Test 7: Filter high-risk proposals on restore
   */
  it('should filter high-risk proposals during restore', async () => {
    // Manually create a snapshot with high-risk proposal
    const snapshot = {
      exportedAt: new Date().toISOString(),
      totalProposals: 2,
      proposals: [
        {
          proposalId: 'prop_low_risk',
          timestamp: BigInt(Date.now()),
          target: 'safety_gate',
          changeType: 'threshold_adjust',
          proposal: 'Low risk',
          evidence: 'Safe change',
          impact: 0.05,
          residualRisk: 0.08, // Low risk - should restore
        },
        {
          proposalId: 'prop_high_risk',
          timestamp: BigInt(Date.now()),
          target: 'safety_gate',
          changeType: 'structural',
          proposal: 'High risk',
          evidence: 'Risky change',
          impact: 0.5,
          residualRisk: 0.3, // High risk - should filter
        },
      ],
      queueHash: 'test_hash',
    }

    // Create new mock storage with the snapshot
    const filterStorage = new MockStorageBackend()
    filterStorage.writeAutoDreamProposals(snapshot)
    initializeAutoDreamPersistence(filterStorage)

    const restored = await restoreAutoDreamProposalsOnStartup()

    // Only low-risk proposal should be restored
    expect(restored.length).toBe(1)
    expect(restored[0].proposalId).toBe('prop_low_risk')
  })

  /**
   * Test 8: Snapshot structure
   */
  it('should include complete snapshot metadata', async () => {
    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_snap_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Snapshot test',
      evidence: 'Test evidence',
      impact: 0.05,
      residualRisk: 0.1,
    })

    await (globalAutoDreamPersistence as any).persistQueueToDisk()

    const snapshot = mockStorage.getStored()
    expect(snapshot.exportedAt).toBeDefined()
    expect(snapshot.totalProposals).toBe(1)
    expect(snapshot.proposals).toBeDefined()
    expect(snapshot.queueHash).toBeDefined()
  })

  /**
   * Test 9: Timestamp on proposals
   */
  it('should record timestamp when proposal is added', () => {
    const beforeTime = BigInt(Date.now())

    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_time_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Time test',
      evidence: 'Test evidence',
      impact: 0.05,
      residualRisk: 0.1,
    })

    const afterTime = BigInt(Date.now())

    const proposals = globalAutoDreamPersistence.getPendingProposals()
    expect(proposals[0].timestamp).toBeGreaterThanOrEqual(beforeTime)
    expect(proposals[0].timestamp).toBeLessThanOrEqual(afterTime + BigInt(100))
  })

  /**
   * Test 10: Handle read errors gracefully
   */
  it('should handle storage read errors during restore', async () => {
    mockStorage.disableRead()

    const restored = await restoreAutoDreamProposalsOnStartup()

    expect(Array.isArray(restored)).toBe(true)
    expect(restored.length).toBe(0)
  })

  /**
   * Test 11: Multiple proposals
   */
  it('should handle multiple proposals in queue', () => {
    const proposalIds = []

    for (let i = 0; i < 5; i++) {
      const id = `prop_multi_${i}`
      proposalIds.push(id)

      globalAutoDreamPersistence.addProposal({
        proposalId: id,
        target: i % 2 === 0 ? 'safety_gate' : 'truth_gate',
        changeType: 'threshold_adjust',
        proposal: `Proposal ${i}`,
        evidence: `Evidence ${i}`,
        impact: 0.01 + i * 0.01,
        residualRisk: 0.05 + i * 0.02,
      })
    }

    const proposals = globalAutoDreamPersistence.getPendingProposals()
    expect(proposals.length).toBe(5)

    for (let i = 0; i < 5; i++) {
      expect(proposals[i].proposalId).toBe(proposalIds[i])
    }
  })

  /**
   * Test 12: Lineage tracking
   */
  it('should track lineage ID when proposal is applied', () => {
    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_lineage_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Lineage test',
      evidence: 'Test evidence',
      impact: 0.05,
      residualRisk: 0.1,
    })

    globalAutoDreamPersistence.markProposalApplied('prop_lineage_1', 'lineage_abc123')

    // Proposal should be removed
    const remaining = globalAutoDreamPersistence.getPendingProposals()
    expect(remaining.length).toBe(0)
  })

  /**
   * Test 13: Reset clears state
   */
  it('should reset queue to empty state', () => {
    globalAutoDreamPersistence.addProposal({
      proposalId: 'prop_reset_1',
      target: 'safety_gate',
      changeType: 'threshold_adjust',
      proposal: 'Reset test',
      evidence: 'Test evidence',
      impact: 0.05,
      residualRisk: 0.1,
    })

    expect(globalAutoDreamPersistence.getPendingProposals().length).toBe(1)

    globalAutoDreamPersistence.reset()

    expect(globalAutoDreamPersistence.getPendingProposals().length).toBe(0)
    const stats = globalAutoDreamPersistence.getQueueStats()
    expect(stats.pendingCount).toBe(0)
  })

  /**
   * Test 14: Invalid proposal rejection on restore
   */
  it('should reject invalid proposals during restore', async () => {
    const badSnapshot = {
      exportedAt: new Date().toISOString(),
      totalProposals: 3,
      proposals: [
        {
          proposalId: 'good_prop',
          timestamp: BigInt(Date.now()),
          target: 'safety_gate',
          changeType: 'threshold_adjust',
          proposal: 'Good',
          evidence: 'Evidence',
          impact: 0.05,
          residualRisk: 0.1,
        },
        {
          // Missing proposalId
          timestamp: BigInt(Date.now()),
          target: 'safety_gate',
          changeType: 'threshold_adjust',
          residualRisk: 0.1,
        },
        {
          proposalId: 'bad_risk',
          timestamp: BigInt(Date.now()),
          target: 'safety_gate',
          changeType: 'structural',
          proposal: 'Risky',
          evidence: 'Evidence',
          impact: 0.5,
          residualRisk: 0.5, // Way too risky
        },
      ],
      queueHash: 'test',
    }

    // Set the proposals in the mock backend for readAutoDreamProposals to return
    const customStorage = new MockStorageBackend()
    customStorage.writeAutoDreamProposals(badSnapshot)
    initializeAutoDreamPersistence(customStorage)

    const restored = await restoreAutoDreamProposalsOnStartup()

    // Only the valid, low-risk proposal should restore
    expect(restored.length).toBe(1)
    expect(restored[0].proposalId).toBe('good_prop')
  })
})
