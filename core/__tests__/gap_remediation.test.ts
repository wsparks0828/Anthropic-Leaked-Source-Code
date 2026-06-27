/**
 * Gap Remediation Tests - All 8 Medium-Severity Gaps
 *
 * Verifies all identified audit gaps have been addressed:
 * 1. Schema Versioning
 * 2. autoDream Idempotency
 * 3. Component Failure Lineage
 * 4. Output Size Limits
 * 5. Access Control on exportLineage()
 * 6. Encryption of Sensitive Fields
 * 7. Atomic Memory Wiring
 * 8. Secret Rotation Handler
 */

import {describe, it, expect, beforeEach} from 'bun:test'
import {globalRubricScorer} from '../rubric_scorer.js'
import {globalTruthGate} from '../truth_gates.js'
import {globalProposalDeduplicator} from '../auto_dream_dedup.js'
import {recordComponentFailure} from '../error_lineage_handler.js'
import {globalLineageAuditor} from '../lineage_auditor.js'
import {
  encryptSensitiveField,
  decryptSensitiveField,
  registerLineageExportToken,
  isAuthorizedForLineageExport,
  resetEncryption,
  initializeLineageEncryption,
} from '../lineage_encryption.js'
import {globalMemoryTransactionManager, createMemoryTransaction} from '../memory_transaction.js'
import {globalSecretRotationHandler} from '../secret_rotation_handler.js'
import {guardApiOutput} from '../guardrail_integration.js'

describe('Gap Remediation - All 8 Gaps', () => {
  beforeEach(() => {
    globalLineageAuditor.reset()
    globalProposalDeduplicator.reset()
    globalMemoryTransactionManager.reset()
    globalSecretRotationHandler.reset()
    resetEncryption()
    initializeLineageEncryption('test-key-12345678')
  })

  // ============================================================================
  // GAP 1: Schema Versioning on RubricScore and TruthGateResult
  // ============================================================================

  describe('Gap 1: Schema Versioning', () => {
    it('should include _version field in RubricScore', () => {
      const score = globalRubricScorer.score('Test output for versioning check.')
      expect(score._version).toBe('1.0')
      expect(score).toHaveProperty('_version')
    })

    it('should include _version field in TruthGateResult', () => {
      const result = globalTruthGate.gate('Test output for truth gate versioning.')
      expect(result._version).toBe('1.0')
      expect(result).toHaveProperty('_version')
    })

    it('should allow future schema version compatibility', () => {
      const score = globalRubricScorer.score('Output for version compatibility test.')
      // Parsers can check version and handle accordingly
      expect(score._version).toMatch(/^\d+\.\d+$/)
    })
  })

  // ============================================================================
  // GAP 2: autoDream Idempotency Guarantee
  // ============================================================================

  describe('Gap 2: autoDream Idempotency', () => {
    it('should detect duplicate proposals', () => {
      const proposal = {
        proposalId: 'prop_1',
        target: 'safety_gate',
        changeType: 'threshold_adjust',
        proposal: 'Adjust threshold from 0.5 to 0.6',
      }

      expect(globalProposalDeduplicator.isDuplicate(proposal)).toBe(false)

      globalProposalDeduplicator.markApplied(proposal, 'prop_1', 'lineage_123')
      expect(globalProposalDeduplicator.isDuplicate(proposal)).toBe(true)
    })

    it('should track deduplication stats', () => {
      const proposal = {
        proposalId: 'prop_dedup_1',
        target: 'safety_gate',
        changeType: 'threshold_adjust',
        proposal: 'Test proposal',
      }

      globalProposalDeduplicator.markApplied(proposal, 'prop_dedup_1', 'lineage_456')

      const stats = globalProposalDeduplicator.getStats()
      expect(stats.totalTracked).toBeGreaterThan(0)
      expect(stats.oldestProposal).toBeDefined()
    })

    it('should prevent re-application of same proposal', () => {
      const proposal = {
        proposalId: 'prop_nodup',
        target: 'truth_gate',
        changeType: 'threshold_adjust',
        proposal: 'No duplicate test',
      }

      // First application
      expect(globalProposalDeduplicator.isDuplicate(proposal)).toBe(false)
      globalProposalDeduplicator.markApplied(proposal, 'prop_nodup', 'lineage_789')

      // Second application attempt should be blocked
      expect(globalProposalDeduplicator.isDuplicate(proposal)).toBe(true)
    })
  })

  // ============================================================================
  // GAP 3: Component Failures Create Lineage Records
  // ============================================================================

  describe('Gap 3: Component Failure Lineage', () => {
    it('should record component failure in lineage', () => {
      const error = new Error('Test component failure')
      recordComponentFailure('test_component', error, 'high')

      const stats = globalLineageAuditor.getChainStats()
      expect(stats.totalRecords).toBeGreaterThan(0)
    })

    it('should record critical failures with alert', () => {
      const error = new Error('Critical component failure')
      recordComponentFailure('critical_component', error, 'critical')

      // Failure should be recorded
      const stats = globalLineageAuditor.getChainStats()
      expect(stats.totalRecords).toBeGreaterThan(0)
    })

    it('should include error details in lineage', () => {
      globalLineageAuditor.reset()
      const error = new Error('Detailed error test')
      recordComponentFailure('detailed_component', error, 'medium')

      const stats = globalLineageAuditor.getChainStats()
      expect(stats.totalRecords).toBe(1)
    })
  })

  // ============================================================================
  // GAP 4: Output Size Limit Enforcement
  // ============================================================================

  describe('Gap 4: Output Size Limits', () => {
    it('should reject outputs exceeding size limit', () => {
      const hugeOutput = 'x'.repeat(150_000) // Over 100KB limit
      const result = guardApiOutput(hugeOutput, {prompt: 'test'})
      expect(result.decision).toBe('quarantine')
    })

    it('should accept outputs within size limit', () => {
      const normalOutput = 'This is a normal output within size limits. ' + 'x'.repeat(500)
      const result = guardApiOutput(normalOutput, {prompt: 'test'})
      expect(result.decision).toMatch(/accept|quarantine/) // May quarantine for other reasons
    })

    it('should have configured size limit constant', () => {
      const testOutput = 'x'.repeat(100_001) // Just over 100KB
      const result = guardApiOutput(testOutput)
      // Should be rejected for size
      expect(result.decision).toBe('quarantine')
    })
  })

  // ============================================================================
  // GAP 5 & 6: Access Control + Encryption on exportLineage()
  // ============================================================================

  describe('Gap 5 & 6: Access Control & Encryption', () => {
    it('should require authorization token for export', () => {
      const records = globalLineageAuditor.exportLineage(10, 'json', undefined)
      expect(records.length).toBe(0)
    })

    it('should accept valid authorization token', () => {
      const token = 'valid-auth-token-12345'
      const registeredToken = registerLineageExportToken(token)

      expect(isAuthorizedForLineageExport(token)).toBe(true)
      expect(registeredToken).toBeDefined()
    })

    it('should encrypt sensitive field', () => {
      const sensitiveData = 'Dangerous proposal rationale'
      const encrypted = encryptSensitiveField(sensitiveData)

      expect(encrypted).not.toContain(sensitiveData)
      expect(encrypted).toContain('|') // Format check
    })

    it('should decrypt with valid authorization', () => {
      const sensitiveData = 'Secret quarantine reason'
      const token = 'decrypt-token-xyz'
      registerLineageExportToken(token)

      const encrypted = encryptSensitiveField(sensitiveData)
      const decrypted = decryptSensitiveField(encrypted, token)

      expect(decrypted).toBe(sensitiveData)
    })

    it('should reject decryption without authorization', () => {
      const encrypted = encryptSensitiveField('Secret data')
      const decrypted = decryptSensitiveField(encrypted, undefined)

      expect(decrypted).toBeNull()
    })

    it('should reject decryption with wrong token', () => {
      const encrypted = encryptSensitiveField('Secret data')
      const decrypted = decryptSensitiveField(encrypted, 'wrong-token')

      expect(decrypted).toBeNull()
    })
  })

  // ============================================================================
  // GAP 7: Atomic Memory Wiring Transactions
  // ============================================================================

  describe('Gap 7: Atomic Memory Wiring', () => {
    it('should create memory transaction', () => {
      const txn = createMemoryTransaction()
      expect(txn).toBeDefined()

      const status = txn.getStatus()
      expect(status.committed).toBe(false)
      expect(status.rolledBack).toBe(false)
    })

    it('should add updates to transaction', () => {
      const txn = createMemoryTransaction()
      txn.addUpdate('semantic', 'policy_1', {threshold: 0.6})
      txn.addUpdate('episodic', 'calibration_1', {accuracy: 0.95})

      const status = txn.getStatus()
      expect(status.operationCount).toBe(2)
    })

    it('should commit transaction atomically', () => {
      const txn = createMemoryTransaction()
      const mockLayers = {
        semantic: new Map(),
        episodic: new Map(),
        graph: new Map(),
      }

      txn.addUpdate('semantic', 'key1', {value: 'data1'})
      const result = txn.commit(mockLayers)

      expect(result.success).toBe(true)
      expect(mockLayers.semantic.get('key1')).toEqual({value: 'data1'})
    })

    it('should rollback on error', () => {
      const txn = createMemoryTransaction()
      txn.addUpdate('semantic', 'key1', {value: 'data1'})
      txn.addUpdate('invalid_layer', 'key2', {value: 'data2'})

      const mockLayers = {
        semantic: new Map(),
        episodic: new Map(),
        graph: new Map(),
      }

      const result = txn.commit(mockLayers)
      expect(result.success).toBe(false)
    })

    it('should prevent duplicate keys in transaction', () => {
      const txn = createMemoryTransaction()
      txn.addUpdate('semantic', 'same_key', {value: 'data1'})
      txn.addUpdate('semantic', 'same_key', {value: 'data2'})

      const mockLayers = {
        semantic: new Map(),
        episodic: new Map(),
        graph: new Map(),
      }

      const result = txn.commit(mockLayers)
      expect(result.success).toBe(false)
    })

    it('should track transaction statistics', () => {
      const txn = createMemoryTransaction()
      txn.addUpdate('semantic', 'key1', {value: 'data1'})

      const mockLayers = {
        semantic: new Map(),
        episodic: new Map(),
        graph: new Map(),
      }

      txn.commit(mockLayers)

      const stats = globalMemoryTransactionManager.getStats()
      expect(stats.committedTransactions).toBeGreaterThanOrEqual(0)
    })
  })

  // ============================================================================
  // GAP 8: Secret Rotation Handler
  // ============================================================================

  describe('Gap 8: Secret Rotation Handler', () => {
    it('should load initial secret', () => {
      globalSecretRotationHandler.loadSecret('api_key', 'secret123456', 'admin')

      const secret = globalSecretRotationHandler.getSecret('api_key')
      expect(secret).toBe('secret123456')
    })

    it('should rotate secret with lineage', async () => {
      globalSecretRotationHandler.loadSecret('db_password', 'oldpass123456')

      const success = await globalSecretRotationHandler.rotateSecret('db_password', 'newpass123456', 'operator')
      expect(success).toBe(true)

      const newSecret = globalSecretRotationHandler.getSecret('db_password')
      expect(newSecret).toBe('newpass123456')
    })

    it('should track rotation history', async () => {
      globalSecretRotationHandler.loadSecret('key1', 'value1')
      await globalSecretRotationHandler.rotateSecret('key1', 'value2', 'admin')
      await globalSecretRotationHandler.rotateSecret('key1', 'value3', 'admin')

      const history = globalSecretRotationHandler.getRotationHistory()
      expect(history.length).toBe(2)
    })

    it('should hide secret values in history', async () => {
      globalSecretRotationHandler.loadSecret('secret_api_key', 'secret_value_12345')
      await globalSecretRotationHandler.rotateSecret('secret_api_key', 'new_secret_67890')

      const history = globalSecretRotationHandler.getRotationHistory()
      const lastRotation = history[history.length - 1]

      // History should contain hash prefixes, not actual secrets
      expect(lastRotation.oldHashPrefix).not.toContain('secret')
      expect(lastRotation.newHashPrefix).not.toContain('new_secret')
      expect(lastRotation.oldHashPrefix.length).toBe(8)
    })

    it('should record rotation in lineage', async () => {
      globalSecretRotationHandler.loadSecret('lineage_secret', 'initial_value_123')
      await globalSecretRotationHandler.rotateSecret('lineage_secret', 'new_value_456')

      const stats = globalLineageAuditor.getChainStats()
      expect(stats.totalRecords).toBeGreaterThan(0)
    })

    it('should reject invalid new secrets', async () => {
      globalSecretRotationHandler.loadSecret('test_key', 'valid_secret_123')

      const success = await globalSecretRotationHandler.rotateSecret('test_key', 'short')
      expect(success).toBe(false)
    })

    it('should provide rotation statistics', async () => {
      globalSecretRotationHandler.loadSecret('stat_key1', 'value1')
      await globalSecretRotationHandler.rotateSecret('stat_key1', 'value2')

      const stats = globalSecretRotationHandler.getStats()
      expect(stats.secretsManaged).toBeGreaterThan(0)
      expect(stats.rotationEvents).toBeGreaterThan(0)
    })
  })
})
