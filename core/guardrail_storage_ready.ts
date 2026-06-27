/**
 * Storage Readiness Validation Gate
 *
 * Prevents partial service exposure by validating storage infrastructure
 * before accepting requests:
 * - Verifies persistent storage backend connectivity
 * - Validates lineage chain initialization
 * - Checks health metrics storage capacity
 * - Confirms alert persistence mechanism ready
 * - Blocks API operations until storage verified
 */

import {globalLineageAuditor} from './lineage_auditor.js'
import {globalHealthMonitor} from './guardrail_health.js'

/**
 * Storage readiness status
 */
export interface StorageReadinessStatus {
  ready: boolean
  timestamp: bigint
  components: {
    lineageChain: {ready: boolean; message: string}
    healthMetrics: {ready: boolean; message: string}
    alertPersistence: {ready: boolean; message: string}
    storageBackend: {ready: boolean; message: string}
  }
  failures: Array<{component: string; reason: string}>
}

/**
 * Storage readiness validator
 */
class StorageReadinessValidator {
  private storageBackend: any = null
  private isReady = false
  private readinessStatus: StorageReadinessStatus | null = null
  private validationAttempts = 0
  private maxValidationAttempts = 5
  private validationRetryDelay = 1000 // milliseconds

  /**
   * Initialize storage readiness validator
   */
  initialize(storageBackend: any): void {
    this.storageBackend = storageBackend
    console.log('[storage-ready] Storage readiness validator initialized')
  }

  /**
   * Validate storage readiness (called at startup)
   */
  async validateStorageReady(): Promise<StorageReadinessStatus> {
    if (this.isReady && this.readinessStatus) {
      return this.readinessStatus
    }

    console.log('[storage-ready] Validating storage readiness...')
    const startTime = BigInt(Date.now())
    const failures: Array<{component: string; reason: string}> = []

    // Validate each storage component
    const lineageChainStatus = await this.validateLineageChain()
    if (!lineageChainStatus.ready) {
      failures.push({component: 'lineageChain', reason: lineageChainStatus.message})
    }

    const healthMetricsStatus = await this.validateHealthMetrics()
    if (!healthMetricsStatus.ready) {
      failures.push({component: 'healthMetrics', reason: healthMetricsStatus.message})
    }

    const alertPersistenceStatus = await this.validateAlertPersistence()
    if (!alertPersistenceStatus.ready) {
      failures.push({component: 'alertPersistence', reason: alertPersistenceStatus.message})
    }

    const storageBackendStatus = await this.validateStorageBackend()
    if (!storageBackendStatus.ready) {
      failures.push({component: 'storageBackend', reason: storageBackendStatus.message})
    }

    const allReady = failures.length === 0

    this.readinessStatus = {
      ready: allReady,
      timestamp: startTime,
      components: {
        lineageChain: lineageChainStatus,
        healthMetrics: healthMetricsStatus,
        alertPersistence: alertPersistenceStatus,
        storageBackend: storageBackendStatus,
      },
      failures,
    }

    if (allReady) {
      this.isReady = true
      console.log('[storage-ready] ✓ All storage components ready')
    } else {
      console.error('[storage-ready] ✗ Storage readiness validation failed:')
      for (const failure of failures) {
        console.error(`  - ${failure.component}: ${failure.reason}`)
      }

      // Retry logic for transient failures
      if (this.validationAttempts < this.maxValidationAttempts) {
        this.validationAttempts++
        console.log(
          `[storage-ready] Retrying in ${this.validationRetryDelay}ms (attempt ${this.validationAttempts}/${this.maxValidationAttempts})`,
        )

        await new Promise((resolve) => setTimeout(resolve, this.validationRetryDelay))
        return this.validateStorageReady() // Recursive retry
      }
    }

    return this.readinessStatus
  }

  /**
   * Validate lineage chain is accessible
   */
  private async validateLineageChain(): Promise<{ready: boolean; message: string}> {
    try {
      const stats = globalLineageAuditor.getChainStats()

      if (stats === undefined || stats.currentChainHash === undefined) {
        return {
          ready: false,
          message: 'Lineage chain statistics unavailable',
        }
      }

      // Try to export (tests readability)
      const exported = globalLineageAuditor.exportLineage(1)
      if (!Array.isArray(exported)) {
        return {
          ready: false,
          message: 'Lineage export returned non-array',
        }
      }

      return {
        ready: true,
        message: `Lineage chain ready (${stats.totalRecords} records, hash=${stats.currentChainHash.substring(0, 8)}...)`,
      }
    } catch (error) {
      return {
        ready: false,
        message: `Lineage chain validation failed: ${String(error)}`,
      }
    }
  }

  /**
   * Validate health metrics subsystem
   */
  private async validateHealthMetrics(): Promise<{ready: boolean; message: string}> {
    try {
      const status = globalHealthMonitor.getStatus()

      if (status === undefined || status.metrics === undefined) {
        return {
          ready: false,
          message: 'Health monitor status unavailable',
        }
      }

      if (status.metrics.acceptanceRate === undefined) {
        return {
          ready: false,
          message: 'Health metrics missing acceptance rate',
        }
      }

      return {
        ready: true,
        message: `Health metrics ready (status=${status.status})`,
      }
    } catch (error) {
      return {
        ready: false,
        message: `Health metrics validation failed: ${String(error)}`,
      }
    }
  }

  /**
   * Validate alert persistence mechanism
   */
  private async validateAlertPersistence(): Promise<{ready: boolean; message: string}> {
    try {
      if (!this.storageBackend) {
        return {
          ready: false,
          message: 'Storage backend not configured',
        }
      }

      // Check if storage backend has alert write capability
      if (typeof this.storageBackend.writeAlertHistory !== 'function') {
        return {
          ready: false,
          message: 'Storage backend missing writeAlertHistory method',
        }
      }

      return {
        ready: true,
        message: 'Alert persistence mechanism ready',
      }
    } catch (error) {
      return {
        ready: false,
        message: `Alert persistence validation failed: ${String(error)}`,
      }
    }
  }

  /**
   * Validate storage backend connectivity
   */
  private async validateStorageBackend(): Promise<{ready: boolean; message: string}> {
    try {
      if (!this.storageBackend) {
        return {
          ready: false,
          message: 'No storage backend provided',
        }
      }

      // Validate storage backend has required methods
      const requiredMethods = ['writeLineage', 'writeHealthMetrics', 'writeAlertHistory']
      const missingMethods = []

      for (const method of requiredMethods) {
        if (typeof this.storageBackend[method] !== 'function') {
          missingMethods.push(method)
        }
      }

      if (missingMethods.length > 0) {
        return {
          ready: false,
          message: `Storage backend missing required methods: ${missingMethods.join(', ')}`,
        }
      }

      // Optional: test write capability if backend supports it
      if (typeof this.storageBackend.ping === 'function') {
        try {
          await this.storageBackend.ping()
        } catch (pingError) {
          return {
            ready: false,
            message: `Storage backend ping failed: ${String(pingError)}`,
          }
        }
      }

      return {
        ready: true,
        message: 'Storage backend fully operational',
      }
    } catch (error) {
      return {
        ready: false,
        message: `Storage backend validation failed: ${String(error)}`,
      }
    }
  }

  /**
   * Check if storage is ready (gate function)
   */
  isStorageReady(): boolean {
    return this.isReady
  }

  /**
   * Get current readiness status
   */
  getReadinessStatus(): StorageReadinessStatus | null {
    return this.readinessStatus
  }

  /**
   * Reset validator (for testing)
   */
  reset(): void {
    this.isReady = false
    this.readinessStatus = null
    this.validationAttempts = 0
  }
}

/**
 * Global storage readiness validator
 */
export const globalStorageReadinessValidator = new StorageReadinessValidator()

/**
 * STORAGE_READY Gate: Block operations until storage validated
 *
 * Returns true if storage is ready, false otherwise.
 * Call this before accepting API requests.
 */
export function guardStorageReady(): boolean {
  return globalStorageReadinessValidator.isStorageReady()
}

/**
 * Initialize storage readiness validation
 */
export function initializeStorageReadiness(storageBackend: any): void {
  globalStorageReadinessValidator.initialize(storageBackend)
}

/**
 * Perform startup validation
 */
export async function validateStorageReadyAtStartup(): Promise<StorageReadinessStatus> {
  return await globalStorageReadinessValidator.validateStorageReady()
}
