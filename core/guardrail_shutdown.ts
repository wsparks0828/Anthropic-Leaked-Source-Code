/**
 * Graceful Shutdown Handler
 *
 * Ensures orderly termination with complete lineage persistence:
 * - Flushes lineage chain to persistent storage
 * - Records shutdown event in immutable audit trail
 * - Prevents data loss on unexpected termination
 * - Validates storage write before acknowledging shutdown
 */

import {globalLineageAuditor} from './lineage_auditor.js'
import {globalAlertManager} from './guardrail_alerts.js'
import {globalHealthMonitor} from './guardrail_health.js'

/**
 * Shutdown manager singleton
 */
class GracefulShutdownManager {
  private isShuttingDown = false
  private shutdownTimeout = 30000 // 30 seconds
  private storageBackend: any = null // Will be injected

  /**
   * Initialize shutdown handler
   */
  initialize(storageBackend: any): void {
    this.storageBackend = storageBackend

    // Register process termination handlers
    process.on('SIGTERM', () => this.handleShutdown('SIGTERM'))
    process.on('SIGINT', () => this.handleShutdown('SIGINT'))
    process.on('uncaughtException', (err) => this.handleCrash('uncaughtException', err))
    process.on('unhandledRejection', (reason) => this.handleCrash('unhandledRejection', reason))
  }

  /**
   * Handle graceful termination signal
   */
  private async handleShutdown(signal: string): Promise<void> {
    if (this.isShuttingDown) {
      console.warn('[shutdown] Already shutting down, ignoring signal')
      return
    }

    this.isShuttingDown = true
    console.log(`[shutdown] Received ${signal}, initiating graceful shutdown`)

    try {
      // Record shutdown event in lineage BEFORE flushing
      const shutdownEventId = `shutdown_${Date.now()}_${Math.random().toString(36).slice(2)}`
      const startTime = BigInt(Date.now() * 1_000_000) // nanoseconds

      // Get current health snapshot for context
      const healthStatus = globalHealthMonitor.getStatus()

      // Add shutdown record to lineage
      const lineageRecord = globalLineageAuditor.addRecord({
        verificationId: shutdownEventId,
        timestamp: startTime,
        decision: 'accept',
        rubricScore: 1.0, // Successful shutdown
        truthVerdict: 'shutdown_nominal',
        lineage: {
          who: 'graceful_shutdown_handler',
          what: {
            before: {status: healthStatus.status},
            after: {shutdownComplete: true},
          },
          when: startTime,
          auth: 'shutdown_signal',
        },
      })

      console.log(`[shutdown] Recorded shutdown event: ${shutdownEventId}`)

      // Flush lineage to persistent storage
      await this.flushLineageToDisk()

      // Flush health metrics
      await this.flushHealthMetricsToDisk(healthStatus)

      // Flush active alerts
      await this.flushAlertHistoryToDisk()

      console.log('[shutdown] All data flushed successfully')
      console.log('[shutdown] Graceful shutdown complete')

      process.exit(0)
    } catch (error) {
      console.error('[shutdown] Error during graceful shutdown:', error)
      this.emergencyShutdown(signal, error)
    }
  }

  /**
   * Handle crashes and unexpected errors
   */
  private async handleCrash(type: string, error: any): Promise<void> {
    if (this.isShuttingDown) {
      console.error(`[shutdown] ${type} during shutdown:`, error)
      process.exit(1)
    }

    this.isShuttingDown = true
    console.error(`[shutdown] Crash detected (${type}):`, error)

    try {
      // Record crash event in lineage for forensics
      const crashEventId = `crash_${Date.now()}_${Math.random().toString(36).slice(2)}`
      const startTime = BigInt(Date.now() * 1_000_000)

      globalLineageAuditor.addRecord({
        verificationId: crashEventId,
        timestamp: startTime,
        decision: 'quarantine',
        rubricScore: 0.0,
        truthVerdict: 'crash_detected',
        lineage: {
          who: 'graceful_shutdown_handler',
          what: {
            before: {},
            after: {crashType: type, errorMessage: String(error)},
          },
          when: startTime,
          auth: 'crash_detection',
        },
      })

      // Emergency flush with shorter timeout
      await Promise.race([
        this.flushLineageToDisk(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Flush timeout')), 5000)),
      ])

      console.log('[shutdown] Crash event recorded and lineage flushed')
    } catch (flushError) {
      console.error('[shutdown] Failed to flush crash event:', flushError)
    }

    process.exit(1)
  }

  /**
   * Flush lineage chain to persistent storage
   */
  private async flushLineageToDisk(): Promise<void> {
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Lineage flush timeout')), this.shutdownTimeout),
    )

    const flushPromise = (async () => {
      if (!this.storageBackend) {
        console.warn('[shutdown] No storage backend configured, skipping lineage flush')
        return
      }

      try {
        // Export full lineage chain
        const lineageRecords = globalLineageAuditor.exportLineage(10000, 'json-ld')
        const stats = globalLineageAuditor.getChainStats()

        // Prepare storage payload
        const payload = {
          exportedAt: new Date().toISOString(),
          totalRecords: stats.totalRecords,
          chainHash: stats.currentChainHash,
          records: lineageRecords,
        }

        // Write to storage atomically
        await this.storageBackend.writeLineage(payload)

        console.log(`[shutdown] Lineage flushed: ${stats.totalRecords} records, hash=${stats.currentChainHash}`)
      } catch (error) {
        console.error('[shutdown] Lineage flush failed:', error)
        throw error
      }
    })()

    return Promise.race([flushPromise, timeoutPromise])
  }

  /**
   * Flush health metrics to persistent storage
   */
  private async flushHealthMetricsToDisk(healthStatus: any): Promise<void> {
    if (!this.storageBackend) {
      return
    }

    try {
      const payload = {
        exportedAt: new Date().toISOString(),
        status: healthStatus.status,
        metrics: healthStatus.metrics,
      }

      await this.storageBackend.writeHealthMetrics(payload)
      console.log('[shutdown] Health metrics flushed')
    } catch (error) {
      console.error('[shutdown] Health metrics flush failed:', error)
      // Non-fatal, continue shutdown
    }
  }

  /**
   * Flush alert history to persistent storage
   */
  private async flushAlertHistoryToDisk(): Promise<void> {
    if (!this.storageBackend) {
      return
    }

    try {
      // Get alert history from alert manager
      // This assumes the alert manager has a method to export history
      const alertData = {
        exportedAt: new Date().toISOString(),
        alerts: globalAlertManager.exportAlertHistory(),
      }

      await this.storageBackend.writeAlertHistory(alertData)
      console.log('[shutdown] Alert history flushed')
    } catch (error) {
      console.error('[shutdown] Alert history flush failed:', error)
      // Non-fatal, continue shutdown
    }
  }

  /**
   * Emergency shutdown fallback (used when graceful shutdown fails)
   */
  private emergencyShutdown(signal: string, error: any): void {
    console.error('[shutdown] EMERGENCY SHUTDOWN - graceful exit failed')
    console.error('[shutdown] Signal:', signal)
    console.error('[shutdown] Error:', error)
    console.error('[shutdown] Data may be lost - manual recovery may be required')

    // Attempt minimal crash dump before exit
    try {
      const crashDump = {
        signal,
        error: String(error),
        timestamp: new Date().toISOString(),
        chainStats: globalLineageAuditor.getChainStats(),
      }

      if (this.storageBackend) {
        this.storageBackend.writeCrashDump(crashDump).catch(() => {
          /* ignored */
        })
      }
    } catch {
      /* ignored */
    }

    process.exit(1)
  }

  /**
   * Check if shutdown is in progress
   */
  isActive(): boolean {
    return this.isShuttingDown
  }
}

/**
 * Global shutdown manager singleton
 */
export const globalShutdownManager = new GracefulShutdownManager()

/**
 * Initialize graceful shutdown (call at application startup)
 */
export function initializeGracefulShutdown(storageBackend: any): void {
  globalShutdownManager.initialize(storageBackend)
  console.log('[shutdown] Graceful shutdown handler initialized')
}
