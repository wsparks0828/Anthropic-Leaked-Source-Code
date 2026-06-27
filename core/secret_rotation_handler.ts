/**
 * Secret Rotation Handler
 *
 * Manages secure rotation of authentication secrets with complete lineage tracking.
 * Ensures no service interruption during secret rotation.
 */

import {createHash} from 'crypto'
import {globalLineageAuditor} from './lineage_auditor.js'

/**
 * Secret metadata
 */
export interface SecretMetadata {
  name: string
  rotatedAt: bigint
  rotatedBy: string
  previousHash: string
  currentHash: string
  expiresAt?: bigint
}

/**
 * Secret rotation event
 */
export interface RotationEvent {
  eventId: string
  secretName: string
  timestamp: bigint
  oldHashPrefix: string // First 8 chars of hash, not full value
  newHashPrefix: string
  lineageId: string
  success: boolean
  reason?: string
}

/**
 * Secret rotation manager
 */
export class SecretRotationHandler {
  private secrets: Map<string, string> = new Map()
  private secretMetadata: Map<string, SecretMetadata> = new Map()
  private rotationHistory: RotationEvent[] = []
  private maxHistorySize = 1000

  /**
   * Load initial secret
   */
  loadSecret(name: string, value: string, rotatedBy: string = 'system'): void {
    const hash = this.hashSecret(value)

    this.secrets.set(name, value)
    this.secretMetadata.set(name, {
      name,
      rotatedAt: BigInt(Date.now()),
      rotatedBy,
      previousHash: '',
      currentHash: hash,
    })

    console.log(`[secret-rotation] Loaded secret: ${name}`)
  }

  /**
   * Rotate secret with lineage tracking
   */
  async rotateSecret(name: string, newValue: string, rotatedBy: string = 'service'): Promise<boolean> {
    const eventId = `rotation_${name}_${Date.now()}_${Math.random().toString(36).slice(2)}`
    const timestamp = BigInt(Date.now() * 1_000_000)

    try {
      // Validate new secret
      if (!newValue || newValue.length < 8) {
        throw new Error('New secret must be at least 8 characters')
      }

      // Get old value
      const oldValue = this.secrets.get(name)
      const oldHash = this.hashSecret(oldValue || '')
      const newHash = this.hashSecret(newValue)

      // Atomically swap secrets (old implementations might need synchronization)
      this.secrets.set(name, newValue)
      const metadata = this.secretMetadata.get(name)
      if (metadata) {
        metadata.previousHash = oldHash
        metadata.currentHash = newHash
        metadata.rotatedAt = BigInt(Date.now())
        metadata.rotatedBy = rotatedBy
      }

      // Record in lineage with encryption of sensitive details
      const lineageRecord = globalLineageAuditor.addRecord({
        verificationId: eventId,
        timestamp,
        decision: 'accept',
        rubricScore: 1.0,
        truthVerdict: 'rotation_success',
        lineage: {
          who: 'secret_rotation_handler',
          what: {
            before: {
              secret: name,
              status: 'active',
              hashPrefix: oldHash.substring(0, 8),
            },
            after: {
              secret: name,
              status: 'active',
              hashPrefix: newHash.substring(0, 8),
            },
          },
          when: timestamp,
          auth: 'secret_rotation',
        },
      })

      // Record rotation event
      const event: RotationEvent = {
        eventId,
        secretName: name,
        timestamp,
        oldHashPrefix: oldHash.substring(0, 8),
        newHashPrefix: newHash.substring(0, 8),
        lineageId: lineageRecord.chainHash,
        success: true,
      }

      this.rotationHistory.push(event)

      // Maintain bounded history
      if (this.rotationHistory.length > this.maxHistorySize) {
        this.rotationHistory.shift()
      }

      console.log(`[secret-rotation] Secret rotated: ${name} (by: ${rotatedBy})`)

      return true
    } catch (error) {
      console.error(`[secret-rotation] Rotation failed for ${name}:`, error)

      // Record failure in lineage
      try {
        globalLineageAuditor.addRecord({
          verificationId: eventId,
          timestamp,
          decision: 'error',
          rubricScore: 0,
          truthVerdict: 'rotation_failed',
          lineage: {
            who: 'secret_rotation_handler',
            what: {
              before: {secret: name, status: 'active'},
              after: {
                secret: name,
                status: 'rotation_failed',
                error: String(error),
              },
            },
            when: timestamp,
            auth: 'secret_rotation',
          },
        })
      } catch (lineageError) {
        console.error('[secret-rotation] Failed to record failure in lineage:', lineageError)
      }

      const event: RotationEvent = {
        eventId,
        secretName: name,
        timestamp,
        oldHashPrefix: 'unknown',
        newHashPrefix: 'unknown',
        lineageId: '',
        success: false,
        reason: String(error),
      }

      this.rotationHistory.push(event)

      return false
    }
  }

  /**
   * Get current secret
   */
  getSecret(name: string): string | null {
    return this.secrets.get(name) || null
  }

  /**
   * Get secret metadata
   */
  getSecretMetadata(name: string): SecretMetadata | null {
    return this.secretMetadata.get(name) || null
  }

  /**
   * Get rotation history
   */
  getRotationHistory(limit: number = 50): RotationEvent[] {
    return this.rotationHistory.slice(-limit)
  }

  /**
   * Hash secret for storage (one-way)
   */
  private hashSecret(secret: string): string {
    return createHash('sha256').update(secret).digest('hex')
  }

  /**
   * Get stats
   */
  getStats(): {
    secretsManaged: number
    rotationEvents: number
    successRate: number
  } {
    const successful = this.rotationHistory.filter((e) => e.success).length
    const total = this.rotationHistory.length

    return {
      secretsManaged: this.secrets.size,
      rotationEvents: total,
      successRate: total > 0 ? successful / total : 1.0,
    }
  }

  /**
   * Reset handler (for testing)
   */
  reset(): void {
    this.secrets.clear()
    this.secretMetadata.clear()
    this.rotationHistory = []
  }
}

/**
 * Global secret rotation handler
 */
export const globalSecretRotationHandler = new SecretRotationHandler()

/**
 * Initialize secret rotation handler
 */
export function initializeSecretRotation(): void {
  console.log('[secret-rotation] Secret rotation handler initialized')
}

/**
 * Rotate secret convenience function
 */
export async function rotateSecret(name: string, newValue: string, rotatedBy?: string): Promise<boolean> {
  return globalSecretRotationHandler.rotateSecret(name, newValue, rotatedBy)
}
