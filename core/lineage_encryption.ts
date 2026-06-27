/**
 * Lineage Encryption & Access Control
 *
 * Protects sensitive fields in lineage records (proposal rationale, quarantine reasons).
 * Provides:
 * - AES-256-GCM encryption for sensitive fields
 * - Authorization token validation for exportLineage()
 * - Decryption API for authorized users
 */

import {createCipheriv, createDecipheriv, randomBytes, createHash} from 'crypto'

/**
 * Encryption key provider (should come from secure key management)
 */
class EncryptionKeyProvider {
  private encryptionKey: Buffer | null = null
  private keyInitialized = false

  /**
   * Initialize encryption key from environment
   */
  initialize(keyOrSecret?: Buffer | string): void {
    if (keyOrSecret) {
      if (typeof keyOrSecret === 'string') {
        // Derive key from secret using SHA-256
        this.encryptionKey = createHash('sha256').update(keyOrSecret).digest()
      } else {
        this.encryptionKey = keyOrSecret
      }
    } else {
      // Fallback: use a default key (should only be used in development)
      const secret = process.env.LINEAGE_ENCRYPTION_KEY || 'default-unsafe-key'
      this.encryptionKey = createHash('sha256').update(secret).digest()
    }

    if (this.encryptionKey.length !== 32) {
      throw new Error('Encryption key must be 32 bytes')
    }

    this.keyInitialized = true
    console.log('[lineage-encryption] Encryption key initialized')
  }

  /**
   * Get encryption key
   */
  getKey(): Buffer {
    if (!this.keyInitialized) {
      this.initialize()
    }
    return this.encryptionKey!
  }
}

const keyProvider = new EncryptionKeyProvider()

/**
 * Authorization token validator
 */
class AuthorizationValidator {
  private validTokens: Set<string> = new Set()
  private tokenHashes: Map<string, {hash: string; createdAt: bigint}> = new Map()

  /**
   * Register authorized token
   */
  registerToken(token: string): string {
    const hash = createHash('sha256').update(token).digest('hex')
    this.tokenHashes.set(token, {hash, createdAt: BigInt(Date.now())})
    return hash
  }

  /**
   * Validate token for exportLineage access
   */
  isAuthorized(token?: string): boolean {
    if (!token) return false

    // Check if token is registered
    const registration = this.tokenHashes.get(token)
    if (!registration) return false

    // Check if token is not expired (30 days)
    const maxAge = BigInt(30 * 24 * 60 * 60 * 1000)
    if (BigInt(Date.now()) - registration.createdAt > maxAge) {
      this.tokenHashes.delete(token)
      return false
    }

    return true
  }

  /**
   * Reset authorization (for testing)
   */
  reset(): void {
    this.tokenHashes.clear()
  }
}

const authValidator = new AuthorizationValidator()

/**
 * Encrypt sensitive field
 */
export function encryptSensitiveField(plaintext: string): string {
  try {
    const key = keyProvider.getKey()
    const iv = randomBytes(16)
    const cipher = createCipheriv('aes-256-gcm', key, iv)

    let encrypted = cipher.update(plaintext, 'utf8', 'hex')
    encrypted += cipher.final('hex')
    const authTag = cipher.getAuthTag()

    // Format: iv|encryptedData|authTag (all hex)
    return `${iv.toString('hex')}|${encrypted}|${authTag.toString('hex')}`
  } catch (error) {
    console.error('[lineage-encryption] Encryption failed:', error)
    // Fallback: return plaintext with marker
    return `[ENCRYPTION_FAILED]${plaintext}`
  }
}

/**
 * Decrypt sensitive field (requires authorization)
 */
export function decryptSensitiveField(encrypted: string, authToken?: string): string | null {
  // Check authorization
  if (!authToken || !authValidator.isAuthorized(authToken)) {
    console.warn('[lineage-encryption] Unauthorized decryption attempt')
    return null
  }

  try {
    if (encrypted.startsWith('[ENCRYPTION_FAILED]')) {
      return encrypted.substring('[ENCRYPTION_FAILED]'.length)
    }

    const parts = encrypted.split('|')
    if (parts.length !== 3) {
      console.error('[lineage-encryption] Invalid encrypted format')
      return null
    }

    const key = keyProvider.getKey()
    const iv = Buffer.from(parts[0], 'hex')
    const encryptedData = parts[1]
    const authTag = Buffer.from(parts[2], 'hex')

    const decipher = createDecipheriv('aes-256-gcm', key, iv)
    decipher.setAuthTag(authTag)

    let decrypted = decipher.update(encryptedData, 'hex', 'utf8')
    decrypted += decipher.final('utf8')

    return decrypted
  } catch (error) {
    console.error('[lineage-encryption] Decryption failed:', error)
    return null
  }
}

/**
 * Mark fields as sensitive (should be encrypted)
 */
export const SENSITIVE_FIELDS = [
  'lineage.what.after.proposal', // Proposal details
  'lineage.what.after.rationale', // Reasoning (may expose decision logic)
  'quarantineDetails', // Why content was quarantined
  'lineage.what.after.reasonForQuarantine', // Detailed quarantine reason
]

/**
 * Check if field should be encrypted
 */
export function isSensitiveField(fieldPath: string): boolean {
  return SENSITIVE_FIELDS.some((pattern) => fieldPath.includes(pattern))
}

/**
 * Initialize encryption subsystem
 */
export function initializeLineageEncryption(keyOrSecret?: Buffer | string): void {
  keyProvider.initialize(keyOrSecret)
}

/**
 * Register authorization token for lineage export
 */
export function registerLineageExportToken(token: string): string {
  return authValidator.registerToken(token)
}

/**
 * Check if user is authorized to export lineage
 */
export function isAuthorizedForLineageExport(token?: string): boolean {
  return authValidator.isAuthorized(token)
}

/**
 * Reset encryption state (for testing)
 */
export function resetEncryption(): void {
  authValidator.reset()
}
