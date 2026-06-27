/**
 * Guardrail/THOTH Corpus Bootstrap.
 *
 * Single coordinated entry point that wires the subsystems together in the correct
 * order. Previously each subsystem had its own initializeX() but nothing called them
 * in concert, so e.g. lineage encryption was only ever initialized lazily with an
 * unsafe default key. This bootstrap closes that wiring gap.
 *
 * Idempotent: safe to call multiple times.
 *
 * Order:
 *   1. Lineage encryption (so sensitive-field protection is active before any record)
 *   2. Secret rotation handler
 *   3. Storage readiness validation       (requires a storage backend)
 *   4. Graceful shutdown signal handlers   (requires a storage backend)
 *   5. Optional: register an external lineage-export authorization token
 */

import {initializeLineageEncryption, registerLineageExportToken} from '../lineage_encryption.js'
import {initializeSecretRotation} from '../secret_rotation_handler.js'
import {initializeStorageReadiness} from '../guardrail_storage_ready.js'
import {initializeGracefulShutdown} from '../guardrail_shutdown.js'

export interface BootstrapOptions {
  /** Encryption key/secret. If omitted, falls back to LINEAGE_ENCRYPTION_KEY env or an unsafe default. */
  encryptionKey?: Buffer | string
  /** Storage backend for shutdown flush + readiness validation. If omitted those stages are skipped. */
  storageBackend?: unknown
  /** Wire process SIGTERM/SIGINT handlers (default true when a backend is given). */
  registerSignalHandlers?: boolean
  /** Optionally register an external lineage-export authorization token at boot. */
  lineageExportToken?: string
}

export interface BootstrapResult {
  encryptionInitialized: boolean
  secretRotationInitialized: boolean
  storageReadinessInitialized: boolean
  shutdownRegistered: boolean
  exportTokenRegistered: boolean
  usedDefaultKey: boolean
}

let booted = false

/**
 * Wire the corpus. Idempotent.
 */
export function bootstrapGuardrailCorpus(opts: BootstrapOptions = {}): BootstrapResult {
  const result: BootstrapResult = {
    encryptionInitialized: false,
    secretRotationInitialized: false,
    storageReadinessInitialized: false,
    shutdownRegistered: false,
    exportTokenRegistered: false,
    usedDefaultKey: false,
  }

  // 1. Encryption — always initialized so sensitive-field protection is active.
  if (!opts.encryptionKey) {
    result.usedDefaultKey = true
    console.warn(
      '[bootstrap] No encryptionKey supplied — lineage encryption will use the env/default key. ' +
        'Supply a real key in production.',
    )
  }
  initializeLineageEncryption(opts.encryptionKey)
  result.encryptionInitialized = true

  // 2. Secret rotation.
  initializeSecretRotation()
  result.secretRotationInitialized = true

  // 3 + 4. Storage-backed stages (only if a backend is provided).
  if (opts.storageBackend !== undefined) {
    initializeStorageReadiness(opts.storageBackend)
    result.storageReadinessInitialized = true

    if (opts.registerSignalHandlers !== false) {
      initializeGracefulShutdown(opts.storageBackend)
      result.shutdownRegistered = true
    }
  }

  // 5. Optional export token.
  if (opts.lineageExportToken) {
    registerLineageExportToken(opts.lineageExportToken)
    result.exportTokenRegistered = true
  }

  booted = true
  return result
}

export function isCorpusBootstrapped(): boolean {
  return booted
}

/** Reset bootstrap flag (testing only). */
export function resetBootstrap(): void {
  booted = false
}
