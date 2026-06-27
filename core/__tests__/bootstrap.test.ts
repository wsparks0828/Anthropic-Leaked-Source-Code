/**
 * Corpus Bootstrap — wiring tests.
 *
 * Verifies the single coordinated bootstrap actually initializes the subsystems
 * (closing the gap where encryption was only ever lazily initialized).
 */

import {describe, it, expect, beforeEach} from 'bun:test'
import {bootstrapGuardrailCorpus, isCorpusBootstrapped, resetBootstrap} from '../thoth/bootstrap.js'
import {
  encryptSensitiveField,
  decryptSensitiveField,
  isAuthorizedForLineageExport,
  resetEncryption,
} from '../lineage_encryption.js'

beforeEach(() => {
  resetBootstrap()
  resetEncryption()
})

describe('Corpus Bootstrap wiring', () => {
  it('initializes encryption so sensitive-field protection is usable end-to-end', () => {
    const token = 'boot-token-1'
    const r = bootstrapGuardrailCorpus({encryptionKey: 'real-bootstrap-key', lineageExportToken: token})

    expect(r.encryptionInitialized).toBe(true)
    expect(r.secretRotationInitialized).toBe(true)
    expect(r.exportTokenRegistered).toBe(true)
    expect(r.usedDefaultKey).toBe(false)

    // The registered token must now authorize, and encrypt→decrypt must round-trip.
    expect(isAuthorizedForLineageExport(token)).toBe(true)
    const ct = encryptSensitiveField('sensitive-proposal-rationale')
    expect(decryptSensitiveField(ct, token)).toBe('sensitive-proposal-rationale')
  })

  it('flags use of the unsafe default key when no key is supplied', () => {
    const r = bootstrapGuardrailCorpus({})
    expect(r.usedDefaultKey).toBe(true)
    expect(r.encryptionInitialized).toBe(true)
  })

  it('skips storage-backed stages when no backend is provided', () => {
    const r = bootstrapGuardrailCorpus({encryptionKey: 'k'})
    expect(r.storageReadinessInitialized).toBe(false)
    expect(r.shutdownRegistered).toBe(false)
  })

  it('wires storage readiness without signal handlers when asked', () => {
    const mockBackend = {writeLineage: async () => {}, exportState: async () => {}}
    const r = bootstrapGuardrailCorpus({
      encryptionKey: 'k',
      storageBackend: mockBackend,
      registerSignalHandlers: false,
    })
    expect(r.storageReadinessInitialized).toBe(true)
    expect(r.shutdownRegistered).toBe(false) // signals intentionally not registered
  })

  it('is idempotent and reports bootstrapped state', () => {
    expect(isCorpusBootstrapped()).toBe(false)
    bootstrapGuardrailCorpus({encryptionKey: 'k'})
    expect(isCorpusBootstrapped()).toBe(true)
    // second call must not throw
    expect(() => bootstrapGuardrailCorpus({encryptionKey: 'k'})).not.toThrow()
    expect(isCorpusBootstrapped()).toBe(true)
  })
})
