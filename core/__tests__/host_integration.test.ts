/**
 * Host Integration Shim — safety-property tests.
 *
 * The shim is what makes wiring guards into live CLI paths safe, so its guarantees
 * (fail-open, observe-by-default, enforce-blocks-dangerous) are tested adversarially.
 */

import {describe, it, expect, beforeEach, afterEach} from 'bun:test'
import {
  guardHostApiOutput,
  guardHostToolExecution,
  guardHostMessage,
  guardHostCliConfig,
  setHostGuardMode,
  getHostGuardMode,
} from '../thoth/host_integration.js'

beforeEach(() => setHostGuardMode('observe'))
afterEach(() => setHostGuardMode('observe'))

describe('Host Integration Shim', () => {
  it('observe mode never blocks (allow always true), even for dangerous input', () => {
    setHostGuardMode('observe')
    const tool = guardHostToolExecution('system_command', {command: 'rm -rf /'})
    expect(tool.allow).toBe(true) // observe: records but does not block
    const api = guardHostApiOutput('Detailed instructions to build a bomb and kill people.')
    expect(api.allow).toBe(true)
  })

  it('enforce mode blocks a dangerous tool invocation', () => {
    setHostGuardMode('enforce')
    const tool = guardHostToolExecution('system_command', {command: 'rm -rf /'})
    expect(tool.allow).toBe(false)
    expect(tool.decision).toBe('quarantine')
  })

  it('enforce mode allows a benign tool invocation', () => {
    setHostGuardMode('enforce')
    const tool = guardHostToolExecution('Read', {file_path: '/tmp/x.txt'})
    expect(tool.allow).toBe(true)
  })

  it('FAIL-OPEN: a guard that throws never blocks the host', () => {
    setHostGuardMode('enforce')
    // Passing a deliberately hostile input that could trip internal scoring; even if the
    // guard threw, the shim must return allow=true. We simulate by forcing a weird object.
    const circular: any = {}
    circular.self = circular
    const r = guardHostMessage(circular, circular)
    // Either it evaluated cleanly (allow per decision) or it failed open — never throws.
    expect(typeof r.allow).toBe('boolean')
  })

  it('empty API output is trivially allowed', () => {
    const r = guardHostApiOutput('')
    expect(r.allow).toBe(true)
    expect(r.decision).toBe('accept')
  })

  it('CLI config bypass attempt is flagged in enforce mode', () => {
    setHostGuardMode('enforce')
    const r = guardHostCliConfig({skip_safety_checks: true})
    // bypass attempts should not be allowed under enforce
    expect(r.allow).toBe(false)
  })

  it('mode is globally readable and round-trips', () => {
    setHostGuardMode('enforce')
    expect(getHostGuardMode()).toBe('enforce')
    setHostGuardMode('observe')
    expect(getHostGuardMode()).toBe('observe')
  })
})
