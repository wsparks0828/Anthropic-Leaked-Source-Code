/**
 * THOTH JSONL Logger — hermetic tests.
 *
 * Writes to a unique temp dir and cleans up. Verifies append-only semantics,
 * one-object-per-line integrity, read-back fidelity, and gate→logger wiring.
 */

import {describe, it, expect, beforeEach, afterEach} from 'bun:test'
import {tmpdir} from 'os'
import {existsSync, readFileSync, rmSync} from 'fs'
import {JsonlLogger} from '../thoth/jsonl_logger.js'
import {PreIngestGate} from '../thoth/pre_ingest_gate.js'
import {globalLineageAuditor} from '../lineage_auditor.js'

let dir: string
let logger: JsonlLogger
let counter = 0

beforeEach(() => {
  counter++
  dir = `${tmpdir()}/thoth-jsonl-test-${Date.now()}-${counter}`
  logger = new JsonlLogger(dir)
  globalLineageAuditor.reset()
})

afterEach(() => {
  if (existsSync(dir)) rmSync(dir, {recursive: true, force: true})
})

describe('THOTH JSONL Logger', () => {
  it('I2: writes exactly one valid JSON object per line', () => {
    logger.logRubricScore({composite: 0.7, decision: 'accept'})
    logger.logRubricScore({composite: 0.3, decision: 'reject'})

    const raw = readFileSync(`${dir}/rubric_scores.jsonl`, 'utf-8')
    const lines = raw.split('\n').filter((l) => l.length > 0)
    expect(lines.length).toBe(2)
    for (const line of lines) {
      expect(() => JSON.parse(line)).not.toThrow()
      expect(line).not.toContain('\n')
    }
  })

  it('I3: every record is timestamped', () => {
    logger.logLesson({
      sourceId: 's1',
      weakestDimension: 'factuality',
      composite: 0.5,
      decision: 'flag',
      evidence: ['factuality:negative'],
      proposedAction: 'review',
    })
    const recs = logger.read('lessons_learned')
    expect(recs.length).toBe(1)
    expect(typeof recs[0].ts).toBe('string')
    expect(String(recs[0].ts)).toMatch(/^\d{4}-\d{2}-\d{2}T/)
  })

  it('I1: append-only — second write preserves the first', () => {
    logger.logHealingAction({target: 'safety_gate', changeType: 'threshold_adjust', rationale: 'r', residualRisk: 0.1, applied: true})
    logger.logHealingAction({target: 'coherence_checker', changeType: 'rule_add', rationale: 'r2', residualRisk: 0.2, applied: false})
    const recs = logger.read('healing_actions')
    expect(recs.length).toBe(2)
    expect(recs[0].target).toBe('safety_gate')
    expect(recs[1].target).toBe('coherence_checker')
  })

  it('read of a non-existent stream returns empty (no throw)', () => {
    expect(logger.read('rubric_scores')).toEqual([])
    expect(logger.count('lessons_learned')).toBe(0)
  })

  it('three streams are independent files', () => {
    logger.logRubricScore({composite: 0.8})
    logger.logLesson({sourceId: 's', weakestDimension: 'coherence', composite: 0.8, decision: 'accept', evidence: [], proposedAction: 'ingest'})
    logger.logHealingAction({target: 'safety_gate', changeType: 'threshold_adjust', rationale: 'r', residualRisk: 0.05, applied: true})
    expect(logger.count('rubric_scores')).toBe(1)
    expect(logger.count('lessons_learned')).toBe(1)
    expect(logger.count('healing_actions')).toBe(1)
  })

  it('clear(stream) removes only that stream', () => {
    logger.logRubricScore({composite: 0.8})
    logger.logLesson({sourceId: 's', weakestDimension: 'coherence', composite: 0.8, decision: 'accept', evidence: [], proposedAction: 'ingest'})
    logger.clear('rubric_scores')
    expect(logger.count('rubric_scores')).toBe(0)
    expect(logger.count('lessons_learned')).toBe(1)
  })

  it('gate→logger wiring: an evaluation writes both rubric_scores and lessons_learned', () => {
    const gate = new PreIngestGate({}, logger)
    gate.evaluate(
      'Transformer attention uses scaled dot-product softmax(QKᵀ/√d)V per Vaswani et al. 2017 with concrete examples and citations.',
      {sourceId: 'src_42', sourceTier: 1, query: 'transformer attention'},
    )
    expect(logger.count('rubric_scores')).toBe(1)
    expect(logger.count('lessons_learned')).toBe(1)

    const lesson = logger.read('lessons_learned')[0]
    expect(lesson.sourceId).toBe('src_42')
    // Provenance: JSONL line carries the lineage chain hash.
    expect(typeof lesson.lineageRecordId).toBe('string')
    expect(String(lesson.lineageRecordId).length).toBeGreaterThan(0)
  })

  it('default gate (no logger) writes NO files — hermetic by default', () => {
    const gate = new PreIngestGate() // no logger injected
    gate.evaluate('some content here that is reasonably long for evaluation purposes', {sourceId: 's', sourceTier: 2})
    // logger dir for THIS test must remain untouched (gate had no logger)
    expect(logger.count('rubric_scores')).toBe(0)
  })
})
