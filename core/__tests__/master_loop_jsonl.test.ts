/**
 * THOTH Master Loop ⇄ JSONL durable-wiring test (hermetic).
 *
 * Proves the default path (logger forwarded into the gate) emits the three
 * durable streams to disk, with no lesson duplication.
 */

import {describe, it, expect, beforeEach, afterEach} from 'bun:test'
import {tmpdir} from 'os'
import {existsSync, rmSync} from 'fs'
import {JsonlLogger} from '../thoth/jsonl_logger.js'
import {MasterLoop} from '../thoth/master_loop.js'
import {globalLineageAuditor} from '../lineage_auditor.js'

let dir: string
let logger: JsonlLogger
let n = 0

beforeEach(() => {
  n++
  dir = `${tmpdir()}/thoth-master-jsonl-${Date.now()}-${n}`
  logger = new JsonlLogger(dir)
  globalLineageAuditor.reset()
})

afterEach(() => {
  if (existsSync(dir)) rmSync(dir, {recursive: true, force: true})
})

const WEAK = 'A short but ingestible sentence about a topic with a little detail included here now.'

describe('Master Loop durable JSONL wiring', () => {
  it('a single cycle writes exactly one rubric_score and one lesson (no duplication)', () => {
    const loop = new MasterLoop({logger})
    loop.runCycle({content: WEAK, sourceId: 'src_a', sourceTier: 2})
    expect(logger.count('rubric_scores')).toBe(1)
    expect(logger.count('lessons_learned')).toBe(1)
  })

  it('a sub-floor reasoned cycle also writes a healing_action', () => {
    const loop = new MasterLoop({logger})
    loop.runCycle({content: WEAK, sourceId: 'src_b', sourceTier: 2})
    // WEAK is tier-2, passes density, but composite is sub-floor → REFLECT_HEAL fires.
    expect(logger.count('healing_actions')).toBeGreaterThanOrEqual(1)
    const heal = logger.read('healing_actions')[0]
    expect(heal).toHaveProperty('applied')
    expect(heal).toHaveProperty('residualRisk')
  })

  it('rubric_scores line carries provenance (lineageRecordId)', () => {
    const loop = new MasterLoop({logger})
    loop.runCycle({content: WEAK, sourceId: 'src_c', sourceTier: 2})
    const score = logger.read('rubric_scores')[0]
    expect(typeof score.lineageRecordId).toBe('string')
    expect(String(score.lineageRecordId).length).toBeGreaterThan(0)
  })

  it('three cycles accumulate three rubric_scores (append-only)', () => {
    const loop = new MasterLoop({logger})
    for (let i = 0; i < 3; i++) loop.runCycle({content: WEAK, sourceId: `src_${i}`, sourceTier: 2})
    expect(logger.count('rubric_scores')).toBe(3)
    expect(logger.count('lessons_learned')).toBe(3)
  })
})
