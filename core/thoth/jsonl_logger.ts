/**
 * THOTH Component F — JSONL Audit Logger.
 *
 * Durable, append-only JSON-Lines sink for the three THOTH audit streams:
 *   - rubric_scores.jsonl   (every rubric/composite score + decision)
 *   - lessons_learned.jsonl (structured lessons w/ source + proposed action)
 *   - healing_actions.jsonl (refinement/healing proposals + apply outcome)
 *
 * Each line is a self-contained JSON object carrying a `lineageRecordId` so the
 * durable JSONL trail is cross-referenceable to the immutable in-memory lineage
 * chain (full provenance). This is the durable counterpart the Pre-Ingest Gate's
 * LessonRecords previously had nowhere to land.
 *
 * INVARIANTS:
 *   I1: append-only — the logger never rewrites or truncates an existing stream
 *       (clear() is explicit, test/operations only)
 *   I2: one valid JSON object per line (newline-terminated) — a reader can
 *       parse line-by-line without a streaming JSON parser
 *   I3: every appended record is timestamped (ISO-8601) for ordering
 */

import {appendFileSync, mkdirSync, existsSync, readFileSync, rmSync} from 'fs'

export type JsonlStream = 'rubric_scores' | 'lessons_learned' | 'healing_actions'

export interface RubricScoreLine {
  sourceId?: string
  composite: number
  dimensions?: Record<string, number>
  decision?: string
  lineageRecordId?: string
}

export interface LessonLine {
  sourceId: string
  weakestDimension: string
  composite: number
  decision: string
  evidence: string[]
  proposedAction: string
  lineageRecordId?: string
}

export interface HealingActionLine {
  target: string
  changeType: string
  rationale: string
  residualRisk: number
  applied: boolean
  lineageRecordId?: string
}

export class JsonlLogger {
  constructor(private readonly dir: string) {}

  private streamPath(stream: JsonlStream): string {
    return `${this.dir}/${stream}.jsonl`
  }

  private ensureDir(): void {
    if (!existsSync(this.dir)) {
      mkdirSync(this.dir, {recursive: true})
    }
  }

  /** Low-level append (I2 + I3 enforced here). */
  private append(stream: JsonlStream, record: Record<string, unknown>): void {
    this.ensureDir()
    const stamped = {ts: new Date().toISOString(), ...record}
    // I2: exactly one JSON object + newline. JSON.stringify never emits a raw newline.
    appendFileSync(this.streamPath(stream), JSON.stringify(stamped) + '\n')
  }

  logRubricScore(line: RubricScoreLine): void {
    this.append('rubric_scores', line as unknown as Record<string, unknown>)
  }

  logLesson(line: LessonLine): void {
    this.append('lessons_learned', line as unknown as Record<string, unknown>)
  }

  logHealingAction(line: HealingActionLine): void {
    this.append('healing_actions', line as unknown as Record<string, unknown>)
  }

  /** Read & parse a stream back into records (verification / replay). */
  read(stream: JsonlStream): Array<Record<string, unknown>> {
    const path = this.streamPath(stream)
    if (!existsSync(path)) return []
    const raw: string = readFileSync(path, 'utf-8')
    return raw
      .split('\n')
      .filter((l: string) => l.trim().length > 0)
      .map((l: string) => JSON.parse(l) as Record<string, unknown>)
  }

  count(stream: JsonlStream): number {
    return this.read(stream).length
  }

  /** Explicit destructive clear — NOT used on hot paths (I1 carve-out). */
  clear(stream?: JsonlStream): void {
    if (stream) {
      const p = this.streamPath(stream)
      if (existsSync(p)) rmSync(p)
      return
    }
    if (existsSync(this.dir)) rmSync(this.dir, {recursive: true, force: true})
  }
}

/** Default global logger; directory is overridable via THOTH_LOG_DIR. */
const DEFAULT_LOG_DIR =
  (typeof process !== 'undefined' && process.env && process.env.THOTH_LOG_DIR) || './.thoth-logs'

export const globalJsonlLogger = new JsonlLogger(DEFAULT_LOG_DIR)
