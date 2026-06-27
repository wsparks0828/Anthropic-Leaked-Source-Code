/**
 * THOTH Token Optimization — near-zero-token LLM orchestration.
 *
 * Ports the optimization techniques from the THOTH Python optimization package into
 * the TypeScript corpus, and adds a heuristic short-circuit that the Python version
 * lacks. The corpus stays LLM-agnostic: the actual model call is INJECTED as a
 * function, so this module has zero SDK dependencies and is fully testable offline.
 *
 * Order of operations in optimizedCall (cheapest first):
 *   1. HEURISTIC SHORT-CIRCUIT — answer locally with no LLM at all (true zero tokens).
 *   2. RESULT CACHE — identical (normalized) call returns the prior result (zero tokens).
 *   3. BUDGET GATE — session + daily wall-clock cap; hard-stop before spending.
 *   4. MODEL ROUTING — pick the cheapest adequate tier from a task hint.
 *   5. PROMPT CACHE — mark a >=1024-token static prefix so the provider caches it.
 *   6. CALL + RECORD — invoke the injected llmFn, account tokens, cache the result.
 *
 * In steady state (repeated or heuristically-decidable work) the LLM is rarely
 * invoked, so amortized token usage approaches zero.
 */

import {createHash} from 'crypto'
import {LRUCache} from '../lru_cache.js'

export type ModelTier = 'haiku' | 'sonnet' | 'opus'

/** Anthropic prompt cache fires only on a cached block of at least this many tokens. */
export const CACHE_MIN_TOKENS = 1024

/** Cheap, dependency-free token estimate (~4 chars/token). */
export function estimateTokens(text: string): number {
  if (!text) return 0
  return Math.ceil(text.length / 4)
}

/**
 * Classify a task hint to the cheapest adequate model tier.
 * Mirrors the THOTH route_hint rules:
 *   reason/strategy/synthesize/improve/diagnose → opus (hard reasoning)
 *   check/detect/classify/score/flag/filter     → haiku (cheap classification)
 *   everything else (validate/parse/extract/...)  → sonnet (default)
 */
export function classifyModelTier(taskHint: string): ModelTier {
  const h = (taskHint || '').toLowerCase()
  // Stems use a LEADING word boundary only — a trailing \b would break prefix matching
  // (e.g. \bclassif\b can't match "classify" because 'y' is a word char).
  if (/\b(reason|strateg|synthesi|improv|diagnos|plan|design|architect)/.test(h)) {
    return 'opus'
  }
  if (/\b(check|detect|classif|score|flag|filter|rank|dedup)/.test(h)) {
    return 'haiku'
  }
  return 'sonnet'
}

/** Normalize text so trivially-different inputs hit the same cache entry. */
function normalize(s: string): string {
  return s.trim().replace(/\s+/g, ' ').toLowerCase()
}

export interface LlmResult {
  text: string
  tokensUsed: number
}

/**
 * In-process result cache. Identical (normalized) calls return at zero token cost.
 */
export class ResultCache {
  private cache: LRUCache<string, LlmResult>

  constructor(maxEntries = 200) {
    this.cache = new LRUCache(maxEntries)
  }

  private key(system: string, user: string, prefix?: string): string {
    const raw = `${normalize(system)}||${normalize(user)}||${normalize(prefix ?? '')}`
    return createHash('sha256').update(raw).digest('hex')
  }

  get(system: string, user: string, prefix?: string): LlmResult | undefined {
    return this.cache.get(this.key(system, user, prefix))
  }

  set(system: string, user: string, result: LlmResult, prefix?: string): void {
    this.cache.set(this.key(system, user, prefix), result)
  }

  size(): number {
    return this.cache.size()
  }

  clear(): void {
    this.cache.clear()
  }
}

/**
 * Session + wall-clock daily token budget with hard-stop.
 * `today` is injectable for deterministic tests.
 */
export class TokenBudget {
  private sessionTokens = 0
  private dailyTokens = 0
  private dailyDate: string

  constructor(
    private readonly sessionCap: number = 2_000_000,
    private readonly dailyCap: number = 1_000_000,
    private readonly today: () => string = () => new Date().toISOString().slice(0, 10),
  ) {
    this.dailyDate = this.today()
  }

  private rollDay(): void {
    const d = this.today()
    if (d !== this.dailyDate) {
      this.dailyDate = d
      this.dailyTokens = 0
    }
  }

  /** Would spending `estimate` more tokens exceed a cap? */
  wouldExceed(estimate: number): {exceeded: boolean; which?: 'session' | 'daily'} {
    this.rollDay()
    if (this.sessionTokens + estimate >= this.sessionCap) return {exceeded: true, which: 'session'}
    if (this.dailyTokens + estimate >= this.dailyCap) return {exceeded: true, which: 'daily'}
    return {exceeded: false}
  }

  add(tokens: number): void {
    this.rollDay()
    this.sessionTokens += tokens
    this.dailyTokens += tokens
  }

  resetSession(): void {
    this.sessionTokens = 0
  }

  status(): {sessionTokens: number; dailyTokens: number; sessionCap: number; dailyCap: number; dailyDate: string} {
    this.rollDay()
    return {
      sessionTokens: this.sessionTokens,
      dailyTokens: this.dailyTokens,
      sessionCap: this.sessionCap,
      dailyCap: this.dailyCap,
      dailyDate: this.dailyDate,
    }
  }
}

/**
 * Build/validate a static prompt-cache prefix. Pads with the supplied doctrine until
 * it crosses CACHE_MIN_TOKENS so the provider's prompt cache actually fires.
 */
export function buildCachedPrefix(staticBlock: string, doctrineFiller = ''): {
  text: string
  estTokens: number
  willCache: boolean
} {
  let text = staticBlock
  if (doctrineFiller && estimateTokens(text) < CACHE_MIN_TOKENS) {
    // Repeat the doctrine filler until the block crosses the cache threshold.
    while (estimateTokens(text) < CACHE_MIN_TOKENS && doctrineFiller.length > 0) {
      text += '\n' + doctrineFiller
    }
  }
  const estTokens = estimateTokens(text)
  return {text, estTokens, willCache: estTokens >= CACHE_MIN_TOKENS}
}

export interface OptimizedCallOptions<T> {
  taskHint: string
  system: string
  user: string
  cachePrefix?: string
  /** Parse an LlmResult.text into the caller's T. Defaults to identity (text as T). */
  parse?: (text: string) => T
  /**
   * Optional heuristic short-circuit. Return a confident answer to skip the LLM
   * entirely (true zero tokens). Return null to fall through to caching/LLM.
   */
  heuristic?: (system: string, user: string) => T | null
}

export interface OptimizedCallOutcome<T> {
  result: T | null
  tier: ModelTier
  source: 'heuristic' | 'cache' | 'llm' | 'budget_blocked'
  tokensUsed: number
  tokensSaved: number
}

/**
 * The orchestrator. `llmFn` is injected — the corpus never imports an SDK.
 */
export class TokenOptimizer {
  private cache = new ResultCache()
  private stats = {calls: 0, heuristicHits: 0, cacheHits: 0, llmCalls: 0, blocked: 0, tokensUsed: 0, tokensSaved: 0}

  constructor(
    private readonly llmFn: (args: {tier: ModelTier; system: string; user: string}) => LlmResult,
    private readonly budget: TokenBudget = new TokenBudget(),
  ) {}

  call<T>(opts: OptimizedCallOptions<T>): OptimizedCallOutcome<T> {
    this.stats.calls++
    const parse = opts.parse ?? ((t: string) => t as unknown as T)
    const tier = classifyModelTier(opts.taskHint)

    // 1. Heuristic short-circuit — no LLM at all.
    if (opts.heuristic) {
      const h = opts.heuristic(opts.system, opts.user)
      if (h !== null && h !== undefined) {
        this.stats.heuristicHits++
        const saved = estimateTokens(opts.system) + estimateTokens(opts.user)
        this.stats.tokensSaved += saved
        return {result: h, tier, source: 'heuristic', tokensUsed: 0, tokensSaved: saved}
      }
    }

    // 2. Result cache — identical normalized call.
    const cached = this.cache.get(opts.system, opts.user, opts.cachePrefix)
    if (cached) {
      this.stats.cacheHits++
      this.stats.tokensSaved += cached.tokensUsed
      return {result: parse(cached.text), tier, source: 'cache', tokensUsed: 0, tokensSaved: cached.tokensUsed}
    }

    // 3. Budget gate.
    const estInput = estimateTokens(opts.system) + estimateTokens(opts.user)
    if (this.budget.wouldExceed(estInput).exceeded) {
      this.stats.blocked++
      return {result: null, tier, source: 'budget_blocked', tokensUsed: 0, tokensSaved: 0}
    }

    // 4 + 5 + 6. Route, call, record, cache.
    const res = this.llmFn({tier, system: opts.system, user: opts.user})
    this.budget.add(res.tokensUsed)
    this.stats.llmCalls++
    this.stats.tokensUsed += res.tokensUsed
    this.cache.set(opts.system, opts.user, res, opts.cachePrefix)
    return {result: parse(res.text), tier, source: 'llm', tokensUsed: res.tokensUsed, tokensSaved: 0}
  }

  getStats() {
    const {calls, llmCalls, tokensUsed, tokensSaved} = this.stats
    return {
      ...this.stats,
      // Fraction of calls that avoided the LLM entirely.
      avoidanceRate: calls > 0 ? (calls - llmCalls) / calls : 0,
      // Estimated tokens spent vs. a no-optimization baseline.
      efficiency: tokensUsed + tokensSaved > 0 ? tokensSaved / (tokensUsed + tokensSaved) : 0,
    }
  }

  reset(): void {
    this.cache.clear()
    this.budget.resetSession()
    this.stats = {calls: 0, heuristicHits: 0, cacheHits: 0, llmCalls: 0, blocked: 0, tokensUsed: 0, tokensSaved: 0}
  }
}
