/**
 * THOTH Token Optimization — tests proving near-zero-token behavior.
 *
 * The injected llmFn counts how many times the real model would have been called,
 * so we can assert the optimizer drives that toward zero.
 */

import {describe, it, expect} from 'bun:test'
import {
  classifyModelTier,
  estimateTokens,
  buildCachedPrefix,
  CACHE_MIN_TOKENS,
  ResultCache,
  TokenBudget,
  TokenOptimizer,
  type ModelTier,
} from '../thoth/token_optimization.js'

describe('Model tier routing', () => {
  it('routes reasoning tasks to opus', () => {
    expect(classifyModelTier('reason about improvement strategy')).toBe('opus')
    expect(classifyModelTier('synthesize a plan')).toBe('opus')
    expect(classifyModelTier('diagnose the failure')).toBe('opus')
  })
  it('routes classification/detection tasks to haiku', () => {
    expect(classifyModelTier('check and detect consistency')).toBe('haiku')
    expect(classifyModelTier('detect hallucination signals')).toBe('haiku')
    expect(classifyModelTier('classify the input')).toBe('haiku')
  })
  it('defaults to sonnet for validate/parse/extract', () => {
    expect(classifyModelTier('validate chunk quality')).toBe('sonnet')
    expect(classifyModelTier('parse and extract goal structure')).toBe('sonnet')
  })
})

describe('Prompt cache prefix', () => {
  it('pads a short static block past the 1024-token cache threshold', () => {
    const short = 'You are THOTH.'
    expect(estimateTokens(short)).toBeLessThan(CACHE_MIN_TOKENS)
    const built = buildCachedPrefix(short, 'Doctrine line repeated to cross the cache threshold for prompt caching. ')
    expect(built.willCache).toBe(true)
    expect(built.estTokens).toBeGreaterThanOrEqual(CACHE_MIN_TOKENS)
  })
  it('reports willCache=false when no filler and block is short', () => {
    const built = buildCachedPrefix('tiny')
    expect(built.willCache).toBe(false)
  })
})

describe('ResultCache', () => {
  it('returns identical results for normalized-equal inputs (zero-token repeat)', () => {
    const c = new ResultCache()
    c.set('SYS', 'Hello World', {text: 'answer', tokensUsed: 100})
    // Different whitespace/case must still hit.
    expect(c.get('sys', '  hello   world ')?.text).toBe('answer')
  })
  it('misses on genuinely different input', () => {
    const c = new ResultCache()
    c.set('s', 'a', {text: 'x', tokensUsed: 1})
    expect(c.get('s', 'b')).toBeUndefined()
  })
})

describe('TokenBudget', () => {
  it('hard-stops when session cap would be exceeded', () => {
    const b = new TokenBudget(1000, 1_000_000)
    b.add(600)
    expect(b.wouldExceed(500).exceeded).toBe(true)
    expect(b.wouldExceed(500).which).toBe('session')
  })
  it('rolls the daily counter at a new date (injected clock)', () => {
    let day = '2026-01-01'
    const b = new TokenBudget(1_000_000, 1000, () => day)
    b.add(900)
    expect(b.status().dailyTokens).toBe(900)
    day = '2026-01-02'
    expect(b.status().dailyTokens).toBe(0) // rolled
  })
})

describe('TokenOptimizer near-zero behavior', () => {
  function makeLlm() {
    const calls: ModelTier[] = []
    const fn = (a: {tier: ModelTier; system: string; user: string}) => {
      calls.push(a.tier)
      return {text: `reply:${a.user}`, tokensUsed: 1000}
    }
    return {fn, calls}
  }

  it('N identical calls cost ONE LLM call; the rest are cache hits', () => {
    const {fn, calls} = makeLlm()
    const opt = new TokenOptimizer(fn)
    for (let i = 0; i < 20; i++) {
      opt.call({taskHint: 'validate chunk quality', system: 'SYS', user: 'same input'})
    }
    expect(calls.length).toBe(1) // only the first actually called the model
    const s = opt.getStats()
    expect(s.llmCalls).toBe(1)
    expect(s.cacheHits).toBe(19)
    expect(s.avoidanceRate).toBeGreaterThan(0.9) // >90% of calls avoided the LLM
  })

  it('heuristic short-circuit avoids the LLM entirely (true zero tokens)', () => {
    const {fn, calls} = makeLlm()
    const opt = new TokenOptimizer(fn)
    const out = opt.call({
      taskHint: 'check consistency',
      system: 'SYS',
      user: 'trivially decidable',
      heuristic: () => 'local-answer',
    })
    expect(out.source).toBe('heuristic')
    expect(out.tokensUsed).toBe(0)
    expect(calls.length).toBe(0)
    expect(out.result).toBe('local-answer')
  })

  it('routes each call to the cheapest adequate tier', () => {
    const {fn, calls} = makeLlm()
    const opt = new TokenOptimizer(fn)
    opt.call({taskHint: 'check and detect consistency', system: 's1', user: 'u1'})
    opt.call({taskHint: 'reason about strategy', system: 's2', user: 'u2'})
    opt.call({taskHint: 'validate quality', system: 's3', user: 'u3'})
    expect(calls).toEqual(['haiku', 'opus', 'sonnet'])
  })

  it('blocks calls that would exceed the budget (hard cap, fail-closed on spend)', () => {
    const {fn} = makeLlm()
    const opt = new TokenOptimizer(fn, new TokenBudget(500, 1_000_000))
    // First call ~ system+user estimate < 500, succeeds and adds 1000 tokens.
    opt.call({taskHint: 'validate', system: 'a', user: 'b'})
    // Now session is over cap; the next (distinct) call must be blocked.
    const out = opt.call({taskHint: 'validate', system: 'c', user: 'd'})
    expect(out.source).toBe('budget_blocked')
    expect(out.result).toBeNull()
  })

  it('reports avoidance + efficiency stats', () => {
    const {fn} = makeLlm()
    const opt = new TokenOptimizer(fn)
    opt.call({taskHint: 'validate', system: 's', user: 'x'}) // llm
    opt.call({taskHint: 'validate', system: 's', user: 'x'}) // cache
    opt.call({taskHint: 'check', system: 's', user: 'y', heuristic: () => 'h'}) // heuristic
    const s = opt.getStats()
    expect(s.llmCalls).toBe(1)
    expect(s.cacheHits).toBe(1)
    expect(s.heuristicHits).toBe(1)
    expect(s.avoidanceRate).toBeCloseTo(2 / 3, 5)
    expect(s.efficiency).toBeGreaterThan(0)
  })
})
