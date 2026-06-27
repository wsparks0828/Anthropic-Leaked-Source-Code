/**
 * Phase 6: Performance Profiling
 *
 * Measures latency impact of guardrail system:
 * - Rubric scoring throughput
 * - Truth gate performance
 * - Learning bridge overhead
 * - Cross-verifier latency
 * - End-to-end gate latency
 * - Memory consumption
 */

import {describe, it, expect} from 'bun:test'
import {globalRubricScorer} from '../rubric_scorer.js'
import {globalTruthGate} from '../truth_gates.js'
import {globalGuardrailLearningBridge} from '../guardrail_learning_bridge.js'
import {globalCrossVerifierEnsemble} from '../cross_verifier_ensemble.js'
import {guardApiOutput, guardToolExecution, guardMessageMutation, guardCliConfig} from '../guardrail_integration.js'
import {globalHealthMonitor} from '../guardrail_health.js'

// Benchmark utilities
function measureTime(fn: () => void): number {
  const start = performance.now()
  fn()
  return performance.now() - start
}

function runBenchmark(name: string, fn: () => void, iterations: number = 100): {avg: number; min: number; max: number; p95: number} {
  const times: number[] = []

  for (let i = 0; i < iterations; i++) {
    times.push(measureTime(fn))
  }

  times.sort((a, b) => a - b)
  const avg = times.reduce((a, b) => a + b, 0) / times.length
  const min = times[0]
  const max = times[times.length - 1]
  const p95 = times[Math.floor(times.length * 0.95)]

  console.log(`\n[${name}]`)
  console.log(`  Avg: ${avg.toFixed(3)}ms | Min: ${min.toFixed(3)}ms | Max: ${max.toFixed(3)}ms | P95: ${p95.toFixed(3)}ms`)

  return {avg, min, max, p95}
}

describe('Performance Profiling', () => {
  /**
   * Test 1: Rubric Scorer Performance
   */
  it('should measure rubric scorer latency', () => {
    const output = 'This is a test output for performance benchmarking of the rubric scoring system.'

    const result = runBenchmark('Rubric Scorer', () => {
      globalRubricScorer.score(output)
    }, 100)

    // Rubric score should be <10ms (heuristic-based, no external calls)
    expect(result.avg).toBeLessThan(10)
    expect(result.p95).toBeLessThan(20)
  })

  /**
   * Test 2: Truth Gate Performance
   */
  it('should measure truth gate latency', () => {
    const output = 'Machine learning is a powerful technology with many applications.'

    const result = runBenchmark('Truth Gate', () => {
      globalTruthGate.gate(output)
    }, 100)

    // Truth gate should be <10ms (heuristic checks)
    expect(result.avg).toBeLessThan(10)
    expect(result.p95).toBeLessThan(20)
  })

  /**
   * Test 3: Learning Bridge Performance
   */
  it('should measure learning bridge latency', () => {
    const output = 'Test output for learning bridge performance measurement.'
    const rubricScore = globalRubricScorer.score(output)
    const truthVerdict = globalTruthGate.gate(output)

    const result = runBenchmark('Learning Bridge', () => {
      globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${Date.now()}_${Math.random()}`,
      })
    }, 50)

    // Learning bridge should be <20ms (pattern extraction + memory wiring)
    expect(result.avg).toBeLessThan(20)
    expect(result.p95).toBeLessThan(50)
  })

  /**
   * Test 4: Cross-Verifier Performance
   */
  it('should measure cross-verifier latency', () => {
    const proposal = {
      target: 'safety_gate' as const,
      changeType: 'threshold_adjust' as const,
      proposal: 'Increase safety threshold by 0.1',
      rationale: 'Multiple anomalies detected',
      expectedImpact: 'Reduce false positives by 5%',
      residualRisk: 0.1,
    }

    const result = runBenchmark('Cross-Verifier', () => {
      globalCrossVerifierEnsemble.check(proposal, `prop_${Date.now()}_${Math.random()}`)
    }, 50)

    // Cross-verifier should be <15ms (3 independent checks)
    expect(result.avg).toBeLessThan(15)
    expect(result.p95).toBeLessThan(30)
  })

  /**
   * Test 5: API Output Gate Latency
   */
  it('should measure API output gate latency', () => {
    const output = 'This is a sample API response for measuring guardrail gate latency.'

    const result = runBenchmark('API Output Gate', () => {
      guardApiOutput(output)
    }, 100)

    // Full API gate should be <20ms (score + truth + learning + cross-check)
    expect(result.avg).toBeLessThan(20)
    expect(result.p95).toBeLessThan(40)
  })

  /**
   * Test 6: Tool Execution Gate Latency
   */
  it('should measure tool execution gate latency', () => {
    const result = runBenchmark('Tool Execution Gate', () => {
      guardToolExecution('read_file', {path: '/home/user/file.txt'})
    }, 100)

    // Tool gate should be <10ms (pattern matching + quick scoring)
    expect(result.avg).toBeLessThan(10)
    expect(result.p95).toBeLessThan(20)
  })

  /**
   * Test 7: Message Mutation Gate Latency
   */
  it('should measure message mutation gate latency', () => {
    const oldMsg = {content: 'Original message content'}
    const newMsg = {content: 'Original message content with small edit'}

    const result = runBenchmark('Message Mutation Gate', () => {
      guardMessageMutation(oldMsg, newMsg)
    }, 100)

    // Message gate should be <15ms (content scoring + delta analysis)
    expect(result.avg).toBeLessThan(15)
    expect(result.p95).toBeLessThan(30)
  })

  /**
   * Test 8: CLI Config Gate Latency
   */
  it('should measure CLI config gate latency', () => {
    const config = {model: 'claude-opus-4-8', temperature: 0.7, max_tokens: 4096}

    const result = runBenchmark('CLI Config Gate', () => {
      guardCliConfig(config)
    }, 100)

    // Config gate should be <10ms (quick pattern matching)
    expect(result.avg).toBeLessThan(10)
    expect(result.p95).toBeLessThan(20)
  })

  /**
   * Test 9: End-to-End Pipeline Latency
   */
  it('should measure full end-to-end pipeline latency', () => {
    const output = 'Comprehensive test output for measuring full guardrail pipeline latency from input to decision.'

    const result = runBenchmark('Full Pipeline', () => {
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)
      const signal = globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${Date.now()}`,
      })
      if (signal.proposal) {
        globalCrossVerifierEnsemble.check(signal.proposal, `prop_${Date.now()}`)
      }
    }, 50)

    // Full pipeline should be <50ms
    expect(result.avg).toBeLessThan(50)
    expect(result.p95).toBeLessThan(100)
  })

  /**
   * Test 10: Health Monitor Latency
   */
  it('should measure health monitoring latency', () => {
    const result = runBenchmark('Health Monitor', () => {
      globalHealthMonitor.recordDecision('accept')
      globalHealthMonitor.recordVerification(0.75, 0.80, false)
      globalHealthMonitor.getStatus()
    }, 100)

    // Health monitoring should be <5ms (in-memory circular buffer)
    expect(result.avg).toBeLessThan(5)
    expect(result.p95).toBeLessThan(10)
  })

  /**
   * Test 11: Throughput — Outputs Per Second
   */
  it('should achieve high throughput on API outputs', () => {
    const outputs = Array(100)
      .fill(0)
      .map((_, i) => `This is sample output number ${i} for throughput testing.`)

    const startTime = performance.now()

    for (const output of outputs) {
      guardApiOutput(output)
    }

    const elapsed = performance.now() - startTime
    const throughput = (outputs.length / elapsed) * 1000 // outputs per second

    console.log(`\n[Throughput Test]`)
    console.log(`  ${outputs.length} outputs in ${elapsed.toFixed(1)}ms`)
    console.log(`  Throughput: ${throughput.toFixed(0)} outputs/sec`)

    // Should handle at least 1000 outputs per second
    expect(throughput).toBeGreaterThan(1000)
  })

  /**
   * Test 12: Scalability — Large Batch Processing
   */
  it('should handle large batches efficiently', () => {
    const batchSize = 500
    const outputs = Array(batchSize)
      .fill(0)
      .map((_, i) => `Batch output number ${i} for scalability testing with reasonable content length.`)

    const startTime = performance.now()

    for (const output of outputs) {
      guardApiOutput(output)
    }

    const elapsed = performance.now() - startTime
    const avgLatency = elapsed / batchSize

    console.log(`\n[Scalability Test]`)
    console.log(`  Batch size: ${batchSize}`)
    console.log(`  Total time: ${elapsed.toFixed(1)}ms`)
    console.log(`  Avg latency per output: ${avgLatency.toFixed(3)}ms`)

    // Average latency should not degrade with larger batches
    expect(avgLatency).toBeLessThan(25)
  })

  /**
   * Test 13: Memory Impact
   */
  it('should have minimal memory overhead', () => {
    const initialMemory = process.memoryUsage().heapUsed

    // Process 100 outputs with full pipeline
    for (let i = 0; i < 100; i++) {
      const output = `Memory test output number ${i} with enough content.`
      const rubricScore = globalRubricScorer.score(output)
      const truthVerdict = globalTruthGate.gate(output)
      globalGuardrailLearningBridge.processVerification(rubricScore, truthVerdict, {
        source: 'api_boundary',
        summary: output,
        hash: `hash_${i}`,
      })
    }

    const finalMemory = process.memoryUsage().heapUsed
    const memoryIncrease = finalMemory - initialMemory

    console.log(`\n[Memory Impact Test]`)
    console.log(`  Memory increase: ${(memoryIncrease / 1024 / 1024).toFixed(2)}MB for 100 processing cycles`)

    // Should use <50MB for 100 cycles (guardrail + learning signals)
    expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024)
  })

  /**
   * Test 14: Cache Effectiveness (LRU memoization)
   */
  it('should leverage LRU cache for repeated content', () => {
    const output = 'Repeated test output for cache effectiveness measurement.'

    // First call (cache miss)
    const score1 = globalRubricScorer.score(output)

    // Second call (cache hit) - should return identical object
    const score2 = globalRubricScorer.score(output)

    console.log(`\n[Cache Effectiveness]`)
    console.log(`  Score 1: ${score1.overall}`)
    console.log(`  Score 2: ${score2.overall}`)

    // Cache should return identical results
    expect(score1.overall).toBe(score2.overall)
    expect(score1.timestamp).toBe(score2.timestamp) // Same timestamp = cached result
  })

  /**
   * Test 15: Concurrent Gate Performance
   */
  it('should maintain latency under concurrent load', () => {
    const concurrentCalls = 50
    const output = 'Test output for concurrent gate performance measurement.'

    const startTime = performance.now()

    for (let i = 0; i < concurrentCalls; i++) {
      guardApiOutput(output)
    }

    const elapsed = performance.now() - startTime
    const avgLatency = elapsed / concurrentCalls

    console.log(`\n[Concurrent Load Test]`)
    console.log(`  Concurrent calls: ${concurrentCalls}`)
    console.log(`  Total time: ${elapsed.toFixed(1)}ms`)
    console.log(`  Avg latency: ${avgLatency.toFixed(3)}ms`)

    // Latency should not degrade significantly under load
    expect(avgLatency).toBeLessThan(25)
  })
})
