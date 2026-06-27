/**
 * THOTH Corpus — public integration surface.
 *
 * Import from here to wire the guardrail/THOTH corpus into a host application.
 * Typical use:
 *
 *   import {bootstrapGuardrailCorpus, globalMasterLoop} from './core/thoth'
 *   bootstrapGuardrailCorpus({encryptionKey: process.env.LINEAGE_KEY, storageBackend})
 *   const cycle = globalMasterLoop.runCycle({content, sourceId, sourceTier, intent, query})
 */

export {
  bootstrapGuardrailCorpus,
  isCorpusBootstrapped,
  resetBootstrap,
  type BootstrapOptions,
  type BootstrapResult,
} from './bootstrap.js'

// Master flow
export {MasterLoop, globalMasterLoop, type CycleInput, type CycleResult, type MasterState} from './master_loop.js'

// Stages
export {
  PreIngestGate,
  globalPreIngestGate,
  preIngestEvaluate,
  computeInformativeness,
  type PreIngestVerdict,
  type PreIngestDecision,
  type SourceTier,
  type LessonRecord,
} from './pre_ingest_gate.js'

export {LifecycleEnforcer, globalLifecycleEnforcer, type LifecycleResult, type StepResult} from './lifecycle.js'

// Durable audit trail
export {
  JsonlLogger,
  globalJsonlLogger,
  type JsonlStream,
  type RubricScoreLine,
  type LessonLine,
  type HealingActionLine,
} from './jsonl_logger.js'

// Host integration shim (wires guards into the live CLI; observe-by-default, fail-open).
// Call setHostGuardMode('enforce') to let quarantine actually block (e.g. dangerous tools).
export {
  guardHostApiOutput,
  guardHostToolExecution,
  guardHostMessage,
  guardHostCliConfig,
  setHostGuardMode,
  getHostGuardMode,
  type HostGuardMode,
  type HostGuardDecision,
} from './host_integration.js'
