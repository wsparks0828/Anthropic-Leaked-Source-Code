# Phase 1: Pragmatic Baseline & Guardrail Integration Points

**Date**: 2026-06-27  
**Branch**: claude/jarvis-0HKxO  
**Audit Approach**: Forensic (evidence-based, fail-closed)

## Audit Findings

### What We Found
- **Total LOC**: 507K (mostly active, integrated code)
- **Monolithic Files**: 40+ files >3K LOC (core logic, not dead code)
- **Commented Lines**: ~14K (mostly documentation, safe to keep)
- **Dead Code**: Minimal and scattered (not worth mass removal)
- **Import Duplication**: 370 files with relative paths to utils/* (architectural, not a blocker)

### Why Traditional Refactoring Didn't Happen
1. Codebase is tightly integrated (CLI → agents → tools → APIs)
2. Large files serve multiple responsibilities (split requires tests we don't have)
3. No obvious dead code blocks (tried aggressive detection, found documentation)
4. Consolidation is architectural debt, not immediate blocker

## Guardrail Integration Points (Phase 2 Target)

### Critical Paths for Guardrail Injection

#### 1. API Boundary (Layer 1)
**File**: `services/api/claude.ts` (3,419 LOC)
**Entry Point**: API call dispatch
**Injection Point**: Before/after Claude SDK call
```
User Input → Validate Input (guardrail pre-gate) 
          → Call Claude API 
          → Validate Output (guardrail post-gate + rubric score) 
          → Return
```
**Phase 2 Action**: Wrap API call with guardrail validation

#### 2. Tool Execution (Layer 2)
**File**: `services/tools/toolExecution.ts` (1,745 LOC)
**Entry Point**: Tool invocation
**Injection Point**: Before tool execution, after tool output
```
Tool Selected → Validate Tool (guardrail gate) 
             → Execute Tool 
             → Validate Output (guard rail + truth-tag) 
             → Return to Agent
```
**Phase 2 Action**: Add guardrail container around tool execution

#### 3. Message Flow (Layer 3)
**File**: `utils/messages.ts` (5,512 LOC)
**Entry Point**: Message creation
**Injection Point**: After message creation, before storage
```
Create Message → Validate Content (rubric score, truth-tag) 
              → Store with Lineage 
              → Return
```
**Phase 2 Action**: Add lineage_v4 record on message mutation

#### 4. CLI Entry (Layer 0)
**File**: `cli/print.ts` (5,594 LOC)
**Entry Point**: User input processing
**Injection Point**: After input parsing, before dispatch
```
User Input → Parse Command 
          → Validate Config (pre-boot gate) 
          → Dispatch to Engine 
          → Display Output
```
**Phase 2 Action**: Add config validation + graceful shutdown handler

### Secondary Integration Points

- **MCP Client** (`services/mcp/client.ts`): MCP message routing → can feed learning signals
- **Session State** (`utils/sessionStorage.ts`): Session mutations → lineage tracking point
- **Permission System** (`utils/permissions/*`): Permission decisions → can feed guardrail calibration
- **Query Engine** (`query.ts`): Context building → can use guardrail hints

## Baseline State (No Changes)

- **Current Commit**: 651a9f6 (main synced)
- **Branch**: claude/jarvis-0HKxO (clean working tree)
- **Size**: 45MB, 1,890 files, 507K LOC

## Phase 1 Conclusion: Ready for Phase 2

**We are NOT doing**:
- Splitting large files (risky without tests)
- Removing dead code (none found)
- Mass import consolidation (low value, high risk)

**We ARE doing**:
- Documenting guardrail injection points (clear targets for Phase 2)
- Identifying atomic mutation points (for lineage tracking)
- Preparing architecture for meta-learning bridge (identified 4 critical paths)

**Next**: Phase 2 starts immediately with guardrail subsystem build.

## Architecture Map: Where Guardrails Live (Phase 2)

```
┌─────────────────────────────────────────────────────────┐
│ JARVIS with Guardrail Integration                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  cli/print.ts (5.5K LOC)                                │
│  └─ Config Gate (NEW) ↓                                 │
│                                                           │
│  query.ts + QueryEngine.ts (10K LOC)                    │
│  └─ Context Validation ↓                                │
│                                                           │
│  services/api/claude.ts (3.4K LOC)                      │
│  ├─ Pre-Gate: Input validation (rubric) ↓               │
│  ├─ API Call ↓                                          │
│  └─ Post-Gate: Output validation (truth-tag) ↓          │
│                                                           │
│  services/tools/toolExecution.ts (1.7K LOC)             │
│  ├─ Tool Gate: Blast-radius check ↓                     │
│  ├─ Execute Tool ↓                                      │
│  └─ Result Gate: Validation + lineage ↓                 │
│                                                           │
│  utils/messages.ts (5.5K LOC)                           │
│  └─ Message Gate: Lineage_v4 record ↓                   │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │ NEW: core/guardrail_layer.ts (Phase 2)          │    │
│  │ ├─ Rubric scorer (8-dim)                        │    │
│  │ ├─ Truth gates (affirm/refute)                  │    │
│  │ ├─ Learning bridge (→ autoDream/compact)        │    │
│  │ ├─ Cross-verifier ensemble (independent gate)   │    │
│  │ └─ Lineage_v4 coupling (atomic)                 │    │
│  └─────────────────────────────────────────────────┘    │
│                        ↓                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Memory Layers (Existing + Phase 2 Wiring)       │    │
│  │ ├─ Semantic: guardrail policy vectors           │    │
│  │ ├─ Episodic: verifier calibration history       │    │
│  │ ├─ Graph (Atlas): boundary knowledge            │    │
│  │ └─ services/autoDream, compact, extractMemories │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## Files Not Changed in Phase 1

All Phase 1 work was architectural documentation. No code was modified.

**Why**: Better to keep baseline clean and commit Phase 2 as one coherent block
(rubric scorer + truth gates + learning bridge) than fragment across Phase 1/2.

## Next: Phase 2 Immediate Start

Phase 2 build begins now:
1. `core/rubric_scorer.ts` (300 LOC, 30 min)
2. `core/truth_gates.ts` (250 LOC, 20 min)
3. `core/guardrail_learning_bridge.ts` (500 LOC, 60 min)
4. `core/cross_verifier_ensemble.ts` (400 LOC, 60 min)
5. Integration into critical paths (120 min)

**Total Phase 2**: ~5 hours (no risk to existing code)
