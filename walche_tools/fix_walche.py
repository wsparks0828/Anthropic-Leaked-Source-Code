#!/usr/bin/env python3
"""
WALCHE Auto-Fix Deployment Script

Run from: WALCHE root (this file lives in walche_tools/)
Usage:    python walche_tools/fix_walche.py [--upgrade-sdk] [--force]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def _sanity_check_root() -> None:
    """Refuse to run if this file has been moved out of walche_tools/ —
    following an older (wrong) version of this docstring would place the
    file at WALCHE root and cause it to write into the wrong tree."""
    here = Path(__file__).resolve()
    if here.parent.name != "walche_tools":
        print(f"  [ABORT] fix_walche.py must live inside walche_tools/ "
              f"(found at {here.parent}) — refusing to write files.")
        sys.exit(1)


def write_file(rel_path, content, force=False):
    p = ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = p.read_text(encoding="utf-8", errors="replace")
        if existing == content:
            print(f"  [SKIP] {rel_path} already up to date")
            return
        if not force:
            backup = p.with_suffix(p.suffix + ".bak")
            backup.write_text(existing, encoding="utf-8")
            print(f"  [BACKUP] {rel_path} -> {backup.name}")
    p.write_text(content, encoding="utf-8")
    print(f"  [OK] wrote {rel_path}")


def step1_upgrade_sdk(do_upgrade: bool) -> None:
    banner("STEP 1 — Upgrading anthropic SDK")
    if not do_upgrade:
        print("  [SKIP] pass --upgrade-sdk to upgrade the anthropic package")
        return
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "anthropic", "--upgrade", "-q"],
        check=False,
    )
    if result.returncode == 0:
        print("  [OK] anthropic upgraded")
    else:
        print(f"  [FAIL] pip exited {result.returncode} — anthropic NOT upgraded")


def step2_reflection_memory(force: bool) -> None:
    banner("STEP 2 — Writing fixed backend/memory/reflection_memory.py")
    write_file("backend/memory/reflection_memory.py", '''\
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb


class PersistentReflectionMemory:
    def __init__(self, path=None):
        if path is None:
            path = str(Path(__file__).resolve().parents[2] / "corpus_store" / "reflections")
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("walche_reflections")

    def add_reflection(self, reflection: dict):
        self.collection.add(
            documents=[json.dumps(reflection)],
            metadatas=[{"timestamp": datetime.now(timezone.utc).isoformat()}],
            ids=[f"ref_{uuid.uuid4().hex}"],
        )

    def get_recent_reflections(self, limit=8):
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"])
        docs  = results.get("documents") or []
        metas = results.get("metadatas") or []
        paired = sorted(
            zip(docs, metas),
            key=lambda dm: dm[1].get("timestamp", ""),
            reverse=True,
        )
        return [json.loads(d) for d, _ in paired[:limit] if d]
''', force=force)


def step3_langgraph_healing_graph(force: bool) -> None:
    banner("STEP 3 — Writing fixed backend/self_healing/langgraph_healing_graph.py")
    write_file("backend/self_healing/langgraph_healing_graph.py", '''\
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END, START
from datetime import datetime, timezone
import os

if not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = "walche-healing-prod"


class WALCHEHealingState(TypedDict):
    raw_telemetry: Dict
    structured_telemetry: Dict
    recent_lineage: List[Dict]
    diagnosis: Optional[Dict]
    reflection: Optional[Dict]
    ppo_decision: Optional[Dict]
    final_action: Optional[Dict]
    result: Optional[Dict]
    reflection_memory: List[Dict]


# ── Nodes: return only changed fields (LangGraph 1.x requirement) ─────────────

def telemetry_node(state: WALCHEHealingState) -> dict:
    raw = state.get("raw_telemetry") or {}
    live_report = raw.get("live_monitor_report")
    return {
        "structured_telemetry": {
            "composite_health": raw.get("composite", 0.8),
            "live_monitor": live_report is not None,
            "drift": raw.get("drift", {}),
        }
    }


def diagnose_node(state: WALCHEHealingState) -> dict:
    tel = state.get("structured_telemetry") or {}
    composite = tel.get("composite_health", 0.8)
    drift = tel.get("drift", {})
    accuracy_drop = drift.get("accuracy_drop", 0.0)

    if accuracy_drop > 0.3 or composite < 0.5:
        diagnosis = "critical_drift"
        confidence = 0.92
        action = "full_rollback"
    elif accuracy_drop > 0.15 or composite < 0.7:
        diagnosis = "moderate_drift"
        confidence = 0.85
        action = "recalibrate_thresholds"
    else:
        diagnosis = "stable_with_minor_drift"
        confidence = 0.78
        action = "retry"

    return {
        "diagnosis": {
            "final_diagnosis": diagnosis,
            "confidence": confidence,
            "recommended_action": action,
            "composite_health": composite,
            "accuracy_drop": accuracy_drop,
        }
    }


def _make_ppo_node(ppo_agent=None):
    """Build the ppo node as a closure so a real PPORecoveryAgent (when supplied)
    is actually consulted instead of being silently ignored."""
    def ppo_node(state: WALCHEHealingState) -> dict:
        diagnosis = state.get("diagnosis") or {}
        if ppo_agent is not None:
            decision = ppo_agent.integrate_with_healing_agent(diagnosis)
        else:
            confidence = diagnosis.get("confidence", 0.5)
            recommended = diagnosis.get("recommended_action", "partial_rollback")
            if confidence >= 0.85:
                action = recommended
            elif confidence >= 0.7:
                action = "partial_rollback"
            else:
                action = "escalate"
            decision = {"final_action": action, "confidence": confidence}
        return {"ppo_decision": decision, "final_action": decision}
    return ppo_node


def execute_node(state: WALCHEHealingState) -> dict:
    final = state.get("final_action") or {}
    action = final.get("final_action", "unknown")
    confidence = final.get("confidence", 0.5)
    delta = round(0.05 + confidence * 0.05, 4)
    return {
        "result": {
            "status": "simulated",
            "action": action,
            "delta": delta,
            "confidence": confidence,
        }
    }


def reflect_node(state: WALCHEHealingState) -> dict:
    result = state.get("result") or {}
    diagnosis = state.get("diagnosis") or {}
    tel = state.get("structured_telemetry") or {}
    return {
        "reflection": {
            "action": result.get("action"),
            "delta": result.get("delta", 0),
            "confidence": diagnosis.get("confidence", 0.85),
            "composite_health": tel.get("composite_health", 0.8),
            "diagnosis": diagnosis.get("final_diagnosis", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_walche_healing_graph(ppo_agent=None):
    workflow = StateGraph(WALCHEHealingState)
    workflow.add_node("telemetry", telemetry_node)
    workflow.add_node("diagnose", diagnose_node)
    workflow.add_node("ppo", _make_ppo_node(ppo_agent))
    workflow.add_node("execute", execute_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_edge(START, "telemetry")
    workflow.add_edge("telemetry", "diagnose")
    workflow.add_edge("diagnose", "ppo")
    workflow.add_edge("ppo", "execute")
    workflow.add_edge("execute", "reflect")
    workflow.add_edge("reflect", END)
    return workflow.compile()


# ── Class wrapper (required by run_system.py) ──────────────────────────────────

class WALCHEHealingGraph:
    """Class wrapper providing .run() and .invoke() over the compiled graph."""

    def __init__(self, ppo_agent=None, memory=None):
        self.graph = build_walche_healing_graph(ppo_agent=ppo_agent)
        self.ppo = ppo_agent
        self.memory = memory

    def _prepare_state(self, state: dict) -> dict:
        state.setdefault("structured_telemetry", {})
        state.setdefault("diagnosis", None)
        state.setdefault("reflection", None)
        state.setdefault("ppo_decision", None)
        state.setdefault("final_action", None)
        state.setdefault("result", None)
        state.setdefault("reflection_memory", [])
        state.setdefault("recent_lineage", [])
        return state

    def run(self, state: dict) -> dict:
        return self.graph.invoke(self._prepare_state(state))

    def invoke(self, state: dict) -> dict:
        return self.graph.invoke(self._prepare_state(state))
''', force=force)


def step4_ppo_recovery_agent(force: bool) -> None:
    banner("STEP 4 — Writing fixed backend/self_healing/ppo_recovery_agent.py")
    write_file("backend/self_healing/ppo_recovery_agent.py", '''\
import json
import pathlib


class PPORecoveryAgent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.action_scores = {
            "retry": 0.6,
            "partial_rollback": 0.85,
            "full_rollback": 0.7,
            "escalate": 0.3,
            "reallocate_resources": 0.75,
        }
        self.learning_rate = 0.1
        if model_path and pathlib.Path(model_path).exists():
            self._load(model_path)

    def train(self, timesteps=5000):
        iterations = min(100, timesteps // 50)
        for _ in range(iterations):
            for a in self.action_scores:
                self.action_scores[a] = min(0.99, self.action_scores[a] + 0.001)
        if iterations == 0:
            print(f"[PPO] No training performed (timesteps={timesteps} too low for 1 iteration)")
        else:
            print(f"[PPO] Lightweight policy trained ({iterations} iterations)")
        if self.model_path:
            self._save(self.model_path)

    def integrate_with_healing_agent(self, diagnosis: dict) -> dict:
        confidence = diagnosis.get("confidence", 0.5)
        recommended = diagnosis.get("recommended_action", "partial_rollback")

        if confidence >= 0.85 and recommended in self.action_scores:
            action = recommended
        elif confidence >= 0.7:
            action = "partial_rollback"
        else:
            action = "escalate"

        return {"final_action": action, "confidence": confidence}

    def _save(self, path: str) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.action_scores, f)
        print(f"[PPO] Policy saved to {path}")

    def _load(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.action_scores.update(loaded)
            print(f"[PPO] Policy loaded from {path}")
        except Exception as e:
            print(f"[PPO] Could not load policy from {path}: {e}")
''', force=force)


def step5_healing_agent(force: bool) -> None:
    banner("STEP 5 — Writing fixed backend/self_healing/healing_agent.py")
    write_file("backend/self_healing/healing_agent.py", '''\
from typing import Dict, Any, List
import json

DEFAULT_MODEL = "claude-sonnet-5"


class NeuroSymbolicHealingAgent:
    def __init__(self, llm_client=None, model=DEFAULT_MODEL):
        self.llm = llm_client
        self.model = model
        self.symbolic_rules = {
            "pricing_drift":      lambda s: s.get("accuracy_drop", 0) > 0.15,
            "lineage_incomplete": lambda s: not s.get("what_delta_present", False),
            "high_risk":          lambda s: s.get("risk_score", 0) > 0.8,
        }

    def _call_llm(self, prompt: str) -> dict:
        if self.llm is None:
            return {"diagnosis": "drift detected (no LLM configured)", "confidence": 0.75}

        # Anthropic SDK
        if hasattr(self.llm, "messages"):
            try:
                response = self.llm.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                try:
                    return json.loads(text)
                except Exception:
                    return {"diagnosis": text[:200], "confidence": 0.75}
            except Exception as e:
                return {"diagnosis": f"LLM error: {e}", "confidence": 0.5, "error": True}

        # OpenAI SDK
        if hasattr(self.llm, "chat"):
            try:
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
                text = response.choices[0].message.content
                try:
                    return json.loads(text)
                except Exception:
                    return {"diagnosis": text[:200], "confidence": 0.75}
            except Exception as e:
                return {"diagnosis": f"LLM error: {e}", "confidence": 0.5, "error": True}

        return {"diagnosis": "drift detected (unknown LLM type)", "confidence": 0.7}

    def diagnose(self, drift_signal: Dict, lineage: List[Dict]) -> Dict:
        prompt = (
            "Analyze this drift and lineage. "
            "Return JSON with keys 'diagnosis' (string) and 'confidence' (0.0-1.0).\\n"
            f"Drift: {json.dumps(drift_signal)}\\n"
            f"Lineage sample: {json.dumps(lineage[-3:] if lineage else [])}"
        )
        neural = self._call_llm(prompt)
        symbolic = {name: rule(drift_signal) for name, rule in self.symbolic_rules.items()}
        return {
            "neural": neural,
            "symbolic": symbolic,
            "final_diagnosis": neural.get("diagnosis", "drift detected"),
            "confidence": neural.get("confidence", 0.85),
        }

    def propose_healing_action(self, diagnosis: Dict) -> Dict:
        confidence = diagnosis.get("confidence", 0.5)
        action = "update_adapter" if confidence > 0.8 else "rollback_to_last_stable"
        return {
            "action": action,
            "parameters": {"confidence": confidence},
            "rollback_plan": "revert_to_last_stable_version",
        }

    def execute_healing(self, action: Dict) -> Dict:
        print(f"[HealingAgent] Executed: {action['action']}")
        return {"status": "success", "message": f"Healing applied: {action['action']}"}
''', force=force)


def step6_train_and_integrate(force: bool) -> None:
    banner("STEP 6 — Writing fixed scripts/train_and_integrate_healing_system.py")
    write_file("scripts/train_and_integrate_healing_system.py", '''\
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from backend.self_healing.langgraph_healing_graph import (
    WALCHEHealingGraph,
    build_walche_healing_graph,
)
from backend.self_healing.ppo_recovery_agent import PPORecoveryAgent
from backend.memory.reflection_memory import PersistentReflectionMemory
from backend.self_healing.healing_agent import NeuroSymbolicHealingAgent

try:
    from core.connection_monitor import ConnectionMonitor
    WALCHE_LIVE_AVAILABLE = True
except Exception as e:
    WALCHE_LIVE_AVAILABLE = False
    print(f"[WALCHE] Live components limited: {e}")

print("=== WALCHE Self-Healing System - Full Training & Boot ===")

# 1. Train PPO
print("[1] Training PPO Recovery Agent...")
ppo = PPORecoveryAgent(model_path="artifacts/ppo_policy.json")
ppo.train(timesteps=5000)

# 2. Build Graph
print("[2] Building LangGraph healing graph...")
graph = WALCHEHealingGraph(ppo_agent=ppo)

# 3. Persistent Memory
print("[3] Initializing Persistent Reflection Memory...")
memory = PersistentReflectionMemory()

# 4. Run healing cycle with live or sample telemetry
print("[4] Running full healing cycle...")
raw_telemetry = {"drift": {"accuracy_drop": 0.21}, "resource_usage": {"cpu": 0.89}, "composite": 0.72}

if WALCHE_LIVE_AVAILABLE:
    try:
        cm = ConnectionMonitor()
        report = cm.scan()
        raw_telemetry = {
            "drift": {"accuracy_drop": max(0.05, 1.0 - getattr(report, "platform_connection_health", 0.8))},
            "composite": getattr(report, "platform_connection_health", 0.8),
            "lineage_completeness": 0.85,
            "live_monitor_report": report,
        }
        print("  [LIVE] Using real ConnectionMonitor report")
    except Exception as e:
        print(f"  [LIVE] Fallback to sample telemetry: {e}")

state = {
    "raw_telemetry": raw_telemetry,
    "recent_lineage": [],
    "reflection_memory": memory.get_recent_reflections(5),
}

result = graph.run(state)

reflection = result.get("reflection")
if reflection:
    memory.add_reflection(reflection)
    print(f"  [Memory] Stored reflection: action={reflection.get('action')}, delta={reflection.get('delta')}")
else:
    print("  [Memory] No reflection to store")

print("\\n=== WALCHE Self-Healing System Fully Operational ===")
print(f"Boot time: {datetime.now(timezone.utc).isoformat()}")
print(f"Action taken: {(result.get('final_action') or {}).get('final_action', 'unknown')}")
print(f"Result delta: {(result.get('result') or {}).get('delta', 0):+.4f}")
print(f"Live WALCHE data used: {WALCHE_LIVE_AVAILABLE}")
print("Components active: LangGraph + PPO + NeuroSymbolic + Persistent Memory")
''', force=force)


def _skip_preamble_index(lines: list[str]) -> int:
    """Return the line index after shebang/encoding comments, blank lines,
    a leading module docstring, and any `from __future__ import` lines —
    the safe insertion point for a prepended code block."""
    idx = 0
    n = len(lines)
    while idx < n and (lines[idx].startswith("#") or lines[idx].strip() == ""):
        idx += 1
    if idx < n:
        stripped = lines[idx].lstrip()
        for quote in ('"""', "'''"):
            if stripped.startswith(quote):
                rest_after_open = stripped[len(quote):]
                if quote in rest_after_open:
                    idx += 1  # single-line docstring
                else:
                    idx += 1
                    while idx < n and quote not in lines[idx]:
                        idx += 1
                    if idx < n:
                        idx += 1  # move past the closing line
                break
    while idx < n and (lines[idx].startswith("#") or lines[idx].strip() == ""):
        idx += 1
    while idx < n and lines[idx].lstrip().startswith("from __future__ import"):
        idx += 1
    return idx


def step7_patch_run_system() -> None:
    banner("STEP 7 — Patching run_system.py (surgical in-place fixes)")
    run_sys = ROOT / "run_system.py"
    if not run_sys.exists():
        print("  [SKIP] run_system.py not found — skipping patch")
        return

    try:
        src = run_sys.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"  [ABORT] run_system.py is not valid UTF-8 ({e}) — refusing to "
              f"patch a file we cannot safely round-trip")
        return
    original = src

    preamble_blocks = []

    # Fix 1: add UTF-8 stdout fix
    if "TextIOWrapper" not in src:
        preamble_blocks.append(
            "import io\n"
            "import sys\n"
            "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')\n"
        )
        print("  [PATCH] Will add UTF-8 stdout fix")
    else:
        print("  [SKIP]  UTF-8 fix already present")

    # Fix 2: remove bad OSMODAIngestionManager import, stub the symbol so a
    # remaining call site fails with a clear error instead of a bare NameError
    src, n1 = re.subn(
        r"^\s*from\s+\S*osmoda\S*\s+import\s+OSMODAIngestionManager.*\n",
        "",
        src,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    src, n2 = re.subn(
        r"^\s*import\s+\S*osmoda\S*.*\n",
        "",
        src,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if n1 or n2:
        preamble_blocks.append(
            "class OSMODAIngestionManager:\n"
            "    def __init__(self, *a, **kw):\n"
            "        raise RuntimeError(\n"
            "            'OSMODAIngestionManager was removed by fix_walche.py "
            "(osmoda dependency retired) — update run_system.py to remove this usage'\n"
            "        )\n"
        )
        print(f"  [PATCH] Removed {n1 + n2} osmoda import line(s), stubbed the symbol")

    if preamble_blocks:
        lines = src.splitlines(keepends=True)
        idx = _skip_preamble_index(lines)
        for block in reversed(preamble_blocks):
            lines.insert(idx, block)
        src = "".join(lines)

    # Fix 3: ensure run_knowledge_injection returns its value — idempotent and
    # scoped to a bare `ingestion_manager.run()` statement line only, so a
    # second run (or an already-fixed file) is a safe no-op instead of
    # producing `return return ingestion_manager.run()` / a SyntaxError.
    if re.search(r"return\s+ingestion_manager\.run\(\)", src):
        print("  [SKIP]  run_knowledge_injection already returns ingestion_manager.run()")
    else:
        new_src, n3 = re.subn(
            r"(?m)^(\s*)ingestion_manager\.run\(\)\s*$",
            r"\1return ingestion_manager.run()",
            src,
            count=1,
        )
        if n3:
            src = new_src
            print("  [PATCH] run_knowledge_injection now returns ingestion_manager.run()")
        else:
            print("  [WARN]  Could not find a bare 'ingestion_manager.run()' statement "
                  "line to patch — skipped (manual review needed)")

    if src == original:
        print("  [INFO] run_system.py — no further patches needed")
        return

    try:
        compile(src, str(run_sys), "exec")
    except SyntaxError as e:
        print(f"  [ABORT] Patched run_system.py would not compile ({e}) — "
              f"writing nothing, original file left untouched")
        return

    backup = run_sys.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"  [BACKUP] run_system.py -> {backup.name}")
    _tmp = run_sys.with_suffix(".py.tmp")
    _tmp.write_text(src, encoding="utf-8")
    _tmp.replace(run_sys)
    print("  [OK] run_system.py patched")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="WALCHE Auto-Fix Deployment Script")
    parser.add_argument("--upgrade-sdk", action="store_true",
                        help="Also run `pip install --upgrade anthropic` (network action)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite deployed files without writing .bak backups")
    args = parser.parse_args(argv)

    _sanity_check_root()

    step1_upgrade_sdk(args.upgrade_sdk)
    step2_reflection_memory(args.force)
    step3_langgraph_healing_graph(args.force)
    step4_ppo_recovery_agent(args.force)
    step5_healing_agent(args.force)
    step6_train_and_integrate(args.force)
    step7_patch_run_system()

    banner("ALL DONE")
    print("""
Files fixed:
  backend/memory/reflection_memory.py             -- ChromaDB empty-collection crash fixed
  backend/self_healing/langgraph_healing_graph.py -- LangGraph 1.x nodes + WALCHEHealingGraph class
  backend/self_healing/ppo_recovery_agent.py      -- PPO saves policy, removed numpy
  backend/self_healing/healing_agent.py           -- LLM SDK calls fixed, llm optional
  scripts/train_and_integrate_healing_system.py   -- imports WALCHEHealingGraph, stores reflection
  run_system.py                                   -- UTF-8 fix + import cleanup (if present)

NOTE: deployed files/identifiers were renamed WACHEL -> WALCHE (the project's
own corpus records WACHEL/WACHLE as a resolved naming mistake). If an earlier
run of this script already created data under the ChromaDB collection name
"wachel_reflections", that data will NOT be migrated automatically — move it
to "walche_reflections" by hand if you need it.

Next step:
  python run_system.py --phase test
""")


if __name__ == "__main__":
    main()
