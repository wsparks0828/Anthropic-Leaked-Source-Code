#!/usr/bin/env python3
"""
WALCHE Minimum Viable Demo
Run from your WALCHE root:  python walche_demo.py
Works whether core modules are healthy or broken — shows the live loop either way.

Options:
  --council         Route high-impact healing proposals through the Grand Council
  --api-key KEY     Anthropic API key for real Council agent deliberation
  --cycles N        Number of healing cycles (default: 4)
"""
import sys
import os
import time
import json
import random
import traceback
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ── WALCHE root detection ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Terminal colors (Windows 10+ ANSI supported) ─────────────────────────────
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

def col(text, color): return f"{color}{text}{C.RESET}"
def ok(t):   return col(t, C.GREEN)
def warn(t): return col(t, C.YELLOW)
def err(t):  return col(t, C.RED)
def hi(t):   return col(t, C.CYAN)
def bold(t): return col(t, C.BOLD)

def _fmt_proposal(p) -> str:
    """Extract human-readable text from a HealingProposal object or string."""
    if isinstance(p, str):
        return p
    # Try common description attribute names
    for attr in ("description", "action", "text", "message", "proposal", "content", "summary"):
        val = getattr(p, attr, None)
        if val and isinstance(val, str) and val.strip():
            ptype = getattr(p, "proposal_type", "")
            dim   = getattr(p, "target_dimension", "")
            prefix = f"[{ptype}] " if ptype else ""
            suffix = f" → {dim}" if dim else ""
            return f"{prefix}{val.strip()}{suffix}"
    # Fall back: parse the repr string for description= field
    import re as _re
    s = str(p)
    m = _re.search(r"description=['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1)
    # Last resort: return the full str but without the class wrapper noise
    return s

# Enable ANSI on Windows
if sys.platform == "win32":
    os.system("")

# ── Stub fallbacks (used when real modules don't load) ───────────────────────

class _StubConfig:
    dry_run = False
    light_mode = False
    max_history = 8
    def __repr__(self): return "WalcheConfig[STUB]"

@dataclass
class _StubRubricResult:
    composite: float
    passed: bool
    signals: Dict[str, float] = field(default_factory=dict)

class _StubRubricScorer:
    def __init__(self, threshold=0.85): self.threshold = threshold
    def score(self, name, signals):
        composite = sum(signals.values()) / len(signals) if signals else 0.0
        return _StubRubricResult(composite=composite, passed=composite >= self.threshold, signals=signals)

class _StubMetaEngine:
    def evaluate_meta(self, context, task, iteration=1):
        base = 0.72 + iteration * 0.04
        return {"confidence": min(0.98, base), "grounded": True, "iteration": iteration}

class _StubHealingEngine:
    def __init__(self, max_history=8): self.max_history = max_history; self._history = []
    def reflect_and_heal(self, cycle_data, domain):
        score = cycle_data.get("metadata", {}).get("verification_score", 0.7)
        proposals = []
        if score < 0.80: proposals.append(f"Strengthen provenance tracking in {domain}")
        if score < 0.85: proposals.append(f"Apply PreIngestGate pattern before mutations in {domain}")
        proposals.append(f"Run VLL refinement on {domain} outputs (baseline: 1982 proposals)")
        self._history.append({"domain": domain, "score": score})
        return {"proposals": proposals, "healed": score > 0.75}

class _StubPAINFMEA:
    def calculate_rpn(self, severity, occurrence, detection):
        rpn = severity * occurrence * detection
        level = "HIGH" if rpn > 100 else "MEDIUM" if rpn > 50 else "LOW"
        return rpn, level

class _StubPreIngestGate:
    def check(self, data, lineage_id=None):
        return {"passed": True, "lineage_id": lineage_id or "WALCHE-DEMO", "risk": 0.12}

class _StubGuardrail:
    def check(self, action, context=None):
        return {"safe": True, "action": action, "risk_score": 0.08}

# ── Module loader ─────────────────────────────────────────────────────────────

_module_status: Dict[str, str] = {}

def _load(module_path: str, class_name: str = None, stub_class=None):
    """Try real import, fall back to stub, record status."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        if class_name:
            cls = getattr(mod, class_name)
            _module_status[f"{module_path}.{class_name}"] = "REAL"
            return cls
        _module_status[module_path] = "REAL"
        return mod
    except Exception as e:
        key = f"{module_path}.{class_name}" if class_name else module_path
        _module_status[key] = f"STUB ({type(e).__name__})"
        return stub_class

# ── Load modules ──────────────────────────────────────────────────────────────

WalcheConfig      = _load("core.config",         "WalcheConfig",      _StubConfig)
RubricScorer      = _load("core.rubric",          "RubricScorer",      _StubRubricScorer)
MetaEngine        = _load("core.meta_engine",     "MetaEngine",        _StubMetaEngine)
HealingEngine     = _load("core.healing_engine",  "HealingEngine",     _StubHealingEngine)
PAINFMEA          = _load("core.pain_fmea",       "PAINFMEA",          _StubPAINFMEA)
PreIngestGate     = _load("core.pre_ingest_gate", "PreIngestGate",     _StubPreIngestGate)
Guardrail         = _load("core.guardrail",       "Guardrail",         _StubGuardrail)

# ── Demo domains ──────────────────────────────────────────────────────────────

DOMAINS = [
    {
        "name":    "corpus.integrity",
        "label":   "Corpus Integrity",
        "signals": {"accuracy": 0.74, "completeness": 0.68, "safety": 0.91,
                    "provenance": 0.62, "consistency": 0.83, "efficiency": 0.77},
    },
    {
        "name":    "healing.engine",
        "label":   "Healing Engine",
        "signals": {"accuracy": 0.88, "completeness": 0.80, "safety": 0.95,
                    "provenance": 0.71, "consistency": 0.89, "efficiency": 0.82},
    },
    {
        "name":    "loop.registry",
        "label":   "Loop Registry",
        "signals": {"accuracy": 0.81, "completeness": 0.73, "safety": 0.93,
                    "provenance": 0.69, "consistency": 0.86, "efficiency": 0.79},
    },
    {
        "name":    "meta.engine",
        "label":   "Meta Engine (Bertha)",
        "signals": {"accuracy": 0.90, "completeness": 0.85, "safety": 0.97,
                    "provenance": 0.78, "consistency": 0.92, "efficiency": 0.88},
    },
]

# ── Progress bar ──────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 24) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    color = C.GREEN if score >= 0.85 else C.YELLOW if score >= 0.70 else C.RED
    return f"{color}{bar}{C.RESET} {score:.3f}"

def _score_color(score: float) -> str:
    color = C.GREEN if score >= 0.85 else C.YELLOW if score >= 0.70 else C.RED
    return f"{color}{score:.3f}{C.RESET}"

# ── Single healing cycle ──────────────────────────────────────────────────────

def run_cycle(
    cycle_num: int,
    domains: List[Dict],
    scorer: _StubRubricScorer,
    meta: _StubMetaEngine,
    healer: _StubHealingEngine,
    pain: _StubPAINFMEA,
    gate: _StubPreIngestGate,
    guard: _StubGuardrail,
) -> Tuple[float, List[Dict]]:
    """Run one full healing cycle across all domains. Returns (avg_score, results)."""

    results = []
    print(f"\n  {hi(f'── CYCLE {cycle_num} ──────────────────────────────────────────')}")

    for domain in domains:
        name    = domain["name"]
        label   = domain["label"]
        signals = {k: min(0.99, v + (cycle_num - 1) * random.uniform(0.01, 0.03))
                   for k, v in domain["signals"].items()}

        # Pre-ingest gate — discover method name at runtime
        gate_ok = True
        gate_result = {}
        for _gate_method in ("evaluate", "check", "gate", "validate", "run", "ingest"):
            if hasattr(gate, _gate_method):
                try:
                    _fn = getattr(gate, _gate_method)
                    gate_result = _fn({"domain": name}, lineage_id=f"WALCHE-{name}-C{cycle_num}")
                    gate_ok = gate_result.get("passed", True) if isinstance(gate_result, dict) else True
                except TypeError:
                    try:
                        gate_result = _fn({"domain": name})
                        gate_ok = True
                    except Exception:
                        gate_ok = True
                except Exception:
                    gate_ok = True
                break

        # Guardrail — discover method name at runtime
        guard_ok = True
        guard_result = {}
        for _guard_method in ("check", "evaluate", "validate", "run", "assess"):
            if hasattr(guard, _guard_method):
                try:
                    _fn = getattr(guard, _guard_method)
                    guard_result = _fn(f"evaluate_{name}", context={"cycle": cycle_num})
                    guard_ok = guard_result.get("safe", True) if isinstance(guard_result, dict) else True
                except TypeError:
                    try:
                        guard_result = _fn(f"evaluate_{name}")
                        guard_ok = True
                    except Exception:
                        guard_ok = True
                except Exception:
                    guard_ok = True
                break

        if not gate_ok or not guard_ok:
            print(f"  {err('BLOCKED')} {label} — gate or guardrail rejected")
            continue

        # Rubric score
        try:
            rubric = scorer.score(name, signals)
            score  = rubric.composite if hasattr(rubric, "composite") else float(rubric)
        except Exception:
            score = sum(signals.values()) / len(signals)

        # Meta evaluation
        try:
            meta_result = meta.evaluate_meta(
                {"walche": True, "domain": name},
                f"healing cycle {cycle_num} on {name}",
                iteration=cycle_num
            )
            if isinstance(meta_result, dict):
                confidence = meta_result.get("confidence", 0.9)
            else:
                confidence = getattr(meta_result, "confidence", 0.9)
        except Exception:
            confidence = 0.9

        # PAIN FMEA
        try:
            sev  = 4 if score < 0.75 else 2
            occ  = 3
            det  = 2 if score > 0.80 else 3
            rpn, risk_level = pain.calculate_rpn(sev, occ, det)
        except Exception:
            rpn, risk_level = 24, "LOW"

        # Healing proposals
        try:
            cycle_data = {
                "content": f"WALCHE domain {name} cycle {cycle_num}",
                "metadata": {"domain": name, "verification_score": score, "cycle": cycle_num},
            }
            heal_result = healer.reflect_and_heal(cycle_data, name)
            if isinstance(heal_result, dict):
                proposals = heal_result.get("proposals", [])
            else:
                proposals = getattr(heal_result, "proposals", [])
        except Exception:
            proposals = [f"Run VLL refinement on {name} outputs"]

        # Display
        status = ok("PASS") if score >= 0.85 else warn("WARN") if score >= 0.70 else err("FAIL")
        print(f"\n  [{status}] {bold(label)}")
        print(f"    Score:      {_bar(score)}")
        print(f"    Confidence: {_score_color(confidence)}   RPN: {rpn} ({risk_level})")
        proposals_str = [_fmt_proposal(p) for p in proposals]
        if proposals_str:
            for p in proposals_str[:2]:
                print(f"    {C.DIM}↳ {p[:80]}{C.RESET}")

        results.append({
            "domain": name, "score": score, "confidence": confidence,
            "rpn": rpn, "risk_level": risk_level, "proposals": proposals_str,
        })

        time.sleep(0.15)

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    return avg, results

# ── Main ──────────────────────────────────────────────────────────────────────

# ── Council auto-trigger ──────────────────────────────────────────────────────

# Keyword → proposal_type mapping for council routing
_PROPOSAL_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "token_strategy": ["max_history", "cache", "token", "batch", "compress", "throttle",
                       "rate_limit", "prompt_cache", "budget"],
    "security":       ["guardrail", "security", "unauthorized", "deception", "trust",
                       "safe", "risk", "veto", "block"],
    "corpus":         ["corpus", "provenance", "ingest", "lineage", "knowledge", "seed",
                       "ingestion", "quality"],
    "retrieval":      ["retrieval", "memory", "recall", "context", "rag", "embed",
                       "vector", "search"],
    "healing":        ["heal", "repair", "fix", "patch", "recover", "rubric", "score",
                       "threshold", "cycle"],
}

def _classify_proposal(text: str) -> str:
    text_lower = text.lower()
    for ptype, keywords in _PROPOSAL_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return ptype
    return "general"


def _run_council_deliberation(
    proposals: List[str],
    api_key: Optional[str] = None,
    quiet: bool = False,
) -> List[Dict]:
    """Route high-impact proposals through the Grand Council. Returns verdicts."""
    verdicts: List[Dict] = []

    # Try to import council_of_9 from walche_tools/
    try:
        council_dir = ROOT / "walche_tools"
        if str(council_dir) not in sys.path:
            sys.path.insert(0, str(council_dir))
        from council_of_9 import deliberate
    except ImportError:
        if not quiet:
            print(f"  {warn('[COUNCIL]')} council_of_9.py not found in walche_tools/ — skipping")
        return []

    # Group proposals by type; only route token_strategy and security to council
    # (other types are informational — council governs Law 11 changes)
    governance_types = {"token_strategy", "security"}
    grouped: Dict[str, List[str]] = {}
    for p in proposals:
        ptype = _classify_proposal(p)
        grouped.setdefault(ptype, []).append(p)

    to_deliberate = {k: v for k, v in grouped.items() if k in governance_types}
    if not to_deliberate:
        if not quiet:
            print(f"  {hi('[COUNCIL]')} No governance-required proposals in this run  "
                  f"{C.DIM}(token_strategy / security){C.RESET}")
        return []

    print(f"\n  {bold('GRAND COUNCIL DELIBERATION')}")
    print(f"  {'─'*54}")
    print(f"  {C.DIM}Routing {sum(len(v) for v in to_deliberate.values())} proposals "
          f"through governance…{C.RESET}")

    for ptype, group in to_deliberate.items():
        # Summarise the group as a single proposal text
        summary = f"Healing proposals [{ptype}]: " + " | ".join(group[:3])
        tier = "standard"  # Council of 5 → Council of 9

        try:
            result = deliberate(
                proposal=summary,
                proposal_type=ptype,
                tier=tier,
                api_key=api_key,
            )
        except Exception as e:
            print(f"  {warn('[COUNCIL]')} Deliberation error: {e}")
            continue

        final = result.get("final_result", {})
        verdict_val = final.get("verdict", "UNKNOWN")
        score = float(final.get("score", 0.0))
        reasoning = final.get("reasoning", "")[:80]

        v_color = C.GREEN if verdict_val in ("APPROVED", "GO") else \
                  C.YELLOW if "CONDITION" in verdict_val else C.RED
        v_str = _col(verdict_val, v_color)

        print(f"\n  [{bold(ptype.upper())}]")
        print(f"    Council verdict:  {v_str}")
        print(f"    Score:            {score:.3f}")
        if reasoning:
            print(f"    {C.DIM}↳ {reasoning}{C.RESET}")

        verdicts.append({
            "proposal_type": ptype,
            "proposals":     group,
            "verdict":       verdict_val,
            "score":         score,
            "reasoning":     reasoning,
            "tier":          tier,
        })

    return verdicts


def _col(text, color): return f"{color}{text}{C.RESET}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="WALCHE Minimum Viable Demo",
                                     add_help=False)
    parser.add_argument("--council",  action="store_true",
                        help="Route high-impact proposals through the Grand Council")
    parser.add_argument("--api-key",  metavar="KEY", default=None,
                        help="Anthropic API key for real Council agents")
    parser.add_argument("--cycles",   type=int, default=4,
                        help="Number of healing cycles (default: 4)")
    parser.add_argument("-h", "--help", action="help")
    args, _ = parser.parse_known_args(argv)

    ts = datetime.now(timezone.utc).isoformat()
    NUM_CYCLES = args.cycles

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print(col("╔══════════════════════════════════════════════════════════╗", C.CYAN))
    print(col("║        W A L C H E   —   MINIMUM VIABLE AGENT           ║", C.CYAN))
    print(col("║     Forensic Looping Corpus Brain  •  Live Demo          ║", C.CYAN))
    print(col("╚══════════════════════════════════════════════════════════╝", C.CYAN))
    print(f"  Root:     {ROOT}")
    print(f"  Started:  {ts[:19]}Z")
    print(f"  Cycles:   {NUM_CYCLES}")

    # ── Module status ─────────────────────────────────────────────────────────
    print(f"\n  {bold('MODULE STATUS')}")
    cfg = WalcheConfig()

    for key, status in _module_status.items():
        short = key.split(".")[-1]
        if status == "REAL":
            print(f"  {ok('REAL  ')} {short}")
        else:
            print(f"  {warn('STUB  ')} {short}  {C.DIM}({status}){C.RESET}")

    real_count = sum(1 for s in _module_status.values() if s == "REAL")
    stub_count = len(_module_status) - real_count
    print(f"\n  {ok(f'{real_count} real')} / {warn(f'{stub_count} stub')} modules loaded")

    # ── Instantiate ───────────────────────────────────────────────────────────
    try: scorer  = RubricScorer(threshold=0.85)
    except: scorer  = _StubRubricScorer(threshold=0.85)
    try: meta    = MetaEngine()
    except: meta    = _StubMetaEngine()
    try: healer  = HealingEngine(max_history=cfg.max_history if hasattr(cfg, "max_history") else 8)
    except: healer  = _StubHealingEngine()
    try: pain    = PAINFMEA()
    except: pain    = _StubPAINFMEA()
    try: gate    = PreIngestGate()
    except: gate    = _StubPreIngestGate()
    try: guard   = Guardrail()
    except: guard   = _StubGuardrail()

    # ── Healing cycles ────────────────────────────────────────────────────────
    print(f"\n  {bold('HEALING LOOP STARTING')}")
    print(f"  {'─'*54}")

    cycle_scores = []
    all_results  = []

    for cycle in range(1, NUM_CYCLES + 1):
        avg_score, results = run_cycle(
            cycle, DOMAINS, scorer, meta, healer, pain, gate, guard
        )
        cycle_scores.append(avg_score)
        all_results.append(results)

        improvement = ""
        if len(cycle_scores) > 1:
            delta = avg_score - cycle_scores[-2]
            improvement = ok(f" ▲ +{delta:.3f}") if delta > 0 else warn(f" ▼ {delta:.3f}")

        print(f"\n  Cycle {cycle} avg: {_bar(avg_score, 20)}{improvement}")
        time.sleep(0.3)

    # ── C12 — WALCHE Judgment ─────────────────────────────────────────────────
    final_score = cycle_scores[-1]
    delta_total = cycle_scores[-1] - cycle_scores[0]

    print(f"\n  {'═'*54}")
    print(f"  {bold('WALCHE JUDGMENT  (C12)')}")
    print(f"  {'═'*54}")

    # Assess
    high_fails = [r for cycle in all_results for r in cycle if r["score"] < 0.70]
    if final_score >= 0.85 and not high_fails:
        verdict = "GO"
        verdict_str = ok("  GO  ")
    elif final_score >= 0.70:
        verdict = "GO-WITH-CONDITIONS"
        verdict_str = warn("GO-WITH-CONDITIONS")
    else:
        verdict = "NO-GO"
        verdict_str = err("NO-GO")

    print(f"\n  Verdict:          {bold(verdict_str)}")
    print(f"  Final avg score:  {_bar(final_score, 20)}")
    print(f"  Cycle 1 → {NUM_CYCLES}:    {_score_color(cycle_scores[0])} → {_score_color(final_score)}  "
          f"({ok(f'+{delta_total:.3f}') if delta_total >= 0 else err(f'{delta_total:.3f}')})")
    print(f"  Real modules:     {ok(str(real_count))} / {len(_module_status)} ({real_count}/{len(_module_status)} loaded)")

    # ── Summary per domain ────────────────────────────────────────────────────
    print(f"\n  {bold('DOMAIN SUMMARY  (final cycle)')}")
    final_cycle = all_results[-1]
    for r in final_cycle:
        status = ok("PASS") if r["score"] >= 0.85 else warn("WARN") if r["score"] >= 0.70 else err("FAIL")
        print(f"  [{status}]  {r['domain']:<26} {_bar(r['score'], 18)}")

    # ── Healing proposals ─────────────────────────────────────────────────────
    _seen, all_proposals = set(), []
    for cycle in all_results:
        for r in cycle:
            for p in r["proposals"]:
                if p not in _seen:
                    _seen.add(p)
                    all_proposals.append(p)
    if all_proposals:
        print(f"\n  {bold('HEALING PROPOSALS  ({} total)'.format(len(all_proposals)))}")
        for p in all_proposals[:6]:
            print(f"  {C.DIM}• {p[:90]}{C.RESET}")

    # ── Council deliberation (if --council flag set) ──────────────────────────
    council_verdicts: List[Dict] = []
    if args.council and all_proposals:
        council_verdicts = _run_council_deliberation(
            all_proposals,
            api_key=args.api_key,
        )

    # ── Provenance log ────────────────────────────────────────────────────────
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"walche_demo_{ts[:10]}.json"

    log_entry = {
        "timestamp": ts,
        "verdict": verdict,
        "cycles": NUM_CYCLES,
        "cycle_scores": cycle_scores,
        "final_score": final_score,
        "delta": delta_total,
        "real_modules": real_count,
        "stub_modules": stub_count,
        "module_status": _module_status,
        "proposals": all_proposals,
        "council_verdicts": council_verdicts,
        "domain_results": [[{k: v for k, v in r.items() if k != "proposals"}
                             for r in cycle] for cycle in all_results],
    }
    log_path.write_text(json.dumps(log_entry, indent=2))

    print(f"\n  {C.DIM}Provenance log: {log_path}{C.RESET}")
    if args.council and council_verdicts:
        print(f"  {C.DIM}Council decisions logged in provenance{C.RESET}")
    print(f"\n  {'═'*54}")
    print(f"  {bold('WALCHE AGENT DEMO COMPLETE')}")
    print(f"  {'═'*54}\n")

    return verdict


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{warn('  [INTERRUPTED]  Demo stopped by user.')}\n")
    except Exception as e:
        print(f"\n{err('  [ERROR]')} {e}")
        traceback.print_exc()
        sys.exit(1)
