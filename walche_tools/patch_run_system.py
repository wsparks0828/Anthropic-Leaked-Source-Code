#!/usr/bin/env python3
"""
patch_run_system.py — Wire the WALCHE Forensic Healing Loop into run_system.py

Run from your WALCHE root:
    python walche_tools\patch_run_system.py

What it does:
  1. Adds PAINFMEA import
  2. Adds compact display helpers for the healing loop
  3. Adds run_healing_loop() as a proper WALCHE phase function
  4. Adds --phase heal to argparse
  5. Wires the heal phase into main() after dream

After patching:
    python run_system.py --phase heal          # healing loop only
    python run_system.py --phase full          # full pipeline including heal
    python run_system.py --phase heal --live   # live mode
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "run_system.py"

if not TARGET.exists():
    print(f"[ERROR] run_system.py not found at {TARGET}")
    print("        Run this script from the WALCHE root directory.")
    sys.exit(1)

src = TARGET.read_text(encoding="utf-8")

# ── Guard: already patched? ───────────────────────────────────────────────────
# Check for the function DEFINITION specifically, not the bare substring —
# a prior partial patch can leave a call site referencing run_healing_loop
# with no definition, and the old substring check treated that broken state
# as "already patched", permanently blocking re-patching.
if "def run_healing_loop(" in src:
    print("[INFO] run_system.py already contains run_healing_loop — nothing to do.")
    sys.exit(0)

# Track whether each prerequisite step for run_healing_loop() actually
# succeeded — steps 1/2/3 must all succeed before step 5 wires a call site
# into main(), or a failed step 3 leaves a NameError call site with no
# definition (and the guard above would then treat that broken file as
# already-patched, permanently blocking a fix).
step1_ok = step2_ok = step3_ok = False

# ── 1. Add PAINFMEA import ────────────────────────────────────────────────────
OLD_IMPORT = "from core.healing_engine import HealingEngine"
NEW_IMPORT = (
    "from core.healing_engine import HealingEngine\n"
    "from core.pain_fmea import PAINFMEA"
)
if "from core.pain_fmea import PAINFMEA" in src:
    step1_ok = True
elif OLD_IMPORT in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    step1_ok = True
    print("[PATCH] Added PAINFMEA import")
else:
    print("[WARN] Could not find HealingEngine import anchor — PAINFMEA import skipped")

# ── 2. Add display helpers (inserted right before BANNER) ─────────────────────
DISPLAY_HELPERS = '''
# ── Healing loop display helpers ─────────────────────────────────────────────
import sys
import os as _os
if sys.platform == "win32":
    _os.system("")
_HC = {"G": "\\033[92m", "Y": "\\033[93m", "R": "\\033[91m",
       "B": "\\033[1m",  "D": "\\033[2m",  "X": "\\033[0m"}

def _hcol(t, c):  return f"{_HC[c]}{t}{_HC['X']}"
def _hbar(v, w=24): n = int(v * w); return "█" * n + "░" * (w - n) + f" {v:.3f}"
def _hsc(v):  return _hcol(f"{v:.3f}", "G" if v >= 0.85 else "Y" if v >= 0.70 else "R")

def _hfp(p) -> str:
    """Extract readable text from a HealingProposal object or string."""
    if isinstance(p, str):
        return p
    for attr in ("description", "action", "text", "message", "summary", "content"):
        val = getattr(p, attr, None)
        if val and isinstance(val, str) and val.strip():
            pt = getattr(p, "proposal_type", "")
            return f"[{pt}] {val.strip()}" if pt else val.strip()
    import re as _re
    m = _re.search(r"description=['\\"](.*?)['\\"](,|\\))", str(p))
    if m:
        return m.group(1)
    return str(p)

'''

BANNER_ANCHOR = 'BANNER = """'
if "_HC = {" in src:
    step2_ok = True
elif BANNER_ANCHOR in src:
    src = src.replace(BANNER_ANCHOR, DISPLAY_HELPERS + BANNER_ANCHOR, 1)
    step2_ok = True
    print("[PATCH] Added display helpers")
else:
    print("[WARN] Could not find BANNER anchor — display helpers skipped")

# ── 3. Add run_healing_loop() function ────────────────────────────────────────
HEALING_LOOP_FN = '''

def run_healing_loop(cfg: WalcheConfig, num_cycles: int = 4) -> dict:
    """Phase: Forensic Healing Loop — multi-cycle rubric scoring, PAIN FMEA, and healing proposals (C12 judgment)."""
    print("\\n[PHASE] Forensic Healing Loop")
    print("[AI_SELF_AUDIT] Running mandatory self-audit before healing loop...")
    try:
        audit = ai_self_audit("healing_loop")
    except Exception as e:
        print(f"[AI_SELF_AUDIT] ai_self_audit() unavailable or failed: {e}")
        audit = {"overall_residual": "unknown"}
    print(f"[AI_SELF_AUDIT] Residual risk: {audit.get('overall_residual', 'unknown')}")

    DOMAINS = [
        ("corpus.integrity", "Corpus Integrity"),
        ("healing.engine",   "Healing Engine"),
        ("loop.registry",    "Loop Registry"),
        ("meta.engine",      "Meta Engine (Bertha)"),
    ]

    try:
        scorer = RubricScorer()
        meta   = MetaEngine(history_size=getattr(cfg, "max_history", 10))
        healer = HealingEngine(max_history=getattr(cfg, "max_history", 5))
        pain   = PAINFMEA()
        gate   = PreIngestGate()
        guard  = Guardrail()
        prov   = ProvenanceLog(persist_path=cfg.provenance_log)
    except Exception as e:
        print(f"[HealingLoop] Module init error: {e}")
        return {"error": str(e), "audit": audit}

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║        W A L C H E   —   FORENSIC HEALING LOOP          ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Cycles: {num_cycles}  |  Domains: {len(DOMAINS)}")
    print("  ──────────────────────────────────────────────────────────")

    all_results  = []
    cycle_scores = []
    _throttle    = cfg.effective_throttle() if callable(getattr(cfg, "effective_throttle", None)) else 0.05

    errors: list = []

    for cycle_num in range(1, num_cycles + 1):
        print(f"\\n  ── CYCLE {cycle_num} {'─' * 44}")
        cycle_results = []

        for name, label in DOMAINS:
            degraded = False

            # ── Rubric score ──────────────────────────────────────────────────
            # PROBE_SIGNALS are a fixed placeholder payload, not a live
            # measurement of this domain — real per-domain instrumentation
            # is not wired up here. All-zero signals previously guaranteed
            # a degenerate composite; these at least exercise the scorer
            # with plausible values, matching walche_demo.py's DOMAINS.
            PROBE_SIGNALS = {
                "accuracy": 0.80, "completeness": 0.75, "consistency": 0.85,
                "safety": 0.90, "provenance": 0.70, "efficiency": 0.80,
                "adaptability": 0.75, "clarity": 0.80,
            }
            try:
                rr    = scorer.score(name, dict(PROBE_SIGNALS))
                score = float(rr.composite) if hasattr(rr, "composite") else float(rr)
            except Exception as e:
                score = 0.75
                degraded = True
                errors.append(f"cycle{cycle_num}:{name}:scorer:{e}")

            # ── MetaEngine update ─────────────────────────────────────────────
            try:
                meta.update(name, score, {})
            except Exception as e:
                degraded = True
                errors.append(f"cycle{cycle_num}:{name}:meta:{e}")

            # ── PAIN FMEA ─────────────────────────────────────────────────────
            try:
                fmea = pain.assess({"domain": name, "score": score, "cycle": cycle_num})
                if isinstance(fmea, dict):
                    rpn, risk_level, confidence = (
                        fmea.get("rpn", 18),
                        fmea.get("risk_level", "LOW"),
                        float(fmea.get("confidence", 0.9)),
                    )
                else:
                    rpn, risk_level, confidence = getattr(fmea, "rpn", 18), getattr(fmea, "risk_level", "LOW"), 0.9
            except Exception as e:
                rpn, risk_level, confidence = 18, "LOW", 0.9
                degraded = True
                errors.append(f"cycle{cycle_num}:{name}:pain_fmea:{e}")

            # ── PreIngestGate ─────────────────────────────────────────────────
            try:
                for _m in ("evaluate", "check", "gate", "validate", "run"):
                    if hasattr(gate, _m):
                        _fn = getattr(gate, _m)
                        try:
                            _fn({"domain": name}, lineage_id=f"WALCHE-{name}-C{cycle_num}")
                        except TypeError:
                            try: _fn({"domain": name})
                            except Exception as e: degraded = True; errors.append(f"cycle{cycle_num}:{name}:gate:{e}")
                        except Exception as e:
                            degraded = True; errors.append(f"cycle{cycle_num}:{name}:gate:{e}")
                        break
            except Exception as e:
                degraded = True; errors.append(f"cycle{cycle_num}:{name}:gate:{e}")

            # ── Guardrail ─────────────────────────────────────────────────────
            try:
                for _m in ("evaluate", "check", "guard", "run"):
                    if hasattr(guard, _m):
                        _fn = getattr(guard, _m)
                        try:
                            _fn({"domain": name, "score": score})
                        except TypeError:
                            try: _fn({"domain": name})
                            except Exception as e: degraded = True; errors.append(f"cycle{cycle_num}:{name}:guard:{e}")
                        except Exception as e:
                            degraded = True; errors.append(f"cycle{cycle_num}:{name}:guard:{e}")
                        break
            except Exception as e:
                degraded = True; errors.append(f"cycle{cycle_num}:{name}:guard:{e}")

            # ── Healing proposals ─────────────────────────────────────────────
            try:
                cycle_data = {
                    "rubric_scores": {name: score},
                    "metadata": {"domain": name, "verification_score": score, "cycle": cycle_num},
                }
                hr = healer.reflect_and_heal(cycle_data, name)
                proposals = hr.get("proposals", []) if isinstance(hr, dict) else getattr(hr, "proposals", [])
            except Exception as e:
                proposals = []
                degraded = True
                errors.append(f"cycle{cycle_num}:{name}:healer:{e}")

            proposals_str = [_hfp(p) for p in proposals]

            # ── Display ───────────────────────────────────────────────────────
            status_key = "PASS" if score >= 0.85 else "WARN" if score >= 0.70 else "FAIL"
            status_col = _hcol(f"[{status_key}]", "G" if status_key == "PASS" else "Y" if status_key == "WARN" else "R")
            degraded_tag = f"  {_hcol('[DEGRADED]', 'R')}" if degraded else ""
            print(f"\\n  {status_col} {_hcol(label, 'B')}{degraded_tag}")
            print(f"    Score:      {_hbar(score)}")
            print(f"    Confidence: {_hsc(confidence)}   RPN: {rpn} ({risk_level})")
            for p in proposals_str[:2]:
                print(f"    {_HC['D']}↳ {p[:80]}{_HC['X']}")

            cycle_results.append({
                "domain":     name,
                "score":      score,
                "confidence": confidence,
                "rpn":        rpn,
                "risk_level": risk_level,
                "proposals":  proposals_str,
                "degraded":   degraded,
            })
            time.sleep(_throttle)

        avg = sum(r["score"] for r in cycle_results) / len(cycle_results)
        cycle_scores.append(avg)
        all_results.append(cycle_results)

        if len(cycle_scores) > 1:
            d = avg - cycle_scores[-2]
            delta_str = f" {'▲' if d >= 0 else '▼'} {d:+.3f}"
        else:
            delta_str = ""
        print(f"\\n  Cycle {cycle_num} avg: {_hbar(avg, 20)}{delta_str}")

    # ── C12 Judgment ──────────────────────────────────────────────────────────
    final_score = cycle_scores[-1]
    delta_total = cycle_scores[-1] - cycle_scores[0]
    final = all_results[-1]
    any_fail  = any(r["score"] < 0.70 for r in final)
    all_pass  = all(r["score"] >= 0.85 for r in final)
    verdict   = "NO-GO" if any_fail else "GO" if all_pass else "GO-WITH-CONDITIONS"
    v_col     = _hcol(verdict, "G" if verdict == "GO" else "Y" if "CONDITIONS" in verdict else "R")

    print(f"\\n  {'═' * 54}")
    print(f"  {_hcol('WALCHE JUDGMENT  (C12)', 'B')}")
    print(f"  {'═' * 54}\\n")
    print(f"  Verdict:            {v_col}")
    print(f"  Final avg score:  {_hbar(final_score, 20)}")
    print(f"  Cycle 1 → {num_cycles}:    {cycle_scores[0]:.3f} → {final_score:.3f}  ({delta_total:+.3f})")
    print()
    print(f"  {_hcol('DOMAIN SUMMARY  (final cycle)', 'B')}")
    for r in final:
        sk  = "PASS" if r["score"] >= 0.85 else "WARN" if r["score"] >= 0.70 else "FAIL"
        sc  = _hcol(f"[{sk}]", "G" if sk == "PASS" else "Y" if sk == "WARN" else "R")
        print(f"  {sc}  {r['domain']:<26} {_hbar(r['score'], 18)}")

    # ── Unique proposals ─────────────────────────────────────────────────────
    seen, unique = set(), []
    for c in all_results:
        for r in c:
            for p in r["proposals"]:
                if p not in seen:
                    seen.add(p); unique.append(p)
    if unique:
        print(f"\\n  {_hcol(f'HEALING PROPOSALS  ({len(unique)} total)', 'B')}")
        for p in unique[:6]:
            print(f"  {_HC['D']}• {p[:90]}{_HC['X']}")

    # ── Provenance ───────────────────────────────────────────────────────────
    try:
        prov.append("healing_loop", final_score, {
            "verdict":      verdict,
            "cycle_scores": cycle_scores,
            "delta":        delta_total,
        })
    except Exception:
        pass

    print(f"\\n  {'═' * 54}")
    print(f"  {_hcol('HEALING LOOP COMPLETE', 'B')}")
    print(f"  {'═' * 54}\\n")

    return {
        "verdict":      verdict,
        "final_score":  final_score,
        "cycle_scores": cycle_scores,
        "delta":        delta_total,
        "domain_results": [
            [{k: v for k, v in r.items() if k != "proposals"} for r in c]
            for c in all_results
        ],
        "audit": audit,
        "errors": errors,
    }

'''

# Insert before run_self_tests
SELF_TESTS_ANCHOR = "\ndef run_self_tests("
if "def run_healing_loop(" in src:
    step3_ok = True
elif SELF_TESTS_ANCHOR in src:
    src = src.replace(SELF_TESTS_ANCHOR, HEALING_LOOP_FN + "\ndef run_self_tests(", 1)
    step3_ok = True
    print("[PATCH] Added run_healing_loop() function")
else:
    print("[WARN] Could not find run_self_tests anchor — function insert skipped")

# ── 4. Add "heal" to argparse choices ─────────────────────────────────────────
OLD_CHOICES = 'choices=["inject","dream","test","audit","full"]'
NEW_CHOICES = 'choices=["inject","dream","heal","test","audit","full"]'
if OLD_CHOICES in src:
    src = src.replace(OLD_CHOICES, NEW_CHOICES, 1)
    print("[PATCH] Added 'heal' to argparse choices")
elif NEW_CHOICES in src:
    print("[INFO]  'heal' already in argparse choices")
else:
    print("[WARN] Could not find argparse choices anchor")

# ── 5. Wire heal into main() after dream block ────────────────────────────────
# Only wire a call site if the function definition and its prerequisites
# (import, display helpers) actually succeeded — otherwise this call site
# references a name that doesn't exist, and the "already patched" guard at
# the top of this script would then treat that broken file as done, making
# the breakage permanent until someone edits run_system.py by hand.
if not (step1_ok and step2_ok and step3_ok):
    print("[ABORT] Skipping step 5 (call-site wiring) — a prerequisite step "
          f"failed (import={step1_ok}, helpers={step2_ok}, function={step3_ok}). "
          "Wiring a call to a possibly-undefined run_healing_loop() would "
          "break run_system.py at runtime.")
elif 'run_healing_loop' in src and 'args.phase in ("full", "heal")' in src:
    print("[INFO]  heal already wired into main()")
else:
    DREAM_BLOCK = (
        '    if args.phase in ("full", "dream"):\n'
        '        results["dream"] = run_dream_cycle(cfg)\n'
        '        time.sleep(cfg.effective_throttle())'
    )
    HEAL_ADDITION = (
        '    if args.phase in ("full", "dream"):\n'
        '        results["dream"] = run_dream_cycle(cfg)\n'
        '        time.sleep(cfg.effective_throttle())\n'
        '\n'
        '    if args.phase in ("full", "heal"):\n'
        '        results["healing_loop"] = run_healing_loop(cfg)\n'
        '        time.sleep(cfg.effective_throttle())'
    )
    if DREAM_BLOCK in src:
        src = src.replace(DREAM_BLOCK, HEAL_ADDITION, 1)
        print("[PATCH] Wired run_healing_loop() into main() after dream phase")
    else:
        print("[WARN] Could not find dream block in main() — heal wiring skipped")

# ── 6. Update docstring usage block ──────────────────────────────────────────
OLD_USAGE = "  python run_system.py --phase dream      # Dream cycle only"
NEW_USAGE = (
    "  python run_system.py --phase dream      # Dream cycle only\n"
    "  python run_system.py --phase heal       # Forensic healing loop (C12 judgment)"
)
if OLD_USAGE in src:
    src = src.replace(OLD_USAGE, NEW_USAGE, 1)
    print("[PATCH] Updated docstring usage block")

# ── Write result ──────────────────────────────────────────────────────────────
try:
    compile(src, str(TARGET), "exec")
except SyntaxError as e:
    print(f"\n[ABORT] Patched run_system.py would not compile ({e}) — "
          f"writing nothing, original file left untouched")
    sys.exit(1)

backup = TARGET.with_suffix(".py.bak")
if backup.exists():
    from datetime import datetime, timezone
    backup = TARGET.with_suffix(f".py.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}")
backup.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")

_tmp = TARGET.with_suffix(".py.tmp")
_tmp.write_text(src, encoding="utf-8")
_tmp.replace(TARGET)

print()
print(f"[DONE] run_system.py patched  (backup: {backup.name})")
print()
print("  Test it:")
print("    python run_system.py --phase heal")
print("    python run_system.py --phase full")
print("    python run_system.py --phase heal --live")
