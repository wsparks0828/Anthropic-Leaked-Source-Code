"""tools/loop_audit.py -- LOOP-ES audit, WALCHE standalone. C0-C12 + L1-L5."""
import importlib
import importlib.util
import subprocess
import py_compile
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Check:
    id: str
    name: str
    status: str          # PASS | WARN | FAIL | SKIP | DEFERRED
    severity: str = ""   # CRITICAL | HIGH | MEDIUM | LOW | INFO
    notes: str = ""
    score: float = 1.0

@dataclass
class AuditReport:
    lineage_id: str
    timestamp: str
    checks: List[Check] = field(default_factory=list)
    overall_status: str = "UNKNOWN"
    composite_score: float = 0.0
    c12_verdict: str = ""

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _try_import(module_path: str, cls_name: Optional[str] = None):
    """Try to import a module and optionally a class. Returns (obj, error_str)."""
    try:
        mod = importlib.import_module(module_path)
        if cls_name:
            obj = getattr(mod, cls_name)
            return obj, None
        return mod, None
    except Exception as e:
        return None, str(e)


def _safe_instantiate(cls, *args, **kwargs):
    """Try to instantiate a class. Returns (instance, error_str)."""
    try:
        return cls(*args, **kwargs), None
    except Exception as e:
        return None, str(e)


def _color(text: str, status: str) -> str:
    colors = {
        "PASS":     "\033[92m",
        "WARN":     "\033[93m",
        "FAIL":     "\033[91m",
        "SKIP":     "\033[90m",
        "DEFERRED": "\033[96m",
    }
    reset = "\033[0m"
    c = colors.get(status, "")
    return f"{c}{text}{reset}"

# ─── C0: Syntax ──────────────────────────────────────────────────────────────

def c0_syntax() -> Check:
    """Syntax-check all Python files under ROOT (excluding venv/node_modules)."""
    errors = []
    exclude = {"venv", "node_modules", ".git", "__pycache__", "site-packages"}
    for py_file in ROOT.rglob("*.py"):
        if any(part in exclude for part in py_file.parts):
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(str(e))
    if errors:
        return Check("C0", "Syntax Check", "FAIL", "HIGH",
                     f"{len(errors)} syntax error(s): " + "; ".join(errors[:3]), 0.0)
    return Check("C0", "Syntax Check", "PASS", "INFO", "All .py files compile OK", 1.0)

# ─── C1: Core Imports ────────────────────────────────────────────────────────

def c1_imports() -> Check:
    """Verify key WALCHE core modules import without error."""
    targets = [
        ("core.meta_engine",       "MetaEngine"),
        ("core.healing_engine",    "HealingEngine"),
        ("core.guardrail",         "Guardrail"),
        ("core.forensic_cognition", None),
        ("core.rubric",             None),
    ]
    failures = []
    for mod, cls in targets:
        _, err = _try_import(mod, cls)
        if err:
            failures.append(f"{mod}: {err}")
    if failures:
        sev = "HIGH" if len(failures) > 2 else "MEDIUM"
        return Check("C1", "Core Imports", "WARN", sev,
                     "; ".join(failures), max(0.3, 1.0 - 0.2 * len(failures)))
    return Check("C1", "Core Imports", "PASS", "INFO", "All core modules importable", 1.0)

# ─── C2: Loop Registry ───────────────────────────────────────────────────────

def c2_loop_registry() -> Check:
    """Check that LoopRegistry loads and reports >= 1 registered loops."""
    cls, err = _try_import("core.loop_registry", "LoopRegistry")
    if err:
        return Check("C2", "Loop Registry", "FAIL", "HIGH",
                     f"Import failed: {err}", 0.0)
    inst, err2 = _safe_instantiate(cls)
    if err2:
        return Check("C2", "Loop Registry", "FAIL", "HIGH",
                     f"Instantiation failed: {err2}", 0.0)
    try:
        loops = inst.list_loops() if hasattr(inst, "list_loops") else getattr(inst, "loops", [])
        count = len(loops) if loops else 0
    except Exception as e:
        return Check("C2", "Loop Registry", "WARN", "MEDIUM",
                     f"list_loops() error: {e}", 0.5)
    if count == 0:
        return Check("C2", "Loop Registry", "WARN", "MEDIUM", "0 loops registered", 0.5)
    return Check("C2", "Loop Registry", "PASS", "INFO", f"{count} loop(s) registered", 1.0)

# ─── C3: Prompt Validation ───────────────────────────────────────────────────

def c3_prompt_validation() -> Check:
    """Check prompt files / prompt manager integrity."""
    prompt_dirs = [
        ROOT / "prompts",
        ROOT / "core" / "prompts",
        ROOT / "backend" / "prompts",
    ]
    found = []
    for d in prompt_dirs:
        if d.exists():
            found.extend(list(d.glob("*.txt")) + list(d.glob("*.md")) + list(d.glob("*.json")))
    if not found:
        return Check("C3", "Prompt Validation", "WARN", "LOW",
                     "No prompt files found in expected directories", 0.7)
    pm, err = _try_import("core.prompt_manager", None)
    if err:
        return Check("C3", "Prompt Validation", "WARN", "LOW",
                     f"{len(found)} prompt file(s); PromptManager not importable: {err}", 0.7)
    return Check("C3", "Prompt Validation", "PASS", "INFO",
                 f"{len(found)} prompt file(s), PromptManager OK", 1.0)

# ─── C4: Memory System ───────────────────────────────────────────────────────

def c4_memory() -> Check:
    """Verify reflection memory and ChromaDB collection."""
    cls, err = _try_import("backend.memory.reflection_memory", "PersistentReflectionMemory")
    if err:
        cls, err2 = _try_import("core.memory", None)
        if err2:
            return Check("C4", "Memory System", "WARN", "MEDIUM",
                         f"Memory module not importable: {err}", 0.5)
        return Check("C4", "Memory System", "WARN", "LOW",
                     "PersistentReflectionMemory not found; core.memory fallback OK", 0.8)
    inst, err3 = _safe_instantiate(cls)
    if err3:
        return Check("C4", "Memory System", "WARN", "MEDIUM",
                     f"PersistentReflectionMemory() failed: {err3}", 0.5)
    return Check("C4", "Memory System", "PASS", "INFO",
                 "PersistentReflectionMemory instantiated OK", 1.0)

# ─── C5: Rubric Scorer ───────────────────────────────────────────────────────

def c5_rubric_scorer() -> Check:
    """Check RubricScorer imports and has expected interface."""
    # core/rubric.py — try RubricScorer then Rubric as fallback class name
    cls, err = _try_import("core.rubric", "RubricScorer")
    if err:
        cls, err_rb = _try_import("core.rubric", "Rubric")
        if err_rb:
            _, err2 = _try_import("core.rubric", None)
            if err2:
                return Check("C5", "Rubric Scorer", "FAIL", "HIGH",
                             f"Import failed: {err2}", 0.0)
            return Check("C5", "Rubric Scorer", "WARN", "MEDIUM",
                         "core.rubric OK but no RubricScorer/Rubric class found", 0.6)
        cls = cls  # Rubric class found
    inst, err3 = _safe_instantiate(cls)
    if err3:
        return Check("C5", "Rubric Scorer", "WARN", "MEDIUM",
                     f"Instantiation failed: {err3}", 0.6)
    if not (hasattr(inst, "score") or hasattr(inst, "evaluate")):
        return Check("C5", "Rubric Scorer", "WARN", "LOW",
                     "Missing score()/evaluate() method", 0.7)
    return Check("C5", "Rubric Scorer", "PASS", "INFO", "RubricScorer OK", 1.0)

# ─── C6: Forensic Logging ────────────────────────────────────────────────────

def c6_forensic_logging() -> Check:
    """Verify forensic logger is importable and logs/ dir is writable."""
    mod, err = _try_import("core.forensic_cognition", None)
    if err:
        mod, err2 = _try_import("backend.forensic_logger", None)
        if err2:
            return Check("C6", "Forensic Logging", "WARN", "MEDIUM",
                         f"ForensicLogger not found: {err}", 0.5)
    log_dir = ROOT / "logs"
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return Check("C6", "Forensic Logging", "WARN", "MEDIUM",
                         f"Cannot create logs/: {e}", 0.6)
    return Check("C6", "Forensic Logging", "PASS", "INFO",
                 "ForensicLogger importable, logs/ dir OK", 1.0)

# ─── C7: Loop Integration ────────────────────────────────────────────────────

def c7_loop_integration() -> Check:
    """Verify WALCHE loop infrastructure via core/loop_registry + core loop modules."""
    # loops/ does not exist as a standalone dir — loop registry lives in core/
    reg_mod, reg_err = _try_import("core.loop_registry", None)
    if reg_err:
        return Check("C7", "Loop Integration", "FAIL", "HIGH",
                     f"core.loop_registry not importable: {reg_err}", 0.0)

    # Spot-check loop-adjacent core modules
    loop_modules = [
        "core.pain_engine", "core.pain_fmea", "core.healing_engine",
        "core.spine_guardian", "core.orchestrator", "core.vll",
    ]
    failures = []
    for mod in loop_modules:
        _, err = _try_import(mod, None)
        if err:
            failures.append(f"{mod.split('.')[-1]}: {err}")
    if failures:
        return Check("C7", "Loop Integration", "WARN", "MEDIUM",
                     f"{len(failures)}/{len(loop_modules)} loop module(s) failed: "
                     + "; ".join(failures[:3]),
                     max(0.4, 1.0 - 0.1 * len(failures)))
    return Check("C7", "Loop Integration", "PASS", "INFO",
                 f"loop_registry + {len(loop_modules)} loop modules OK", 1.0)

# ─── C8: Guardrail Interface ─────────────────────────────────────────────────

def c8_guardrail() -> Check:
    """Verify Guardrail has expected .check() or .evaluate() interface."""
    cls, err = _try_import("core.guardrail", "Guardrail")
    if err:
        return Check("C8", "Guardrail Interface", "FAIL", "HIGH",
                     f"Import failed: {err}", 0.0)
    inst, err2 = _safe_instantiate(cls)
    if err2:
        return Check("C8", "Guardrail Interface", "WARN", "MEDIUM",
                     f"Instantiation failed: {err2}", 0.5)
    has_check    = hasattr(inst, "check")
    has_evaluate = hasattr(inst, "evaluate")
    if not (has_check or has_evaluate):
        return Check("C8", "Guardrail Interface", "WARN", "HIGH",
                     "Guardrail missing check() and evaluate() methods", 0.4)
    method = "check" if has_check else "evaluate"
    return Check("C8", "Guardrail Interface", "PASS", "INFO",
                 f"Guardrail.{method}() present", 1.0)

# ─── C9: Self-Tests ──────────────────────────────────────────────────────────

def c9_self_tests() -> Check:
    """Run any WALCHE self-test scripts and report pass rate."""
    test_paths = [
        ROOT / "tests",
        ROOT / "test",
        ROOT / "self_tests",
        ROOT / "backend" / "tests",
    ]
    test_files = []
    for td in test_paths:
        if td.exists():
            test_files.extend(td.glob("test_*.py"))
            test_files.extend(td.glob("*_test.py"))
    if not test_files:
        return Check("C9", "Self-Tests", "SKIP", "LOW",
                     "No test files found", 1.0)
    passed = failed = 0
    for tf in test_files[:6]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(tf), "-x", "-q", "--tb=no"],
                capture_output=True, text=True, timeout=30, cwd=str(ROOT)
            )
            if result.returncode == 0:
                passed += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    total = passed + failed
    rate = passed / total if total else 1.0
    status = "PASS" if rate >= 0.75 else "WARN" if rate >= 0.5 else "FAIL"
    sev    = "INFO" if status == "PASS" else "MEDIUM" if status == "WARN" else "HIGH"
    return Check("C9", "Self-Tests", status, sev,
                 f"{passed}/{total} self-tests passed", rate)

# ─── C10: Provenance Log ─────────────────────────────────────────────────────

def c10_provenance() -> Check:
    """Check provenance/audit log module and log file existence."""
    mod, err = _try_import("core.provenance", None)
    if err:
        mod, err2 = _try_import("backend.provenance_log", None)
        if err2:
            log_files = (list((ROOT / "logs").glob("provenance*.json"))
                         + list((ROOT / "logs").glob("audit*.json"))
                         + list(ROOT.glob("provenance*.json")))
            if log_files:
                return Check("C10", "Provenance Log", "WARN", "LOW",
                             f"Module not importable ({err}); {len(log_files)} log file(s) found", 0.7)
            return Check("C10", "Provenance Log", "WARN", "MEDIUM",
                         f"Provenance module not importable: {err}", 0.5)
    return Check("C10", "Provenance Log", "PASS", "INFO",
                 "Provenance module importable", 1.0)

# ─── C11: Architecture Health ────────────────────────────────────────────────

def c11_arch_health() -> Check:
    """Aggregate architecture health: key dirs, file counts, venv."""
    issues = []
    for d in ["core", "walche_tools", "backend"]:
        if not (ROOT / d).exists():
            issues.append(f"missing {d}/")
    venv_ok = (ROOT / "venv" / "Scripts" / "python.exe").exists() \
           or (ROOT / "venv" / "bin" / "python").exists()
    if not venv_ok:
        issues.append("venv not found")
    py_count = sum(
        1 for f in ROOT.rglob("*.py")
        if "venv" not in str(f) and "node_modules" not in str(f)
    )
    if py_count < 5:
        issues.append(f"only {py_count} .py file(s) found")
    if issues:
        return Check("C11", "Architecture Health", "WARN", "MEDIUM",
                     "; ".join(issues), max(0.4, 1.0 - 0.15 * len(issues)))
    return Check("C11", "Architecture Health", "PASS", "INFO",
                 f"Structure OK, {py_count} source files", 1.0)

# ─── C12: WALCHE Judgment (STANDALONE) ───────────────────────────────────────

def c12_walche_judgment(checks: List[Check], composite: float) -> Check:
    """
    WALCHE-native judgment layer.
    Sources: MetaEngine (Bertha), PAIN FMEA adversarial scan, HealingEngine.
    NO external agents / councils.  Produces GO / GO-WITH-CONDITIONS / NO-GO.
    """
    notes = []
    verdict = "GO"

    # ── MetaEngine (Bertha) ───────────────────────────────────────────────────
    me_cls, me_err = _try_import("core.meta_engine", "MetaEngine")
    if me_err:
        notes.append("MetaEngine:unavailable")
        verdict = "GO-WITH-CONDITIONS"
    else:
        me_inst, me_ie = _safe_instantiate(me_cls)
        if me_ie:
            notes.append(f"MetaEngine:init_fail")
            verdict = "GO-WITH-CONDITIONS"
        else:
            conf = True
            if hasattr(me_inst, "verify"):
                try:
                    conf = bool(me_inst.verify())
                except Exception as ve:
                    notes.append(f"MetaEngine:verify_err:{ve}")
                    conf = False
            elif hasattr(me_inst, "get_status"):
                try:
                    st = me_inst.get_status()
                    conf = st.get("healthy", True) if isinstance(st, dict) else bool(st)
                except Exception:
                    conf = False
            notes.append(f"MetaEngine:{'OK' if conf else 'LOW'}")
            if not conf:
                verdict = "GO-WITH-CONDITIONS"

    # ── PAIN FMEA adversarial scan ────────────────────────────────────────────
    critical_fails = [c for c in checks if c.status == "FAIL" and c.severity == "CRITICAL"]
    high_fails     = [c for c in checks if c.status == "FAIL" and c.severity == "HIGH"]
    pain_items     = []
    if critical_fails:
        pain_items.append(f"CRITICAL_FAIL:{[c.id for c in critical_fails]}")
        verdict = "NO-GO"
    if high_fails:
        pain_items.append(f"HIGH_FAIL:{[c.id for c in high_fails]}")
        if verdict == "GO":
            verdict = "GO-WITH-CONDITIONS"
        if len(high_fails) >= 3:
            verdict = "NO-GO"
    if composite < 0.40:
        pain_items.append(f"COMPOSITE_LOW:{composite:.2f}")
        verdict = "NO-GO"
    elif composite < 0.65:
        pain_items.append(f"COMPOSITE_WARN:{composite:.2f}")
        if verdict == "GO":
            verdict = "GO-WITH-CONDITIONS"
    notes.append("PAIN_FMEA:" + ("|".join(pain_items) if pain_items else "CLEAR"))

    # ── HealingEngine proposals ───────────────────────────────────────────────
    he_cls, he_err = _try_import("core.healing_engine", "HealingEngine")
    if he_err:
        _, he_err2 = _try_import("backend.self_healing.healing_agent", None)
        notes.append("HealingEngine:unavailable" if he_err2 else "HealingEngine:fallback_module")
    else:
        he_inst, he_ie = _safe_instantiate(he_cls)
        if he_ie:
            notes.append("HealingEngine:init_fail")
        else:
            proposals = 0
            if hasattr(he_inst, "propose_fixes"):
                try:
                    fixes = he_inst.propose_fixes(checks)
                    proposals = len(fixes) if fixes else 0
                except Exception as e:
                    notes.append(f"HealingEngine:propose_err:{e}")
            elif hasattr(he_inst, "get_proposals"):
                try:
                    fixes = he_inst.get_proposals()
                    proposals = len(fixes) if fixes else 0
                except Exception:
                    proposals = 0
            notes.append(f"HealingEngine:{proposals}_proposal(s)")

    # ── Final verdict ─────────────────────────────────────────────────────────
    sc     = 1.0 if verdict == "GO" else 0.7 if verdict == "GO-WITH-CONDITIONS" else 0.0
    sev    = "INFO" if verdict == "GO" else "MEDIUM" if verdict == "GO-WITH-CONDITIONS" else "CRITICAL"
    status = "PASS" if verdict == "GO" else "WARN" if verdict == "GO-WITH-CONDITIONS" else "FAIL"

    return Check(
        "C12", "WALCHE Judgment", status, sev,
        "Verdict:" + verdict + "|" + "|".join(notes),
        sc,
    )

# ─── L1-L5: Learning Checks (DEFERRED) ───────────────────────────────────────

def _l_deferred(lid: str, name: str) -> Check:
    return Check(lid, name, "DEFERRED", "INFO",
                 "No production corpus — deferred until corpus available", 1.0)

# ─── Runner ───────────────────────────────────────────────────────────────────

def run_audit(lineage_id: Optional[str] = None) -> AuditReport:
    ts  = datetime.now(timezone.utc).isoformat()
    lid = lineage_id or f"WALCHE-AUDIT-{ts[:10]}"
    report = AuditReport(lineage_id=lid, timestamp=ts)

    print(f"\n{'='*62}")
    print(f"  WALCHE LOOP-ES AUDIT  |  {ts[:19]}Z")
    print(f"  Lineage: {lid}")
    print(f"{'='*62}\n")

    # C0-C11
    arch_fns = [
        c0_syntax, c1_imports, c2_loop_registry, c3_prompt_validation,
        c4_memory, c5_rubric_scorer, c6_forensic_logging, c7_loop_integration,
        c8_guardrail, c9_self_tests, c10_provenance, c11_arch_health,
    ]
    for fn in arch_fns:
        chk = fn()
        report.checks.append(chk)
        bar = _color(f"[{chk.status:^8}]", chk.status)
        print(f"  {chk.id:<4} {chk.name:<28} {bar}  {chk.notes[:78]}")

    # Composite score C0-C11
    scored    = [c for c in report.checks if c.status not in ("SKIP", "DEFERRED")]
    composite = sum(c.score for c in scored) / len(scored) if scored else 0.0
    print(f"\n  Composite (C0-C11): {composite:.2f}")

    # C12 WALCHE Judgment
    c12 = c12_walche_judgment(report.checks, composite)
    report.checks.append(c12)
    bar = _color(f"[{c12.status:^8}]", c12.status)
    print(f"\n  {c12.id:<4} {c12.name:<28} {bar}")
    verdict = c12.notes.split("|")[0].replace("Verdict:", "")
    report.c12_verdict = verdict
    for detail in c12.notes.split("|")[1:]:
        if detail:
            print(f"       ↳ {detail}")

    # L1-L5
    print()
    l_checks = [
        _l_deferred("L1", "Corpus Ingestion"),
        _l_deferred("L2", "Schema Alignment"),
        _l_deferred("L3", "Loop Signal Quality"),
        _l_deferred("L4", "Reinforcement Health"),
        _l_deferred("L5", "Emergent Pattern Integrity"),
    ]
    for lc in l_checks:
        report.checks.append(lc)
        bar = _color(f"[{lc.status:^8}]", lc.status)
        print(f"  {lc.id:<4} {lc.name:<28} {bar}  {lc.notes[:58]}")

    # Overall
    all_fails = [c for c in report.checks if c.status == "FAIL"]
    all_warns = [c for c in report.checks if c.status == "WARN"]
    if verdict == "NO-GO" or any(c.severity == "CRITICAL" for c in all_fails):
        report.overall_status = "NO-GO"
    elif verdict == "GO-WITH-CONDITIONS" or all_fails or len(all_warns) > 3:
        report.overall_status = "GO-WITH-CONDITIONS"
    else:
        report.overall_status = "GO"

    report.composite_score = composite

    status_color = "PASS" if report.overall_status == "GO" else \
                   "WARN" if "CONDITIONS" in report.overall_status else "FAIL"
    print(f"\n{'='*62}")
    print(f"  OVERALL: {_color(report.overall_status, status_color)}")
    print(f"  Composite: {composite:.2f}  |  FAIL: {len(all_fails)}  WARN: {len(all_warns)}")
    print(f"{'='*62}\n")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="WALCHE LOOP-ES Audit — C0-C12 architectural checks + L1-L5 learning checks"
    )
    parser.add_argument("--lineage-id", default=None,
                        help="Custom lineage ID for this audit run")
    parsed = parser.parse_args()
    run_audit(lineage_id=parsed.lineage_id)
