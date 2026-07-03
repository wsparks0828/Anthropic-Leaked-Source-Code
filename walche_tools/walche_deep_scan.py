"""tools/walche_deep_scan.py -- WALCHE Full Platform Scan
Covers: syntax, imports, interfaces, AST bug patterns, loop registry,
module connectivity, config, tests, dead code, circular imports, EPM contamination.
Run from WALCHE root: venv\Scripts\python tools\walche_deep_scan.py
"""
import ast
import importlib
import importlib.util
import json
import py_compile
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXCLUDE_DIRS = {"venv", "node_modules", ".git", "__pycache__", "site-packages",
                "wachle-dashboard", "dist", ".pytest_cache"}

# ─── Data ────────────────────────────────────────────────────────────────────

@dataclass
class Issue:
    category: str      # SYNTAX | IMPORT | INTERFACE | BUG | LOOP | CONFIG | CONTAM | TEST
    severity: str      # CRITICAL | HIGH | MEDIUM | LOW | INFO
    file: str
    line: int
    message: str
    suggestion: str = ""

@dataclass
class ScanReport:
    timestamp: str
    root: str
    issues: List[Issue] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)
    loop_registry: Dict = field(default_factory=dict)
    module_map: Dict = field(default_factory=dict)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _py_files(subdir: str = "") -> List[Path]:
    base = ROOT / subdir if subdir else ROOT
    return [
        f for f in base.rglob("*.py")
        if not any(p in EXCLUDE_DIRS for p in f.parts)
    ]

def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)

def _try_import(mod: str, cls: str = None):
    try:
        m = importlib.import_module(mod)
        if cls:
            return getattr(m, cls), None
        return m, None
    except Exception as e:
        return None, str(e)

def _read_ast(path: Path):
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        return ast.parse(src, filename=str(path)), src, None
    except SyntaxError as e:
        return None, "", str(e)
    except Exception as e:
        return None, "", str(e)

# ─── S1: Syntax ──────────────────────────────────────────────────────────────

def scan_syntax() -> List[Issue]:
    issues = []
    for f in _py_files():
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            issues.append(Issue("SYNTAX", "CRITICAL", _rel(f), 0,
                                str(e)[:200], "Fix syntax before any other work"))
    return issues

# ─── S2: Imports ─────────────────────────────────────────────────────────────

CORE_MODULES = [
    ("core.config",           "WalcheConfig"),
    ("core.meta_engine",      "MetaEngine"),
    ("core.healing_engine",   "HealingEngine"),
    ("core.guardrail",        "Guardrail"),
    ("core.forensic_cognition", None),
    ("core.rubric",           "RubricScorer"),
    ("core.pain_fmea",        "PAINFMEA"),
    ("core.pain_engine",      None),
    ("core.loop_registry",    None),
    ("core.provenance",       None),
    ("core.pre_ingest_gate",  None),
    ("core.vector_store",     None),
    ("core.orchestrator",     None),
    ("core.spine_guardian",   None),
    ("core.vll",              None),
    ("core.nine_step",        None),
    ("core.routing_cache",    None),
    ("core.structure_feedback", None),
    ("core.connection_monitor", None),
    ("core.grok_monitor",     None),
    ("backend.memory.reflection_memory", "PersistentReflectionMemory"),
]

def scan_imports() -> Tuple[List[Issue], Dict]:
    issues = []
    module_map = {}
    for mod, cls in CORE_MODULES:
        obj, err = _try_import(mod, cls)
        if err:
            sev = "CRITICAL" if "core.meta_engine" in mod or "core.healing_engine" in mod else "HIGH"
            issues.append(Issue("IMPORT", sev, mod.replace(".", "/") + ".py", 0,
                                f"Cannot import {mod}" + (f".{cls}" if cls else "") + f": {err}",
                                "Check file exists and has no internal import errors"))
            module_map[mod] = {"status": "FAIL", "error": err}
        else:
            module_map[mod] = {"status": "OK", "class": cls}
    return issues, module_map

# ─── S3: Interface Validation ────────────────────────────────────────────────

INTERFACE_CHECKS = {
    "core.meta_engine.MetaEngine":         ["evaluate_meta"],
    "core.healing_engine.HealingEngine":   ["reflect_and_heal"],
    "core.guardrail.Guardrail":            ["check", "evaluate"],   # either
    "core.rubric.RubricScorer":            ["score"],
    "core.pain_fmea.PAINFMEA":             ["calculate_rpn"],
    "core.pre_ingest_gate.PreIngestGate":  ["check", "gate", "validate"],  # any
}

def scan_interfaces() -> List[Issue]:
    issues = []
    for dotpath, required_methods in INTERFACE_CHECKS.items():
        parts = dotpath.rsplit(".", 1)
        if len(parts) != 2:
            continue
        mod_path, cls_name = parts
        cls, err = _try_import(mod_path, cls_name)
        if err:
            continue  # already caught in S2
        try:
            inst = cls()
        except Exception as e:
            issues.append(Issue("INTERFACE", "HIGH",
                                mod_path.replace(".", "/") + ".py", 0,
                                f"{cls_name}() instantiation failed: {e}",
                                "Check __init__ signature and dependencies"))
            continue
        # For guardrail/gate: require at least ONE of the listed methods
        found = [m for m in required_methods if hasattr(inst, m)]
        if not found:
            issues.append(Issue("INTERFACE", "HIGH",
                                mod_path.replace(".", "/") + ".py", 0,
                                f"{cls_name} missing all of: {required_methods}",
                                f"Add one of {required_methods} to {cls_name}"))
    return issues

# ─── S4: AST Bug Patterns ────────────────────────────────────────────────────

MUTABLE_DEFAULTS = (ast.List, ast.Dict, ast.Set)

class BugVisitor(ast.NodeVisitor):
    def __init__(self, src_lines: List[str], filename: str):
        self.issues: List[Issue] = []
        self.src = src_lines
        self.filename = filename
        self._imports: Set[str] = set()
        self._used: Set[str] = set()

    def visit_Import(self, node):
        for alias in node.names:
            self._imports.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self._imports.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self._used.add(node.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.issues.append(Issue("BUG", "MEDIUM", self.filename, node.lineno,
                                     "Bare except: catches ALL exceptions including KeyboardInterrupt/SystemExit",
                                     "Use 'except Exception as e:' instead"))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for default in node.args.defaults + node.args.kw_defaults:
            if default and isinstance(default, MUTABLE_DEFAULTS):
                self.issues.append(Issue("BUG", "MEDIUM", self.filename, node.lineno,
                                         f"Mutable default argument in '{node.name}': {ast.dump(default)[:60]}",
                                         "Use None as default, initialise inside function"))
        self.generic_visit(node)

    def visit_Assert(self, node):
        # assert (a, b) always True
        if isinstance(node.test, ast.Tuple):
            self.issues.append(Issue("BUG", "HIGH", self.filename, node.lineno,
                                     "assert with a tuple is always True (did you mean 'assert a, b'?)",
                                     "Remove parentheses: assert condition, message"))
        self.generic_visit(node)

    def visit_Compare(self, node):
        # x == None / x != None
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(comp, ast.Constant) and comp.value is None:
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    self.issues.append(Issue("BUG", "LOW", self.filename, node.lineno,
                                             "Comparison to None with == / != (use 'is' / 'is not')",
                                             "Replace with 'is None' or 'is not None'"))
        self.generic_visit(node)

    def visit_Try(self, node):
        for handler in node.handlers:
            if handler.name is None and handler.type is not None:
                pass  # fine
        self.generic_visit(node)


def scan_ast_bugs() -> List[Issue]:
    issues = []
    for f in _py_files():
        tree, src, err = _read_ast(f)
        if err or tree is None:
            continue
        lines = src.splitlines()
        visitor = BugVisitor(lines, _rel(f))
        try:
            visitor.visit(tree)
        except Exception:
            pass
        issues.extend(visitor.issues)
    return issues

# ─── S5: EPM / External Contamination ────────────────────────────────────────

CONTAM_PATTERNS = [
    (r"EPM-STARK",   "CRITICAL", "EPM-STARK reference inside WALCHE — violates standalone constraint"),
    (r"WOM-STARK",   "CRITICAL", "WOM-STARK reference inside WALCHE — violates standalone constraint"),
    (r"C:\\EPM",     "CRITICAL", "Hardcoded EPM path inside WALCHE"),
    (r"C:/EPM",      "CRITICAL", "Hardcoded EPM path inside WALCHE"),
    (r"estc2_live_apply", "HIGH", "ESTC2 apply script referenced inside WALCHE"),
    (r"from wom",    "HIGH",     "WOM import inside WALCHE"),
    (r"import wom",  "HIGH",     "WOM import inside WALCHE"),
]

def scan_contamination() -> List[Issue]:
    issues = []
    for f in _py_files():
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for lineno, line in enumerate(src.splitlines(), 1):
            for pattern, sev, msg in CONTAM_PATTERNS:
                if pattern.lower() in line.lower():
                    issues.append(Issue("CONTAM", sev, _rel(f), lineno,
                                        f"{msg}: {line.strip()[:100]}",
                                        "Move this file to EPM-STARK side or remove reference"))
    return issues

# ─── S6: Loop Registry ───────────────────────────────────────────────────────

EXPECTED_LOOP_COUNT = 18

def scan_loop_registry() -> Tuple[List[Issue], Dict]:
    issues = []
    registry_info = {"status": "UNKNOWN", "count": 0, "loops": [], "connections": 0}

    mod, err = _try_import("core.loop_registry")
    if err:
        issues.append(Issue("LOOP", "CRITICAL", "core/loop_registry.py", 0,
                            f"loop_registry not importable: {err}",
                            "Ensure core/loop_registry.py exists and is error-free"))
        return issues, registry_info

    # Try to find loop count
    cls, _ = _try_import("core.loop_registry", "LoopRegistry")
    if cls:
        try:
            inst = cls()
            loops = inst.list_loops() if hasattr(inst, "list_loops") else getattr(inst, "loops", [])
            count = len(loops) if loops else 0
            registry_info.update({"status": "OK", "count": count,
                                   "loops": [str(l) for l in (loops or [])]})
            if count < EXPECTED_LOOP_COUNT:
                issues.append(Issue("LOOP", "HIGH", "core/loop_registry.py", 0,
                                    f"Only {count}/{EXPECTED_LOOP_COUNT} loops registered",
                                    f"Register all {EXPECTED_LOOP_COUNT} WALCHE loops"))
            else:
                registry_info["status"] = "FULL"
        except Exception as e:
            issues.append(Issue("LOOP", "MEDIUM", "core/loop_registry.py", 0,
                                f"LoopRegistry instantiation error: {e}", ""))
    else:
        # Try module-level attributes
        attrs = {k: v for k, v in vars(mod).items() if not k.startswith("_")}
        loops = attrs.get("LOOPS") or attrs.get("loops") or attrs.get("REGISTRY") or []
        count = len(loops) if hasattr(loops, "__len__") else 0
        registry_info.update({"status": "MODULE_OK", "count": count})
        if count == 0:
            issues.append(Issue("LOOP", "MEDIUM", "core/loop_registry.py", 0,
                                "No LoopRegistry class found; module-level loops also empty", ""))

    # Check connections (42 expected)
    conn_attr = None
    for attr in ["connections", "CONNECTIONS", "edges", "EDGES"]:
        obj = getattr(mod, attr, None)
        if obj is not None:
            conn_attr = obj
            break
    if conn_attr is not None:
        registry_info["connections"] = len(conn_attr) if hasattr(conn_attr, "__len__") else 0
    return issues, registry_info

# ─── S7: Circular Import Detection ───────────────────────────────────────────

def scan_circular_imports() -> List[Issue]:
    issues = []
    # Build import graph from AST
    graph: Dict[str, Set[str]] = defaultdict(set)
    mod_to_file: Dict[str, Path] = {}

    for f in _py_files():
        rel = _rel(f).replace("\\", "/").replace("/", ".").removesuffix(".py")
        mod_to_file[rel] = f
        tree, _, err = _read_ast(f)
        if err or tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("core.") or node.module.startswith("backend."):
                    graph[rel].add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("core.") or alias.name.startswith("backend."):
                        graph[rel].add(alias.name)

    # DFS cycle detection
    def has_cycle(start: str, visited: Set[str], stack: Set[str]) -> Optional[List[str]]:
        visited.add(start)
        stack.add(start)
        for dep in graph.get(start, set()):
            if dep not in visited:
                result = has_cycle(dep, visited, stack)
                if result:
                    return [start] + result
            elif dep in stack:
                return [start, dep]
        stack.discard(start)
        return None

    visited: Set[str] = set()
    for mod in list(graph.keys()):
        if mod not in visited:
            cycle = has_cycle(mod, visited, set())
            if cycle:
                issues.append(Issue("BUG", "HIGH",
                                    mod.replace(".", "/") + ".py", 0,
                                    f"Circular import detected: {' → '.join(cycle)}",
                                    "Refactor to break the cycle (use lazy imports or dependency injection)"))
    return issues

# ─── S8: Config Validation ───────────────────────────────────────────────────

REQUIRED_CONFIG_ATTRS = [
    "dry_run", "light_mode", "max_history",
]

def scan_config() -> List[Issue]:
    issues = []
    cls, err = _try_import("core.config", "WalcheConfig")
    if err:
        issues.append(Issue("CONFIG", "HIGH", "core/config.py", 0,
                            f"WalcheConfig not importable: {err}", ""))
        return issues
    try:
        cfg = cls()
    except Exception as e:
        issues.append(Issue("CONFIG", "HIGH", "core/config.py", 0,
                            f"WalcheConfig() failed: {e}", ""))
        return issues
    for attr in REQUIRED_CONFIG_ATTRS:
        if not hasattr(cfg, attr):
            issues.append(Issue("CONFIG", "MEDIUM", "core/config.py", 0,
                                f"WalcheConfig missing attribute: {attr}",
                                f"Add '{attr}' to WalcheConfig"))
    return issues

# ─── S9: Test Coverage Gaps ──────────────────────────────────────────────────

def scan_test_gaps() -> List[Issue]:
    issues = []
    tested: Set[str] = set()
    test_dirs = [ROOT / "tests", ROOT / "test"]
    for td in test_dirs:
        if td.exists():
            for tf in td.rglob("test_*.py"):
                stem = tf.stem.replace("test_", "")
                tested.add(stem)

    core_modules = [f.stem for f in (ROOT / "core").glob("*.py")
                    if f.name != "__init__.py"] if (ROOT / "core").exists() else []
    for mod in core_modules:
        if mod not in tested and mod not in ("config",):
            issues.append(Issue("TEST", "LOW", f"core/{mod}.py", 0,
                                f"No test file found for core.{mod}",
                                f"Create tests/test_{mod}.py"))
    return issues

# ─── S10: Dead / Unreachable Module Files ────────────────────────────────────

def scan_orphan_scripts() -> List[Issue]:
    issues = []
    known_entrypoints = {"run_system", "loop_audit", "walche_validate", "walche_apply",
                         "walche_deep_scan", "__init__", "conftest"}
    orphan_prefixes = ("fix_", "temp_", "patch_", "diag_", "test_", "auto_")
    for f in (ROOT).glob("*.py"):
        if f.stem.startswith(orphan_prefixes) or f.stem not in known_entrypoints:
            # Check if imported anywhere
            stem = f.stem
            referenced = False
            for pf in _py_files():
                try:
                    src = pf.read_text(encoding="utf-8", errors="ignore")
                    if stem in src and pf != f:
                        referenced = True
                        break
                except Exception:
                    pass
            if not referenced:
                issues.append(Issue("BUG", "LOW", _rel(f), 0,
                                    f"Root-level script '{f.name}' appears unused/orphaned",
                                    "Delete if no longer needed, or move to tools/ if still required"))
    return issues

# ─── S11: Self-Tests ─────────────────────────────────────────────────────────

def run_tests() -> Tuple[List[Issue], Dict]:
    issues = []
    stats = {"run": 0, "passed": 0, "failed": 0}
    test_dirs = [ROOT / "tests", ROOT / "test"]
    test_files = []
    for td in test_dirs:
        if td.exists():
            test_files.extend(list(td.glob("test_*.py"))[:10])

    for tf in test_files:
        stats["run"] += 1
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", str(tf), "-x", "-q", "--tb=short"],
                capture_output=True, text=True, timeout=45, cwd=str(ROOT)
            )
            if r.returncode == 0:
                stats["passed"] += 1
            else:
                stats["failed"] += 1
                first_failure = r.stdout[:300] + r.stderr[:200]
                issues.append(Issue("TEST", "HIGH", _rel(tf), 0,
                                    f"Test failures: {first_failure.strip()[:250]}",
                                    "Fix failing tests before deployment"))
        except subprocess.TimeoutExpired:
            stats["failed"] += 1
            issues.append(Issue("TEST", "MEDIUM", _rel(tf), 0, "Test timed out (>45s)", ""))
        except Exception as e:
            stats["failed"] += 1
            issues.append(Issue("TEST", "MEDIUM", _rel(tf), 0, f"Test runner error: {e}", ""))
    return issues, stats

# ─── Runner ──────────────────────────────────────────────────────────────────

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
SEV_COLOR = {
    "CRITICAL": "\033[91m", "HIGH": "\033[91m",
    "MEDIUM":   "\033[93m", "LOW":  "\033[94m", "INFO": "\033[90m",
}
RESET = "\033[0m"

def _c(text: str, sev: str) -> str:
    return SEV_COLOR.get(sev, "") + text + RESET


def main():
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*70}")
    print(f"  WALCHE FULL PLATFORM SCAN  |  {ts[:19]}Z")
    print(f"  Root: {ROOT}")
    print(f"{'='*70}\n")

    all_issues: List[Issue] = []

    steps = [
        ("S1  Syntax Check",          scan_syntax),
        ("S2  Import Validation",      lambda: scan_imports()[0]),
        ("S3  Interface Validation",   scan_interfaces),
        ("S4  AST Bug Detection",      scan_ast_bugs),
        ("S5  EPM Contamination",      scan_contamination),
        ("S6  Loop Registry",          lambda: scan_loop_registry()[0]),
        ("S7  Circular Imports",       scan_circular_imports),
        ("S8  Config Validation",      scan_config),
        ("S9  Test Coverage Gaps",     scan_test_gaps),
        ("S10 Orphan Scripts",         scan_orphan_scripts),
    ]

    step_counts = {}
    for label, fn in steps:
        print(f"  Running {label}...", end=" ", flush=True)
        try:
            found = fn()
        except Exception as e:
            found = [Issue("BUG", "HIGH", "scanner", 0, f"Scanner error in {label}: {e}", "")]
        all_issues.extend(found)
        step_counts[label.strip()] = len(found)
        crits = sum(1 for i in found if i.severity == "CRITICAL")
        highs = sum(1 for i in found if i.severity == "HIGH")
        print(f"{len(found)} issue(s)" + (f"  [{crits} CRITICAL, {highs} HIGH]" if crits or highs else ""))

    # Run tests separately (slow)
    print(f"  Running S11 Self-Tests...", end=" ", flush=True)
    test_issues, test_stats = run_tests()
    all_issues.extend(test_issues)
    step_counts["S11 Self-Tests"] = len(test_issues)
    print(f"{test_stats['passed']}/{test_stats['run']} passed")

    # Collect supplementary info
    _, module_map = scan_imports()
    _, loop_info = scan_loop_registry()

    # Sort by severity
    all_issues.sort(key=lambda i: (SEVERITY_ORDER.get(i.severity, 9), i.category, i.file, i.line))

    # ── Print grouped report ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  ISSUES BY SEVERITY")
    print(f"{'─'*70}")

    by_sev: Dict[str, List[Issue]] = defaultdict(list)
    for iss in all_issues:
        by_sev[iss.severity].append(iss)

    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        group = by_sev.get(sev, [])
        if not group:
            continue
        print(f"\n  {_c(sev, sev)} ({len(group)})")
        for iss in group[:30]:  # cap display per severity
            loc = f"{iss.file}:{iss.line}" if iss.line else iss.file
            print(f"    [{iss.category}] {loc}")
            print(f"      {iss.message[:120]}")
            if iss.suggestion:
                print(f"      → {iss.suggestion[:100]}")
        if len(group) > 30:
            print(f"    ... and {len(group)-30} more")

    # ── Loop Registry Summary ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  LOOP REGISTRY")
    print(f"{'─'*70}")
    print(f"  Status:      {loop_info.get('status', 'UNKNOWN')}")
    print(f"  Loops found: {loop_info.get('count', 0)} / {EXPECTED_LOOP_COUNT} expected")
    print(f"  Connections: {loop_info.get('connections', 'N/A')}")
    if loop_info.get("loops"):
        for lp in loop_info["loops"][:18]:
            print(f"    • {lp}")

    # ── Module Map ────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  MODULE STATUS")
    print(f"{'─'*70}")
    for mod, info in module_map.items():
        status = info["status"]
        color = "\033[92m" if status == "OK" else "\033[91m"
        print(f"  {color}{status:4}{RESET}  {mod}")

    # ── Totals ────────────────────────────────────────────────────────────────
    criticals = sum(1 for i in all_issues if i.severity == "CRITICAL")
    highs     = sum(1 for i in all_issues if i.severity == "HIGH")
    mediums   = sum(1 for i in all_issues if i.severity == "MEDIUM")
    lows      = sum(1 for i in all_issues if i.severity == "LOW")

    print(f"\n{'='*70}")
    print(f"  TOTAL ISSUES: {len(all_issues)}")
    print(f"  CRITICAL: {criticals}  HIGH: {highs}  MEDIUM: {mediums}  LOW: {lows}")
    print(f"  Tests: {test_stats['passed']}/{test_stats['run']} passed")
    print(f"{'='*70}\n")

    # ── JSON report ───────────────────────────────────────────────────────────
    report_path = ROOT / "audit" / f"walche_scan_{ts[:10]}.json"
    report_path.parent.mkdir(exist_ok=True)
    report = {
        "timestamp": ts,
        "root": str(ROOT),
        "totals": {
            "critical": criticals, "high": highs,
            "medium": mediums, "low": lows, "total": len(all_issues),
        },
        "test_stats": test_stats,
        "loop_registry": loop_info,
        "module_map": module_map,
        "issues": [
            {
                "category": i.category, "severity": i.severity,
                "file": i.file, "line": i.line,
                "message": i.message, "suggestion": i.suggestion,
            }
            for i in all_issues
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report: {report_path}\n")

    return report


if __name__ == "__main__":
    main()
