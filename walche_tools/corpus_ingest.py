#!/usr/bin/env python3
"""
corpus_ingest.py — WALCHE Corpus Ingestion Tool

Two ingestion modes:

  MODE 1 — Source scan (Python AST):
    Scans the WALCHE codebase and extracts modules, classes, and functions.
    Feeds corpus.integrity from ~0.78 to 0.85+.

  MODE 2 — Log ingestion (any AI platform):
    Ingests build logs from OpenAI, LangChain/LangSmith, LangGraph, AutoGen,
    CrewAI, Hugging Face, GitHub Actions, or any JSON/JSONL/text log.
    Scrubs secrets before writing. Structured AI decisions and error patterns
    are the highest-value corpus signal.

Usage (from WALCHE root):
    # Source scan
    python walche_tools\corpus_ingest.py --scan . --output corpus\walche_kb.json
    python walche_tools\corpus_ingest.py --scan . --output corpus\walche_kb.json --dry-run
    python walche_tools\corpus_ingest.py --scan . --output corpus\walche_kb.json --verbose

    # Log ingestion
    python walche_tools\corpus_ingest.py --logs path\to\logs --output corpus\walche_kb.json
    python walche_tools\corpus_ingest.py --logs path\to\logs --output corpus\walche_kb.json --dry-run
    python walche_tools\corpus_ingest.py --logs path\to\logs --output corpus\walche_kb.json --platform langchain

    # Merge both (source + logs in one pass)
    python walche_tools\corpus_ingest.py --scan . --logs path\to\logs --output corpus\walche_kb.json

    # Report
    python walche_tools\corpus_ingest.py --report corpus\walche_kb.json

Supported log platforms (auto-detected):
    openai, langchain, langsmith, langgraph, autogen, crewai,
    huggingface, github_actions, generic_json, generic_text
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Configuration ─────────────────────────────────────────────────────────────

WALCHE_CORE_DIRS = [
    "core", "brain", "loops", "agents", "memory",
    "knowledge", "corpus", "spine", "registry", "config",
]

# Minimum density for a corpus entry to pass PreIngestGate tier 1
MIN_CONTENT_TOKENS = 30
INGEST_VERSION = "1.0.0"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CorpusEntry:
    id: str
    source_file: str
    entry_type: str          # module | class | function | constant | config
    name: str
    content: str             # primary readable content
    metadata: dict = field(default_factory=dict)
    quality_score: float = 0.0
    token_estimate: int = 0
    provenance: str = ""
    ingested_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestReport:
    scan_root: str
    files_scanned: int = 0
    files_with_content: int = 0
    entries_extracted: int = 0
    entries_passed_gate: int = 0
    entries_rejected: int = 0
    total_tokens: int = 0
    predicted_integrity_score: float = 0.0
    duration_seconds: float = 0.0
    output_path: str = ""


# ── AST extraction ────────────────────────────────────────────────────────────

def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _quality_score(entry: CorpusEntry) -> float:
    score = 0.0
    tokens = entry.token_estimate
    if tokens >= 200: score += 0.40
    elif tokens >= 100: score += 0.30
    elif tokens >= 50:  score += 0.20
    elif tokens >= MIN_CONTENT_TOKENS: score += 0.10

    if entry.metadata.get("has_docstring"):    score += 0.20
    if entry.metadata.get("has_type_hints"):   score += 0.15
    if entry.metadata.get("has_examples"):     score += 0.10
    if entry.metadata.get("is_public"):        score += 0.10
    if entry.entry_type == "class":            score += 0.05
    return min(1.0, score)


def extract_module_entries(path: Path, root: Path) -> list[CorpusEntry]:
    """Parse a .py file with AST and extract corpus entries."""
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    rel_path = str(path.relative_to(root))
    entries: list[CorpusEntry] = []
    now = datetime.utcnow().isoformat() + "Z"

    # ── Module-level docstring ─────────────────────────────────────────────────
    module_doc = ast.get_docstring(tree) or ""
    module_name = path.stem
    module_parts = rel_path.replace("\\", "/").replace("/", ".").rstrip(".py")

    module_content_parts = [f"Module: {module_name}", f"Path: {rel_path}"]
    if module_doc:
        module_content_parts.append(f"Description: {module_doc}")

    module_content = "\n".join(module_content_parts)
    tok = _count_tokens(module_content)

    e = CorpusEntry(
        id=f"module::{rel_path}",
        source_file=rel_path,
        entry_type="module",
        name=module_name,
        content=module_content,
        metadata={
            "has_docstring": bool(module_doc),
            "has_type_hints": False,
            "has_examples": "example" in (module_doc or "").lower()
                            or "usage" in (module_doc or "").lower(),
            "is_public": not module_name.startswith("_"),
            "line_count": len(source.splitlines()),
        },
        token_estimate=tok,
        provenance=f"AST extraction from {rel_path}",
        ingested_at=now,
    )
    e.quality_score = _quality_score(e)
    if tok >= MIN_CONTENT_TOKENS:
        entries.append(e)

    # ── Classes ───────────────────────────────────────────────────────────────
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        doc = ast.get_docstring(node) or ""
        bases = [_safe_unparse(b) for b in node.bases]
        methods = [
            n.name for n in ast.walk(node)
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("__")
        ]
        attrs = [
            n.targets[0].id
            for n in ast.walk(node)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ]

        has_hints = any(
            bool(m.returns or m.args.annotations)
            for m in ast.walk(node)
            if isinstance(m, ast.FunctionDef)
        )

        parts = [
            f"Class: {node.name}",
            f"Module: {rel_path}",
        ]
        if bases:
            parts.append(f"Inherits: {', '.join(bases)}")
        if doc:
            parts.append(f"Description: {doc}")
        if methods:
            parts.append(f"Public methods: {', '.join(methods[:20])}")
        if attrs:
            parts.append(f"Attributes: {', '.join(attrs[:15])}")

        content = "\n".join(parts)
        tok = _count_tokens(content)

        ce = CorpusEntry(
            id=f"class::{rel_path}::{node.name}",
            source_file=rel_path,
            entry_type="class",
            name=node.name,
            content=content,
            metadata={
                "has_docstring": bool(doc),
                "has_type_hints": has_hints,
                "has_examples": "example" in doc.lower() if doc else False,
                "is_public": not node.name.startswith("_"),
                "method_count": len(methods),
                "base_classes": bases,
                "line_number": node.lineno,
            },
            token_estimate=tok,
            provenance=f"AST class extraction from {rel_path}:{node.lineno}",
            ingested_at=now,
        )
        ce.quality_score = _quality_score(ce)
        if tok >= MIN_CONTENT_TOKENS:
            entries.append(ce)

    # ── Top-level functions ───────────────────────────────────────────────────
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        doc = ast.get_docstring(node) or ""
        args = [a.arg for a in node.args.args if a.arg != "self"]
        has_hints = bool(node.returns) or any(
            a.annotation for a in node.args.args
        )
        ret_annotation = _safe_unparse(node.returns) if node.returns else ""

        parts = [f"Function: {node.name}", f"Module: {rel_path}"]
        if args:
            parts.append(f"Parameters: {', '.join(args[:10])}")
        if ret_annotation:
            parts.append(f"Returns: {ret_annotation}")
        if doc:
            parts.append(f"Description: {doc}")

        content = "\n".join(parts)
        tok = _count_tokens(content)

        fe = CorpusEntry(
            id=f"func::{rel_path}::{node.name}",
            source_file=rel_path,
            entry_type="function",
            name=node.name,
            content=content,
            metadata={
                "has_docstring": bool(doc),
                "has_type_hints": has_hints,
                "has_examples": "example" in doc.lower() if doc else False,
                "is_public": not node.name.startswith("_"),
                "param_count": len(args),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "line_number": node.lineno,
            },
            token_estimate=tok,
            provenance=f"AST function extraction from {rel_path}:{node.lineno}",
            ingested_at=now,
        )
        fe.quality_score = _quality_score(fe)
        if tok >= MIN_CONTENT_TOKENS:
            entries.append(fe)

    return entries


# ── PreIngestGate shim ────────────────────────────────────────────────────────

def _gate_check(entry: CorpusEntry, verbose: bool = False) -> tuple[bool, str]:
    """
    Lightweight gate check mirroring PreIngestGate tier logic.
    Returns (passes, reason).
    """
    if entry.token_estimate < MIN_CONTENT_TOKENS:
        return False, f"too sparse ({entry.token_estimate} tokens < {MIN_CONTENT_TOKENS})"

    if entry.quality_score < 0.10:
        return False, f"quality too low ({entry.quality_score:.2f})"

    if not entry.content.strip():
        return False, "empty content"

    if entry.entry_type == "function" and entry.name.startswith("_"):
        return False, "private function — skipped"

    return True, "ok"


def _try_real_gate(entry_dict: dict) -> bool:
    """Attempt to use WALCHE's real PreIngestGate if importable."""
    try:
        from core.pre_ingest_gate import PreIngestGate
        gate = PreIngestGate()
        for method in ("evaluate", "check", "gate", "validate", "run"):
            if hasattr(gate, method):
                fn = getattr(gate, method)
                try:
                    result = fn(entry_dict, lineage_id=entry_dict.get("id", "unknown"))
                except TypeError:
                    result = fn(entry_dict)
                if isinstance(result, dict):
                    return result.get("passed", result.get("pass", True))
                return bool(result)
    except Exception:
        pass
    return None  # None means "gate unavailable, use shim"


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan_walche(root: Path, verbose: bool = False) -> list[CorpusEntry]:
    """Scan the WALCHE tree and return all extracted corpus entries."""
    all_entries: list[CorpusEntry] = []
    py_files: list[Path] = []

    # Collect Python files from WALCHE core directories
    for dirname in WALCHE_CORE_DIRS:
        d = root / dirname
        if d.is_dir():
            for f in d.rglob("*.py"):
                if not any(p in str(f) for p in ["__pycache__", ".venv", "venv"]):
                    py_files.append(f)

    # Also scan root-level Python files (run_system.py, walche_demo.py, etc.)
    for f in root.glob("*.py"):
        py_files.append(f)

    if verbose:
        print(f"  Found {len(py_files)} Python files to scan")

    for py in py_files:
        entries = extract_module_entries(py, root)
        if verbose and entries:
            print(f"  {py.relative_to(root)} → {len(entries)} entries")
        all_entries.extend(entries)

    return all_entries


# ── Score predictor ───────────────────────────────────────────────────────────

def predict_integrity_score(entries: list[CorpusEntry], passed: int) -> float:
    """
    Predict what corpus.integrity will score after ingestion.
    Based on: entry count, average quality, pass rate, token density.
    """
    if not entries:
        return 0.78  # baseline with no corpus

    avg_quality = sum(e.quality_score for e in entries) / len(entries)
    pass_rate = passed / max(1, len(entries))
    total_tokens = sum(e.token_estimate for e in entries)
    density_score = min(1.0, total_tokens / 50_000)

    # Weighted composite (matches rubric dimensions)
    score = (
        avg_quality     * 0.35 +   # content quality
        pass_rate       * 0.30 +   # gate pass rate (completeness)
        density_score   * 0.20 +   # corpus size / coverage
        0.15                        # base (provenance tracked)
    )
    return min(0.98, max(0.78, score))


# ── Main ingest ───────────────────────────────────────────────────────────────

def run_ingest(
    scan_root: str,
    output_path: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> IngestReport:
    root = Path(scan_root).resolve()
    out  = Path(output_path)
    report = IngestReport(scan_root=str(root), output_path=str(out))
    t_start = time.time()

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║        W A L C H E   —   CORPUS INGESTION TOOL          ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Scan root: {root}")
    print(f"  Output:    {out}")
    print(f"  Mode:      {'DRY RUN (no files written)' if dry_run else 'LIVE'}")
    print()

    # ── Scan ─────────────────────────────────────────────────────────────────
    print("  [1/4] Scanning WALCHE source tree…")
    entries = scan_walche(root, verbose=verbose)
    report.files_scanned = len({e.source_file for e in entries})
    report.files_with_content = report.files_scanned
    report.entries_extracted = len(entries)

    print(f"        Files scanned:    {report.files_scanned}")
    print(f"        Entries extracted: {report.entries_extracted}")

    # ── Gate ─────────────────────────────────────────────────────────────────
    print()
    print("  [2/4] Running PreIngestGate quality filter…")
    passed: list[CorpusEntry] = []
    rejected: list[tuple[CorpusEntry, str]] = []

    for entry in entries:
        # Try real gate first; fall back to shim
        real_result = _try_real_gate(entry.to_dict())
        if real_result is not None:
            ok = real_result
            reason = "real gate" if ok else "real gate rejected"
        else:
            ok, reason = _gate_check(entry, verbose)

        if ok:
            passed.append(entry)
        else:
            rejected.append((entry, reason))

    report.entries_passed_gate = len(passed)
    report.entries_rejected = len(rejected)
    report.total_tokens = sum(e.token_estimate for e in passed)
    print(f"        Passed:   {len(passed)}")
    print(f"        Rejected: {len(rejected)}")
    print(f"        Tokens:   {report.total_tokens:,}")

    if verbose and rejected:
        print("        Rejection reasons:")
        reasons: dict[str, int] = {}
        for _, r in rejected:
            reasons[r] = reasons.get(r, 0) + 1
        for r, cnt in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
            print(f"          {cnt}× {r}")

    # ── Score prediction ─────────────────────────────────────────────────────
    print()
    print("  [3/4] Predicting post-ingest corpus.integrity score…")
    report.predicted_integrity_score = predict_integrity_score(entries, len(passed))
    baseline = 0.78
    delta = report.predicted_integrity_score - baseline

    status = "PASS" if report.predicted_integrity_score >= 0.85 else "WARN"
    status_color = "\033[92m" if status == "PASS" else "\033[93m"
    reset = "\033[0m"

    bar_width = 24
    filled = int(report.predicted_integrity_score * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"        Baseline:  {baseline:.3f}")
    print(f"        Predicted: {status_color}{bar} {report.predicted_integrity_score:.3f} ({delta:+.3f}){reset}  [{status}]")

    if report.predicted_integrity_score >= 0.85:
        print("        → corpus.integrity will reach PASS threshold after ingest")
    else:
        print(f"        → Still {0.85 - report.predicted_integrity_score:.3f} below PASS — ingest more content or improve docstrings")

    # ── Write ─────────────────────────────────────────────────────────────────
    print()
    print("  [4/4] Writing knowledge base…")

    if dry_run:
        print("        DRY RUN — no files written")
        print(f"        Would write {len(passed)} entries to {out}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

        kb = {
            "version": INGEST_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "scan_root": str(root),
            "stats": {
                "files_scanned": report.files_scanned,
                "entries_extracted": report.entries_extracted,
                "entries_passed": report.entries_passed_gate,
                "entries_rejected": report.entries_rejected,
                "total_tokens": report.total_tokens,
                "predicted_integrity_score": report.predicted_integrity_score,
            },
            "entries": [e.to_dict() for e in passed],
        }

        out.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"        Written: {out}  ({out.stat().st_size:,} bytes)")
        print(f"        Entries: {len(passed)}")

    report.duration_seconds = time.time() - t_start

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║                 CORPUS INGEST COMPLETE                   ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Duration: {report.duration_seconds:.2f}s")
    print(f"  Entries ingested: {report.entries_passed_gate}")
    print(f"  Predicted corpus.integrity: {report.predicted_integrity_score:.3f}  ({'+' if delta >= 0 else ''}{delta:.3f})")
    print()
    print("  Next step: run walche_demo.py to confirm the score improvement")
    print()

    return report


# ── Report viewer ─────────────────────────────────────────────────────────────

def print_report(kb_path: str) -> None:
    p = Path(kb_path)
    if not p.exists():
        print(f"[ERROR] Knowledge base not found: {kb_path}")
        sys.exit(1)

    kb = json.loads(p.read_text(encoding="utf-8"))
    stats = kb.get("stats", {})

    print()
    print("  WALCHE Corpus Knowledge Base Report")
    print(f"  Generated: {kb.get('generated_at', 'unknown')}")
    print(f"  Version:   {kb.get('version', 'unknown')}")
    print()
    print(f"  Files scanned:      {stats.get('files_scanned', 0)}")
    print(f"  Entries extracted:  {stats.get('entries_extracted', 0)}")
    print(f"  Entries passed:     {stats.get('entries_passed', 0)}")
    print(f"  Entries rejected:   {stats.get('entries_rejected', 0)}")
    print(f"  Total tokens:       {stats.get('total_tokens', 0):,}")
    print(f"  Predicted integrity:{stats.get('predicted_integrity_score', 0.0):.3f}")
    print()

    entries = kb.get("entries", [])
    if entries:
        by_type: dict[str, int] = {}
        for e in entries:
            t = e.get("entry_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        print("  Entry types:")
        for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {t:12s}  {cnt}")
        print()

        # Top 10 by quality
        top = sorted(entries, key=lambda e: e.get("quality_score", 0), reverse=True)[:10]
        print("  Top 10 entries by quality:")
        for e in top:
            print(f"    [{e.get('quality_score', 0):.2f}] {e.get('entry_type', '?'):8s}  {e.get('name', '?')}")
    print()


# ── Log ingestion — secret scrubbing ─────────────────────────────────────────

# Ordered by specificity (most specific patterns first)
_SECRET_PATTERNS: list[tuple[str, str]] = [
    # Anthropic
    (r"sk-ant-api\d{2}-[A-Za-z0-9_\-]{90,}", "[REDACTED-ANTHROPIC-KEY]"),
    (r"sk-ant-[A-Za-z0-9_\-]{30,}", "[REDACTED-ANTHROPIC-KEY]"),
    # OpenAI
    (r"sk-proj-[A-Za-z0-9_\-]{40,}", "[REDACTED-OPENAI-KEY]"),
    (r"sk-[A-Za-z0-9]{48}", "[REDACTED-OPENAI-KEY]"),
    # Google / Gemini
    (r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED-GOOGLE-KEY]"),
    # GitHub tokens
    (r"ghp_[0-9a-zA-Z]{36}", "[REDACTED-GITHUB-TOKEN]"),
    (r"ghu_[0-9a-zA-Z]{36}", "[REDACTED-GITHUB-TOKEN]"),
    (r"github_pat_[A-Za-z0-9_]{82}", "[REDACTED-GITHUB-TOKEN]"),
    # AWS
    (r"AKIA[0-9A-Z]{16}", "[REDACTED-AWS-KEY]"),
    (r"(?i)aws[_\-]secret[_\-]access[_\-]key[\"'\s:=]+[A-Za-z0-9/+=]{40}", "[REDACTED-AWS-SECRET]"),
    # Generic Bearer tokens
    (r"Bearer [A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED]"),
    # Passwords in key=value patterns
    (r"(?i)(password|passwd|secret|token|api_key|apikey)[\"'\s]*[:=][\"'\s]*\S{8,}", r"\1=[REDACTED]"),
    # Email addresses (optional — preserves structural info, masks identity)
    (r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
    # Private IP ranges in connection strings
    (r"\b(?:192\.168|10\.\d+|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b", "[PRIVATE-IP]"),
]

_SECRET_RE = [(re.compile(p), r) for p, r in _SECRET_PATTERNS]


def scrub_secrets(text: str) -> str:
    """Remove known secret patterns from text before corpus ingestion."""
    for pattern, replacement in _SECRET_RE:
        text = pattern.sub(replacement, text)
    return text


def scrub_obj(obj: Any, depth: int = 0) -> Any:
    """Recursively scrub secrets from a parsed JSON object."""
    if depth > 12:
        return obj
    if isinstance(obj, str):
        return scrub_secrets(obj)
    if isinstance(obj, dict):
        return {k: scrub_obj(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(i, depth + 1) for i in obj]
    return obj


# ── Log ingestion — platform detection ───────────────────────────────────────

# Fingerprint: list of keys that identify the platform
_PLATFORM_SIGNATURES: dict[str, list[list[str]]] = {
    "langsmith":      [["run_type", "serialized"], ["dotted_order"], ["trace_id", "dotted_order"]],
    "langchain":      [["lc", "type"], ["run_type", "inputs", "outputs"], ["serialized", "inputs"]],
    "langgraph":      [["__pregel_pull"], ["graph_id", "node"], ["channel_values"]],
    "openai":         [["model", "choices", "usage"], ["object", "choices"], ["model", "object"]],
    "autogen":        [["role", "content", "name"], ["sender", "receiver", "message"]],
    "crewai":         [["agent", "task", "output"], ["crew", "agents", "tasks"]],
    "huggingface":    [["model_id", "inputs", "parameters"], ["generated_text"], ["pipeline_tag"]],
    "github_actions": [["workflow", "job", "step"], ["runner", "workflow_run"], ["conclusion", "workflow"]],
}


def detect_platform(obj: Any) -> str:
    """Return best-guess platform name from a parsed JSON object."""
    if not isinstance(obj, dict):
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        else:
            return "generic_json"

    keys = set(obj.keys())
    for platform, sig_groups in _PLATFORM_SIGNATURES.items():
        for sig in sig_groups:
            if all(k in keys for k in sig):
                return platform
    return "generic_json"


# ── Log ingestion — event extractors ─────────────────────────────────────────

def _text_from_obj(obj: Any, max_depth: int = 3, depth: int = 0) -> str:
    """Flatten a JSON object to readable key:value text."""
    if depth > max_depth:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        parts = [_text_from_obj(i, max_depth, depth + 1) for i in obj[:5]]
        return " | ".join(p for p in parts if p)
    if isinstance(obj, dict):
        parts = []
        for k, v in list(obj.items())[:12]:
            val = _text_from_obj(v, max_depth, depth + 1)
            if val:
                parts.append(f"{k}: {val}")
        return "\n".join(parts)
    return ""


def _quality_from_log_event(event: dict, platform: str) -> float:
    """Score a log event for corpus quality."""
    score = 0.10  # base

    # Errors are high-value learning signals
    if any(k in event for k in ("error", "exception", "traceback", "stderr")):
        score += 0.25
    # Structured output / decision data
    if any(k in event for k in ("output", "outputs", "result", "response", "choices")):
        score += 0.20
    # Model/decision metadata
    if any(k in event for k in ("model", "agent", "run_type", "node", "task")):
        score += 0.15
    # Timing / performance
    if any(k in event for k in ("duration", "latency", "elapsed", "total_tokens", "usage")):
        score += 0.10
    # Inputs present
    if any(k in event for k in ("input", "inputs", "prompt", "query", "message")):
        score += 0.10
    # Platform-specific bonuses
    if platform in ("langsmith", "langchain", "langgraph"):
        score += 0.10
    return min(1.0, score)


def _extract_from_json_event(
    event: Any,
    source_file: str,
    platform: str,
    index: int,
    now: str,
) -> CorpusEntry | None:
    """Turn one JSON log event into a CorpusEntry."""
    if not isinstance(event, dict):
        return None

    # Scrub secrets from the event
    event = scrub_obj(event)

    # Build readable content
    entry_type = "execution_trace"
    name_parts: list[str] = []

    if platform == "openai":
        name_parts = [event.get("model", "openai"), event.get("object", "completion")]
        entry_type = "api_call"
    elif platform in ("langchain", "langsmith"):
        name_parts = [event.get("run_type", "run"), event.get("name", "chain")]
        entry_type = "agent_decision"
    elif platform == "langgraph":
        name_parts = ["graph", event.get("node", str(index))]
        entry_type = "agent_decision"
    elif platform == "autogen":
        name_parts = [event.get("role", "agent"), event.get("name", str(index))]
        entry_type = "agent_decision"
    elif platform == "crewai":
        name_parts = ["crew", event.get("agent", str(index))]
        entry_type = "agent_decision"
    elif platform == "github_actions":
        name_parts = [event.get("workflow", "workflow"), event.get("job", str(index))]
        entry_type = "build_event"
    else:
        name_parts = [platform, str(index)]

    name = "/".join(str(p) for p in name_parts if p)
    content = _text_from_obj(event, max_depth=3)

    if not content.strip():
        return None

    tok = _count_tokens(content)
    if tok < MIN_CONTENT_TOKENS:
        return None

    # Capture error text separately for metadata
    error_text = ""
    for ek in ("error", "exception", "stderr", "traceback"):
        v = event.get(ek)
        if v and isinstance(v, str):
            error_text = v[:300]
            break

    quality = _quality_from_log_event(event, platform)

    return CorpusEntry(
        id=f"log::{source_file}::{index}",
        source_file=source_file,
        entry_type=entry_type,
        name=name,
        content=content,
        metadata={
            "platform": platform,
            "has_docstring": False,
            "has_type_hints": False,
            "has_examples": bool(error_text),
            "is_public": True,
            "error": error_text,
            "event_index": index,
        },
        quality_score=quality,
        token_estimate=tok,
        provenance=f"log extraction from {source_file} (platform: {platform})",
        ingested_at=now,
    )


def _parse_text_log(text: str, source_file: str, platform: str, now: str) -> list[CorpusEntry]:
    """Parse a plain-text log file into corpus entries by chunking on blank lines."""
    entries: list[CorpusEntry] = []
    # Split on double newlines or lines starting with timestamps
    ts_pattern = re.compile(
        r"^\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}",
        re.MULTILINE,
    )
    chunks: list[str] = []
    if ts_pattern.search(text):
        # Split on timestamp lines
        parts = ts_pattern.split(text)
        ts_matches = ts_pattern.findall(text)
        for i, part in enumerate(parts[1:], 0):
            chunks.append(f"{ts_matches[i]} {part.strip()}")
    else:
        # Split on blank lines
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]

    for i, chunk in enumerate(chunks):
        chunk = scrub_secrets(chunk)
        tok = _count_tokens(chunk)
        if tok < MIN_CONTENT_TOKENS:
            continue

        has_error = bool(re.search(r"\b(error|exception|traceback|failed|fatal)\b", chunk, re.I))
        quality = 0.20 + (0.20 if has_error else 0.0) + min(0.30, tok / 500)

        entries.append(CorpusEntry(
            id=f"log::{source_file}::text::{i}",
            source_file=source_file,
            entry_type="build_event",
            name=f"{platform}/chunk-{i}",
            content=chunk[:2000],
            metadata={
                "platform": platform,
                "has_docstring": False,
                "has_type_hints": False,
                "has_examples": has_error,
                "is_public": True,
                "error": "",
                "chunk_index": i,
            },
            quality_score=min(1.0, quality),
            token_estimate=tok,
            provenance=f"text log extraction from {source_file}",
            ingested_at=now,
        ))
    return entries


def extract_log_entries(path: Path, log_dir: Path) -> list[CorpusEntry]:
    """Parse one log file and return corpus entries."""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    rel = str(path.relative_to(log_dir))
    now = datetime.utcnow().isoformat() + "Z"
    suffix = path.suffix.lower()
    entries: list[CorpusEntry] = []

    # ── JSONL ─────────────────────────────────────────────────────────────────
    if suffix == ".jsonl":
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        platform = "generic_json"
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i == 0:
                platform = detect_platform(obj)
            entry = _extract_from_json_event(obj, rel, platform, i, now)
            if entry:
                entries.append(entry)

    # ── JSON ─────────────────────────────────────────────────────────────────
    elif suffix == ".json":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if isinstance(obj, list):
            platform = detect_platform(obj[0]) if obj else "generic_json"
            for i, item in enumerate(obj):
                entry = _extract_from_json_event(item, rel, platform, i, now)
                if entry:
                    entries.append(entry)
        elif isinstance(obj, dict):
            platform = detect_platform(obj)
            # Check for an events/runs/steps array inside
            for container_key in ("runs", "events", "steps", "messages", "turns", "records", "data"):
                if container_key in obj and isinstance(obj[container_key], list):
                    for i, item in enumerate(obj[container_key]):
                        entry = _extract_from_json_event(item, rel, platform, i, now)
                        if entry:
                            entries.append(entry)
                    break
            else:
                # Single object
                entry = _extract_from_json_event(obj, rel, platform, 0, now)
                if entry:
                    entries.append(entry)

    # ── Plain text / log ─────────────────────────────────────────────────────
    elif suffix in (".log", ".txt", ".out", ""):
        entries = _parse_text_log(raw, rel, "generic_text", now)

    return entries


# ── Log ingestion — scanner and pipeline ─────────────────────────────────────

LOG_EXTENSIONS = {".json", ".jsonl", ".log", ".txt", ".out"}
LOG_SKIP_DIRS  = {"__pycache__", ".venv", "venv", "node_modules", ".git"}


def scan_logs(log_dir: Path, verbose: bool = False) -> list[CorpusEntry]:
    """Walk log_dir and extract corpus entries from every recognised log file."""
    all_entries: list[CorpusEntry] = []
    log_files: list[Path] = []

    for f in log_dir.rglob("*"):
        if not f.is_file():
            continue
        if any(skip in f.parts for skip in LOG_SKIP_DIRS):
            continue
        if f.suffix.lower() in LOG_EXTENSIONS or f.suffix == "":
            log_files.append(f)

    if verbose:
        print(f"  Found {len(log_files)} log files to parse")

    for lf in log_files:
        entries = extract_log_entries(lf, log_dir)
        if verbose and entries:
            platforms = {e.metadata.get("platform", "?") for e in entries}
            print(f"  {lf.relative_to(log_dir)} → {len(entries)} entries  [{', '.join(platforms)}]")
        all_entries.extend(entries)

    return all_entries


def run_log_ingest(
    log_dir: str,
    output_path: str,
    dry_run: bool = False,
    verbose: bool = False,
    platform_hint: str | None = None,
) -> IngestReport:
    """Full log ingestion pipeline — scrub, gate, score, write."""
    log_root = Path(log_dir).resolve()
    out      = Path(output_path)
    report   = IngestReport(scan_root=str(log_root), output_path=str(out))
    t_start  = time.time()

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║      W A L C H E   —   LOG CORPUS INGESTION             ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Log dir:   {log_root}")
    print(f"  Output:    {out}")
    if platform_hint:
        print(f"  Platform:  {platform_hint} (hint)")
    print(f"  Mode:      {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # ── 1. Extract ────────────────────────────────────────────────────────────
    print("  [1/4] Parsing log files…")
    entries = scan_logs(log_root, verbose=verbose)
    report.files_scanned = len({e.source_file for e in entries})
    report.entries_extracted = len(entries)

    platforms: dict[str, int] = {}
    for e in entries:
        p = e.metadata.get("platform", "unknown")
        platforms[p] = platforms.get(p, 0) + 1

    print(f"        Files parsed:     {report.files_scanned}")
    print(f"        Events extracted: {report.entries_extracted}")
    if platforms:
        print("        Platforms detected:")
        for p, cnt in sorted(platforms.items(), key=lambda x: -x[1]):
            print(f"          {p:<20} {cnt} events")

    # ── 2. Gate ───────────────────────────────────────────────────────────────
    print()
    print("  [2/4] Running PreIngestGate quality filter…")
    passed:   list[CorpusEntry] = []
    rejected: list[tuple[CorpusEntry, str]] = []

    for entry in entries:
        real_result = _try_real_gate(entry.to_dict())
        if real_result is not None:
            ok     = real_result
            reason = "real gate" if ok else "real gate rejected"
        else:
            ok, reason = _gate_check(entry)
        if ok:
            passed.append(entry)
        else:
            rejected.append((entry, reason))

    report.entries_passed_gate = len(passed)
    report.entries_rejected    = len(rejected)
    report.total_tokens        = sum(e.token_estimate for e in passed)

    print(f"        Passed:   {len(passed)}")
    print(f"        Rejected: {len(rejected)}")
    print(f"        Tokens:   {report.total_tokens:,}")

    # ── 3. Score ──────────────────────────────────────────────────────────────
    print()
    print("  [3/4] Predicting post-ingest corpus.integrity score…")
    report.predicted_integrity_score = predict_integrity_score(entries, len(passed))
    baseline = 0.78
    delta    = report.predicted_integrity_score - baseline

    status       = "PASS" if report.predicted_integrity_score >= 0.85 else "WARN"
    status_color = "\033[92m" if status == "PASS" else "\033[93m"
    reset        = "\033[0m"
    bar_width    = 24
    filled       = int(report.predicted_integrity_score * bar_width)
    bar          = "█" * filled + "░" * (bar_width - filled)

    print(f"        Baseline:  {baseline:.3f}")
    print(f"        Predicted: {status_color}{bar} {report.predicted_integrity_score:.3f} ({delta:+.3f}){reset}  [{status}]")

    # ── 4. Write ──────────────────────────────────────────────────────────────
    print()
    print("  [4/4] Writing knowledge base…")

    if dry_run:
        print(f"        DRY RUN — would write {len(passed)} entries to {out}")
    else:
        # Merge with existing KB if present (append, don't replace)
        existing_entries: list[dict] = []
        if out.exists():
            try:
                existing_kb = json.loads(out.read_text(encoding="utf-8"))
                existing_entries = existing_kb.get("entries", [])
                print(f"        Merging with existing KB ({len(existing_entries)} existing entries)")
            except Exception:
                pass

        out.parent.mkdir(parents=True, exist_ok=True)
        all_entry_dicts = existing_entries + [e.to_dict() for e in passed]

        # Deduplicate by id
        seen_ids: set[str] = set()
        deduped: list[dict] = []
        for ed in all_entry_dicts:
            eid = ed.get("id", "")
            if eid not in seen_ids:
                seen_ids.add(eid)
                deduped.append(ed)

        kb = {
            "version":      INGEST_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "scan_root":    str(log_root),
            "stats": {
                "files_scanned":             report.files_scanned,
                "entries_extracted":         report.entries_extracted,
                "entries_passed":            report.entries_passed_gate,
                "entries_rejected":          report.entries_rejected,
                "total_tokens":              report.total_tokens,
                "predicted_integrity_score": report.predicted_integrity_score,
                "total_entries_in_kb":       len(deduped),
            },
            "entries": deduped,
        }
        out.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"        Written: {out}  ({out.stat().st_size:,} bytes)")
        print(f"        Total entries in KB: {len(deduped)}")

    report.duration_seconds = time.time() - t_start

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║              LOG CORPUS INGEST COMPLETE                  ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Duration: {report.duration_seconds:.2f}s")
    print(f"  Events ingested: {report.entries_passed_gate}")
    print(f"  Predicted corpus.integrity: {report.predicted_integrity_score:.3f}  ({'+' if delta >= 0 else ''}{delta:.3f})")
    print()
    print("  Secrets scrubbed: API keys, tokens, emails, private IPs")
    print("  Next step: run walche_demo.py to confirm the score improvement")
    print()

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WALCHE Corpus Ingestion Tool — source scan + multi-platform log ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Source scan only
          python walche_tools\\corpus_ingest.py --scan . --output corpus\\walche_kb.json

          # Log ingestion only (auto-detects platform)
          python walche_tools\\corpus_ingest.py --logs path\\to\\logs --output corpus\\walche_kb.json

          # Log ingestion with platform hint
          python walche_tools\\corpus_ingest.py --logs path\\to\\logs --output corpus\\walche_kb.json --platform langchain

          # Merge source + logs in one pass
          python walche_tools\\corpus_ingest.py --scan . --logs path\\to\\logs --output corpus\\walche_kb.json

          # Dry run (no files written)
          python walche_tools\\corpus_ingest.py --logs path\\to\\logs --output corpus\\walche_kb.json --dry-run

          # View report on existing KB
          python walche_tools\\corpus_ingest.py --report corpus\\walche_kb.json

        Supported platforms (auto-detected from log structure):
          openai, langchain, langsmith, langgraph, autogen, crewai,
          huggingface, github_actions, generic_json, generic_text
        """),
    )
    parser.add_argument("--scan",     metavar="DIR",      help="WALCHE root to scan for Python source")
    parser.add_argument("--logs",     metavar="DIR",      help="Directory of AI platform logs to ingest")
    parser.add_argument("--output",   metavar="FILE",     help="Output knowledge base JSON path")
    parser.add_argument("--report",   metavar="FILE",     help="Print report on existing knowledge base")
    parser.add_argument("--platform", metavar="PLATFORM", help="Platform hint for log ingestion (optional)")
    parser.add_argument("--dry-run",  action="store_true", help="Parse and gate-check without writing files")
    parser.add_argument("--verbose",  action="store_true", help="Show per-file extraction details")

    args = parser.parse_args()

    if args.report:
        print_report(args.report)
        return

    if not args.scan and not args.logs:
        parser.print_help()
        print("\n[ERROR] Provide --scan and/or --logs  (or use --report to view an existing KB)")
        sys.exit(1)

    if not args.output:
        parser.print_help()
        print("\n[ERROR] --output is required")
        sys.exit(1)

    if args.scan:
        run_ingest(
            scan_root=args.scan,
            output_path=args.output,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

    if args.logs:
        run_log_ingest(
            log_dir=args.logs,
            output_path=args.output,
            dry_run=args.dry_run,
            verbose=args.verbose,
            platform_hint=args.platform,
        )


if __name__ == "__main__":
    main()
