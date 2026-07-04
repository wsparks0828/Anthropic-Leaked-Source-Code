#!/usr/bin/env python3
"""
provenance_viewer.py — WALCHE Provenance Log Viewer + Score Trend Tracker

Reads all walche_demo_*.json provenance logs, builds a cross-session score
history, and shows trends against the WALCHE baseline (0.9133, osModa).

Usage (from WALCHE root):
    python walche_tools\provenance_viewer.py
    python walche_tools\provenance_viewer.py --sessions 10
    python walche_tools\provenance_viewer.py --domain corpus.integrity
    python walche_tools\provenance_viewer.py --export
    python walche_tools\provenance_viewer.py --export --quiet
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

WALCHE_BASELINE  = 0.9133          # osModa benchmark, 2026-06-29
PASS_THRESHOLD   = 0.85
WARN_THRESHOLD   = 0.70
HISTORY_FILE     = "corpus/score_history.jsonl"
LOG_GLOB         = "logs/walche_demo_*.json"

# Also read provenance logs written by run_system.py heal phase
HEAL_LOG_GLOB    = "logs/provenance_log*.json"

# ── Terminal colors ───────────────────────────────────────────────────────────

class C:
    G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    CY = "\033[96m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"

def _col(t, c): return f"{c}{t}{C.X}"
def _sc(v):
    c = C.G if v >= PASS_THRESHOLD else C.Y if v >= WARN_THRESHOLD else C.R
    return _col(f"{v:.3f}", c)
def _bar(v, w=20):
    n = int(v * w)
    c = C.G if v >= PASS_THRESHOLD else C.Y if v >= WARN_THRESHOLD else C.R
    return f"{c}{'█'*n}{'░'*(w-n)}{C.X}"
def _verdict_col(v):
    if v == "GO":                  return _col(v, C.G)
    if v == "GO-WITH-CONDITIONS":  return _col(v, C.Y)
    return _col(v, C.R)


# ── Sparkline ─────────────────────────────────────────────────────────────────

_SPARK = " ▁▂▃▄▅▆▇█"

def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    rng = hi - lo or 1.0
    chars = []
    for v in values:
        idx = int((v - lo) / rng * (len(_SPARK) - 1))
        chars.append(_SPARK[idx])
    return "".join(chars)


# ── Log discovery and parsing ─────────────────────────────────────────────────

def _load_log(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_session(log: dict, source: str) -> dict | None:
    """Normalise a raw log dict into a standard session record."""
    ts_raw = log.get("timestamp") or log.get("generated_at") or log.get("date") or ""
    if not ts_raw:
        return None

    # Normalise timestamp to ISO date string
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        date_str = ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        date_str = ts_raw[:16]

    verdict       = log.get("verdict", "UNKNOWN")
    final_score   = float(log.get("final_score", log.get("score", 0.0)))
    cycle_scores  = log.get("cycle_scores", [final_score])
    delta         = float(log.get("delta", 0.0))
    real_modules  = int(log.get("real_modules", 0))
    proposals     = log.get("proposals", [])

    # Extract domain scores from last cycle
    domain_scores: dict[str, float] = {}
    domain_results = log.get("domain_results", [])
    if domain_results:
        last_cycle = domain_results[-1]
        for entry in last_cycle:
            if isinstance(entry, dict):
                name  = entry.get("domain", "")
                score = float(entry.get("score", entry.get("final_score", 0.0)))
                if name:
                    domain_scores[name] = score

    return {
        "date":          date_str,
        "timestamp":     ts_raw,
        "source":        source,
        "verdict":       verdict,
        "final_score":   final_score,
        "cycle_scores":  cycle_scores,
        "delta":         delta,
        "real_modules":  real_modules,
        "proposals":     proposals,
        "domain_scores": domain_scores,
    }


def discover_sessions(root: Path) -> list[dict]:
    """Find and parse all provenance log files under root."""
    sessions: list[dict] = []
    for pattern in (LOG_GLOB, HEAL_LOG_GLOB):
        for path in sorted(root.glob(pattern)):
            raw = _load_log(path)
            if raw:
                session = _parse_session(raw, path.name)
                if session:
                    sessions.append(session)

    # Sort oldest → newest
    sessions.sort(key=lambda s: s["timestamp"])
    # Deduplicate by timestamp
    seen: set[str] = set()
    unique: list[dict] = []
    for s in sessions:
        key = s["timestamp"][:16]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ── Score history persistence ─────────────────────────────────────────────────

def load_history(root: Path) -> list[dict]:
    hist_path = root / HISTORY_FILE
    if not hist_path.exists():
        return []
    records: list[dict] = []
    for line in hist_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def save_history(root: Path, sessions: list[dict]) -> None:
    hist_path = root / HISTORY_FILE
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(s, ensure_ascii=False) for s in sessions]
    hist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_history(existing: list[dict], discovered: list[dict]) -> list[dict]:
    """Merge discovered sessions into existing history, no duplicates."""
    seen = {s["timestamp"][:16] for s in existing}
    merged = list(existing)
    for s in discovered:
        if s["timestamp"][:16] not in seen:
            seen.add(s["timestamp"][:16])
            merged.append(s)
    merged.sort(key=lambda s: s["timestamp"])
    return merged


# ── Display ───────────────────────────────────────────────────────────────────

def print_session_table(sessions: list[dict], n: int = 10) -> None:
    shown = sessions[-n:] if len(sessions) > n else sessions
    print(f"\n  {'DATE':<18} {'VERDICT':<22} {'SCORE':<8} {'ΔBASE':<9} {'MODS':<6} TREND")
    print(f"  {'─'*18} {'─'*22} {'─'*8} {'─'*9} {'─'*6} {'─'*10}")
    for s in shown:
        delta_base = s["final_score"] - WALCHE_BASELINE
        db_str = f"{delta_base:+.3f}"
        db_col = C.G if delta_base >= 0 else C.R
        trend  = sparkline(s["cycle_scores"])
        verd   = _verdict_col(s["verdict"])
        print(
            f"  {s['date']:<18} {verd:<30} "
            f"{_sc(s['final_score']):<16} "
            f"{_col(db_str, db_col):<17} "
            f"{s['real_modules']:<6} {C.D}{trend}{C.X}"
        )


def print_domain_table(sessions: list[dict], domain_filter: str | None = None) -> None:
    # Collect all domain names seen
    all_domains: set[str] = set()
    for s in sessions:
        all_domains.update(s.get("domain_scores", {}).keys())

    if domain_filter:
        all_domains = {d for d in all_domains if domain_filter in d}

    if not all_domains:
        print("  No domain score data found in logs.")
        return

    for domain in sorted(all_domains):
        scores = [s["domain_scores"][domain] for s in sessions if domain in s.get("domain_scores", {})]
        if not scores:
            continue
        latest = scores[-1]
        delta  = scores[-1] - scores[0] if len(scores) > 1 else 0.0
        trend  = sparkline(scores[-8:])
        db     = latest - WALCHE_BASELINE
        db_str = f"{db:+.3f}"
        db_col = C.G if db >= 0 else C.R

        status = "PASS" if latest >= PASS_THRESHOLD else "WARN" if latest >= WARN_THRESHOLD else "FAIL"
        sc     = C.G if status == "PASS" else C.Y if status == "WARN" else C.R

        print(
            f"  [{_col(status, sc)}]  {domain:<28} "
            f"{_bar(latest, 16)} {_sc(latest)}  "
            f"vs baseline {_col(db_str, db_col)}  "
            f"{C.D}{trend}{C.X}"
        )


def print_proposals_summary(sessions: list[dict], n: int = 5) -> None:
    recent = sessions[-3:] if len(sessions) >= 3 else sessions
    seen: set[str] = set()
    unique: list[str] = []
    for s in recent:
        for p in s.get("proposals", []):
            if p not in seen:
                seen.add(p)
                unique.append(p)
    if not unique:
        return
    print(f"\n  {C.B}RECENT HEALING PROPOSALS{C.X}  (last {len(recent)} sessions)")
    for p in unique[:n]:
        print(f"  {C.D}• {p[:88]}{C.X}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import os as _os
    if sys.platform == "win32":
        _os.system("")

    parser = argparse.ArgumentParser(
        description="WALCHE Provenance Log Viewer + Score Trend Tracker",
    )
    parser.add_argument("--sessions", type=int, default=10,
                        help="Number of recent sessions to show (default: 10)")
    parser.add_argument("--domain",   metavar="NAME",
                        help="Filter domain table by name substring")
    parser.add_argument("--export",   action="store_true",
                        help="Update corpus/score_history.jsonl from discovered logs")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress display output (use with --export)")
    args = parser.parse_args()

    root = Path.cwd()

    # ── Discover ──────────────────────────────────────────────────────────────
    discovered = discover_sessions(root)
    existing   = load_history(root)
    sessions   = merge_history(existing, discovered)

    if args.export:
        save_history(root, sessions)
        if not args.quiet:
            print(f"  Exported {len(sessions)} sessions → {root / HISTORY_FILE}")

    if args.quiet:
        return

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", C.CY))
    print(_col("║      W A L C H E   —   PROVENANCE VIEWER                ║", C.CY))
    print(_col("╚══════════════════════════════════════════════════════════╝", C.CY))
    print(f"  Baseline:  {_sc(WALCHE_BASELINE)}  (osModa, 2026-06-29)")
    print(f"  Sessions:  {len(sessions)} found")
    print(f"  Log dir:   {root / 'logs'}")
    print(f"  History:   {root / HISTORY_FILE}")

    if not sessions:
        print()
        print(f"  {_col('No provenance logs found.', C.Y)}")
        print(f"  Run walche_demo.py first to generate a log.")
        print()
        return

    # ── Session table ─────────────────────────────────────────────────────────
    print(f"\n  {C.B}SESSION HISTORY{C.X}  (last {min(args.sessions, len(sessions))} of {len(sessions)})")
    print_session_table(sessions, n=args.sessions)

    # ── Score trend ───────────────────────────────────────────────────────────
    final_scores = [s["final_score"] for s in sessions]
    if len(final_scores) >= 2:
        overall_delta = final_scores[-1] - final_scores[0]
        trend_str = sparkline(final_scores)
        dc = C.G if overall_delta >= 0 else C.R
        print(f"\n  {C.B}SCORE TREND{C.X}  {C.D}{trend_str}{C.X}  "
              f"({final_scores[0]:.3f} → {final_scores[-1]:.3f}  "
              f"{_col(f'{overall_delta:+.3f}', dc)})")

    # ── Domain breakdown ──────────────────────────────────────────────────────
    print(f"\n  {C.B}DOMAIN SCORES{C.X}  (latest session vs baseline {WALCHE_BASELINE})")
    print_domain_table(sessions, domain_filter=args.domain)

    # ── Recent proposals ──────────────────────────────────────────────────────
    print_proposals_summary(sessions)

    # ── Baseline gap ─────────────────────────────────────────────────────────
    latest_score = sessions[-1]["final_score"] if sessions else 0.0
    gap = WALCHE_BASELINE - latest_score
    print()
    if gap <= 0:
        print(f"  {_col('✓  WALCHE score meets or exceeds baseline', C.G)}")
    elif gap <= 0.05:
        print(f"  {_col(f'↑  {gap:.3f} below baseline — close', C.Y)}  "
              f"(feed corpus or improve docstrings)")
    else:
        print(f"  {_col(f'↑  {gap:.3f} below baseline — feed corpus + run healing cycles', C.R)}")

    print()
    if args.export:
        print(f"  {C.D}Score history updated: {root / HISTORY_FILE}{C.X}")
    else:
        print(f"  {C.D}Run with --export to update corpus/score_history.jsonl{C.X}")
    print()


if __name__ == "__main__":
    main()
