#!/usr/bin/env python3
"""
walche_status.py — WALCHE Single-Command Health Dashboard

Shows current WALCHE state at a glance: domain scores, last verdict, corpus
stats, council decisions, and score delta vs baseline.

Usage (from WALCHE root):
    python walche_tools/walche_status.py
    python walche_tools/walche_status.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

WALCHE_BASELINE = 0.9133
PASS_THRESHOLD  = 0.85
WARN_THRESHOLD  = 0.70

# ── Colors ────────────────────────────────────────────────────────────────────

class C:
    G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    CY = "\033[96m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"

def _col(t, c): return f"{c}{t}{C.X}"
def _sc(v):
    c = C.G if v >= PASS_THRESHOLD else C.Y if v >= WARN_THRESHOLD else C.R
    return _col(f"{v:.3f}", c)
def _bar(v, w=18):
    n = int(v * w)
    c = C.G if v >= PASS_THRESHOLD else C.Y if v >= WARN_THRESHOLD else C.R
    return f"{c}{'█'*n}{'░'*(w-n)}{C.X}"
def _verdict_col(v):
    if v == "GO":                  return _col("GO", C.G)
    if v == "GO-WITH-CONDITIONS":  return _col("GO-WITH-CONDITIONS", C.Y)
    return _col(v, C.R)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _age(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - ts
        secs = int(diff.total_seconds())
        if secs < 60:       return f"{secs}s ago"
        if secs < 3600:     return f"{secs//60}m ago"
        if secs < 86400:    return f"{secs//3600}h ago"
        return f"{secs//86400}d ago"
    except Exception:
        return ""


def _latest_log(root: Path) -> dict | None:
    logs = sorted(root.glob("logs/walche_demo_*.json"))
    for path in reversed(logs):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _score_history(root: Path) -> list[float]:
    path = root / "corpus/score_history.jsonl"
    if not path.exists():
        return []
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            scores.append(float(rec.get("final_score", 0.0)))
        except Exception:
            pass
    return scores[-8:]


def _council_log(root: Path) -> list[dict]:
    path = root / "logs/grand_council_decisions.jsonl"
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records[-5:]


def _vll_state(root: Path) -> dict | None:
    return _load_json(root / "corpus/vll_state.json")


def _corpus_kb(root: Path) -> dict | None:
    return _load_json(root / "corpus/walche_kb.json")


def _monitor_status(root: Path) -> str | None:
    path = root / "logs/walche_monitor.log"
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[-1].strip() if lines else None
    except Exception:
        return None

# ── Sections ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n  {C.B}{title}{C.X}")
    print(f"  {'─'*52}")


def _print_last_run(log: dict) -> None:
    ts   = log.get("timestamp") or ""
    age  = _age(ts)
    v    = log.get("verdict", "UNKNOWN")
    fs   = float(log.get("final_score", 0.0))
    d    = float(log.get("delta", 0.0))
    rm   = int(log.get("real_modules", 0))
    sm   = int(log.get("stub_modules", 0))
    cyc  = log.get("cycle_scores", [])
    d_col = C.G if d >= 0 else C.R

    print(f"  Timestamp:  {ts[:19]}  {C.D}({age}){C.X}")
    print(f"  Verdict:    {_verdict_col(v)}")
    print(f"  Score:      {_bar(fs)} {_sc(fs)}  ({_col(f'{d:+.3f}', d_col)} this run)")
    print(f"  Modules:    {_col(str(rm), C.G)} real  /  {_col(str(sm), C.Y)} stub")
    if len(cyc) >= 2:
        trend = " → ".join(f"{s:.3f}" for s in cyc)
        print(f"  Cycles:     {C.D}{trend}{C.X}")


def _print_domains(log: dict) -> None:
    domain_results = log.get("domain_results", [])
    if not domain_results:
        print(f"  {C.D}No domain data in last run{C.X}")
        return
    last_cycle = domain_results[-1]
    for entry in last_cycle:
        if not isinstance(entry, dict):
            continue
        name  = entry.get("domain", "?")
        score = float(entry.get("score", 0.0))
        db    = score - WALCHE_BASELINE
        db_c  = C.G if db >= 0 else C.R
        status = "PASS" if score >= PASS_THRESHOLD else "WARN" if score >= WARN_THRESHOLD else "FAIL"
        sc = C.G if status == "PASS" else C.Y if status == "WARN" else C.R
        print(f"  [{_col(status, sc)}]  {name:<28} {_bar(score)} {_sc(score)}  "
              f"vs baseline {_col(f'{db:+.3f}', db_c)}")


def _print_corpus(kb: dict) -> None:
    stats = kb.get("stats", {})
    n     = stats.get("total_entries_in_kb", stats.get("entries_passed", 0))
    tok   = stats.get("total_tokens", 0)
    score = float(stats.get("predicted_integrity_score", 0.0))
    gen   = kb.get("generated_at", "")[:10]
    print(f"  Entries:    {n:,}  ({tok:,} tokens)")
    print(f"  Predicted integrity: {_sc(score)}")
    print(f"  Generated:  {gen}")


def _print_council(decisions: list[dict]) -> None:
    if not decisions:
        print(f"  {C.D}No council decisions logged yet{C.X}")
        return
    for d in reversed(decisions[-3:]):
        ts  = (d.get("timestamp") or "")[:10]
        t   = d.get("proposal_type", d.get("type", "?"))
        v   = str(d.get("verdict") or d.get("final_verdict") or "?")
        vu  = v.upper()
        # Council's verdict vocabulary is APPROVED/REJECTED/DEADLOCKED/ESCALATED,
        # distinct from the healing loop's GO/GO-WITH-CONDITIONS/NO-GO — both are
        # handled here since this panel can show either kind of decision.
        if "APPROVE" in vu or vu == "GO":
            vc = C.G
        elif "CONDITION" in vu:
            vc = C.Y
        elif "REJECT" in vu or vu == "NO-GO" or "DENY" in vu or "VETO" in vu:
            vc = C.R
        elif vu == "DEADLOCKED":
            vc = C.Y   # pending/no-consensus, not a rejection
        elif vu == "ESCALATED":
            vc = C.CY  # pending escalation, not a rejection
        else:
            vc = C.D   # unknown verdict — neutral, not red
        print(f"  {ts}  [{t:<18}]  {_col(v, vc)}")


def _print_vll(state: dict) -> None:
    adj = state.get("adjusted_weights", {})
    processed = state.get("proposals_processed", 0)
    updated   = state.get("last_updated", "")[:10]
    print(f"  Proposals processed: {processed}")
    print(f"  Last updated:        {updated}")
    if adj:
        deltas = {k: adj[k] - state.get("baseline_weights", {}).get(k, 1.0)
                  for k in adj}
        top = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        for dim, delta in top:
            dc = C.G if delta > 0 else C.R
            print(f"  {dim:<22} {_col(f'{delta:+.3f}', dc)}")


def _print_trend(scores: list[float]) -> None:
    if not scores:
        print(f"  {C.D}No history yet — run walche_tools\\provenance_viewer.py --export{C.X}")
        return
    _SPARK = " ▁▂▃▄▅▆▇█"
    lo, hi = min(scores), max(scores)
    rng = hi - lo or 1.0
    spark = "".join(_SPARK[int((v - lo) / rng * (len(_SPARK) - 1))] for v in scores)
    delta = scores[-1] - scores[0] if len(scores) > 1 else 0.0
    dc = C.G if delta >= 0 else C.R
    print(f"  {C.D}{spark}{C.X}  {scores[0]:.3f} → {_sc(scores[-1])}  "
          f"{_col(f'{delta:+.3f}', dc)}  ({len(scores)} sessions)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if sys.platform == "win32":
        os.system("")

    parser = argparse.ArgumentParser(description="WALCHE Health Dashboard")
    parser.add_argument("--json", action="store_true",
                        help="Output status as JSON instead of formatted display")
    parser.add_argument("--root", metavar="PATH", default=None,
                        help="WALCHE root directory (default: auto-detected from this file's location)")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    last_log  = _latest_log(root)
    kb        = _corpus_kb(root)
    council   = _council_log(root)
    vll       = _vll_state(root)
    trend     = _score_history(root)
    monitor   = _monitor_status(root)

    if args.json:
        status = {
            "last_run":         last_log,
            "corpus_kb_stats":  ({**kb["stats"], "generated_at": kb.get("generated_at")} if isinstance(kb.get("stats"), dict) else None) if kb else None,
            "council_decisions":council,
            "vll_state":        vll,
            "score_trend":      trend,
            "baseline":         WALCHE_BASELINE,
        }
        print(json.dumps(status, indent=2))
        return

    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", C.CY))
    print(_col("║              W A L C H E   —   STATUS                   ║", C.CY))
    print(_col("╚══════════════════════════════════════════════════════════╝", C.CY))
    print(f"  Root:      {root}")
    print(f"  Baseline:  {_sc(WALCHE_BASELINE)}  (osModa, 2026-06-29)")

    # Last run
    _section("LAST RUN")
    if last_log:
        _print_last_run(last_log)
    else:
        print(f"  {C.Y}No walche_demo log found — run walche_demo.py first{C.X}")

    # Domains
    _section("DOMAIN SCORES  (last run, final cycle)")
    if last_log:
        _print_domains(last_log)
    else:
        print(f"  {C.D}No data{C.X}")

    # Score trend
    _section("SCORE TREND")
    _print_trend(trend)

    # Corpus
    _section("CORPUS KNOWLEDGE BASE")
    if kb:
        _print_corpus(kb)
    else:
        print(f"  {C.D}No KB found — run: python walche_tools\\corpus_ingest.py --scan . --output corpus\\walche_kb.json{C.X}")

    # Council
    _section("GRAND COUNCIL  (recent decisions)")
    _print_council(council)

    # VLL
    _section("VLL  (Verification Learning Loop)")
    if vll:
        _print_vll(vll)
    else:
        print(f"  {C.D}No VLL state — run: python walche_tools\\vll_engine.py --apply{C.X}")

    # Monitor
    if monitor:
        _section("MONITOR")
        print(f"  {C.D}{monitor}{C.X}")

    # Quick fixes
    _section("NEXT ACTIONS")
    if last_log:
        fs = float(last_log.get("final_score", 0.0))
        if fs < PASS_THRESHOLD:
            gap = PASS_THRESHOLD - fs
            print(f"  {C.Y}↑ corpus.integrity {gap:.3f} below PASS — feed corpus or run more cycles{C.X}")
        sm = int(last_log.get("stub_modules", 0))
        if sm > 0:
            print(f"  {C.Y}↑ {sm} modules still on stubs — check WALCHE install{C.X}")
    if not kb:
        print(f"  {C.D}• python walche_tools\\corpus_ingest.py --scan . --output corpus\\walche_kb.json{C.X}")
    if not vll:
        print(f"  {C.D}• python walche_tools\\vll_engine.py --apply{C.X}")
    if not council:
        print(f"  {C.D}• python walche_demo.py --council    (enable governance loop){C.X}")

    print()


if __name__ == "__main__":
    main()
