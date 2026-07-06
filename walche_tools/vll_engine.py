#!/usr/bin/env python3
"""
vll_engine.py — WALCHE Verification Learning Loop Engine

The VLL closes the self-improvement loop:
  Healing proposals → Council approval → VLL applies learning → Rubric weights
  update → Domain scores improve → Verdict converges toward GO.

How it works:
  1. Reads accepted council decisions from logs/grand_council_decisions.jsonl
     and council_verdicts embedded in walche_demo provenance logs.
  2. Maps each accepted proposal to the WALCHE signal dimensions it targets.
  3. Computes weight adjustments: accepted proposals increase dimension weight,
     rejected proposals decrease it (or leave it unchanged).
  4. Persists learned weights to corpus/vll_state.json.
  5. walche_demo.py and walche_monitor.py load vll_state.json and apply
     weight multipliers to signal values before rubric scoring.

Usage (from WALCHE root):
    python walche_tools\vll_engine.py --status
    python walche_tools\vll_engine.py --apply
    python walche_tools\vll_engine.py --apply --dry-run
    python walche_tools\vll_engine.py --reset
    python walche_tools\vll_engine.py --history
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ──────────────────────────────���──────────────────────────────────

VLL_STATE_FILE    = "corpus/vll_state.json"
COUNCIL_LOG_FILE  = "logs/grand_council_decisions.jsonl"
DEMO_LOG_GLOB     = "logs/walche_demo_*.json"

VLL_VERSION       = "1.0.0"
WALCHE_BASELINE   = 0.9133
MAX_WEIGHT        = 1.40    # cap weight multipliers
MIN_WEIGHT        = 0.60    # floor
LEARNING_RATE     = 0.03    # per approved proposal
DECAY_RATE        = 0.01    # per rejected proposal

# Signal dimensions tracked by WALCHE's rubric
ALL_DIMENSIONS = [
    "accuracy", "completeness", "consistency", "safety",
    "provenance", "efficiency", "adaptability", "clarity",
]

# Map proposal keywords → targeted signal dimensions
DIMENSION_MAP: dict[str, list[str]] = {
    # Corpus / data quality
    "provenance":    ["provenance", "accuracy", "consistency"],
    "corpus":        ["completeness", "provenance", "accuracy"],
    "ingest":        ["completeness", "accuracy"],
    "lineage":       ["provenance", "consistency"],
    "quality":       ["accuracy", "completeness", "clarity"],

    # Healing / rubric
    "heal":          ["accuracy", "consistency", "adaptability"],
    "rubric":        ["accuracy", "completeness"],
    "threshold":     ["accuracy", "safety"],
    "score":         ["accuracy", "adaptability"],
    "cycle":         ["consistency", "efficiency"],

    # Performance / efficiency
    "efficiency":    ["efficiency"],
    "cache":         ["efficiency", "adaptability"],
    "token":         ["efficiency"],
    "batch":         ["efficiency", "adaptability"],
    "compress":      ["efficiency"],
    "retrieval":     ["accuracy", "completeness", "efficiency"],
    "memory":        ["accuracy", "completeness"],

    # Safety / security
    "safety":        ["safety"],
    "guardrail":     ["safety", "consistency"],
    "security":      ["safety", "accuracy"],
    "trust":         ["safety", "provenance"],
    "veto":          ["safety"],

    # Clarity / completeness
    "docstring":     ["clarity", "completeness"],
    "documentation": ["clarity", "completeness"],
    "clarity":       ["clarity"],
    "complete":      ["completeness"],
    "adapt":         ["adaptability"],

    # VLL itself
    "vll":           ["adaptability", "accuracy"],
    "learn":         ["adaptability", "accuracy"],
    "improve":       ["accuracy", "efficiency", "adaptability"],
}


# ── Colors ────────────────────────────────────────────────────────────────��───

class C:
    G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    CY = "\033[96m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"

def _col(t, c):  return f"{c}{t}{C.X}"
def _ts() -> str: return datetime.now(timezone.utc).isoformat()


def _decision_key(d: dict) -> str:
    """Stable dedup/processed-tracking key for a council decision. Used by both
    collect_decisions() and apply_learning() so they can never diverge — demo-log
    verdicts carry no timestamp, so _source (the demo log filename) is required
    to distinguish them."""
    return (
        f"{(d.get('timestamp') or '')[:16]}:"
        f"{d.get('proposal_type') or ''}:"
        f"{d.get('verdict') or ''}:"
        f"{d.get('_source') or ''}"
    )


# ── State I/O ────────────────���────────────────────────────────────────────────

def load_state(root: Path) -> dict:
    path = root / VLL_STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as _e:
            print(f"  [VLL] WARNING: {path.name} could not be read ({_e}) — starting fresh")
    # Default state
    return {
        "version":           VLL_VERSION,
        "created_at":        _ts(),
        "last_updated":      _ts(),
        "proposals_processed": 0,
        "approved_count":    0,
        "rejected_count":    0,
        "baseline_weights":  {d: 1.0 for d in ALL_DIMENSIONS},
        "adjusted_weights":  {d: 1.0 for d in ALL_DIMENSIONS},
        "dimension_history": {d: [] for d in ALL_DIMENSIONS},
        "learning_log":      [],
    }


def save_state(root: Path, state: dict) -> None:
    path = root / VLL_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    _tmp = path.with_suffix(".tmp")
    _tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(_tmp, path)


# ── Decision source readers ───────────────────────────────────────────────────

def _read_council_log(root: Path) -> list[dict]:
    path = root / COUNCIL_LOG_FILE
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


def _read_demo_council_verdicts(root: Path) -> list[dict]:
    """Extract council_verdicts embedded in walche_demo provenance logs."""
    verdicts: list[dict] = []
    for path in sorted(root.glob(DEMO_LOG_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for cv in data.get("council_verdicts", []):
                cv["_source"] = path.name
                verdicts.append(cv)
        except Exception:
            pass
    return verdicts


def collect_decisions(root: Path) -> list[dict]:
    """Gather all council decisions from all sources."""
    decisions = _read_council_log(root) + _read_demo_council_verdicts(root)
    # Deduplicate by a rough key
    seen: set[str] = set()
    unique: list[dict] = []
    for d in decisions:
        key = _decision_key(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


# ── VLL computation ───────────────────────────────��─────────────────────��─────

def _map_proposal_to_dimensions(proposal_text: str, proposal_type: str) -> list[str]:
    """Return list of signal dimensions targeted by this proposal."""
    proposal_type = proposal_type or ""
    text = (str(proposal_text or "") + " " + proposal_type).lower()
    dims: set[str] = set()

    for keyword, targets in DIMENSION_MAP.items():
        if keyword in text:
            dims.update(targets)

    # Fallback: proposal_type directly names a dimension
    if proposal_type in ALL_DIMENSIONS:
        dims.add(proposal_type)

    # If nothing matched, target the most general dimensions
    if not dims:
        dims = {"accuracy", "adaptability"}

    return list(dims)


def _is_approved(verdict: str) -> bool:
    v = verdict.upper()
    return "APPROVE" in v or v in ("GO", "YES", "PASS", "RATIFIED", "ACCEPTED")


def _is_rejected(verdict: str) -> bool:
    v = verdict.upper()
    return "REJECT" in v or "NO-GO" in v or "DENY" in v or "VETO" in v


def apply_learning(
    state: dict,
    decisions: list[dict],
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """Apply council decisions to update signal weights. Returns updated state."""
    import copy
    new_state = copy.deepcopy(state)
    weights   = new_state["adjusted_weights"]
    log_entries: list[dict] = []
    n_approved       = 0
    n_rejected       = 0
    n_already        = 0   # decision was already applied in a prior --apply run
    n_no_verdict     = 0   # decision carries no verdict at all
    n_undecided      = 0   # verdict present but neither approved nor rejected
                            # (DEADLOCKED / ESCALATED are pending, not learning signal)

    # Track which decisions have already been applied
    already_processed: set[str] = {
        entry.get("decision_key", "")
        for entry in new_state.get("learning_log", [])
    }

    for decision in decisions:
        # Build a stable key for this decision — must match collect_decisions()
        # exactly (including _source) or cross-run learning silently stops.
        dec_key = _decision_key(decision)
        if dec_key in already_processed:
            n_already += 1
            continue

        verdict = decision.get("verdict") or decision.get("final_verdict") or ""
        if not verdict:
            n_no_verdict += 1
            continue

        approved = _is_approved(verdict)
        rejected = _is_rejected(verdict)
        if not approved and not rejected:
            n_undecided += 1
            continue

        # Count once per decision, not once per dimension/proposal fanout —
        # otherwise approved_count/rejected_count overstate reality several-fold.
        if approved:
            n_approved += 1
        else:
            n_rejected += 1

        # Get all proposals from this decision
        proposals = decision.get("proposals") or []
        ptype     = decision.get("proposal_type") or decision.get("type") or "general"

        dims: list[str] = []
        for proposal in (proposals if proposals else [ptype]):
            dims = _map_proposal_to_dimensions(str(proposal), ptype)
            for dim in dims:
                if dim not in weights:
                    continue
                old = weights[dim]
                if approved:
                    new = min(MAX_WEIGHT, old + LEARNING_RATE)
                else:
                    new = max(MIN_WEIGHT, old - DECAY_RATE)

                if not dry_run:
                    weights[dim] = round(new, 4)
                    hist = new_state["dimension_history"].setdefault(dim, [])
                    hist.append({
                        "ts":    _ts(),
                        "from":  old,
                        "to":    new,
                        "delta": round(new - old, 4),
                        "cause": f"{'APPROVED' if approved else 'REJECTED'} [{ptype}]",
                    })
                    del hist[:-500]  # cap unbounded growth

        log_entries.append({
            "decision_key":  dec_key,
            "timestamp":     _ts(),
            "proposal_type": ptype,
            "verdict":       verdict,
            "approved":      approved,
            "dimensions":    dims,
            "dry_run":       dry_run,
        })

    if not dry_run:
        new_state["adjusted_weights"]  = weights
        new_state["last_updated"]      = _ts()
        new_state["proposals_processed"] += len(log_entries)
        new_state["approved_count"]    += n_approved
        new_state["rejected_count"]    += n_rejected
        new_state["learning_log"].extend(log_entries)
        del new_state["learning_log"][:-500]  # cap unbounded growth

    return new_state, {
        "new_decisions":   len(log_entries),
        "already_applied": n_already,
        "no_verdict":       n_no_verdict,
        "undecided":        n_undecided,
        "n_approved":       n_approved,
        "n_rejected":       n_rejected,
        "total_processed":  new_state["proposals_processed"],
    }


# ── Display ──────────────────────────────���───────────────────────────────���────

def _bar(v: float, w: int = 16) -> str:
    n = int((v - 0.6) / 0.8 * w)
    n = max(0, min(w, n))
    c = C.G if v >= 1.05 else C.Y if v >= 1.00 else C.R
    return f"{c}{'█'*n}{'░'*(w-n)}{C.X}"


def print_status(state: dict) -> None:
    weights = state.get("adjusted_weights", {})
    base    = state.get("baseline_weights", {d: 1.0 for d in ALL_DIMENSIONS})

    print()
    print(_col("╔════════��══════════════════��══════════════════════════════╗", C.CY))
    print(_col("║      W A L C H E   —   VLL ENGINE STATUS                ║", C.CY))
    print(_col("╚═══════════════════════���══════════════════════════════════╝", C.CY))
    print(f"  Version:             {state.get('version', '?')}")
    print(f"  Last updated:        {state.get('last_updated', 'never')[:19]}")
    print(f"  Proposals processed: {state.get('proposals_processed', 0)}")
    print(f"  Approved / Rejected: {state.get('approved_count', 0)} / {state.get('rejected_count', 0)}")
    print()
    print(f"  {C.B}DIMENSION WEIGHTS{C.X}  (1.000 = baseline, > 1 = reinforced, < 1 = penalized)")
    print(f"  {'─'*52}")
    for dim in ALL_DIMENSIONS:
        w   = weights.get(dim, 1.0)
        b   = base.get(dim, 1.0)
        d   = w - b
        dc  = C.G if d > 0.005 else C.R if d < -0.005 else C.D
        print(f"  {dim:<16}  {_bar(w)} {_col(f'{w:.4f}', dc)}  ({_col(f'{d:+.4f}', dc)})")
    print()


def print_history(state: dict, dim_filter: str | None = None) -> None:
    history = state.get("dimension_history", {})
    for dim, entries in sorted(history.items()):
        if dim_filter and dim_filter not in dim:
            continue
        if not entries:
            continue
        recent = entries[-5:]
        print(f"\n  {C.B}{dim}{C.X}")
        for e in recent:
            dc    = C.G if e.get("delta", 0) > 0 else C.R
            delta_str = f"{e.get('delta', 0):+.4f}"
            cause = e.get("cause", "")
            print(f"    {e.get('ts', '')[:16]}  {e.get('from', 0):.4f} → {e.get('to', 0):.4f}  "
                  f"{_col(delta_str, dc)}  {C.D}{cause}{C.X}")


# ── Main ──────────────────────────��───────────────────────────────────────────

def main() -> None:
    if sys.platform == "win32":
        os.system("")

    parser = argparse.ArgumentParser(
        description="WALCHE VLL Engine — closes the self-improvement loop",
    )
    parser.add_argument("--status",   action="store_true",
                        help="Show current VLL state and weight adjustments")
    parser.add_argument("--apply",    action="store_true",
                        help="Read council decisions and apply learning to weights")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would change without writing anything")
    parser.add_argument("--reset",    action="store_true",
                        help="Reset all weights to baseline 1.000")
    parser.add_argument("--history",  action="store_true",
                        help="Show per-dimension weight history")
    parser.add_argument("--dim",      metavar="NAME",
                        help="Filter --history to a specific dimension")
    parser.add_argument("--root",     metavar="PATH", default=None,
                        help="WALCHE root directory (default: auto-detected from this file's location)")
    args = parser.parse_args()

    root  = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    state = load_state(root)

    if args.reset:
        state["adjusted_weights"]   = {d: 1.0 for d in ALL_DIMENSIONS}
        state["dimension_history"]  = {d: [] for d in ALL_DIMENSIONS}
        state["learning_log"]       = []
        state["proposals_processed"] = 0
        state["approved_count"]     = 0
        state["rejected_count"]     = 0
        state["last_updated"]       = _ts()
        save_state(root, state)
        print(f"  {_col('VLL weights reset to baseline.', C.Y)}")
        return

    if args.history:
        print_history(state, dim_filter=args.dim)
        return

    if args.apply or args.dry_run:
        decisions = collect_decisions(root)
        print(f"\n  Found {len(decisions)} total council decisions")

        if not decisions:
            print(f"  {C.Y}No decisions yet.{C.X}")
            print(f"  {C.D}Run: python walche_demo.py --council{C.X}")
            print(f"  {C.D}Or:  python walche_tools\\council_of_9.py --proposal \"...\" --type token_strategy{C.X}")
            print()
            return

        new_state, summary = apply_learning(state, decisions, dry_run=args.dry_run)

        print()
        print(f"  {C.B}VLL LEARNING SUMMARY{C.X}  {'(DRY RUN)' if args.dry_run else ''}")
        print(f"  {'─'*52}")
        print(f"  New decisions processed: {summary['new_decisions']}")
        print(f"  Already applied before:  {summary['already_applied']}")
        print(f"  No verdict / unparsable: {summary['no_verdict']}")
        print(f"  Undecided (DEADLOCKED/ESCALATED, no learning signal): {summary['undecided']}")
        print(f"  Approved proposals:      {_col(str(summary['n_approved']), C.G)}")
        print(f"  Rejected proposals:      {_col(str(summary['n_rejected']), C.R)}")
        print(f"  Total lifetime:          {summary['total_processed']}")

        # Show weight changes
        old_w = state.get("adjusted_weights", {})
        new_w = new_state.get("adjusted_weights", {})
        changed = [(d, old_w.get(d, 1.0), new_w.get(d, 1.0))
                   for d in ALL_DIMENSIONS
                   if abs(new_w.get(d, 1.0) - old_w.get(d, 1.0)) > 0.0001]
        if changed:
            print()
            print(f"  {C.B}WEIGHT CHANGES:{C.X}")
            for dim, old, new in changed:
                d  = new - old
                dc = C.G if d > 0 else C.R
                print(f"  {dim:<16}  {old:.4f} → {new:.4f}  {_col(f'{d:+.4f}', dc)}")

        if not args.dry_run:
            save_state(root, new_state)
            print()
            print(f"  {_col('VLL state saved.', C.G)}  {root / VLL_STATE_FILE}")
            print(f"  walche_demo.py and walche_monitor.py will use updated weights on next run.")
        else:
            print()
            print(f"  {_col('Dry run — no changes written.', C.Y)}")
        print()
        state = new_state

    if args.status or not any([args.apply, args.dry_run, args.reset, args.history]):
        print_status(state)


if __name__ == "__main__":
    main()
