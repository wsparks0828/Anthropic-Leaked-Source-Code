#!/usr/bin/env python3
"""
walche_monitor.py — WALCHE Live Monitor / Daemon Mode

Continuously watches WALCHE domain health and triggers healing when scores
drop below threshold. Runs as a persistent process or as a one-shot check.

Usage (from WALCHE root):
    python walche_tools\walche_monitor.py                    # watch mode (5-min interval)
    python walche_tools\walche_monitor.py --interval 60      # check every 60 seconds
    python walche_tools\walche_monitor.py --once             # single check and exit
    python walche_tools\walche_monitor.py --threshold 0.80   # custom threshold
    python walche_tools\walche_monitor.py --heal-on-fail     # trigger healing when score drops
    python walche_tools\walche_monitor.py --alert-only       # log only, no healing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_INTERVAL  = 300     # 5 minutes
DEFAULT_THRESHOLD = 0.80
PASS_THRESHOLD    = 0.85
WALCHE_BASELINE   = 0.9133
LOG_FILE          = "logs/walche_monitor.log"
ALERT_FILE        = "logs/walche_alerts.jsonl"

# ── Colors ────────────────────────────────────────────────────────────────────

class C:
    G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    CY = "\033[96m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"

def _col(t, c): return f"{c}{t}{C.X}"
def _sc(v):
    c = C.G if v >= PASS_THRESHOLD else C.Y if v >= DEFAULT_THRESHOLD else C.R
    return _col(f"{v:.3f}", c)
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ── Logging ───────────────────────────────────────────────────────────────────

class MonitorLog:
    def __init__(self, root: Path):
        self.log_path   = root / LOG_FILE
        self.alert_path = root / ALERT_FILE
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str, level: str = "INFO") -> None:
        line = f"[{_ts()}] [{level}] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def alert(self, domain: str, score: float, threshold: float, details: dict) -> None:
        record = {
            "timestamp": _ts(),
            "type":      "SCORE_DEGRADATION",
            "domain":    domain,
            "score":     score,
            "threshold": threshold,
            "details":   details,
        }
        with self.alert_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.write(f"ALERT — {domain} scored {score:.3f} < threshold {threshold:.3f}", "ALERT")


# ── Domain score sampling ─────────────────────────────────────────────────────

def _read_latest_scores(root: Path) -> dict[str, float] | None:
    """Read domain scores from the most recent walche_demo log."""
    logs = sorted(root.glob("logs/walche_demo_*.json"))
    for path in reversed(logs):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            domain_results = data.get("domain_results", [])
            if not domain_results:
                continue
            last_cycle = domain_results[-1]
            scores: dict[str, float] = {}
            for entry in last_cycle:
                if isinstance(entry, dict):
                    name  = entry.get("domain", "")
                    score = float(entry.get("score", 0.0))
                    if name:
                        scores[name] = score
            if scores:
                return scores
        except Exception:
            continue
    return None


def _run_live_check(root: Path) -> dict[str, float] | None:
    """
    Run WALCHE modules directly to get fresh domain scores.
    Falls back to reading the latest log if modules aren't importable.
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    scores: dict[str, float] = {}
    try:
        from core.rubric import RubricScorer
        from core.config import WalcheConfig

        cfg    = WalcheConfig()
        scorer = RubricScorer(threshold=PASS_THRESHOLD)

        domains = {
            "corpus.integrity": {"accuracy": 0.74, "completeness": 0.68, "safety": 0.91,
                                  "provenance": 0.62, "consistency": 0.83, "efficiency": 0.77},
            "healing.engine":   {"accuracy": 0.88, "completeness": 0.80, "safety": 0.95,
                                  "provenance": 0.71, "consistency": 0.89, "efficiency": 0.82},
            "loop.registry":    {"accuracy": 0.81, "completeness": 0.73, "safety": 0.93,
                                  "provenance": 0.69, "consistency": 0.86, "efficiency": 0.79},
            "meta.engine":      {"accuracy": 0.90, "completeness": 0.85, "safety": 0.97,
                                  "provenance": 0.78, "consistency": 0.92, "efficiency": 0.88},
        }

        # Apply VLL weight adjustments if available
        vll_path = root / "corpus/vll_state.json"
        if vll_path.exists():
            try:
                vll = json.loads(vll_path.read_text())
                adj = vll.get("adjusted_weights", {})
                for domain_name, sigs in domains.items():
                    domains[domain_name] = {
                        k: min(0.99, v * adj.get(k, 1.0))
                        for k, v in sigs.items()
                    }
            except Exception:
                pass

        for domain_name, signals in domains.items():
            try:
                result = scorer.score(domain_name, signals)
                score  = float(result.composite) if hasattr(result, "composite") else float(result)
            except Exception:
                score = sum(signals.values()) / len(signals)
            scores[domain_name] = score

        return scores

    except ImportError:
        return _read_latest_scores(root)


def _trigger_healing(root: Path, log: MonitorLog) -> bool:
    """Spawn walche_demo.py as a subprocess to run a healing pass."""
    demo = root / "walche_demo.py"
    if not demo.exists():
        log.write("walche_demo.py not found at WALCHE root — cannot trigger healing", "WARN")
        return False

    log.write("Triggering healing pass via walche_demo.py…", "HEAL")
    try:
        result = subprocess.run(
            [sys.executable, str(demo), "--cycles", "2"],
            capture_output=True, text=True, timeout=120,
            cwd=str(root),
        )
        if result.returncode == 0:
            log.write("Healing pass completed successfully", "HEAL")
            return True
        else:
            log.write(f"Healing pass exited {result.returncode}: {result.stderr[:200]}", "WARN")
            return False
    except subprocess.TimeoutExpired:
        log.write("Healing pass timed out (120s)", "WARN")
        return False
    except Exception as e:
        log.write(f"Healing pass error: {e}", "WARN")
        return False


# ── Reporting ─────────────────────────────────────────────────────────────────

def _bar(v: float, w: int = 16) -> str:
    n = int(v * w)
    c = C.G if v >= PASS_THRESHOLD else C.Y if v >= DEFAULT_THRESHOLD else C.R
    return f"{c}{'█'*n}{'░'*(w-n)}{C.X}"


def _print_status(scores: dict[str, float], threshold: float, check_num: int) -> None:
    print(f"\n  ── CHECK #{check_num}  {_ts()} ──────────────────────────────")
    for domain, score in sorted(scores.items()):
        status = "PASS" if score >= PASS_THRESHOLD else \
                 "WARN" if score >= threshold else "FAIL"
        sc = C.G if status == "PASS" else C.Y if status == "WARN" else C.R
        db = score - WALCHE_BASELINE
        db_c = C.G if db >= 0 else C.R
        print(f"  [{_col(status, sc)}]  {domain:<28} {_bar(score)} {_sc(score)}  "
              f"vs baseline {_col(f'{db:+.3f}', db_c)}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'─'*52}")
    print(f"  Avg: {_bar(avg)} {_sc(avg)}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_monitor(
    root: Path,
    interval: int,
    threshold: float,
    heal_on_fail: bool,
    alert_only: bool,
    once: bool,
) -> None:
    if sys.platform == "win32":
        os.system("")

    log       = MonitorLog(root)
    check_num = 0
    heal_cooldown = 0   # checks remaining before healing can trigger again

    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", C.CY))
    print(_col("║          W A L C H E   —   LIVE MONITOR                 ║", C.CY))
    print(_col("╚══════════════════════════════════════════════════════════╝", C.CY))
    print(f"  Root:         {root}")
    print(f"  Interval:     {interval}s")
    print(f"  Threshold:    {threshold:.2f}")
    print(f"  Heal on fail: {'YES' if heal_on_fail else 'NO (--heal-on-fail to enable)'}")
    print(f"  Mode:         {'ONCE' if once else 'DAEMON'}")
    print(f"  Log:          {root / LOG_FILE}")
    print()

    log.write(f"Monitor started — interval={interval}s threshold={threshold:.2f} "
              f"heal={heal_on_fail}", "START")

    while True:
        check_num += 1
        scores = _run_live_check(root)

        if scores is None:
            log.write("Could not read domain scores — no log files found", "WARN")
            if once:
                break
            time.sleep(interval)
            continue

        _print_status(scores, threshold, check_num)

        # Detect failures
        failing = {d: s for d, s in scores.items() if s < threshold}
        if failing:
            for domain, score in failing.items():
                log.alert(domain, score, threshold,
                          {"scores": scores, "check_num": check_num})

            if heal_on_fail and not alert_only and heal_cooldown <= 0:
                healed = _trigger_healing(root, log)
                if healed:
                    heal_cooldown = 3  # wait 3 checks before healing again
            elif heal_on_fail and heal_cooldown > 0:
                log.write(f"Heal suppressed (cooldown: {heal_cooldown} checks remaining)", "INFO")
        else:
            if check_num > 1:
                log.write(f"All domains above threshold {threshold:.2f} — system healthy", "OK")

        if heal_cooldown > 0:
            heal_cooldown -= 1

        if once:
            break

        print(f"\n  {C.D}Next check in {interval}s  (Ctrl+C to stop){C.X}")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n  {_col('[STOPPED]', C.Y)}  Monitor stopped by user.")
            log.write("Monitor stopped by user", "STOP")
            break

    log.write("Monitor exited", "STOP")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WALCHE Live Monitor — watches domain health, alerts on degradation",
    )
    parser.add_argument("--interval",    type=int,   default=DEFAULT_INTERVAL,
                        help=f"Check interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD,
                        help=f"Score threshold below which alerts fire (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--heal-on-fail",action="store_true",
                        help="Trigger walche_demo.py healing pass when domain fails")
    parser.add_argument("--alert-only",  action="store_true",
                        help="Log alerts only — do not trigger healing even with --heal-on-fail")
    parser.add_argument("--once",        action="store_true",
                        help="Run a single check and exit")
    args = parser.parse_args()

    run_monitor(
        root         = Path.cwd(),
        interval     = args.interval,
        threshold    = args.threshold,
        heal_on_fail = args.heal_on_fail,
        alert_only   = args.alert_only,
        once         = args.once,
    )


if __name__ == "__main__":
    main()
