# WALCHE Deep Scan — Pass 3 Forensic Audit (Fable 5)

**Date:** 2026-07-04
**Auditor:** Claude (Fable 5 deep-scan pass) — scan and log ONLY; no fixes applied
**Executor:** Repair instructions in this document are written for Sonnet to apply
**Scope:** All 15 files in `walche_tools/` (~8,760 lines), repo structure, data files,
cross-file contracts, plus live execution checks. Every finding below was verified
against actual file content; findings verified by executing code are marked `[EXECUTED]`.
Findings from parallel audit subagents were spot-verified before inclusion.

**Branch for repairs:** `claude/session-01ht1jmqadwsdphy19maevvl-7qlsw7`
**Rules for the repair session:** fix in the order given (Tier 0 → Tier 3), one commit
per tier, run `python3 -m py_compile` on every touched file before each commit,
and do NOT change behavior beyond what each instruction states.

---

## EXECUTIVE SUMMARY

Pass 3 found **~130 defects** that survived the first two audit passes
(9 CRITICAL, ~30 HIGH, ~50 MEDIUM, ~40 LOW). The most important discoveries,
in order of damage to the platform:

1. **`fix_walche.py` cannot run at all** — a hard `SyntaxError` at line 3 (`C:\Users\...`
   in a non-raw docstring; `\U` is parsed as a unicode escape). The file has never
   parsed: it is the only tool with no `.pyc` in `__pycache__`. `[EXECUTED]`
2. **The self-improvement loop is structurally open end-to-end.**
   (a) `walche_demo.py` NEVER reads `corpus/vll_state.json` — learned VLL weights are
   computed and saved but never applied to demo scoring, contradicting `vll_engine.py`'s
   own docstring (lines 15–17). (b) With stub modules, `_classify_proposal` can never
   emit `token_strategy` or `security`, so `--council` never deliberates anything and
   `logs/grand_council_decisions.jsonl` is never written by demo runs. (c) VLL therefore
   learns from nothing. The loop the platform is named for does not close anywhere.
3. **The second-pass fix created a regression:** `walche_demo.py` now appends minimal
   rows to `corpus/score_history.jsonl`, but `provenance_viewer.py` crashes with
   `KeyError: 'cycle_scores'` on any history containing those rows, and its `--export`
   dedup keeps the impoverished row over the rich one. `[EXECUTED]`
4. **The VLL dedup fix from pass 2 is only half-applied:** `_source` was added to the
   `collect_decisions` key but NOT to `apply_learning`'s `dec_key`, so cross-run learning
   still collapses all same-type/same-verdict demo verdicts to one.
5. **`corpus_ingest.py --scan` destroys the KB:** full-replace write with no merge; a
   scan that extracts zero entries (the default at this repo root) overwrites a
   populated `walche_kb.json` with `entries: []`.
6. **`walche_apply.py` and `walche_validate.py` fabricate their outputs:** hardcoded
   `decision = "APPLY"` ignoring computed risk, fabricated provenance records
   (`"docs": 50, "vll": 1982` fiction), synthetic keyword-sniff "validation" scores
   presented as measured, and a fake verification gate injected non-idempotently into
   another repo's source file. Under CLAUDE.md Laws 2–4 this is the highest-priority
   repair class in the codebase.
7. **Council governance has dead escalation paths and biased tallies:** Council-of-9
   ESCALATED is a terminal dead end; any judge can force ESCALATE (overriding hard
   vetoes); local mode can never vote REJECT on most panels; C9 panels are seeded by
   unseeded `random.shuffle` (irreproducible governance).
8. **The two deployment tools can destroy `run_system.py` on the owner's machine:**
   `fix_walche.py` Fix 3 writes a literal SyntaxError (`return return …`) into it on
   any second run — with no backup — and deploys the resolved-and-retired "WACHEL"
   typo back into ChromaDB collection names and class names; `patch_run_system.py`
   has a partial-patch lockout that leaves a `NameError` call site AND then refuses
   to re-patch, and its embedded verdict logic reports GO-WITH-CONDITIONS when a
   domain hard-FAILs.

Structural facts (verified): `core/`, `backend/`, `logs/`, `tests/` do **not** exist in
this repo — every `core.*` import fails here; the platform can only fully run on the
owner's Windows installation. The repo hosts three unrelated projects (WALCHE tools,
the "Print Perfect" Node app + Dockerfile, and the leaked-source TypeScript corpus).

---

## TIER 0 — SHOWSTOPPERS (fix first)

### T0-1 · `fix_walche.py:1–5` · CRITICAL · File cannot parse `[EXECUTED]`
`SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position
47-48: truncated \UXXXXXXXX escape` — line 3 `Run from: C:\Users\wspar\Desktop\WALCHE\`
inside a normal (non-raw) docstring. The entire tool is dead; it has never run.
**Repair (Sonnet):** make the module docstring a raw string (`r"""` … `"""`) or escape
the backslashes. Then apply findings T0-2 and section F below before considering the
file usable.

### T0-2 · `fix_walche.py` (whole file) · CRITICAL · Executes at import, no main guard
The entire script body (pip install at line 28, all file writes) runs at module level.
Importing it for any reason executes a pip upgrade and overwrites `backend/` files.
**Repair:** wrap all steps in `def main():` + `if __name__ == "__main__": main()`.

### T0-3 · `walche_demo.py` · CRITICAL · VLL weights never applied — learning loop open
`grep vll walche_tools/walche_demo.py` → zero hits. `vll_engine.py:15-17` documents that
the demo loads `corpus/vll_state.json` and applies weight multipliers before rubric
scoring. It does not. The entire learn→improve→GO convergence story is disconnected.
**Repair:** in `walche_demo.py`, before the cycle loop in `main()`, load
`ROOT / "corpus" / "vll_state.json"` (guard with try/except, default `{}`), take
`adjusted_weights`, and inside `run_cycle` multiply each signal:
`signals = {k: min(0.99, v * vll_weights.get(k, 1.0)) for k, v in signals.items()}`
(pass `vll_weights` down as a parameter). Mirror the exact pattern already used in
`walche_monitor.py:131-142`.

### T0-4 · `walche_demo.py:329-347` · CRITICAL · `--council` can never trigger with stubs
Stub healer proposals ("Strengthen provenance tracking in X", "Apply PreIngestGate
pattern…", "Run VLL refinement…") always classify as `corpus`/`healing`/`general` —
never `token_strategy` or `security` (the only two types routed to council at line 371).
Verified by walking every stub proposal string through `_PROPOSAL_TYPE_KEYWORDS` in
dict order. Consequence: council deliberation, council log, dashboard council panel,
and VLL learning are all dead in any stub-mode run.
**Repair (choose deliberately, tell the operator which):** either (a) widen
`governance_types` to include `corpus` and `healing` so real proposals reach council,
or (b) make the stub healer emit at least one governance-typed proposal when scores
warrant (e.g. "Adjust token budget threshold" / "Tighten guardrail veto risk"). Option
(a) is the honest fix: it routes what the system actually produces.

### T0-5 · `vll_engine.py:240-244` · CRITICAL · Pass-2 dedup fix incomplete — cross-run learning still dead
`collect_decisions` (line 179) now includes `_source` in its key, but `apply_learning`
builds `dec_key` WITHOUT `_source`. Demo-log verdicts carry no `timestamp`, so the first
processed verdict of a given `(type, verdict)` writes a `dec_key` like
`:token_strategy:APPROVED` into `learning_log`, and every future demo run's identical
verdict is skipped as "already processed". VLL still learns ~once per (type, verdict)
pair, ever.
**Repair:** add `f":{decision.get('_source', '')}"` to `dec_key` (lines 240-244) so it
matches the `collect_decisions` key exactly. Consider extracting one
`def _decision_key(d): ...` used by both sites so they can never diverge again.

### T0-6 · `walche_demo.py` + `provenance_viewer.py` · CRITICAL · score_history.jsonl schema conflict `[EXECUTED]`
Two writers now exist: the demo appends `{timestamp, final_score, verdict}`;
`provenance_viewer.py --export` rewrites the file with rich session records
(`date, cycle_scores, real_modules, domain_scores, …`). Confirmed by execution:
`provenance_viewer.print_session_table` raises `KeyError: 'cycle_scores'` on any
history containing a demo-appended row. Additionally `merge_history` (line 173-181)
prefers the EXISTING (impoverished) row over the discovered rich one, so `--export`
permanently degrades history quality instead of repairing it.
**Repair:**
1. In `walche_demo.py:594-597`, write the full session-record schema (match
   `provenance_viewer._parse_session` output: `date`, `timestamp`, `source`
   (log filename), `verdict`, `final_score`, `cycle_scores`, `delta`,
   `real_modules`, `proposals` (may be `[]` to keep rows small), `domain_scores`).
2. In `provenance_viewer.py:187-202` and `219`, make every field access tolerant:
   `s.get("cycle_scores", [s.get("final_score", 0.0)])`, `s.get("date", s.get("timestamp","")[:16])`,
   `s.get("real_modules", 0)`, `s.get("domain_scores", {})` — so legacy minimal rows
   can never crash the viewer.
3. In `merge_history`, when keys collide prefer the record with MORE fields
   (e.g. `if len(s) > len(existing_by_key[k]): replace`).

### T0-7 · `corpus_ingest.py:485-504` · CRITICAL · `--scan` full-replace write destroys merged KB `[SPOT-VERIFIED]`
`run_ingest` writes `"entries": [e.to_dict() for e in passed]` — no merge with the
existing KB (unlike `run_log_ingest:1017-1037`, which merges). At this repo root a
default `--scan .` extracts ZERO entries (no `.py` under `WALCHE_CORE_DIRS`) yet still
overwrites a populated `corpus/walche_kb.json` with `entries: []`.
**Repair:** in `run_ingest`, load the existing KB (same code path as
`run_log_ingest:1019-1025`), merge/dedupe by entry id keeping non-source entries,
and refuse to overwrite a non-empty KB with 0 extracted entries unless a new
`--force` flag is passed. Print what was kept vs replaced.

### T0-8 · `walche_apply.py` / `walche_validate.py` · CRITICAL · Import-time crash everywhere but owner's box + EPM-STARK contamination `[SPOT-VERIFIED]`
Both files: `WALCHE_DEFAULT = Path(r"C:\Users\wspar\Desktop\WALCHE_FULL_20260630_214341")`
with a `RuntimeError` at import if `WALCHE_ROOT` env is unset (line 17-24); both import
`core.*` at module level (crash — `core/` absent); both hardcode `C:\EPM-STARK` and
`mkdir` it AT IMPORT — on Linux this literally creates a junk directory named
`C:\EPM-STARK/...` in the CWD (agent-verified by execution). Docstrings say the files
"live in `C:\EPM-STARK\tools\`" — a direct violation of the CLAUDE.md constraint
"WALCHE must remain standalone — no EPM-STARK references inside WALCHE core".
**Repair:** (1) default root to `Path(__file__).resolve().parent.parent` with env
override; (2) wrap `core.*` imports in try/except with a clear exit message;
(3) move ALL `mkdir` calls inside `main()`; (4) **decision for the operator (flagged
per Law 2 — human-resolvable):** these two files belong to EPM-STARK per their own
headers — either relocate them to the EPM-STARK repo or strip every EPM-STARK path
and make targets configurable. Do not silently pick; ask.

---

## TIER 1 — HIGH: wrong results, dead features, dishonest output

### A. `walche_demo.py`
- **T1-A1 (HIGH)** `line 466`: `cfg = WalcheConfig()` is NOT wrapped in try/except
  (unlike lines 480-491). On the owner's machine, a real `core.config.WalcheConfig`
  whose constructor raises kills the whole demo. **Repair:** wrap in
  `try: cfg = WalcheConfig()` / `except Exception: cfg = _StubConfig()`.
- **T1-A2 (HIGH)** `lines 405-413`: if council returns `final_verdict: None`,
  `"CONDITION" in verdict_val` raises TypeError, and the whole deliberation's verdict
  is lost via the outer per-proposal catch. **Repair:**
  `verdict_val = str(result.get("final_verdict") or result.get("verdict") or "UNKNOWN")`.
- **T1-A3 (MEDIUM)** `line 447`: `parse_known_args` silently swallows typo'd flags
  (`--cylces 8` → runs 4 cycles, no error). **Repair:** use `parse_args()`.
- **T1-A4 (MEDIUM)** `line 25`: `ROOT = Path(__file__).parent.parent` without
  `.resolve()` — wrong ROOT if run via relative path on Python ≤3.10 or imported.
  **Repair:** `Path(__file__).resolve().parent.parent` (align with the other tools).
- **T1-A5 (MEDIUM)** `lines 480-491`: six bare `except:` clauses (catch
  SystemExit/KeyboardInterrupt too). **Repair:** `except Exception:`.
- **T1-A6 (LOW)** `line 215`: unseeded `random.uniform` guarantees scores climb每 run —
  provenance logs record nondeterministic synthetic improvement. **Repair:** seed from
  a `--seed` flag (default fixed) and document that demo signals are synthetic.
- **T1-A7 (LOW)** `line 539`: prints the real/total module count twice in one line.
  **Repair:** drop the duplicated `({real_count}/{len(_module_status)} loaded)`.

### B. `vll_engine.py`
- **T1-B1 (HIGH)** `line 107`: `_ts()` returns `…+00:00Z` — an INVALID ISO timestamp
  (`isoformat()` already appends the offset). `[EXECUTED]` `walche_status._age()`
  silently returns `''` for every VLL timestamp. **Repair:** drop the `+ "Z"`:
  `return datetime.now(timezone.utc).isoformat()`.
- **T1-B2 (HIGH)** `lines 179, 241`: `d.get('timestamp', '')[:16]` crashes with
  TypeError when a record has `"timestamp": null` (get default does not apply to
  explicit null). One bad council-log line kills `--apply` entirely. **Repair:**
  `(d.get('timestamp') or '')[:16]` at both sites (and same pattern in
  `walche_status.py:188` `d.get("timestamp", "")[:10]`).
- **T1-B3 (MEDIUM)** `lines 271-275`: `n_approved`/`n_rejected` count
  per-proposal-per-dimension, then persist into `approved_count` — the stats overstate
  reality by ~6× (Law 4 issue: misleading numbers). **Repair:** count once per decision
  (increment outside the `for dim` / `for proposal` loops).
- **T1-B4 (MEDIUM)** `line 387`: `root = Path.cwd()` — run from anywhere but WALCHE
  root and it silently finds nothing / writes state to the wrong tree. **Repair:**
  `root = Path(__file__).resolve().parent.parent` (keep an optional `--root` override).
  Same defect in `walche_status.py:236`, `walche_monitor.py:305`,
  `provenance_viewer.py:276` — fix all four identically.
- **T1-B5 (LOW)** `line 218`: `apply_learning` annotated `-> dict` but returns a tuple.
  **Repair:** `-> tuple[dict, dict]`.
- **T1-B6 (LOW)** `learning_log`/`dimension_history` grow unboundedly in
  `vll_state.json`. **Repair:** cap history lists (e.g. keep last 500 entries).
- **T1-B7 (LOW)** `line 190`: `proposal_text + " " + proposal_type` raises TypeError if
  `proposal_type` is explicitly `null` in a record. **Repair:** `str(proposal_type or "")`.

### C. `walche_monitor.py`
- **T1-C1 (HIGH)** `lines 103-155`: only `ImportError` falls back to logs; ANY other
  exception (e.g. `WalcheConfig()` raising) propagates and kills the daemon loop
  (line 243 is unguarded). **Repair:** change `except ImportError:` to
  `except Exception:` (or wrap the body after the imports in its own try/except that
  falls back to `_read_latest_scores`).
- **T1-C2 (HIGH — design honesty)** `lines 119-128`: the "live check" scores hardcoded
  static signals (duplicated from the demo's DOMAINS). `corpus.integrity` averages
  0.758 — permanently below the 0.80 default threshold — so the daemon alert-spams
  `logs/walche_alerts.jsonl` and re-triggers healing forever, and "healing" cannot ever
  change these constants. The monitor does not measure anything real. **Repair
  (minimum honest fix):** read the latest demo log's final-cycle scores as the primary
  source (`_read_latest_scores`) and use the static table only as a labeled
  `"synthetic"` fallback; include `"source": "log"|"synthetic"` in every alert record.
  Flag to operator that a true live metric source does not exist yet.
- **T1-C3 (MEDIUM)** `line 134`: `vll_path.read_text()` missing `encoding="utf-8"` —
  on Windows cp1252 this can throw and the bare except silently drops VLL weights.
  **Repair:** add `encoding="utf-8"`.
- **T1-C4 (LOW)** failed healing sets no cooldown (line 262-264) — a permanently
  failing heal respawns a 120s subprocess every interval. **Repair:** set
  `heal_cooldown = 3` on failure too (or a separate failure backoff).
- **T1-C5 (LOW)** `line 47`: `_ts()` returns a non-ISO format, inconsistent with every
  other tool's timestamps (breaks any future parsing of alert records). **Repair:**
  use `datetime.now(timezone.utc).isoformat()`.
- **T1-C6 (LOW)** `line 116`: `cfg` assigned, never used. **Repair:** delete.
- **T1-C7 (LOW)** no validation of `--interval` ≥ 1 (0/negative → busy loop).
  **Repair:** `parser.error` if `interval < 1`.

### D. `walche_status.py`
- **T1-D1 (MEDIUM)** `line 236`: `root = Path.cwd()` — see T1-B4.
- **T1-D2 (MEDIUM)** `line 188`: `d.get("timestamp", "")[:10]` — null-timestamp crash,
  see T1-B2.
- **T1-D3 (LOW)** `lines 190-192`: council verdict colorizer checks `"GO"`/`"CONDITION"`
  which council NEVER produces (its set is APPROVED/REJECTED/DEADLOCKED/ESCALATED);
  DEADLOCKED and ESCALATED render red, indistinguishable from REJECTED. **Repair:**
  add explicit cases: DEADLOCKED → yellow, ESCALATED → cyan/yellow.
- **T1-D4 (LOW)** `line 9-10`: docstring contains unescaped `\w` (invalid escape;
  SyntaxWarning on 3.12+ — same defect class that killed fix_walche.py). **Repair:**
  make the docstring raw (`r"""`). Same for `walche_monitor.py`, `vll_engine.py`
  (whose `\v` in lines 20-24 is a live VERTICAL TAB character in the docstring),
  `corpus_ingest.py:19-32`, `walche_deep_scan.py:4`.

### E. `walche_server.py`
- **T1-E1 (LOW)** `line 119`: comment says "suppress default access log" but the
  override prints every request — misleading comment (Law 4). **Repair:** fix comment
  to "custom access log format".
- **T1-E2 (LOW)** `line 25`: `os` imported but unused. **Repair:** remove.
- **T1-E3 (LOW)** `do_GET` 500-handler can write a JSON error into a stream that
  already has partial HTML headers (if `_send_html` fails mid-write). Harmless in
  practice; note only — no repair required.
- **T1-E4 (MEDIUM)** No `/api/alerts` endpoint: `walche_monitor` writes
  `logs/walche_alerts.jsonl` but no surface reads it — alerts are invisible to the
  operator console. **Repair (optional feature, confirm with operator):** add
  `/api/alerts` mirroring `_council_decisions()` over the alerts file.

### F. `fix_walche.py` (beyond T0-1/T0-2 — agent audit, behavioral claims verified by execution)
- **T1-F1 (CRITICAL)** `lines 500-505`: Fix 3's regex blindly inserts `return ` before
  `ingestion_manager.run()`; if the line is `result = ingestion_manager.run()` it
  produces `result = return …` (SyntaxError), and on a SECOND run (or an already-fixed
  file) it produces `return return …` — and STEP 7 makes NO backup. One re-run
  permanently breaks `run_system.py`. **Repair:** skip if
  `re.search(r"return\s+ingestion_manager\.run\(\)", src)` already matches; anchor the
  substitution to a whole statement with MULTILINE inside the located function only;
  write a `.bak` first; after ANY rewrite, `compile(new_src, str(target), "exec")` and
  abort on SyntaxError.
- **T1-F2 (CRITICAL)** STEPS 2–6 unconditionally overwrite five backend/scripts files
  with embedded stale snapshots — no existence/diff check, no backup — silently
  reverting any newer fixed versions every run (and it runs on IMPORT — see T0-2).
  **Repair:** skip when identical; write `<target>.bak` when different; require
  `--force`.
- **T1-F3 (CRITICAL)** ~14 sites: embedded payloads are riddled with the stale
  **"WACHEL"** misspelling the project explicitly resolved
  (`corpus/session_decisions_and_changes.json` line 140): ChromaDB collection
  `"wachel_reflections"` (line 44), LangSmith project `"wachel-healing-prod"`
  (line 73), class names `WACHELHealingState`/`WACHELHealingGraph`/`build_wachel_…`
  which the embedded train script imports by name — running this deploys the typo
  back and silently orphans reflection data into a parallel collection. **Repair:**
  rename all WACHEL/wachel identifiers to WALCHE/walche across payloads; flag the
  ChromaDB collection rename to the operator as a one-time manual data migration
  (human-resolvable — do not auto-migrate).
- **T1-F4 (HIGH)** `lines 26-29`: unconditional `pip install --upgrade` at import time
  with `check=False` and an unconditional "[OK] anthropic upgraded" — a false success
  report even when pip fails, and an unconfirmed external network action (CLAUDE.md
  hard stop). **Repair:** move behind `main()` + an opt-in `--upgrade-sdk` flag; check
  `returncode` and report honestly.
- **T1-F5 (HIGH)** `lines 465-481`: Fix 1 inserts the UTF-8 shim ABOVE the module
  docstring (verified: displaces `__doc__`, and produces a hard SyntaxError when the
  target uses `from __future__ import`). **Repair:** extend the skip loop to pass over
  an initial docstring and `from __future__` lines before inserting.
- **T1-F6 (HIGH)** `lines 456-511` + `461`: STEP 7 reads with `errors="replace"` and
  writes back — any non-UTF-8 byte becomes U+FFFD permanently; no backup, non-atomic
  write. **Repair:** strict `encoding="utf-8"` read (abort on decode error), `.bak`,
  tmp+`os.replace`.
- **T1-F7 (HIGH)** docstring says "Run from: C:\Users\wspar\Desktop\WALCHE\" with the
  file at WALCHE root — but `ROOT = parent.parent` means that placement writes
  `backend/`/`scripts/` onto the Desktop's PARENT. **Repair:** fix the docstring
  (run from `walche_tools/`); add a sanity check that ROOT contains `walche_tools/`
  or `run_system.py` before writing anything, abort otherwise.
- **T1-F8 (MEDIUM×5, LOW×4)** Fix 2's osmoda import-strip leaves dangling
  `OSMODAIngestionManager` call sites (ImportError becomes runtime NameError);
  embedded `get_recent_reflections` returns an arbitrary slice, not recent entries
  (no timestamp metadata — store/sort by timestamp); embedded ChromaDB path
  `"corpus_store/reflections"` is CWD-relative (anchor absolute); embedded
  healing_agent hardcodes retired model `claude-3-5-sonnet-20241022` (every call
  fails and is folded into a "valid" 0.5-confidence diagnosis — parameterize the
  model, surface errors distinctly); embedded "PPO integration" never uses the
  trained agent (`ppo_node` re-implements a hardcoded policy; the "Fully Operational"
  banner misrepresents it); embedded `execute_node` fabricates
  `{"status": "success", "delta": …}` without executing anything (label
  `"simulated"` or wire to a real executor); `open()` without encoding in embedded
  ppo agent; `datetime.utcnow()` ×2 in payloads; dead `import os`.

### G. `patch_run_system.py` (agent audit, verified by execution)
- **T1-G1 (HIGH)** `lines 34-36 vs 297-352`: partial-patch lockout — the six patch
  steps are independent, failures only WARN, and the result is written regardless.
  If step 5 (call site) succeeds while step 3 (function def) fails, the written
  `run_system.py` calls `run_healing_loop(cfg)` with no definition (NameError), and
  the re-run guard (`if "run_healing_loop" in src`) then sees the dangling call and
  exits "nothing to do" — the broken state is permanent. **Repair:** track per-step
  success; abort without writing if any required anchor failed; strengthen the guard
  to `def run_healing_loop(`.
- **T1-G2 (HIGH)** `lines 239-241` (embedded): verdict logic —
  `"GO" if all_pass else "GO-WITH-CONDITIONS" if some_warn else "NO-GO"` — a domain
  that outright FAILs (score < 0.70) yields GO-WITH-CONDITIONS whenever any other
  domain is in the WARN band (verified with `[0.40, 0.75, 0.75, 0.75]`). **Repair:**
  `any_fail = any(r["score"] < 0.70 …)`; `verdict = "NO-GO" if any_fail else "GO" if
  all_pass else "GO-WITH-CONDITIONS"` (this mirrors walche_demo.py's correct logic).
- **T1-G3 (MEDIUM×4)** embedded healing loop scores hardcoded ALL-ZERO signals and
  converts any scorer exception into a synthetic 0.75 (broken module renders as
  healthy WARN — log the exception, use a `None`/"ERROR" sentinel);
  `ai_self_audit("healing_loop")` called unguarded (NameError if target lacks it —
  wrap and default); non-atomic `TARGET.write_text` (tmp+`os.replace`, same pattern
  as walche_demo.py:589-591); pervasive `except Exception: pass` defaults
  (`rpn=18, confidence=0.9`) print green PASS rows over a fully broken stack (count
  and print `[DEGRADED]` markers, include an `errors` list).
- **T1-G4 (LOW×4)** negative cycle delta renders "▲ +-0.050" (compute sign properly);
  embedded DISPLAY_HELPERS uses `sys` without importing it; `_hsc` colors confidence
  on score thresholds; re-run after hand-editing silently overwrites the only `.bak`.
- **T1-G5 (HIGH — cross-file)** `fix_walche.py` STEP 7 and `patch_run_system.py` are
  mutually hazardous on `run_system.py` in either order (fix-then-patch backs up the
  already-corrupted version; patch-then-fix re-corrupts with no backup). **Repair:**
  both must compile-check before writing and refuse on failure; document a single
  canonical order in PENDING_COMMANDS.md.

### H. `provenance_viewer.py` (my findings + agent audit; crash verified by execution)
- **T1-H1 (CRITICAL)** = T0-6: `KeyError: 'cycle_scores'` on every non-`--quiet` run
  once the demo has appended to score_history.jsonl. Also `merge_history` seeds `seen`
  from existing history FIRST, so the minimal row permanently shadows the rich log
  record.
- **T1-H2 (HIGH)** `line 276`: `root = Path.cwd()` — see T1-B4; `--export` from the
  wrong CWD creates a spurious `corpus/score_history.jsonl` elsewhere.
- **T1-H3 (HIGH)** `lines 166-170`: `save_history` truncate-writes the history file
  non-atomically (crash mid-write destroys ALL cross-session history) and re-schemas
  a file another tool appends to. **Repair:** tmp+`os.replace`; unify on the FULL
  session schema everywhere (see T0-6 repair) so both writers agree.
- **T1-H4 (MEDIUM×2)** `logs/provenance_log*.json` parsed with dict-only assumptions,
  no try/except — a JSON-array provenance log crashes discovery with AttributeError
  (type-check `isinstance(raw, dict)`, iterate lists); dedupe key `timestamp[:16]`
  is minute-resolution — two runs in the same minute silently drop one (dedupe on
  full timestamp, or `(timestamp, source)` — NOTE: fixing this requires the same
  key change in `discover_sessions` line 142 and `merge_history` line 175).
- **T1-H5 (LOW×4)** column alignment computed on ANSI-colored strings (pad before
  colorizing); unused `timezone` import; `sparkline` renders a flat series as blank
  spaces (`_SPARK[0]` is a space — special-case `rng == 0`); unknown verdicts render
  red, indistinguishable from NO-GO (add a neutral branch).

---

## TIER 2 — council_of_9.py (agent audit, key claims consumer-verified)

- **T2-1 (CRITICAL)** `lines 1221-1226`: Council-of-9 `ESCALATED` is a terminal dead
  end — the Full Grand Council can never be convened from the ratification tier; the
  proposal is neither approved nor rejected and VLL ignores it. **Repair:** after the
  C9 result, if `r9.verdict == "ESCALATED"`, run `full_grand_council(...)`, store it in
  `results`, and take `final_verdict` from it (same for the `c9_only` tier).
- **T2-2 (CRITICAL)** `line 1223`: C9 exclusion list recomputed via a fresh
  `select_judges` call instead of the judges actually seated in C5 — double-vote risk
  the moment selection gains any nondeterminism (which it already has, see T2-3).
  **Repair:** add `judge_ids: List[int]` to `CouncilResult`, populate it in
  `_run_council`, pass `exclude_ids=r5.judge_ids`.
- **T2-3 (HIGH)** `line 1033`: unseeded `random.shuffle` fills 5 of 9 ratification
  seats — governance verdicts are irreproducible run-to-run. **Repair:** deterministic
  fill: `remaining.sort(key=lambda j: j.id)` (or a seeded local `random.Random`).
- **T2-4 (HIGH)** `lines 1084-1091`: ANY judge's ESCALATE forces escalation
  (special power `escalate_to_full_council` never checked), and escalation is
  evaluated BEFORE hard vetoes, so a stray ESCALATE overrides a constitutional veto.
  **Repair:** count ESCALATE only from judges whose
  `special_power == "escalate_to_full_council"`; treat others' ESCALATE as ABSTAIN;
  check hard vetoes first; ignore ESCALATE when already in `full_grand_council`.
- **T2-5 (HIGH)** `lines 927-948`: local (no-API) mode can only vote REJECT if
  `judge.bias == "reject"` — most panels structurally cannot reject anything.
  **Repair:** add symmetric branch `elif adjusted < 0.45: vote = "REJECT"` before the
  ABSTAIN fallthrough.
- **T2-6 (HIGH)** local mode never emits ESCALATE at all — whistleblower power is dead
  code without an API key. **Repair:** emit ESCALATE from the whistleblower judge on
  omission-keyword hits, or print a loud "local mode cannot escalate" notice at start.
- **T2-7 (HIGH)** `lines 909-914`: relevance scoring matches focus words as raw
  substrings (`"io"` matches "action") — inflates APPROVED verdicts on letter
  coincidences. **Repair:** tokenize with `re.findall(r"[a-z0-9]+", p_lower)` and test
  set membership.
- **T2-8 (MEDIUM)** `lines 1266-1291`: when `core.provenance` IS importable (owner's
  box), the JSONL that vll/status/dashboard read is NEVER written — decisions vanish
  from the learning loop. **Repair:** append to
  `logs/grand_council_decisions.jsonl` unconditionally; treat ProvenanceLog as an
  additional sink; log (don't swallow) sink failures.
- **T2-9 (MEDIUM)** `line 1284`: fallback log regenerates its timestamp instead of
  using the deliberation `ts` — return dict and log disagree; breaks VLL dedup
  windows. **Repair:** pass `ts` into `_write_provenance` and write it.
- **T2-10 (MEDIUM)** `deliberate()` returns no `score`/`confidence` key while
  `walche_demo.py:407` looks for one — every council verdict logs score 0.0 forever.
  **Repair:** include `"confidence"` per council in the `councils` sub-dicts and a
  top-level aggregate `"score"`.
- **T2-11 (MEDIUM)** the fallback JSONL writer is itself unguarded — a disk error
  aborts `deliberate()` AFTER the verdict was announced; the caller loses the verdict.
  **Repair:** wrap the fallback write in try/except, print loudly, still return.
- **T2-12 (MEDIUM)** full-council quorum (24/47) counts abstentions as de-facto
  rejections; judge 46 always abstains; dominant outcome is DEADLOCKED (pocket veto).
  **Repair:** majority of non-abstaining votes with a minimum participation count.
- **T2-13 (MEDIUM)** Sovereign tiebreaker fires only on exact tie, only toward
  APPROVED, can approve below quorum, and the judge is almost never seated in C5.
  **Repair:** explicit tie rule honoring the Sovereign's actual vote in both
  directions; never emit APPROVED below quorum.
- **T2-14 (MEDIUM)** hard-veto check precedes the `abstain_until_ratified` special
  case — Law-5 Placeholder judge (mandated to always ABSTAIN) can vote REJECT.
  **Repair:** check `abstain_until_ratified` before the veto loop.
- **T2-15 (LOW×8)** dead `reject = max(reject, 1)`; malformed sentinel
  `Judge(0,"",...)` (latent KeyError); API CONFIDENCE not clamped to [0,1]; blanket
  `except Exception` silently downgrades all 47 judges to local mode on missing
  API package; unused imports `field`/`Tuple` + duplicate `os` import; unlocked JSONL
  append under concurrency; returned `tier` is the requested tier, not the deciding
  one; ESCALATED renders red (indistinguishable from REJECTED) in status/dashboard.
  **Repairs:** as stated in each clause; all are one-to-three-line changes.

---

## TIER 2b — corpus_ingest.py (agent audit; C1/H2 spot-verified by me)

- **T2b-1 (CRITICAL)** = T0-7 above (full-replace KB write).
- **T2b-2 (HIGH)** `line 130`: UTF-8-BOM files silently skipped (`ast.parse` chokes on
  U+FEFF; bare `except SyntaxError` hides it). **Repair:** `encoding="utf-8-sig"`.
- **T2b-3 (HIGH)** source-scan path never calls `scrub_secrets` (only the log pipeline
  does — lines 710/799). Hardcoded keys/passwords in docstrings land verbatim in the
  KB. **Repair:** scrub `module_content`/`content` at lines ~151/214/260 before
  building each entry.
- **T2b-4 (HIGH)** `line 343`: exclusion via substring on the ABSOLUTE path — an
  ancestor dir containing "venv" excludes everything (silent empty scan → T0-7 wipe);
  `.git`/`node_modules`/`build` are NOT excluded. **Repair:** compare path PARTS
  relative to root against a set.
- **T2b-5 (HIGH)** per-file failures swallowed with zero accounting;
  `files_scanned`/`files_with_content` computed identically from extracted entries.
  **Repair:** count actual files globbed, record and print `files_failed` + names.
- **T2b-6 (MEDIUM×15)** async methods invisible (`AsyncFunctionDef` unhandled in class
  extraction and hint detection); `ast.walk` in ClassDef harvests locals as
  "attributes"; `AnnAssign` (dataclass fields) never counted; class-id collisions for
  nested same-name classes; OS-dependent ids (`\` vs `/` — use `.as_posix()`); corrupt
  existing KB silently replaced (rename to `.corrupt-<ts>` instead); merge dedup keeps
  the STALE copy (new entries should win); log-ingest stats describe the run not the
  KB (`total_tokens` wrong after merge); extensionless binaries ingested as logs
  (sniff for `\x00`); `--platform` hint accepted, printed, never used; no root
  anchoring (relative `--output` lands under CWD — anchor to
  `Path(__file__).resolve().parent.parent`); cp1252 UnicodeEncodeError on redirected
  Windows stdout (add `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`);
  no file-size cap (multi-GB read); PreIngestGate import attempted per-entry (memoize
  once at module level); text-log token counts measured before the `[:2000]`
  truncation (truncate first).
- **T2b-7 (LOW×12)** `errors="ignore"` silently deletes non-UTF-8 chars (use
  `"replace"` + warning counter); text-log preamble chunk dropped; JSONL platform
  detection only from line 0; real gate fails open on unknown dict shape (fall back to
  `_gate_check` instead); `_try_real_gate -> bool` returns None; integrity floor
  `max(0.78, score)` makes a bad corpus structurally unreportable (remove the floor);
  run_ingest stats omit `total_entries_in_kb` (add it); fixed `.tmp` name collides
  between concurrent runs; `--logs corpus` self-ingests `walche_kb.json` (skip the
  output path); `has_examples` inconsistent between module and class/function paths;
  posonly/kwonly/*args/**kwargs excluded from parameter extraction; unused `verbose`
  param in `_gate_check`.

---

## TIER 2c — walche_dashboard.html (agent audit)

- **T2c-1 (HIGH)** lines 1058, 1062-1063: last remaining XSS-class sink —
  `${n.toLocaleString()}` etc. into `innerHTML` where `n` comes UNVALIDATED from
  `corpus_kb_stats` (a string value in a tampered/corrupt `walche_kb.json` passes
  through raw). **Repair:** coerce: `Number(kb.total_entries_in_kb ?? kb.entries_passed ?? 0)
  .toLocaleString()` and `Number(kb.total_tokens || 0).toLocaleString()`; apply the same
  coercion at the line-941 textContent site for display sanity.
- **T2c-2 (MEDIUM)** line 604: `timeAgo()` appends `'Z'` to `+00:00` timestamps →
  Invalid Date → the "Last Run … ago" field is permanently blank for every real log.
  **Repair:** `const hasTz = /(?:Z|[+-]\d{2}:?\d{2})$/.test(ts); new Date(hasTz ? ts : ts+'Z')`.
- **T2c-3 (MEDIUM)** council panel: `/api/status` only carries the last 5 decisions
  (status.py line 105) but the panel labels it "(N total)" and the node shows a
  plateaued "5dec"; the richer `/api/council` (last 20) is never fetched. **Repair:**
  fetch `/api/council` for the panel (with `response.ok` + array checks) or relabel to
  "last 5".
- **T2c-4 (MEDIUM)** lines 856-867: per-module real/stub orb coloring is positional
  fiction derived from a count. **Repair:** send `module_status` (already in the demo
  log!) through `/api/status` → color orbs by name; fallback: neutral color + counts.
- **T2c-5 (MEDIUM)** API `baseline` field ignored — baseline hardcoded at line 578 and
  in static HTML (line 562). **Repair:** adopt `data.baseline` when numeric; update
  `#stat-baseline`.
- **T2c-6 (MEDIUM)** stale-data ghosting: on `{error}` responses and null transitions,
  old KPI/gap/integrity values remain rendered with no staleness marker. **Repair:**
  in the error branch and null branches, explicitly render '—' into
  `stat-integ`/`gap-label` etc.
- **T2c-7 (LOW×4)** `openDetailPanel` crashes on non-numeric `cycle_scores` elements
  (filter to finite numbers — mirror the buildSpeakText guard); no fetch timeout
  (AbortController at 35s — backend subprocess can take 30s); `#stat-corpus` color
  never reset after a "No KB" episode (reset `style.color` in the success branch);
  dead `lr.cycles` key + `parseFloat(x || 0)` NaN-leak idiom (use a
  `num(v, d=0)` helper).

---

## TIER 2d — walche_deep_scan.py / walche_validate.py / walche_apply.py (agent audit; EPM items spot-verified)

`walche_validate.py` / `walche_apply.py` Tier-0 items are T0-8. Remaining:

- **T2d-1 (CRITICAL)** `walche_apply.py:164-187`: appends a fake gate function
  (`walche_gated_send` — prints "OK (risk low per PAIN)" and `return True`, checks
  NOTHING) into `tools/send_to_corpus.py` non-idempotently — every run stacks another
  duplicate def into a pipeline that uploads to a corpus. **Repair:** sentinel check
  before append (`if "WALCHE PERSONA-LED APPLY" in content: skip`); replace the stub
  body with a real `PreIngestGate` call or `raise NotImplementedError` — never a
  hardcoded `True`.
- **T2d-2 (HIGH)** `walche_apply.py:157`: `decision = "APPLY"` hardcoded, ignoring the
  computed risk directly above it (and contradicting the risk<0.25 gate used in the
  sibling function). **Repair:**
  `decision = "APPLY" if risk < 0.25 and score.composite > 0.80 else "PROPOSE"`, and
  only mutate files when decision == "APPLY".
- **T2d-3 (HIGH)** `walche_apply.py:201-211`: provenance record hardcodes fictional
  platform state (`"docs": 50, "vll": 1982, "master": 1.0, "residual": 0.0`) and logs
  `"APPLIED (manual)"` even when both apply functions early-returned on SKIP.
  **Repair:** return status objects from the apply functions; write provenance only
  from those; delete the invented numbers.
- **T2d-4 (HIGH)** `walche_apply.py:71-79`: reads the target file then never uses it;
  signals are constants — the "analysis" outputs APPLY/risk-0.16 for ANY input
  including a file of syntax errors. **Repair:** derive signals from content or label
  the record `"signals_source": "static-template"`.
- **T2d-5 (HIGH)** `walche_apply.py:196`: `cfg.dry_run = False` forced unconditionally,
  no flag, no confirmation — contradicts the CLAUDE.md hard-stop on unconfirmed
  destructive actions. **Repair:** default dry-run True; require `--live`.
- **T2d-6 (HIGH)** `walche_validate.py:103-119, 153-174`: "validation" scores are
  hardcoded constants bumped by substring sniffs of the first 2,500 chars (a
  `# TODO: add error handling` comment RAISES the safety score), recorded with no
  synthetic marker. **Repair:** add `"signals_source": "heuristic-keyword-stub"` at
  minimum; longer-term replace with real analysis.
- **T2d-7 (HIGH)** `walche_validate.py:173`: `delta_vs_baseline` computed against
  0.78, not the platform baseline 0.9133 — every validation looks like an improvement
  when it is 0.11 below baseline. **Repair:** use 0.9133 (import a shared constant).
- **T2d-8 (HIGH)** `walche_deep_scan.py:255-274`: contamination patterns are matched
  as LITERAL substrings, so the regex-style `r"C:\\EPM"` NEVER matches real
  `C:\EPM...` strings (agent-verified by execution) — the EPM contamination detector
  is blind to its primary target. Also the scanner flags ITSELF (its own pattern
  table) as ~7 CRITICALs every run. **Repair:** proper `re.search(pattern, line,
  re.IGNORECASE)` with correct patterns; skip the scanner's own file.
- **T2d-9 (MEDIUM×7, LOW×6)** deep_scan: ROOT missing `.resolve()`; double execution
  of `scan_imports`/`scan_loop_registry` (double side effects); pytest subprocess can
  orphan grandchildren (process-group kill needed); silent `[:10]` test truncation +
  glob/rglob discovery mismatch; circular-import graph misses `from core import x`
  edges and `__init__` normalization; `py_compile` writes `__pycache__` all over the
  scanned tree and misses `OSError` (use `ast.parse` instead); non-atomic date-only
  report filename (same-day runs clobber); typo'd exclude entry `"wachle-dashboard"`;
  orphan-detection boolean makes prefix list dead code (`or` should be `and`) +
  substring reference check; dead `visit_Try` / unused-import tracker; async/lambda
  mutable-default gap; unconditional ANSI + non-cp1252 glyphs (UnicodeEncodeError on
  redirected Windows stdout AFTER scan work, before report write); blanket
  `except: pass` around visitor drops all findings for a file; docstring path
  `tools/` + Windows-only venv path.

---

## TIER 3 — loop_audit.py, walche_server.py residuals, structure & repo hygiene

### loop_audit.py
- **T3-1 (MEDIUM)** the audit report is never persisted (print-only) and the process
  always exits 0 even on NO-GO — nothing can gate on it, no provenance (Law 8).
  **Repair:** write `logs/loop_audit_<ts>.json` (atomic tmp+replace) with the full
  `AuditReport` (dataclasses via `asdict`), and `sys.exit(1)` when overall is NO-GO
  (`2` for GO-WITH-CONDITIONS optional).
- **T3-2 (MEDIUM)** `c0_syntax` compiles EVERY `.py` under the repo root (all three
  unrelated projects) and writes `__pycache__` everywhere as a side effect; exclude
  set lacks `".venv"`. **Repair:** syntax-check with `ast.parse` (no bytecode), add
  `".venv"`, and scope the walk to `walche_tools/` + `core/` + `backend/` only.
- **T3-3 (LOW)** `line 12`: ROOT missing `.resolve()` (see T1-A4). `line 1`: docstring
  says `tools/loop_audit.py`. `line 189`: dead `cls = cls`. `line 341`: venv check
  ignores `.venv/`. c9 double-counts files matching both `test_*.py` and `*_test.py`.
  **Repairs:** as stated; all trivial.
- **T3-4 (INFO)** In this repo, C1/C2/C5/C7/C8 always FAIL (no `core/`) → C12 is
  permanently NO-GO. Accurate, but the report never states "core/ is missing" as the
  root cause. **Repair:** pre-flight check that emits a single clear
  "core/ directory absent — module checks skipped" line.

### Structure / repo / data
- **T3-5 (HIGH — structural)** `core/`, `backend/`, `logs/`, `tests/` do not exist in
  this repo. The "platform" in git is tools-only; nothing here can run end-to-end,
  and every tool degrades to stub/fallback modes silently. **Repair (operator
  decision, flag per Law 2):** either commit the WALCHE core from
  `C:\Users\wspar\Desktop\WALCHE_FULL_20260630_214341` into this repo (recommended —
  it is the platform's only backup) or document loudly in a README that this repo is
  tools-only.
- **T3-6 (MEDIUM — security)** `PENDING_COMMANDS.md` instructs the operator to
  download executable code with `-SkipCertificateCheck` (TLS validation disabled).
  **Repair:** remove the flag from all queued commands; raw.githubusercontent.com has
  a valid certificate — if the owner's box has a corporate MITM proxy, fix the trust
  store instead.
- **T3-7 (LOW)** `Dockerfile` does `COPY . .` — the Print Perfect image ships the
  entire WALCHE corpus, leaked-source tree, and PENDING_COMMANDS (with its
  server address). **Repair:** add a `.dockerignore` entry for `walche_tools/`,
  `corpus/`, `*.md` queues, or COPY only `Print Perfet/`.
- **T3-8 (LOW)** `walche_server.py` binds localhost by default (good) but sets
  `Access-Control-Allow-Origin: *` on all JSON — any website open in the operator's
  browser can read the dashboard API while the server runs. **Repair:** drop the CORS
  header or restrict to `http://localhost:<port>`.
- **T3-9 (INFO)** Three unrelated projects share one repo (WALCHE tools, Print
  Perfect app, leaked-source TS corpus). Not a defect per se; noted because every
  scanning tool in walche_tools currently walks all of it (see T3-2, T2b-4).

---

## CROSS-FILE CONTRACT MAP (verified state after this audit)

| Producer | Artifact | Consumers | Status |
|---|---|---|---|
| walche_demo.py | logs/walche_demo_*.json | status, server, monitor, provenance_viewer, vll | OK (schema consistent) |
| walche_demo.py | corpus/score_history.jsonl (append) | status (OK), provenance_viewer (**CRASHES** — T0-6) | BROKEN |
| provenance_viewer --export | corpus/score_history.jsonl (rewrite) | status, itself | DEGRADES data (T0-6) |
| council_of_9.py | logs/grand_council_decisions.jsonl | vll, status, server | Never written by demo runs (T0-4); skipped when core.provenance importable (T2-8) |
| vll_engine.py | corpus/vll_state.json | monitor (OK), status (OK), **demo (NEVER READS — T0-3)** | BROKEN |
| corpus_ingest.py | corpus/walche_kb.json | status, dashboard | `--scan` wipes merged KB (T0-7) |
| walche_monitor.py | logs/walche_alerts.jsonl | **nobody** (T1-E4) | ORPHAN |
| loop_audit.py | (nothing persisted) | — | ORPHAN (T3-1) |

---

## REPAIR ORDER FOR SONNET

1. **Commit 1 (Tier 0):** T0-1 → T0-8 exactly as specified. T0-8 item 4 and T3-5
   require operator decisions — implement the safe parts (root default, import guards,
   mkdir into main) and ASK about relocation/core-commit; do not decide unilaterally.
2. **Commit 2 (Tier 1):** all T1 items. Mechanical; no design decisions except T1-C2
   (implement the labeled-fallback version described).
3. **Commit 3 (Tier 2/2b/2c/2d):** council, ingest, dashboard, deep-scan/validate/apply
   fixes. For T2d items that change what apply/validate CLAIM (honesty fixes), prefer
   labeling data as synthetic over inventing new analysis.
4. **Commit 4 (Tier 3):** loop_audit persistence/exit codes, repo hygiene.
5. After each commit: `for f in walche_tools/*.py; do python3 -m py_compile "$f"; done`
   must pass with zero output, plus re-run the two `[EXECUTED]` reproductions in this
   document to confirm they no longer reproduce.
6. Push to `claude/session-01ht1jmqadwsdphy19maevvl-7qlsw7` only.

---

*Pass 3 complete. Nothing was modified in this pass except the creation of this report
and the corpus session log entry.*
