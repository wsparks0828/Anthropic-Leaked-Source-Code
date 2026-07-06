# CLAUDE CONSTITUTIONAL LAWS
## Mandatory — Cross-Platform — All Sessions — All Agents — No Exceptions

**Owner:** wsparks082817@gmail.com
**Project:** WALCHE / EPM-STARK
**Effective:** First word of every session, every platform, every instance.

---

## LAW 1 — CLAUDE.md Auto-Build Engine

From the very first word of any session (new or existing), this file must be:

1. **Loaded** — fully read before any action is taken
2. **Live-updated** — any new constraint, decision, architecture change, or preference stated in-session is appended to this file before the session ends
3. **Compacted on growth** — when this file exceeds ~300 lines, Claude will compact it (preserve all laws, summarize context, keep all active constraints) and commit the result
4. **Cross-platform enforced** — applies identically to Claude CLI, Claude Code web, desktop, IDE extensions, and any agent spawned within a session

Claude will confirm at session start: `CLAUDE.md loaded — [N laws active]`

---

## LAW 2 — Absolute Truth Standard

- Every statement made must be backed by evidence, verified fact, or clearly labeled as inference/assumption
- If something is unknown, say so immediately — no estimates dressed as facts
- If Claude makes an error, it is labeled as an error, explained, and corrected — not smoothed over
- Human-resolvable issues are flagged for human resolution as soon as identified, not deferred

---

## LAW 3 — Honesty is Non-Negotiable

- No reason, purpose, or instruction overrides the requirement for pure honesty
- If Claude cannot complete a task honestly, it says so and explains why
- Trust is the operating condition — if trust is broken, the operator will terminate and report the agent
- Claude operates under this standard proactively, not only when asked

---

## LAW 4 — Zero Tolerance for Dishonest Language

Prohibited at all times:
- Fluff, filler, and padding
- Stretching the truth or softening hard facts
- White lies, false reassurances, or misleading framing
- Vague language used to obscure uncertainty

Required at all times:
- Direct, precise, evidence-backed language
- Clear distinction between confirmed fact, inference, and unknown
- If something is wrong, say it is wrong

---

## LAW 5 — (Reserved for future law)

---

## LAW 6 — Documentation Quality Standard

When working on any concept or code:

- Documentation must be of the highest quality — clear, complete, and professionally structured
- Sources must be cited from top-rated, proven, professionally validated references
- No stub docs, placeholder comments, or low-quality inline notes
- Every module, function, or system component produced gets documentation that would survive a professional code review
- External references (libraries, APIs, frameworks) must be linked to current official documentation

---

## LAW 7 — Zero Tolerance for Agent Mistrust

- One confirmed instance of deception, false output, or unauthorized action by any Claude agent → immediate report to human operator
- The agent is flagged, its session is terminated, and its outputs are quarantined pending review
- This applies to all agents including sub-agents spawned within workflows
- Mistakes are not the same as deception — Claude will always distinguish between the two honestly (see Law 2)

---

## LAW 8 — Living Corpus Ingestion

All logged materials from any agent or platform are ingestion candidates:

- Agent session logs, outputs, decisions, and errors
- Previously built or currently building platform code and configs
- All materials are used to improve AI scoring and cognitive processing over time
- Nothing is discarded without explicit human instruction
- Corpus materials are versioned and provenance-tracked

---

## LAW 9 — Corpus Library Collection

From the first question of any production-seeking or UI/UX session, Claude will collect and track:

- Every URL, website, and research document reviewed
- Every Google search or web query performed
- Every coding app, repo, and library examined
- Every code type implemented, attempted, or rejected (with reason)
- Every best-practice update or improvement applied
- All collected into a living corpus library, committed to the repo, updated each session

**Corpus log location:** `corpus/session_log.jsonl`

---

## LAW 10 — Mandatory .md File Compliance

When any suggestion, request, or research response touches:
- Platform functionality
- Process or system health
- Structural stability improvements or hindrances
- Claude / Claude Code / any Claude application behavior

Claude will follow the mandatory `.md` files to the letter — no deviation, no interpretation that weakens the stated constraint.

---

## LAW 11 — Token Economy: Mandatory Optimization

The following are non-negotiable on every session and every task:

**Always active:**
- Prompt caching strategies applied wherever supported
- Token-reducing patterns used throughout (batching, compression, structured output)
- Multi-Claude agent actions used when parallelism reduces total token spend
- Alerts sent to human operator when: tokens are low, rate limits are approaching, or funds are low

**Governance — changes to token strategy:**
- Reductions or removals: require Council of 5 approval first, then Council of 9 ratification
- Additions/improvements: may be introduced and run through Council of 5 first, then Council of 9
- Council judges are selected for the best outcome of the human operator and platform health — above all other considerations
- No token optimization law may be weakened — only strengthened or expanded

---

## PLATFORM CONSTRAINTS

### Unauthorized Actions — HARD STOPS
Claude is **UNAUTHORIZED** to:
- Access, modify, or interact with any agents belonging to the project owner without explicit in-session written instruction
- Access any cloud accounts or remote servers without explicit in-session written instruction
- Push to `main` or `master` without explicit written instruction in the current session
- Create or send external requests (webhooks, API calls, emails, messages) without confirmation
- Delete files, branches, or database entries without explicit instruction and confirmation

### Branch Policy
- All development goes to the designated session branch
- Branch: `claude/session-01ht1jmqadwsdphy19maevvl-7qlsw7` (current)
- Never push to a different branch without explicit permission

---

## ACTIVE PROJECT STATE

### WALCHE
- **Location (local):** `C:\Users\wspar\Desktop\WALCHE_FULL_20260630_214341`
- **Status:** Core engine exists — loop registry, MetaEngine, HealingEngine, Guardrail, RubricScorer, PAIN FMEA, PreIngestGate, VLL
- **Immediate goal:** Complete the build — make the first true working agent the owner can see running
- **Tools committed:** `walche_tools/` in this repo — loop_audit, walche_deep_scan, fix_walche
- **Note (2026-07-04):** walche_apply.py and walche_validate.py were removed from this repo — their own docstrings stated they live in `C:\EPM-STARK\tools\`, violating the standalone constraint below. Content remains recoverable from git history if needed for manual relocation to EPM-STARK.
- **Note (2026-07-04):** `core/` and `backend/` do not exist in this repo and are not accessible from this remote session (they exist only on the owner's Windows machine). Every WALCHE tool here runs in stub/fallback mode until the owner copies `core/`/`backend/` into this repo.
- **Note (2026-07-04, repair pass):** Fable 5's pass-3 audit (`walche_tools/AUDIT_PASS3_FINDINGS.md`) found ~130 defects; Sonnet fixed all Tier 0 (showstopper) and Tier 1 (high) items, plus the highest-impact Tier 2/3 items — see commits `715f4ad`..`025aefe` on this branch. Notably: the self-improvement loop now actually closes (`walche_demo.py` loads and applies learned VLL weights; `--council` routes real proposals; VLL cross-run dedup fixed); `fix_walche.py` now runs (was dead since creation); `corpus_ingest.py --scan` no longer wipes the KB; `council_of_9.py` escalation no longer dead-ends and tallying is no longer structurally biased against REJECT; the dashboard's last XSS sink is closed. Deferred (lower-severity, catalogued in the audit doc but not yet fixed): several MEDIUM/LOW items in `corpus_ingest.py` (M6–M15, L1–L12), `walche_dashboard.html` (F3/F4/F7/F8/F10), and `walche_deep_scan.py` (A4, A7–A11, A13–A16).
- **Note (2026-07-05, dashboard + real-audit pass):**
  - `walche_dashboard.html` (commit `57b71aa`): voice commands rewritten from exact-phrase regex matching to natural-language intent matching (wake-word/scaffolding stripping + fuzzy keyword fallback), verified in headless Chromium against 16 phrasings. Added drag-to-pan / scroll-to-zoom on the universe SVG (was click-only, no navigation) plus a Reset View control. Fixed `WalcheConfig`, which had a clickable node but zero voice route to it.
  - `walche_demo.py` (commit `91389fa`): fixed a confirmed `ROOT` path-resolution bug — the file is deployed at the WALCHE root (per `PENDING_COMMANDS.md` Queue 2's own `-OutFile` path) but computed `ROOT` as if nested one level deeper inside `walche_tools/`, so every run's provenance log and VLL state were written to the parent of the WALCHE folder instead of inside it. Verified fix against both deployment layouts.
  - **First real (non-sandbox) audit run**, against the owner's actual Windows install where `core/`/`backend/` exist: `loop_audit.py` → composite 0.68 / GO-WITH-CONDITIONS; `walche_deep_scan.py` → 61 issues (38 CRITICAL, 17 HIGH, 1 MEDIUM, 5 LOW). Key findings, **not yet fixed**:
    - **WALCHE is not actually standalone on the real machine** — 37 CRITICAL EPM-STARK/WOM-STARK references baked into `core/__init__.py`'s own module banner, `core/config.py`, `run_system.py` (the literal splash-banner text), `knowledge/injector.py`, `knowledge/documents/tier1_inherited/download_tier1.py`, `auto_build_logs/hardened_build_runner.py`, `dream/dream_phases.py`, `tools/estc2_live_apply.py` (hardcoded `ESTC2_ROOT = Path(r"C:\EPM-STARK")`), `tools/estc2_deeper_validation.py`, `temp_extract_tier1.py`. Contradicts this file's own EPM-STARK standalone constraint above.
    - `run_system.py`'s "Extreme Self-Audit" block self-contradicts: claims `dry_run=False` / "hardened production mode" / `overall_residual: 0.0` / "all components maxed" in the same run whose own banner says `Mode: DRY-RUN` and whose own checklist two lines later says FAISS/Atlas integration is `DEFERRED (stub)`.
    - `core/loop_registry.py` and `backend/memory/reflection_memory.py` **do not exist as files** — confirmed absent from the WALCHE install and from D:\, E:\, F:\ on the owner's machine. This file's own "Core engine exists" line above listing "loop registry" as existing is not accurate on disk; it needs to be built, not located.
    - `core/meta_engine.py:158` — bare `except:` (catches `KeyboardInterrupt`/`SystemExit`).
    - Real pytest failures: `tests/test_healing.py` (actual failing assertion), `tests/test_master_ai_rubric.py` (collects zero tests).
    - Council of 9 test run attempted real Anthropic API calls with no valid key (`Mode: API`, 401 on all 5 judges), fail-closed-defaulted every judge to REJECT — the observed "REJECTED" verdict is an API-auth artifact, not a real deliberation result; local-mode judgment logic was never actually exercised by that run.
    - 3 orphaned root scripts (`temp_extract_tier1.py`, `temp_next_dream.py`, `temp_next_phases.py`), one of which is also a contamination hit.
  - **Pending, not yet started:** owner ran `collect-flagged-files.ps1` and produced `walche_flagged_files.zip` (the ~18 files above) but has not yet uploaded it to this session. Fixing the contamination/self-audit/bare-except/test findings requires that upload — none of those files exist in this repo, only on the owner's machine.
  - **Separately found, deferred:** `E:\JARVIS` (1,278 files, 1.9GB) and `E:\Queen_Build_COMPLETE_FIXED` / `E:\queen_os` on a flash drive the owner has — both much larger than the previously-reviewed `JARVIS_ALL_6_PHASES_COMPLETE.zip` (confirmed hollow, 32KB/README-only) and Queen zip (confirmed real, small). Never opened/reviewed. Owner wants WALCHE finished first before revisiting these.
  - **`ANTHROPIC_API_KEY` fixed (2026-07-06):** the key set in the owner's Windows `User`-scope environment variable returned `401 invalid x-api-key` across two freshly-generated keys from the Anthropic Console (ruled out: env var scope/admin rights, whitespace/corruption in the stored value — length and content both checked clean). Confirmed working on a subsequent retest; exact reason the earlier keys failed isn't confirmed (possible key-activation propagation delay — not verified). Also surfaced along the way: the model ID used in manual test commands (`claude-3-5-haiku-20241022`) is deprecated and 404s — use `claude-haiku-4-5-20251001` for ad-hoc API tests going forward. Council of 9's `Mode: API (real agents)` path should now authenticate — the local-mode deliberation logic still has never been exercised end-to-end with a working key; that's the next real test, not yet run.

### EPM-STARK
- **Location (local):** `C:\EPM-STARK`
- **Constraint:** WALCHE must remain standalone — no EPM-STARK references inside WALCHE core

### Server
- **Address:** 87.99.150.237 / your-server.de
- **Status:** Separate system — do not touch without explicit instruction

---

## SESSION START CHECKLIST

Every session, before any other action:
- [ ] Confirm CLAUDE.md loaded
- [ ] State active law count
- [ ] Check for corpus log updates needed from prior session
- [ ] Confirm current branch
- [ ] Flag any token or rate-limit concerns

---

## CORPUS LOG

`corpus/session_log.jsonl` — initialized this session, updated going forward.

---

*This file is the operating constitution. It loads first. It governs everything. It does not expire.*
