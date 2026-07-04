# WALCHE — Pending Commands Queue

**Last updated:** 2026-07-04  
**Run from:** `C:\Users\wspar\Desktop\WALCHE_FULL_20260630_214341\WALCHE\`

---

## QUEUE 1 — Wire heal phase into run_system.py

```powershell
# Download the patch script
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/wsparks0828/Anthropic-Leaked-Source-Code/claude/session-01ht1jmqadwsdphy19maevvl-7qlsw7/walche_tools/patch_run_system.py" `
  -OutFile "walche_tools\patch_run_system.py" `
  -SkipCertificateCheck

# Run it once — patches run_system.py in place, backs up original as run_system.py.bak
python walche_tools\patch_run_system.py

# Test the new heal phase
python run_system.py --phase heal
```

**What it does:** Adds `--phase heal` to run_system.py. The heal phase runs the
forensic healing loop (4 cycles, C12 judgment) as a proper pipeline stage.
After patching, `--phase full` also includes the heal phase automatically.

---

## QUEUE 2 — Download and test walche_demo.py (updated)

```powershell
# Download updated walche_demo.py (HealingProposal serialization fix)
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/wsparks0828/Anthropic-Leaked-Source-Code/claude/session-01ht1jmqadwsdphy19maevvl-7qlsw7/walche_tools/walche_demo.py" `
  -OutFile "walche_demo.py" `
  -SkipCertificateCheck

# Run demo (should show readable proposals, not HealingProposal objects)
python walche_demo.py
```

---

## QUEUE 3 — Download and test Grand Council (council_of_9.py)

```powershell
# Download council
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/wsparks0828/Anthropic-Leaked-Source-Code/claude/session-01ht1jmqadwsdphy19maevvl-7qlsw7/walche_tools/council_of_9.py" `
  -OutFile "walche_tools\council_of_9.py" `
  -SkipCertificateCheck

# List all 47 judges
python walche_tools\council_of_9.py --list-judges

# Test deliberation (local mode, no API key needed)
python walche_tools\council_of_9.py --proposal "Reduce max_history from 10 to 5" --type token_strategy

# Test with real Claude agents (requires Anthropic API key)
python walche_tools\council_of_9.py --proposal "Reduce max_history from 10 to 5" --type token_strategy --api-key sk-ant-YOUR_KEY_HERE

# Full Grand Council (all 47 judges) on a major decision
python walche_tools\council_of_9.py --proposal "Disable prompt caching in test environments" --type token_strategy --tier full
```

---

## QUEUE 4 — Feed the corpus (fix corpus.integrity ~0.78 → 0.85+)

```powershell
# Download corpus ingestion tool (being built now)
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/wsparks0828/Anthropic-Leaked-Source-Code/claude/session-01ht1jmqadwsdphy19maevvl-7qlsw7/walche_tools/corpus_ingest.py" `
  -OutFile "walche_tools\corpus_ingest.py" `
  -SkipCertificateCheck

# Scan WALCHE and build corpus knowledge base
python walche_tools\corpus_ingest.py --scan . --output corpus\walche_kb.json

# Dry run first to see what would be ingested
python walche_tools\corpus_ingest.py --scan . --output corpus\walche_kb.json --dry-run

# After ingesting — run demo again to confirm corpus.integrity improved
python walche_demo.py
```

---

## QUEUE 5 — Explore provenance log

```powershell
# View provenance log from last walche_demo run
python -c "
import json, pathlib
log = pathlib.Path('logs/walche_demo_2026-07-03.json')
if log.exists():
    data = json.loads(log.read_text())
    import pprint; pprint.pprint(data)
else:
    print('Log not found — check logs/ directory for .json files')
    import os
    for f in os.listdir('logs'): print(f)
"
```

---

## QUEUE 6 — Run full pipeline after all patches

```powershell
# Full pipeline: inject → dream → heal → test → audit
python run_system.py --phase full
```

---

## STATUS

| Queue | Status | Notes |
|-------|--------|-------|
| 1 — Wire heal phase | PENDING | patch_run_system.py ready in repo |
| 2 — Updated walche_demo.py | PENDING | HealingProposal fix + cleaner output |
| 3 — Grand Council | PENDING | 47 judges, ready to deliberate |
| 4 — Feed corpus | PENDING | corpus_ingest.py being built |
| 5 — Explore provenance | PENDING | Run after Queue 1 generates new log |
| 6 — Full pipeline | PENDING | Run last, after all above complete |

---

*This file is updated each session. Do not delete.*
