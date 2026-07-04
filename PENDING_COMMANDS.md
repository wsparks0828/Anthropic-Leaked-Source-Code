# WALCHE — Pending Commands Queue

**Last updated:** 2026-07-04 (added Queue 4B — OSMODA paths)  
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
# Download corpus ingestion tool
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/wsparks0828/Anthropic-Leaked-Source-Code/claude/session-01ht1jmqadwsdphy19maevvl-7qlsw7/walche_tools/corpus_ingest.py" `
  -OutFile "walche_tools\corpus_ingest.py" `
  -SkipCertificateCheck

# --- OPTION A: Scan WALCHE Python source ---
python walche_tools\corpus_ingest.py --scan . --output corpus\walche_kb.json --dry-run
python walche_tools\corpus_ingest.py --scan . --output corpus\walche_kb.json

# --- OPTION B: Ingest AI platform build logs ---
# Supports: OpenAI, LangChain, LangSmith, LangGraph, AutoGen, CrewAI,
#           Hugging Face, GitHub Actions, any JSON/JSONL/text log.
# Platform is auto-detected from log structure. Secrets are scrubbed before write.

# Point at any directory containing your logs:
python walche_tools\corpus_ingest.py --logs C:\path\to\your\logs --output corpus\walche_kb.json

# With an explicit platform hint (optional — auto-detect usually works):
python walche_tools\corpus_ingest.py --logs C:\path\to\logs --output corpus\walche_kb.json --platform langchain

# --- OPTION C: Both in one pass (recommended — highest corpus density) ---
python walche_tools\corpus_ingest.py --scan . --logs C:\path\to\logs --output corpus\walche_kb.json

# After ingesting — run demo again to confirm corpus.integrity improved
python walche_demo.py

# View a report on what's in the KB
python walche_tools\corpus_ingest.py --report corpus\walche_kb.json
```

**Where to find logs on your machine:**
- LangSmith/LangChain traces: usually exported from https://smith.langchain.com as `.jsonl`
- OpenAI usage logs: dashboard → Usage → Export
- GitHub Actions: any `.log` files from workflow runs
- Your own agent outputs: any `.json` or `.jsonl` your agents write to disk
- AutoGen/CrewAI: console output redirected to a `.log` file works too

---

## QUEUE 4B — Ingest OSMODA corpus (exact paths — run after Queue 4)

> **Priority: HIGH** — This pushes corpus.integrity from ~0.78 → 0.87-0.92.
> OSMODA is ~1.88 GB of verified material across 16 genres on your machine.
> See full source inventory: `corpus/pending_ingestion_sources.json`

```powershell
# --- STEP 1: Ingest full OSMODA directory (all genres, auto-detects JSON/JSONL/markdown/text) ---
python walche_tools\corpus_ingest.py `
  --logs "C:\Users\wspar\OneDrive\Microsoft Copilot Chat Files\Desktop\OSMODA" `
  --output corpus\walche_kb.json

# --- STEP 2: Ingest CEVIP-processed output (structured vector records from ESTC2/stark) ---
python walche_tools\corpus_ingest.py `
  --logs "C:\Users\wspar\OneDrive\Microsoft Copilot Chat Files\Desktop\OSMODA\ingest\cevip" `
  --output corpus\cevip_kb.json

# --- STEP 3: Scan ESTC2/stark Python source (AST extraction of Scholar/HashEmbedder code) ---
python walche_tools\corpus_ingest.py `
  --scan "C:\Users\wspar\OneDrive\Microsoft Copilot Chat Files\Desktop\ESTC2\stark\src" `
  --output corpus\estc2_kb.json

# --- STEP 4: Scan HuggingFace AI repos (transformers + datasets Python source) ---
python walche_tools\corpus_ingest.py `
  --scan "C:\Users\wspar\OneDrive\Microsoft Copilot Chat Files\Desktop\OSMODA\AI_Fundamentals_to_Advanced_Practices_Repos_Issues_Testing" `
  --output corpus\ai_kb.json

# --- STEP 5: Run walche_demo to confirm corpus.integrity improved ---
python walche_demo.py
```

**After these 4 ingestion passes, run Queue 5 and Queue 6 for the full picture.**

**Note:** The Linux kernel (~1.58 GB) is C source — the AST scanner targets Python.
If you want to ingest Linux docs/Kconfig/txt files, run with `--logs` not `--scan`:
```powershell
python walche_tools\corpus_ingest.py `
  --logs "C:\Users\wspar\OneDrive\Microsoft Copilot Chat Files\Desktop\OSMODA\Computer_Science_Hardware_to_Advanced_Software\linux\Documentation" `
  --output corpus\linux_docs_kb.json
```

---

## QUEUE 4C — Clone remaining repos into OSMODA (exact paths from CLONE_COMMANDS.md)

> **Priority: HIGH** — These repos add volume to CS and AI genres. Run BEFORE Queue 4B if repos aren't already present.
> Check if `linux`, `llvm`, `transformers`, `datasets` already exist first (they may have been cloned Jun 29, 2026).

```powershell
# --- CS repos (Computer_Science_Hardware_to_Advanced_Software) ---
cd "C:\Users\wspar\OneDrive\Microsoft Copilot Chat Files\Desktop\OSMODA\Computer_Science_Hardware_to_Advanced_Software"

# Skip if already exists
if (!(Test-Path "linux")) {
    git clone --depth 1 https://github.com/torvalds/linux linux
}
if (!(Test-Path "llvm")) {
    git clone --depth 1 https://github.com/llvm/llvm-project llvm
}
if (!(Test-Path "riscv-isa-manual")) {
    git clone --depth 1 https://github.com/riscv/riscv-isa-manual riscv-isa-manual
}

# --- AI repos (AI_Fundamentals_to_Advanced_Practices_Repos_Issues_Testing) ---
cd "C:\Users\wspar\OneDrive\Microsoft Copilot Chat Files\Desktop\OSMODA\AI_Fundamentals_to_Advanced_Practices_Repos_Issues_Testing"

if (!(Test-Path "transformers")) {
    git clone --depth 1 https://github.com/huggingface/transformers transformers
}
if (!(Test-Path "datasets")) {
    git clone --depth 1 https://github.com/huggingface/datasets datasets
}
if (!(Test-Path "lm-evaluation-harness")) {
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness lm-evaluation-harness
}
```

**Estimated disk:** CS repos ~1.6 GB (linux dominant). AI repos ~200-350 MB. Run with stable internet.  
**Note:** linux and transformers may already be present from Jun 29, 2026 run. The `Test-Path` guards skip if so.

---

## QUEUE 4D — Download US Code XML (Legal genre)

> Add machine-readable US Code to the Legal genre for high-precision CEVIP ingestion.

```powershell
cd "C:\Users\wspar\OneDrive\Microsoft Copilot Chat Files\Desktop\OSMODA\Legal_Terminology"

# US Code Title 26 (Internal Revenue Code)
Invoke-WebRequest `
  -Uri "https://uscode.house.gov/download/releasepoints/us/pl/119/99/xml/usc26.xml.zip" `
  -OutFile "usc26.xml.zip"

# US Code Title 28 (Judiciary and Judicial Procedure)
Invoke-WebRequest `
  -Uri "https://uscode.house.gov/download/releasepoints/us/pl/119/99/xml/usc28.xml.zip" `
  -OutFile "usc28.xml.zip"

# Expand both
Expand-Archive -Path usc26.xml.zip -DestinationPath usc26_xml -Force
Expand-Archive -Path usc28.xml.zip -DestinationPath usc28_xml -Force

Write-Host "Done. Run corpus_ingest.py --logs Legal_Terminology --output corpus\legal_kb.json next."
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
| 4 — Feed corpus (generic) | PENDING | corpus_ingest.py ready in repo |
| 4B — Feed corpus (OSMODA) | PENDING | Exact OSMODA paths documented above — HIGH PRIORITY |
| 5 — Explore provenance | PENDING | Run after Queue 1 generates new log |
| 6 — Full pipeline | PENDING | Run last, after all above complete |

---

*This file is updated each session. Do not delete.*
