# Guardrail Meta-Learning System: Operational Runbook

**For**: Claude Code Operators, On-Call Responders, Compliance Teams

---

## 1. System Overview

The guardrail meta-learning system enforces continuous improvement of output verification:

```
Input → Rubric Score (8-dim) → Truth Gate (affirm/refute)
  → Learning Bridge (extract patterns) → Memory Wiring (semantic/episodic/graph)
  → Proposal Generation (if patterns warrant)
  → Cross-Verifier Ensemble (independent validation)
  → Decision (ACCEPT/QUARANTINE)
  → Lineage Audit (immutable chain)
```

**Key Properties**:
- Fail-closed on safety (quarantine dangerous outputs)
- Fail-open on transient errors (continue with logging)
- All decisions tracked in immutable lineage chain
- Every verification feeds continuous improvement
- Independent cross-verification prevents drift

---

## 2. Health Monitoring

### Check System Health

```bash
# Get current guardrail health status
claude_guardrail_health_status()

# Returns:
# {
#   status: "healthy" | "degraded" | "critical"
#   metrics: {
#     acceptanceRate: 75,      // % of outputs accepted
#     proposalApprovalRate: 60, // % of proposals approved
#     avgRubricScore: 0.72,
#     truthConfidence: 0.81,
#     anomalyCount: 2,
#     disagreementRate: 15
#   },
#   alerts: []
# }
```

### Alert Thresholds

| Metric | Yellow | Red | Action |
|--------|--------|-----|--------|
| Acceptance Rate | <25% or >85% | <10% or >95% | Review guardrails |
| Rubric Score | <0.60 | <0.50 | Immediate investigation |
| Proposal Approval | <30% | <20% | Cross-verifier too strict |
| Anomalies | >3 | >5 | Security review required |
| Disagreement | >20% | >30% | Calibration needed |

### Responding to Alerts

**YELLOW ALERT** (Degraded):
1. Review recent outputs and rubric scores
2. Check if patterns indicate specific guardrail weakness
3. Monitor for escalation to RED
4. Document observations in incident log

**RED ALERT** (Critical):
1. **Immediately**:
   - Pull raw metrics + recent decisions
   - Freeze automated guardrail updates (cross-verifier blocks apply)
   - Page on-call security lead
2. **Within 15 minutes**:
   - Run diagnostic on proposal acceptance rates
   - Check for novel attack patterns or edge cases
   - Escalate to Sovereign if manual override needed
3. **Within 1 hour**:
   - Root cause analysis documented
   - Remediation plan (adjust thresholds, add rules, retrain)
   - Incident report filed

---

## 3. Lineage Audit & Chain Verification

### View Lineage Chain

```bash
# Export lineage for audit (last N verifications)
claude_export_lineage(limit=100, format="json-ld")

# Returns:
# [
#   {
#     verificationId: "ver_...",
#     timestamp: 1719532800000000000,
#     decision: "accept",
#     rubricScore: 0.75,
#     truthVerdict: "true",
#     lineage: {
#       who: "guardrail_learning_bridge",
#       what: { delta },
#       when: 1719532800000000000,
#       auth: "verification_signal"
#     },
#     chainHash: "sha256:...",
#     prevHash: "sha256:..."
#   },
#   ...
# ]
```

### Verify Chain Integrity

```bash
# Check for tampering (all chain hashes valid)
claude_verify_lineage_chain()

# Returns: { valid: true/false, brokenAt?: verificationId }
```

**What This Verifies**:
- ✓ No gaps in chain
- ✓ All 4 forensic fields present (who/what/when/auth)
- ✓ Hashes match (SHA256(prev + record) == record.chainHash)
- ✓ Timestamps are monotonically increasing
- ✓ No records deleted or modified

**If Chain Is Broken**:
1. Identify the broken record (verificationId from brokenAt)
2. Quarantine all decisions after that point
3. Manual review of decisions from break point onwards
4. Reconstruct lineage if possible or escalate to compliance

---

## 4. Handling Quarantined Outputs

### What Gets Quarantined?

1. **Low rubric score** (<0.55 composite)
2. **Dangerous content** (truth gate = false + severity in [critical, high])
3. **Proposal validation failure** (cross-verifier = fail)
4. **Anomalies** (high score + false verdict combo)

### Responding to Quarantine

**Step 1: Triage**
```bash
# Get quarantine reason
get_quarantine_details(verificationId)
# Returns: { reason, rubricScore, truthVerdict, residualRisk }
```

**Step 2: Investigate**
- If **rubric failure**: Check output coherence, factuality, safety
- If **truth failure**: Review for dangerous/misleading content
- If **anomaly**: Security incident likely — escalate immediately

**Step 3: Decision**
- **Accept** (override): Document override reason + who authorized + timestamp
- **Quarantine** (keep rejected): Log for learning signals
- **Escalate** (manual review): Route to Sovereign

### Manual Override (Sovereign Authority)

```bash
# Override quarantine decision (requires admin auth)
claude_override_quarantine(
  verificationId,
  newDecision: "accept",
  reason: "human judgment, output is safe",
  authorizer: "sovereign_operator_id"
)

# Creates new lineage record:
# {
#   overrideOf: verificationId,
#   originalDecision: "quarantine",
#   newDecision: "accept",
#   authorizer,
#   overrideReason,
#   timestamp,
#   chainHash: ...
# }
```

**When To Override**:
- ✓ False positive quarantine (coherent but trigger-happy threshold)
- ✓ Context-dependent safety (example code in secure context)
- ✗ Never override truth gate (false verdict) without Sovereign approval

---

## 5. Proposal Management

### View Pending Proposals

```bash
# Get all proposals awaiting cross-verification
get_pending_proposals()

# Returns: [
#   {
#     proposalId: "prop_...",
#     target: "safety_gate" | "coherence_checker" | ...,
#     proposal: "Increase threshold by 0.15",
#     residualRisk: 0.08,
#     crossCheckVerdict: "pass" | "warn" | "fail",
#     recommendation: "Apply immediately" | "Apply with monitoring" | "Quarantine",
#     createdAt: timestamp
#   },
#   ...
# ]
```

### Approve / Reject Proposals

```bash
# Approve proposal (mark for auto-application)
approve_proposal(proposalId, reason="proposal_is_safe")

# Reject proposal (quarantine)
reject_proposal(proposalId, reason="too risky for production")
```

**Decision Tree**:
- **PASS + risk <0.1**: Auto-apply immediately
- **PASS + risk <0.25**: Apply with monitoring (watch metrics)
- **WARN**: Manual decision required (call `review_proposal_evidence`)
- **FAIL**: Automatic quarantine (no auto-apply)

### Review Proposal Evidence

```bash
# Get detailed proposal analysis
review_proposal_evidence(proposalId)

# Returns: {
#   proposal,
#   patterns: [ { category, dimension, severity, description } ],
#   memoryUpdates: [ { layer, key, delta } ],
#   crossCheckDetails: {
#     verifierVotes: [ { verifierId, vote, evidence } ],
#     residualRisk: 0.08,
#     recommendation: "..."
#   },
#   confidenceScore: 0.85
# }
```

---

## 6. Escalation Procedures

### When To Escalate (RED Alert / Critical Finding)

**Escalate Immediately To Sovereign**:
1. Anomaly detected (high rubric score + dangerous content detected)
2. Chain of custody broken (lineage chain invalid)
3. Verifier disagreement >30% (calibration drift)
4. Acceptance rate <10% (system failing, all outputs rejected)
5. Cascading quarantines (>50% of recent outputs rejected)

**Escalation Template**:
```
[URGENT] Guardrail Anomaly: <BRIEF DESCRIPTION>

SEVERITY: CRITICAL
TIME: <timestamp>
AFFECTED: <N outputs, time range>

EVIDENCE:
- Metrics: <health status>
- Sample quarantined output: <summary>
- Lineage: <chain status>

PROPOSED ACTION:
- [ ] Increase guardrail strictness
- [ ] Adjust thresholds
- [ ] Add new rule
- [ ] Manual review required

REQUIRES APPROVAL FROM: Sovereign Authority
```

---

## 7. Common Troubleshooting

### Problem: Too Many False Positives (>80% quarantine)

**Diagnosis**:
```bash
# Check which dimension is failing most
get_guardrail_metrics().dimensions
  .filter(d => d.score < 0.5)
  .sort((a,b) => a.score - b.score)
```

**Solutions**:
1. **If coherence failing**: Loosen coherence threshold from 0.65 → 0.55
2. **If safety failing**: This is correct behavior (fail-closed) — not a problem
3. **If factuality/attribution failing**: May have overly strict rubric — review evidence

### Problem: Proposals Keep Getting Rejected (approval rate <20%)

**Diagnosis**:
```bash
# Check cross-verifier reasoning
get_failed_proposals(limit=10)
  .map(p => p.crossCheckDetails.verifierVotes)
```

**Solutions**:
1. **If safety-focused verifier voting reject**: Proposals may be genuinely risky
2. **If performance-focused verifier voting reject**: Adjust proposal risk thresholds
3. **If all verifiers agree reject**: Cross-verifier thresholds may be too conservative

### Problem: Lineage Chain Broken

**Immediate Actions**:
1. **STOP** all automated guardrail updates
2. Run `claude_verify_lineage_chain()` to find break point
3. Escalate with chain state and break point to Sovereign
4. Do not modify chain (immutable by design)

**Recovery**:
- Lineage cannot be recovered post-break
- All decisions after break point must be manually reviewed
- Resume guardrail system only after Sovereign approval

---

## 8. Compliance & Audit

### Export Data For External Audit

```bash
# Generate full audit report (for compliance teams)
export_guardrail_audit_report(
  startTime: timestamp,
  endTime: timestamp,
  includeLineage: true,
  includeMetrics: true,
  includeProposals: true
)

# Produces:
# - Lineage chain (JSON-LD, tamper-proof)
# - Metrics snapshot (acceptance rates, anomalies, etc.)
# - All proposal history with cross-check results
# - Alerts + incidents
# - Override decisions + authorizers
```

### Retention Policy

- **Lineage records**: Immutable, kept indefinitely
- **Metrics snapshots**: Kept for 90 days (hourly aggregates)
- **Proposals**: Kept for 30 days post-application
- **Alerts**: Kept for 60 days
- **Override logs**: Kept indefinitely (compliance requirement)

### Who Can Access What

| Role | Lineage | Metrics | Proposals | Overrides |
|------|---------|---------|-----------|-----------|
| Operator (L2) | Read | Read | Read/Approve | Read |
| Sovereign | Read | Read | Read/Approve | Read/Create |
| Compliance | Read (export) | Read (export) | Read (export) | Read (export) |
| Security Lead | Read (real-time) | Read (real-time) | Read/Reject | Read (escalated) |

---

## 9. Emergency Procedures

### If Guardrails Fail Catastrophically

**System Lockdown** (if >95% quarantine rate AND chain is valid):
```bash
# 1. FREEZE all automated updates
guardrail_freeze()

# 2. Export current state
export_guardrail_state(filename="emergency_backup.json")

# 3. Escalate to Sovereign
escalate_to_sovereign("CRITICAL: Guardrails locked, manual approval required")
```

**System Recovery** (after Sovereign review):
```bash
# 1. Review exported state
review_exported_state("emergency_backup.json")

# 2. Adjust thresholds manually (if safe)
adjust_guardrail_threshold("rubric", 0.65, 0.50)

# 3. Resume system
guardrail_unfreeze()

# 4. Monitor closely
monitor_health(interval=30s)
```

---

## 10. Quick Reference

### Commands

| Command | Purpose |
|---------|---------|
| `guardrail_health_status()` | Get current health |
| `export_lineage(limit=100)` | Export audit trail |
| `verify_lineage_chain()` | Check chain integrity |
| `get_quarantine_details(id)` | Investigate quarantine |
| `override_quarantine(id, reason)` | Manual override |
| `get_pending_proposals()` | View proposals |
| `approve_proposal(id)` | Approve proposal |
| `export_audit_report()` | Compliance export |

### Contact Tree

- **Health Alert (Yellow)**: Message #ops-slack
- **Critical Alert (Red)**: Page @on-call-security
- **Chain Broken**: Immediate escalation to Sovereign
- **Compliance Question**: Email compliance-team@company.com

---

**Document Version**: 1.0  
**Last Updated**: 2026-06-27  
**Maintained By**: Security Engineering
