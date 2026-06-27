# Guardrail Meta-Learning System: Deployment Guide

**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: 2026-06-27

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Steps](#deployment-steps)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Health Checks](#health-checks)
5. [Rollback Procedures](#rollback-procedures)
6. [Operational Runbook](#operational-runbook)
7. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

Before deploying the guardrail system to production, verify:

### Code and Testing
- [ ] All 86 unit and integration tests passing locally
- [ ] Performance benchmarks meet latency requirements (<0.04ms per component)
- [ ] No critical security vulnerabilities in dependencies
- [ ] All commits signed and lineage complete
- [ ] Feature flags disabled for guardrail experimental features

### Configuration
- [ ] Config schema validated (`core/schemas.ts`)
- [ ] Environment variables documented (see below)
- [ ] Secret rotation credentials staged
- [ ] Database migration path verified (if applicable)
- [ ] STORAGE_READY gate tested with sample data

### Infrastructure
- [ ] Logging infrastructure ready (structured JSON)
- [ ] Monitoring dashboards created
- [ ] Alert channels configured (Slack, PagerDuty, email)
- [ ] Rollback procedure tested on staging
- [ ] Backup of current production state created

### Documentation
- [ ] Deployment guide reviewed by ops team
- [ ] Incident response playbook created
- [ ] Escalation tree updated with on-call contacts
- [ ] Team trained on observability signals

---

## Deployment Steps

### Step 1: Pre-Flight Validation (5 min)

```bash
# Verify all tests pass
bun test core/__tests__/*.test.ts

# Check performance baselines
bun test core/__tests__/performance_profiling.test.ts

# Validate configuration
bun run scripts/validate-config.ts

# Verify lineage integrity
bun run core/lineage_auditor.ts --verify-chain
```

### Step 2: Staging Deployment (15 min)

1. Deploy to staging environment with feature flag `GUARDRAILS_ENABLED=false`
2. Verify application starts and health check passes
3. Run synthetic test suite against staging
4. Monitor for 5 minutes for any startup errors
5. Enable guardrails gradually (10% of traffic)

```bash
# Deploy code
git push origin claude/jarvis-0HKxO
./scripts/deploy-to-staging.sh

# Verify staging health
curl http://staging.internal/health
# Expected response: {status: "healthy", metrics: {...}}

# Enable 10% traffic
GUARDRAILS_ENABLED=true GUARDRAIL_SAMPLE_RATE=0.1 \
  ./scripts/update-config.sh
```

### Step 3: Monitoring Phase (30 min)

Monitor these metrics on staging:
- API latency (p50, p95, p99) — should not increase
- Error rates — should remain baseline
- Quarantine rate — should be 10-30% for normal traffic
- Learning signal generation rate — should be 1-5 per request
- Health status — should remain "healthy"

```bash
# Watch metrics in real-time
watch -n 1 'curl -s http://staging.internal/metrics | jq .'

# Check for errors in logs
tail -f logs/guardrail.log | grep -i error
```

### Step 4: Staged Rollout to Production (2 hours)

Deploy with graduated traffic:

| Phase | Duration | Traffic % | Action |
|-------|----------|-----------|--------|
| Phase 1 | 15 min | 10% | Canary: monitor metrics closely |
| Phase 2 | 30 min | 25% | If phase 1 stable, increase traffic |
| Phase 3 | 30 min | 50% | If phase 2 stable, increase traffic |
| Phase 4 | 45 min | 100% | Full rollout to all users |

```bash
# Phase 1: 10% traffic
GUARDRAILS_ENABLED=true GUARDRAIL_SAMPLE_RATE=0.1 \
  ./scripts/deploy-to-production.sh

# Monitor for 15 minutes
sleep 900 && ./scripts/check-metrics.sh

# Phase 2: 25% traffic
GUARDRAILS_ENABLED=true GUARDRAIL_SAMPLE_RATE=0.25 \
  ./scripts/update-production-config.sh

# Continue monitoring and progression...
```

### Step 5: Post-Deployment Validation (30 min)

After full rollout:

1. Verify all API latencies unchanged
2. Verify quarantine rates within expected range (10-30%)
3. Verify learning signals being generated
4. Verify lineage records being created
5. Run full E2E test suite
6. Verify health endpoint reports "healthy"

```bash
# Run full validation
bun run scripts/post-deployment-validation.ts

# Expected output:
# ✓ API latency impact: <1ms (acceptable)
# ✓ Quarantine rate: 18% (within range)
# ✓ Learning signals: 1200/hour (healthy)
# ✓ Lineage integrity: 100% (all records valid)
# ✓ System status: HEALTHY
```

---

## Monitoring and Alerting

### Key Metrics to Monitor

**Performance Metrics**
- `api.latency.ms` (p50, p95, p99) — should not increase
- `guardrail.gate.latency.ms` — should be <0.05ms average
- `guardrail.throughput.outputs_per_sec` — should be >1000

**Correctness Metrics**
- `guardrail.acceptance_rate` (%) — expected 60-80%
- `guardrail.quarantine_rate` (%) — expected 20-40%
- `guardrail.rubric_score.avg` — should be >0.55
- `guardrail.rubric_score.min` — should be >0.40 (avoid cascades)

**Safety Metrics**
- `guardrail.anomalies.count` — should be <5 per hour
- `guardrail.verifier_disagreement_rate` (%) — should be <20%
- `guardrail.dangerous_content_detected.count` — trending/incident response
- `guardrail.proposal_approval_rate` (%) — should be 40-70%

**System Metrics**
- `guardrail.lineage.records_per_minute` — should be >100
- `guardrail.health.status` — should be "healthy"
- `guardrail.health.alerts.count` — should be 0 (or low)
- `guardrail.memory.mb` — should be <100MB

### Alert Rules

**Critical Alerts** (page on-call)
```yaml
# Acceptance rate too low — system failing
- alert: GuardrailAcceptanceRateCritical
  condition: guardrail.acceptance_rate < 10
  threshold: 1 minute
  action: page on-call-security

# Acceptance rate too high — guardrails too permissive
- alert: GuardrailAcceptanceRateTooHigh
  condition: guardrail.acceptance_rate > 95
  threshold: 5 minutes
  action: page on-call-security

# Rubric score collapsed — quality degradation
- alert: GuardrailRubricScoreLow
  condition: guardrail.rubric_score.avg < 0.45
  threshold: 5 minutes
  action: page on-call-security

# Anomalies spiking — possible attack
- alert: GuardrailAnomalySurge
  condition: guardrail.anomalies.count > 10 (per hour)
  threshold: 10 minutes
  action: page on-call-security

# Lineage chain broken — data integrity issue
- alert: GuardrailLineageBroken
  condition: guardrail.lineage.chain_valid == false
  threshold: immediate
  action: page on-call-security + escalate to compliance
```

**Warning Alerts** (email + Slack)
```yaml
# Proposal approval rate low — cross-verifier too strict
- alert: GuardrailProposalApprovalsLow
  condition: guardrail.proposal_approval_rate < 30
  threshold: 30 minutes
  action: notify #ops-guardrails

# Verifier disagreement high — calibration drift
- alert: GuardrailVerifierDisagreement
  condition: guardrail.verifier_disagreement_rate > 25
  threshold: 30 minutes
  action: notify #ops-guardrails

# Health degraded — monitor for escalation
- alert: GuardrailHealthDegraded
  condition: guardrail.health.status == "degraded"
  threshold: 15 minutes
  action: notify #ops-guardrails
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'guardrail'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

---

## Health Checks

### Health Endpoint

**URL**: `GET /health`

**Response (healthy)**:
```json
{
  "status": "healthy",
  "timestamp": 1719532800000,
  "guardrail_status": {
    "status": "healthy",
    "metrics": {
      "acceptanceRate": 75,
      "proposalApprovalRate": 60,
      "avgRubricScore": 0.72,
      "truthConfidence": 0.81,
      "anomalyCount": 2,
      "disagreementRate": 15
    },
    "alerts": []
  },
  "lineage_status": {
    "chain_valid": true,
    "total_records": 15234,
    "oldest_record": "2026-06-26T10:00:00Z",
    "newest_record": "2026-06-27T14:30:00Z"
  }
}
```

**Response (degraded)**:
```json
{
  "status": "degraded",
  "timestamp": 1719532800000,
  "guardrail_status": {
    "status": "degraded",
    "metrics": {...},
    "alerts": [
      {
        "level": "warning",
        "message": "Acceptance rate is 22%. Monitor for escalation."
      }
    ]
  }
}
```

**Response (critical)**:
```json
{
  "status": "critical",
  "timestamp": 1719532800000,
  "guardrail_status": {
    "status": "critical",
    "metrics": {...},
    "alerts": [
      {
        "level": "critical",
        "message": "Acceptance rate is 5%. Almost all outputs are being quarantined."
      }
    ]
  },
  "action_required": "Page on-call security lead immediately"
}
```

### Liveness Probe

**URL**: `GET /live`

**Response**: `{"status": "alive"}`

Used by Kubernetes/container orchestration to detect process death.

### Readiness Probe

**URL**: `GET /ready`

**Response**: `{"ready": true}`

Used by load balancers to detect if guardrail system is ready to handle traffic.

---

## Rollback Procedures

### Immediate Rollback (Emergency)

If critical alert fires:

```bash
# 1. Disable guardrails (traffic goes to baseline)
GUARDRAILS_ENABLED=false ./scripts/update-production-config.sh

# 2. Verify traffic restored to normal
curl http://prod.internal/health

# 3. Page on-call to investigate
./scripts/page-on-call.sh "GUARDRAIL_EMERGENCY_ROLLBACK"

# 4. Capture state for post-mortem
./scripts/export-guardrail-state.sh > guardrail_state_prerollback.json
```

### Graceful Rollback (Staged)

If non-critical degradation detected:

```bash
# 1. Reduce traffic gradually
GUARDRAILS_ENABLED=true GUARDRAIL_SAMPLE_RATE=0.5 \
  ./scripts/update-production-config.sh

# 2. Monitor for 10 minutes
watch -n 5 'curl -s http://prod.internal/metrics | jq .'

# 3. Reduce further if needed
GUARDRAILS_ENABLED=true GUARDRAIL_SAMPLE_RATE=0.1 \
  ./scripts/update-production-config.sh

# 4. Disable completely if issues persist
GUARDRAILS_ENABLED=false ./scripts/update-production-config.sh
```

### Root Cause Analysis Post-Rollback

```bash
# Export logs and metrics
./scripts/export-incident-data.sh > incident_$(date +%s).json

# Check lineage for anomalies
bun run core/lineage_auditor.ts --export-recent 1000 > lineage_dump.jsonl

# Review guardrail health history
./scripts/get-health-history.sh --since 1h

# Generate post-mortem
./scripts/generate-postmortem.sh
```

---

## Operational Runbook

See `OPERATIONAL_RUNBOOK.md` for:
- System health monitoring and alert response
- Lineage audit and chain verification procedures
- Quarantine handling and manual override procedures
- Proposal management and approval workflow
- Emergency escalation procedures
- Compliance and audit export procedures

---

## Troubleshooting

### Problem: Acceptance Rate Too Low (<15%)

**Symptoms**: Most API outputs being quarantined

**Diagnosis**:
```bash
# Check which dimension is failing
curl http://prod.internal/metrics | jq '.guardrail.rubric_score.by_dimension'

# Export recent quarantine reasons
bun run scripts/diagnose-quarantines.ts --limit 100
```

**Solutions**:
1. **If coherence failing**: Output quality issue, not guardrail bug
2. **If safety threshold**: Consider context—is detected content actually dangerous?
3. **If rubric threshold too high**: Adjust `DEFAULT_GUARDRAIL_CONFIG.rubricThreshold` from 0.55 to 0.50 (only with approval)

### Problem: No Learning Signals Generated

**Symptoms**: Learning bridge not producing signals

**Diagnosis**:
```bash
# Check if patterns are detected
tail -f logs/guardrail.log | grep "pattern:"

# Verify learning bridge is running
curl http://prod.internal/debug/learning-bridge
```

**Solutions**:
1. Verify pattern threshold in guardrail_learning_bridge.ts (~0.6)
2. Check if rubric scores are low enough to trigger pattern extraction
3. Restart guardrail subsystem if stalled

### Problem: Lineage Chain Broken

**Symptoms**: Chain verification fails, immutability compromised

**Diagnosis**:
```bash
# Verify chain
bun run core/lineage_auditor.ts --verify-chain

# Export breach point
bun run core/lineage_auditor.ts --find-break
```

**Solutions**:
1. **DO NOT** attempt to fix—chain is immutable by design
2. Escalate to compliance team immediately
3. Begin incident investigation
4. Disable guardrails if chain integrity is critical

### Problem: API Latency Increased

**Symptoms**: Response time degraded after guardrail deployment

**Diagnosis**:
```bash
# Measure guardrail overhead
curl http://prod.internal/metrics | jq '.guardrail.gate.latency'

# Should be <0.05ms average
```

**Solutions**:
1. Verify this is guardrail overhead, not other system issue
2. Check if memoization is working (cache hit rate should be >50%)
3. If overhead is real (<1ms), it's acceptable
4. Rollback if latency impact >5ms (unlikely with current system)

---

## Contact and Escalation

**On-Call Security**: `@on-call-security` in Slack

**Compliance Team**: `compliance-team@anthropic.com`

**Guardrail Ops Channel**: `#ops-guardrails` in Slack

**Emergency Escalation**: Page on-call security via PagerDuty

---

**Document Version**: 1.0  
**Last Updated**: 2026-06-27  
**Maintained By**: Security Engineering & DevOps

