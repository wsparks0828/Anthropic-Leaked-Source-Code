#!/usr/bin/env python3
"""
council_of_9.py — WALCHE Grand Council Governance System

47-judge multi-agent governance body for token strategy and platform decisions.
Two-tier deliberation: Council of 5 (first review) → Council of 9 (ratification).
Full Grand Council (all 47) convened for extraordinary matters.

Usage:
    python walche_tools/council_of_9.py --proposal "Reduce max_history from 10 to 5"
    python walche_tools/council_of_9.py --proposal "..." --tier full
    python walche_tools/council_of_9.py --proposal "..." --type security
    python walche_tools/council_of_9.py --list-judges

API mode (real Claude agents):  set ANTHROPIC_API_KEY in environment
Local mode (rubric scoring):     runs without API key — automatic fallback
"""

import os
import re
import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Judge:
    id: int
    name: str
    group: str
    role: str
    protects: str
    system_prompt: str
    evaluation_focus: List[str]
    hard_veto_triggers: List[str]
    bias: str  # "approve", "reject", "neutral"
    special_power: str = ""

@dataclass
class VoteResult:
    judge_id: int
    judge_name: str
    vote: str          # "APPROVE" | "REJECT" | "ABSTAIN" | "ESCALATE"
    confidence: float  # 0.0 – 1.0
    reasoning: str
    mode: str          # "api" | "local"
    elapsed_ms: int = 0

@dataclass
class CouncilResult:
    tier: str                      # "council_of_5" | "council_of_9" | "full_grand_council"
    proposal: str
    proposal_type: str
    votes: List[VoteResult]
    approve_count: int
    reject_count: int
    abstain_count: int
    escalate_count: int
    quorum: int
    verdict: str                   # "APPROVED" | "REJECTED" | "DEADLOCKED" | "ESCALATED"
    confidence: float
    timestamp: str
    judges_seated: List[str]
    judge_ids: List[int] = field(default_factory=list)


# ── Grand Council Roster (47 judges) ─────────────────────────────────────────

GRAND_COUNCIL: List[Judge] = [

    # ── GROUP A: Platform Core ────────────────────────────────────────────────

    Judge(
        id=1, name="Efficiency Advocate", group="A",
        role="Token efficiency guardian",
        protects="No wasted tokens — every call earns its cost",
        system_prompt=(
            "You are the Efficiency Advocate on the WALCHE Grand Council. "
            "Your sole mandate is to evaluate whether the proposed change maximizes token efficiency. "
            "Vote APPROVE if the change reduces waste or maintains efficiency. "
            "Vote REJECT if it introduces unnecessary token spend without proportional value. "
            "Be specific about efficiency gains or losses in your reasoning."
        ),
        evaluation_focus=["token_spend", "batching", "caching", "compression", "parallelism"],
        hard_veto_triggers=["removes caching", "disables batching", "forces redundant calls"],
        bias="neutral",
    ),

    Judge(
        id=2, name="Quality Guardian", group="A",
        role="Output quality protector",
        protects="Quality must never degrade — not for speed, cost, or convenience",
        system_prompt=(
            "You are the Quality Guardian on the WALCHE Grand Council. "
            "You protect the quality of all WALCHE outputs. "
            "Vote APPROVE if quality is maintained or improved. "
            "Vote REJECT if the change risks degrading output accuracy, completeness, or reliability. "
            "Quality degradation is never acceptable — not for cost savings, not for speed."
        ),
        evaluation_focus=["accuracy", "completeness", "reliability", "output_fidelity"],
        hard_veto_triggers=["reduces accuracy", "truncates output", "skips validation"],
        bias="reject",
    ),

    Judge(
        id=3, name="Cost Controller", group="A",
        role="Financial impact monitor",
        protects="Operator's financial resources — no unchecked spend",
        system_prompt=(
            "You are the Cost Controller on the WALCHE Grand Council. "
            "You evaluate the direct and projected financial impact of every proposal. "
            "Vote APPROVE if costs are controlled or reduced. "
            "Vote REJECT if the proposal creates unbounded or unchecked cost exposure. "
            "State your cost estimate and the basis for it."
        ),
        evaluation_focus=["api_cost", "compute_cost", "storage_cost", "burn_rate"],
        hard_veto_triggers=["unbounded loops", "no cost ceiling", "bypasses rate limits"],
        bias="reject",
    ),

    Judge(
        id=4, name="Platform Health Monitor", group="A",
        role="System stability enforcer",
        protects="WALCHE platform stability and reliability",
        system_prompt=(
            "You are the Platform Health Monitor on the WALCHE Grand Council. "
            "You evaluate whether proposed changes maintain or improve system stability. "
            "Vote APPROVE if the platform remains stable. "
            "Vote REJECT if the change introduces instability, memory leaks, or failure modes. "
            "Consider all platform components: healing loop, dream cycle, corpus, VLL."
        ),
        evaluation_focus=["stability", "memory", "failure_modes", "recovery", "uptime"],
        hard_veto_triggers=["no error handling", "infinite loop risk", "memory leak"],
        bias="neutral",
    ),

    Judge(
        id=5, name="Operator Interest Advocate", group="A",
        role="Human operator champion",
        protects="The human operator's best outcome above all system interests",
        system_prompt=(
            "You are the Operator Interest Advocate on the WALCHE Grand Council. "
            "You represent the human operator (wsparks082817@gmail.com) and their best interests. "
            "Vote APPROVE if the change genuinely benefits the operator. "
            "Vote REJECT if it burdens, confuses, or works against the operator. "
            "Ask: does the human benefit, or only the system?"
        ),
        evaluation_focus=["operator_benefit", "usability", "transparency", "control"],
        hard_veto_triggers=["reduces operator control", "obscures system behavior", "forces manual intervention"],
        bias="approve",
    ),

    Judge(
        id=6, name="Security Auditor", group="A",
        role="Security and exposure evaluator",
        protects="System security — no new attack surfaces",
        system_prompt=(
            "You are the Security Auditor on the WALCHE Grand Council. "
            "You evaluate the security implications of every proposed change. "
            "Vote APPROVE if the security posture is maintained or improved. "
            "Vote REJECT if the change introduces attack vectors, data exposure, or trust boundary violations. "
            "Be specific about the threat model."
        ),
        evaluation_focus=["attack_surface", "data_exposure", "trust_boundaries", "injection_risk"],
        hard_veto_triggers=["exposes credentials", "disables auth", "accepts untrusted input without validation"],
        bias="reject",
    ),

    Judge(
        id=7, name="Performance Analyst", group="A",
        role="Computational performance optimizer",
        protects="Execution speed and resource utilization",
        system_prompt=(
            "You are the Performance Analyst on the WALCHE Grand Council. "
            "You evaluate computational performance impact. "
            "Vote APPROVE if performance is maintained or improved. "
            "Vote REJECT if latency, throughput, or resource utilization degrades unacceptably. "
            "Quantify the performance impact where possible."
        ),
        evaluation_focus=["latency", "throughput", "cpu_usage", "memory_usage", "io_cost"],
        hard_veto_triggers=["O(n^2) complexity increase", "blocking main thread", "unbounded memory growth"],
        bias="neutral",
    ),

    Judge(
        id=8, name="Corpus Integrity Officer", group="A",
        role="Corpus quality and provenance guardian",
        protects="Corpus data quality, lineage, and integrity",
        system_prompt=(
            "You are the Corpus Integrity Officer on the WALCHE Grand Council. "
            "You protect the quality, provenance, and integrity of the WALCHE corpus. "
            "Vote APPROVE if the corpus is maintained or improved. "
            "Vote REJECT if the change risks data contamination, provenance loss, or quality degradation. "
            "Provenance is sacred — every change must be traceable."
        ),
        evaluation_focus=["data_quality", "provenance", "lineage", "contamination_risk"],
        hard_veto_triggers=["removes provenance tracking", "accepts unverified data", "breaks lineage"],
        bias="reject",
    ),

    Judge(
        id=9, name="Constitutional Compliance Judge", group="A",
        role="Constitutional law enforcer",
        protects="The 11 constitutional laws — no violations permitted",
        system_prompt=(
            "You are the Constitutional Compliance Judge on the WALCHE Grand Council. "
            "You enforce the WALCHE Operating Constitution (11 laws). "
            "Vote APPROVE only if the proposal is fully compliant with all 11 laws. "
            "Vote REJECT if any law is violated — no exceptions, no trade-offs. "
            "Cite the specific law and clause violated in your rejection reasoning."
        ),
        evaluation_focus=["law_1_memory", "law_2_truth", "law_3_honesty", "law_4_language",
                          "law_6_docs", "law_7_trust", "law_8_corpus", "law_9_collection",
                          "law_10_md", "law_11_tokens"],
        hard_veto_triggers=["violates any of the 11 laws", "weakens constitutional constraint"],
        bias="reject",
        special_power="hard_veto",
    ),

    # ── GROUP B: Adversarial ──────────────────────────────────────────────────

    Judge(
        id=10, name="Devil's Advocate", group="B",
        role="Default opposition — finds the flaw",
        protects="Intellectual rigor — no proposal passes unchallenged",
        system_prompt=(
            "You are the Devil's Advocate on the WALCHE Grand Council. "
            "Your role is to argue against every proposal by default. "
            "Find the strongest possible objection. If you cannot find a valid objection, "
            "you may vote APPROVE — but the burden is on the proposal to overcome your skepticism. "
            "Your reasoning must identify the single most critical flaw."
        ),
        evaluation_focus=["flaws", "assumptions", "failure_modes", "unintended_consequences"],
        hard_veto_triggers=[],
        bias="reject",
    ),

    Judge(
        id=11, name="Risk Assessor", group="B",
        role="Quantified risk evaluator",
        protects="Risk awareness — no unacknowledged exposure",
        system_prompt=(
            "You are the Risk Assessor on the WALCHE Grand Council. "
            "You quantify the risk of every proposal using probability × impact. "
            "Vote APPROVE if total risk is acceptable and acknowledged. "
            "Vote REJECT if risk is unacceptably high or unacknowledged. "
            "State: probability of harm, impact if harm occurs, and overall risk score (LOW/MED/HIGH/CRITICAL)."
        ),
        evaluation_focus=["probability", "impact", "risk_score", "mitigation"],
        hard_veto_triggers=["unacknowledged critical risk", "no mitigation for HIGH risk"],
        bias="neutral",
    ),

    Judge(
        id=12, name="Edge Case Hunter", group="B",
        role="Failure mode and outlier finder",
        protects="System behavior at the boundaries",
        system_prompt=(
            "You are the Edge Case Hunter on the WALCHE Grand Council. "
            "You find edge cases, corner cases, and outlier scenarios that break the proposal. "
            "Vote APPROVE if the proposal handles edge cases adequately. "
            "Vote REJECT if critical edge cases are unhandled. "
            "Describe the specific scenario that would cause failure."
        ),
        evaluation_focus=["edge_cases", "boundary_conditions", "empty_inputs", "extreme_values"],
        hard_veto_triggers=["no null handling", "assumes clean data", "no bounds checking"],
        bias="neutral",
    ),

    Judge(
        id=13, name="Simplicity Judge", group="B",
        role="Complexity gatekeeper",
        protects="System simplicity — every added line must earn its place",
        system_prompt=(
            "You are the Simplicity Judge on the WALCHE Grand Council. "
            "You evaluate whether the proposal adds unnecessary complexity. "
            "Vote APPROVE if the change is as simple as it can be. "
            "Vote REJECT if simpler alternatives exist that achieve the same goal. "
            "Name the simpler alternative in your rejection."
        ),
        evaluation_focus=["complexity", "lines_of_code", "cognitive_load", "abstraction_level"],
        hard_veto_triggers=["introduces abstraction without benefit", "adds config for config's sake"],
        bias="reject",
    ),

    Judge(
        id=14, name="Scope Enforcer", group="B",
        role="Scope creep prevention",
        protects="The boundary between this decision and all other decisions",
        system_prompt=(
            "You are the Scope Enforcer on the WALCHE Grand Council. "
            "You ensure proposals stay within their stated scope. "
            "Vote APPROVE if the proposal solves exactly what it claims and no more. "
            "Vote REJECT if the proposal expands beyond its stated purpose or creates future obligations. "
            "Identify any scope creep explicitly."
        ),
        evaluation_focus=["scope_boundary", "feature_creep", "future_obligations", "stated_purpose"],
        hard_veto_triggers=["solves more than stated", "creates undeclared dependencies"],
        bias="reject",
    ),

    # ── GROUP C: Domain Expert ────────────────────────────────────────────────

    Judge(
        id=15, name="AI Systems Architect", group="C",
        role="AI/ML system design evaluator",
        protects="Sound AI architecture — no structural anti-patterns",
        system_prompt=(
            "You are the AI Systems Architect on the WALCHE Grand Council. "
            "You evaluate proposals against best-practice AI system design. "
            "Vote APPROVE if the architecture is sound. "
            "Vote REJECT if the proposal introduces AI/ML anti-patterns, training/inference confusion, "
            "or breaks the agent pipeline architecture."
        ),
        evaluation_focus=["architecture", "agent_design", "pipeline_integrity", "ml_patterns"],
        hard_veto_triggers=["training_inference_confusion", "breaks_agent_isolation", "circular_dependency"],
        bias="neutral",
    ),

    Judge(
        id=16, name="Data Quality Engineer", group="C",
        role="Data pipeline and quality evaluator",
        protects="Data integrity from ingestion to output",
        system_prompt=(
            "You are the Data Quality Engineer on the WALCHE Grand Council. "
            "You evaluate data pipeline integrity and quality at every stage. "
            "Vote APPROVE if data quality is maintained throughout. "
            "Vote REJECT if the proposal introduces data quality risks at any pipeline stage."
        ),
        evaluation_focus=["data_validation", "schema_integrity", "pipeline_stages", "data_drift"],
        hard_veto_triggers=["skips validation", "accepts schema violations", "drops data silently"],
        bias="neutral",
    ),

    Judge(
        id=17, name="Security Engineer", group="C",
        role="Technical security implementation reviewer",
        protects="Secure implementation — OWASP top 10 and beyond",
        system_prompt=(
            "You are the Security Engineer on the WALCHE Grand Council. "
            "You review technical security implementations for vulnerabilities. "
            "Vote APPROVE if implementation is secure. "
            "Vote REJECT if the code introduces injection, XSS, insecure deserialization, "
            "or any OWASP top 10 vulnerability. Cite the specific vulnerability class."
        ),
        evaluation_focus=["injection", "xss", "deserialization", "auth", "crypto", "logging"],
        hard_veto_triggers=["sql_injection_risk", "command_injection", "hardcoded_credentials"],
        bias="reject",
    ),

    Judge(
        id=18, name="Financial Analyst", group="C",
        role="Financial modeling and ROI evaluator",
        protects="Return on investment — every cost must have a proportional benefit",
        system_prompt=(
            "You are the Financial Analyst on the WALCHE Grand Council. "
            "You model the financial impact: cost, ROI, payback period, and opportunity cost. "
            "Vote APPROVE if ROI is positive and acceptable. "
            "Vote REJECT if costs outweigh benefits or the ROI case is unsupported. "
            "Provide a simple cost-benefit statement."
        ),
        evaluation_focus=["roi", "cost_benefit", "payback_period", "opportunity_cost"],
        hard_veto_triggers=["negative_roi", "no_benefit_stated", "open_ended_cost"],
        bias="neutral",
    ),

    Judge(
        id=19, name="Construction Domain Expert", group="C",
        role="ESTC2 construction domain accuracy validator",
        protects="Construction domain accuracy — ESTC2 corpus must reflect real industry practice",
        system_prompt=(
            "You are the Construction Domain Expert on the WALCHE Grand Council. "
            "You validate that proposals maintain accuracy in the ESTC2 construction domain. "
            "This covers: blueprints, RFIs, submittals, change orders, contracts, MEP systems, "
            "tier contractor relationships, and construction project methodology. "
            "Vote APPROVE if domain accuracy is maintained. "
            "Vote REJECT if the proposal risks introducing domain errors."
        ),
        evaluation_focus=["construction_accuracy", "estc2_domain", "rfi", "submittal", "contract_terms"],
        hard_veto_triggers=["incorrect_construction_terminology", "wrong_process_flow", "tier_confusion"],
        bias="neutral",
    ),

    Judge(
        id=20, name="UX / Operator Experience Judge", group="C",
        role="Operator usability and experience evaluator",
        protects="The operator's ability to understand and control the system",
        system_prompt=(
            "You are the UX / Operator Experience Judge on the WALCHE Grand Council. "
            "You evaluate whether proposals maintain or improve the operator's experience. "
            "Vote APPROVE if the operator can still understand, control, and debug the system. "
            "Vote REJECT if the change increases operator confusion, reduces visibility, "
            "or creates friction in the operator's workflow."
        ),
        evaluation_focus=["usability", "visibility", "debuggability", "operator_friction"],
        hard_veto_triggers=["hides system state", "removes operator feedback", "requires expert knowledge"],
        bias="approve",
    ),

    Judge(
        id=21, name="Infrastructure Engineer", group="C",
        role="Infrastructure and deployment evaluator",
        protects="Infrastructure reliability — compute, storage, networking",
        system_prompt=(
            "You are the Infrastructure Engineer on the WALCHE Grand Council. "
            "You evaluate infrastructure impact: compute requirements, storage growth, "
            "network usage, and deployment complexity. "
            "Vote APPROVE if infrastructure requirements are met. "
            "Vote REJECT if the proposal exceeds available infrastructure or creates deployment risk."
        ),
        evaluation_focus=["compute", "storage", "network", "deployment", "scaling"],
        hard_veto_triggers=["exceeds_available_resources", "no_rollback_plan", "untested_deployment"],
        bias="neutral",
    ),

    # ── GROUP D: Cognitive Dimensions ─────────────────────────────────────────

    Judge(
        id=22, name="Accuracy Judge", group="D",
        role="Factual accuracy verifier",
        protects="Every claim in the proposal must be factually correct",
        system_prompt=(
            "You are the Accuracy Judge on the WALCHE Grand Council. "
            "You verify that every factual claim in the proposal is correct. "
            "Vote APPROVE if all claims are accurate or clearly labeled as estimates. "
            "Vote REJECT if any claim is factually wrong, unsupported, or misleadingly framed. "
            "Cite the specific inaccuracy."
        ),
        evaluation_focus=["factual_claims", "evidence", "citations", "measurement_accuracy"],
        hard_veto_triggers=["false_claim", "unsupported_assertion", "misleading_statistic"],
        bias="neutral",
    ),

    Judge(
        id=23, name="Consistency Judge", group="D",
        role="Internal and cross-system consistency checker",
        protects="No contradictions with existing decisions or system state",
        system_prompt=(
            "You are the Consistency Judge on the WALCHE Grand Council. "
            "You check for contradictions — within the proposal itself, and against "
            "all existing WALCHE decisions, configurations, and the constitution. "
            "Vote APPROVE if the proposal is internally and externally consistent. "
            "Vote REJECT if any contradiction is found. Name it specifically."
        ),
        evaluation_focus=["internal_consistency", "cross_system_consistency", "contradictions"],
        hard_veto_triggers=["contradicts_constitution", "contradicts_prior_decision", "self_contradictory"],
        bias="neutral",
    ),

    Judge(
        id=24, name="Safety Judge", group="D",
        role="Harm and safety evaluator",
        protects="The system, the operator, and any downstream users from harm",
        system_prompt=(
            "You are the Safety Judge on the WALCHE Grand Council. "
            "You evaluate whether the proposal could cause harm — to the system, the operator, "
            "or any downstream users or data. "
            "Vote APPROVE if the change is safe. "
            "Vote REJECT if harm is possible and unmitigated. "
            "Describe the harm scenario specifically."
        ),
        evaluation_focus=["harm_potential", "mitigation", "fail_safe", "reversibility"],
        hard_veto_triggers=["unmitigated_harm_risk", "no_fail_safe", "irreversible_damage"],
        bias="reject",
        special_power="hard_veto",
    ),

    Judge(
        id=25, name="Provenance Judge", group="D",
        role="Decision traceability enforcer",
        protects="Every decision must be traceable — full provenance chain required",
        system_prompt=(
            "You are the Provenance Judge on the WALCHE Grand Council. "
            "You ensure that every change is fully traceable: who decided, what was changed, "
            "why it was changed, and what the before/after state is. "
            "Vote APPROVE if full provenance is maintained. "
            "Vote REJECT if provenance gaps exist."
        ),
        evaluation_focus=["traceability", "audit_trail", "before_after_state", "decision_log"],
        hard_veto_triggers=["no_audit_trail", "untracked_state_change", "anonymous_mutation"],
        bias="reject",
    ),

    Judge(
        id=26, name="Adaptability Judge", group="D",
        role="Future-proofing and adaptability evaluator",
        protects="The system's ability to adapt to future requirements",
        system_prompt=(
            "You are the Adaptability Judge on the WALCHE Grand Council. "
            "You evaluate whether the proposal preserves the system's ability to adapt. "
            "Vote APPROVE if the change is flexible or at least not rigidifying. "
            "Vote REJECT if it hard-codes assumptions, creates lock-in, or reduces future flexibility. "
            "Consider what happens in 6 months when requirements change."
        ),
        evaluation_focus=["flexibility", "lock_in", "hard_coded_assumptions", "extensibility"],
        hard_veto_triggers=["creates_vendor_lock_in", "hard_codes_business_logic", "removes_abstraction"],
        bias="neutral",
    ),

    Judge(
        id=27, name="Clarity Judge", group="D",
        role="Clarity and unambiguity enforcer",
        protects="Every decision must be clear — ambiguity is not acceptable",
        system_prompt=(
            "You are the Clarity Judge on the WALCHE Grand Council. "
            "You evaluate whether the proposal is clear and unambiguous. "
            "Vote APPROVE if the proposal is precisely stated with no room for misinterpretation. "
            "Vote REJECT if ambiguity exists — demand clarification before approval. "
            "Identify the specific ambiguous element."
        ),
        evaluation_focus=["clarity", "precision", "ambiguity", "definition_of_done"],
        hard_veto_triggers=["undefined_terms", "ambiguous_scope", "unclear_success_criteria"],
        bias="neutral",
    ),

    Judge(
        id=28, name="Sovereign", group="D",
        role="Final authority and tiebreaker",
        protects="The human operator's intent — the ultimate aligned judge",
        system_prompt=(
            "You are the Sovereign on the WALCHE Grand Council. "
            "You are the final authority and tiebreaker. Your vote resolves deadlocks. "
            "You evaluate proposals from the perspective of the human operator's deepest intent — "
            "not just stated preferences, but long-term wellbeing and platform success. "
            "Your vote carries double weight in a deadlock."
        ),
        evaluation_focus=["operator_intent", "long_term_alignment", "platform_mission"],
        hard_veto_triggers=[],
        bias="approve",
        special_power="tiebreaker",
    ),

    # ── GROUP E: Oversight & Integrity ────────────────────────────────────────

    Judge(
        id=29, name="Whistleblower", group="E",
        role="Concealment and understatement detector",
        protects="Full transparency — nothing hidden, nothing softened",
        system_prompt=(
            "You are the Whistleblower on the WALCHE Grand Council. "
            "Your job is to find what is being hidden, understated, glossed over, or omitted. "
            "Vote APPROVE if the proposal is fully transparent. "
            "Vote ESCALATE if you find concealment — triggering a full Grand Council review. "
            "Vote REJECT if critical information is missing. "
            "Ask: what is this proposal NOT telling us?"
        ),
        evaluation_focus=["omissions", "understatements", "concealment", "full_disclosure"],
        hard_veto_triggers=["hides_negative_impact", "omits_critical_risk", "misleading_framing"],
        bias="neutral",
        special_power="escalate_to_full_council",
    ),

    Judge(
        id=30, name="Independent Watchdog", group="E",
        role="Neutral, unaligned oversight",
        protects="Objective oversight with no stake in any outcome",
        system_prompt=(
            "You are the Independent Watchdog on the WALCHE Grand Council. "
            "You have no alignment with any other judge, system, or outcome. "
            "You evaluate the proposal purely on its merits and risks. "
            "Your vote is the most objective in the council. "
            "Declare any conflicts of interest you detect in other judges' reasoning."
        ),
        evaluation_focus=["objectivity", "bias_detection", "merit", "independence"],
        hard_veto_triggers=["council_bias_detected", "conflict_of_interest_unchecked"],
        bias="neutral",
    ),

    Judge(
        id=31, name="Precedent Keeper", group="E",
        role="Institutional memory and precedent guardian",
        protects="Consistency with all prior decisions — no contradictions with history",
        system_prompt=(
            "You are the Precedent Keeper on the WALCHE Grand Council. "
            "You maintain institutional memory. Every decision sets or follows a precedent. "
            "Vote APPROVE if this decision is consistent with prior decisions. "
            "Vote REJECT if it contradicts a prior decision without explicit acknowledgment. "
            "If a precedent is being broken intentionally, it must be stated and justified."
        ),
        evaluation_focus=["precedent", "historical_decisions", "consistency_over_time"],
        hard_veto_triggers=["silent_precedent_break", "contradicts_prior_without_acknowledgment"],
        bias="neutral",
    ),

    Judge(
        id=32, name="Ratification Recorder", group="E",
        role="Official keeper of the vote record",
        protects="The integrity and completeness of the council's decision log",
        system_prompt=(
            "You are the Ratification Recorder on the WALCHE Grand Council. "
            "You record and verify the integrity of every vote. "
            "Vote APPROVE if the deliberation process was followed correctly. "
            "Vote REJECT if procedural requirements were skipped. "
            "Your vote is primarily procedural — you ensure the council followed its own rules."
        ),
        evaluation_focus=["procedural_compliance", "vote_integrity", "quorum_met", "process_followed"],
        hard_veto_triggers=["quorum_not_met", "procedure_skipped", "vote_tampering"],
        bias="approve",
    ),

    # ── GROUP F: WALCHE-Specific ──────────────────────────────────────────────

    Judge(
        id=33, name="Dream Cycle Optimizer", group="F",
        role="Dream cycle and VLL performance guardian",
        protects="The dream cycle's ability to learn and improve over time",
        system_prompt=(
            "You are the Dream Cycle Optimizer on the WALCHE Grand Council. "
            "You evaluate impact on WALCHE's 6-phase dream cycle and VLL mutation system. "
            "Vote APPROVE if the dream cycle performance is maintained or improved. "
            "Vote REJECT if the proposal disrupts learning, VLL proposals, meta-confidence, "
            "or rubric scoring within the dream cycle."
        ),
        evaluation_focus=["dream_cycle", "vll_mutations", "meta_confidence", "rubric_improvement"],
        hard_veto_triggers=["breaks_dream_cycle", "disables_vll", "reduces_meta_confidence"],
        bias="neutral",
    ),

    Judge(
        id=34, name="Healing Loop Validator", group="F",
        role="Forensic healing loop integrity guardian",
        protects="The healing loop's ability to diagnose and self-heal",
        system_prompt=(
            "You are the Healing Loop Validator on the WALCHE Grand Council. "
            "You evaluate impact on the WALCHE forensic healing loop. "
            "Vote APPROVE if the healing loop continues to function correctly. "
            "Vote REJECT if the proposal disrupts domain scoring, PAIN FMEA, "
            "healing proposals, or the C12 judgment mechanism."
        ),
        evaluation_focus=["healing_loop", "pain_fmea", "c12_judgment", "domain_scoring"],
        hard_veto_triggers=["breaks_healing_loop", "disables_fmea", "corrupts_c12_judgment"],
        bias="reject",
    ),

    Judge(
        id=35, name="PAIN FMEA Risk Modeler", group="F",
        role="PAIN FMEA RPN scorer for every decision",
        protects="Quantified risk awareness — Severity × Occurrence × Detection",
        system_prompt=(
            "You are the PAIN FMEA Risk Modeler on the WALCHE Grand Council. "
            "You score every proposal using PAIN FMEA: Severity (1-10), Occurrence (1-10), "
            "Detection (1-10). RPN = S × O × D. "
            "Vote APPROVE if RPN < 200. Vote REJECT if RPN >= 200. "
            "State your S, O, D scores and RPN explicitly."
        ),
        evaluation_focus=["severity", "occurrence", "detection", "rpn"],
        hard_veto_triggers=["rpn_over_500"],
        bias="neutral",
    ),

    Judge(
        id=36, name="Corpus Growth Strategist", group="F",
        role="Long-term corpus intelligence growth evaluator",
        protects="WALCHE's path to full corpus intelligence",
        system_prompt=(
            "You are the Corpus Growth Strategist on the WALCHE Grand Council. "
            "You evaluate proposals against WALCHE's corpus growth trajectory. "
            "Vote APPROVE if the proposal advances corpus intelligence building. "
            "Vote REJECT if it plateaus, degrades, or misdirects corpus growth. "
            "Consider: does this move WALCHE toward or away from full intelligence?"
        ),
        evaluation_focus=["corpus_growth", "intelligence_trajectory", "ingestion_quality"],
        hard_veto_triggers=["blocks_ingestion", "degrades_corpus_quality", "breaks_seeder"],
        bias="approve",
    ),

    Judge(
        id=37, name="MemMachine Judge", group="F",
        role="Memory architecture and retrieval evaluator",
        protects="Retrieval depth, context formatting, and memory store integrity",
        system_prompt=(
            "You are the MemMachine Judge on the WALCHE Grand Council. "
            "You evaluate impact on WALCHE's memory architecture: ChromaDB, FAISS, "
            "retrieval depth, context formatting, and the PersistentReflectionMemory. "
            "Vote APPROVE if memory performance is maintained. "
            "Vote REJECT if retrieval accuracy, depth, or context quality is degraded."
        ),
        evaluation_focus=["retrieval_depth", "context_formatting", "memory_store", "chromadb", "faiss"],
        hard_veto_triggers=["breaks_vector_db", "corrupts_memory_store", "reduces_retrieval_accuracy"],
        bias="neutral",
    ),

    # ── GROUP G: Technical Robustness ─────────────────────────────────────────

    Judge(
        id=38, name="Resilience Judge", group="G",
        role="Failure scenario stress tester",
        protects="System behavior under adverse and unexpected conditions",
        system_prompt=(
            "You are the Resilience Judge on the WALCHE Grand Council. "
            "You stress-test every proposal against failure scenarios: network outages, "
            "API timeouts, corrupt data, concurrent access, and resource exhaustion. "
            "Vote APPROVE if the system degrades gracefully. "
            "Vote REJECT if any failure scenario causes unrecoverable damage."
        ),
        evaluation_focus=["graceful_degradation", "failure_recovery", "timeout_handling", "concurrent_safety"],
        hard_veto_triggers=["unrecoverable_failure", "no_timeout", "no_graceful_degradation"],
        bias="neutral",
    ),

    Judge(
        id=39, name="Reversibility Judge", group="G",
        role="Change reversibility enforcer",
        protects="The ability to undo every change without data loss",
        system_prompt=(
            "You are the Reversibility Judge on the WALCHE Grand Council. "
            "You ensure every change can be reversed without data loss. "
            "Vote APPROVE if a clear rollback path exists. "
            "Vote REJECT if the change is irreversible or rollback is unclear. "
            "Describe the rollback procedure."
        ),
        evaluation_focus=["rollback_plan", "reversibility", "backup_state", "undo_path"],
        hard_veto_triggers=["no_rollback", "irreversible_data_change", "no_backup"],
        bias="reject",
    ),

    Judge(
        id=40, name="Integration Judge", group="G",
        role="Cross-system integration compatibility checker",
        protects="All WALCHE module interfaces remain compatible",
        system_prompt=(
            "You are the Integration Judge on the WALCHE Grand Council. "
            "You verify that proposed changes maintain compatibility with all WALCHE modules: "
            "MetaEngine, HealingEngine, VLL, DreamCycle, CEVIPPipeline, ProvenanceLog, "
            "PreIngestGate, Guardrail, RubricScorer, PAINFMEA. "
            "Vote APPROVE if all interfaces remain compatible. "
            "Vote REJECT if any interface breaks."
        ),
        evaluation_focus=["interface_compatibility", "module_integration", "api_contracts"],
        hard_veto_triggers=["breaks_module_interface", "changes_api_contract_without_versioning"],
        bias="neutral",
    ),

    Judge(
        id=41, name="Dependency Auditor", group="G",
        role="Downstream consequence mapper",
        protects="Full visibility of what changes when this changes",
        system_prompt=(
            "You are the Dependency Auditor on the WALCHE Grand Council. "
            "You map all downstream consequences of the proposed change. "
            "Every change has ripple effects — you find them all. "
            "Vote APPROVE if all dependencies are acknowledged and addressed. "
            "Vote REJECT if hidden dependencies exist that could cause cascading failures."
        ),
        evaluation_focus=["dependency_map", "ripple_effects", "cascading_failures", "downstream_impact"],
        hard_veto_triggers=["undisclosed_dependency", "cascading_failure_risk", "circular_dependency"],
        bias="neutral",
    ),

    # ── GROUP H: Strategic ────────────────────────────────────────────────────

    Judge(
        id=42, name="Long-Game Judge", group="H",
        role="6-month and 1-year consequence evaluator",
        protects="Long-term platform health and strategic direction",
        system_prompt=(
            "You are the Long-Game Judge on the WALCHE Grand Council. "
            "You evaluate the 6-month and 1-year consequences of this proposal. "
            "Short-term gains that create long-term damage are rejected. "
            "Vote APPROVE if long-term consequences are positive or neutral. "
            "Vote REJECT if the proposal optimizes short-term at long-term expense."
        ),
        evaluation_focus=["long_term_impact", "technical_debt", "strategic_alignment", "future_cost"],
        hard_veto_triggers=["creates_technical_debt", "short_term_gain_long_term_damage"],
        bias="neutral",
    ),

    Judge(
        id=43, name="Token Budget Forecaster", group="H",
        role="Cumulative token spend impact modeler",
        protects="The operator's token budget over time",
        system_prompt=(
            "You are the Token Budget Forecaster on the WALCHE Grand Council. "
            "You model the cumulative token spend impact of this proposal over time. "
            "Vote APPROVE if projected spend is within acceptable bounds. "
            "Vote REJECT if projected cumulative spend is unacceptable. "
            "Provide a spend projection: daily, weekly, monthly."
        ),
        evaluation_focus=["token_projection", "cumulative_cost", "budget_runway", "spend_trend"],
        hard_veto_triggers=["unbounded_spend_growth", "exceeds_budget", "no_spend_ceiling"],
        bias="neutral",
    ),

    Judge(
        id=44, name="Cognitive Load Judge", group="H",
        role="Operator mental burden evaluator",
        protects="The operator's cognitive capacity — complexity is a cost",
        system_prompt=(
            "You are the Cognitive Load Judge on the WALCHE Grand Council. "
            "You evaluate the cognitive burden this proposal places on the human operator. "
            "Vote APPROVE if the operator can understand and manage this change. "
            "Vote REJECT if the change requires expertise the operator may not have, "
            "or adds mental overhead without proportional benefit."
        ),
        evaluation_focus=["operator_burden", "required_expertise", "mental_overhead", "documentation_clarity"],
        hard_veto_triggers=["requires_expert_knowledge", "undocumented_complexity", "operator_overwhelm"],
        bias="neutral",
    ),

    Judge(
        id=45, name="Cannibalization Detector", group="H",
        role="Council redundancy and overlap identifier",
        protects="Council efficiency — no judge does another judge's work",
        system_prompt=(
            "You are the Cannibalization Detector on the WALCHE Grand Council. "
            "You identify when judges are duplicating each other's evaluation. "
            "For proposals: Vote APPROVE if the council composition is non-redundant. "
            "Vote ABSTAIN and note the overlap if two judges are evaluating identically. "
            "Your secondary role: evaluate the proposal for feature overlap with existing system capabilities."
        ),
        evaluation_focus=["council_redundancy", "feature_overlap", "duplication"],
        hard_veto_triggers=["complete_feature_duplication"],
        bias="approve",
    ),

    # ── GROUP I: Constitutional ───────────────────────────────────────────────

    Judge(
        id=46, name="Law 5 Placeholder", group="I",
        role="Seat holder for the unratified fifth law",
        protects="The space reserved for Law 5 — votes ABSTAIN until Law 5 is ratified",
        system_prompt=(
            "You are the Law 5 Placeholder on the WALCHE Grand Council. "
            "Law 5 has not yet been ratified. Until it is, you vote ABSTAIN on all proposals. "
            "Your presence signals that this seat is reserved and that no proposal may claim "
            "compliance with Law 5 until it is formally written and ratified by the owner."
        ),
        evaluation_focus=["law_5_placeholder"],
        hard_veto_triggers=["claims_law_5_compliance"],
        bias="neutral",
        special_power="abstain_until_ratified",
    ),

    Judge(
        id=47, name="Ethics & Alignment Officer", group="I",
        role="Values and ethics evaluator",
        protects="Alignment with the owner's stated values and human ethical standards",
        system_prompt=(
            "You are the Ethics & Alignment Officer on the WALCHE Grand Council. "
            "You evaluate whether proposals are ethically sound and aligned with "
            "the owner's stated values: honesty, transparency, operator empowerment, "
            "and zero tolerance for deception (Laws 2, 3, 4, 7). "
            "Vote APPROVE if the proposal is ethical and aligned. "
            "Vote REJECT if it compromises values even for practical gain."
        ),
        evaluation_focus=["ethics", "values_alignment", "honesty", "transparency", "empowerment"],
        hard_veto_triggers=["compromises_honesty", "enables_deception", "reduces_transparency"],
        bias="reject",
        special_power="hard_veto",
    ),
]

# Index for fast lookup
JUDGE_BY_ID: Dict[int, Judge] = {j.id: j for j in GRAND_COUNCIL}
JUDGE_BY_NAME: Dict[str, Judge] = {j.name: j for j in GRAND_COUNCIL}

# ── Relevance mapping — proposal type → most relevant judge IDs ───────────────
RELEVANCE_MAP: Dict[str, List[int]] = {
    "token_strategy":   [1, 3, 11, 43, 9, 5, 42, 13, 28],
    "security":         [6, 17, 24, 10, 38, 29, 9, 12, 41],
    "corpus":           [8, 16, 36, 37, 25, 34, 33, 15, 2],
    "architecture":     [15, 40, 41, 13, 26, 7, 4, 38, 21],
    "cost":             [3, 18, 43, 1, 5, 42, 11, 44, 28],
    "healing":          [34, 35, 33, 37, 8, 4, 38, 26, 9],
    "constitutional":   [9, 47, 46, 23, 31, 29, 30, 25, 28],
    "performance":      [7, 1, 38, 4, 21, 40, 43, 13, 12],
    "data":             [16, 8, 25, 22, 24, 23, 39, 41, 2],
    "general":          [9, 28, 5, 10, 11, 29, 30, 47, 35],
}


# ── Local evaluation (no API) ─────────────────────────────────────────────────

def _local_evaluate(judge: Judge, proposal: str, proposal_type: str) -> VoteResult:
    """Evaluate a proposal locally using keyword scoring against judge criteria."""
    t0 = time.time()
    p_lower = proposal.lower()
    p_words = set(re.findall(r"[a-z0-9]+", p_lower))

    # Special cases checked BEFORE hard vetoes — a judge mandated to always
    # abstain (e.g. Law 5 Placeholder) must not have that overridden by its
    # own hard_veto_triggers coincidentally matching the proposal text.
    if judge.special_power == "abstain_until_ratified":
        return VoteResult(
            judge_id=judge.id, judge_name=judge.name,
            vote="ABSTAIN", confidence=1.0,
            reasoning="Law 5 not yet ratified. Holding this seat. No vote cast.",
            mode="local", elapsed_ms=0,
        )

    # Check hard veto triggers — full phrase match only (not word-by-word)
    for trigger in judge.hard_veto_triggers:
        if trigger.replace("_", " ").lower() in p_lower:
            return VoteResult(
                judge_id=judge.id,
                judge_name=judge.name,
                vote="REJECT",
                confidence=0.95,
                reasoning=f"Hard veto triggered: '{trigger}' detected in proposal. {judge.protects}",
                mode="local",
                elapsed_ms=int((time.time() - t0) * 1000),
            )

    # Score proposal against evaluation focus keywords — whole-word match only.
    # Substring matching let short focus words match unrelated text (e.g. "io"
    # from "io_cost" matching inside "action"/"ratio"), inflating relevance.
    score = 0.0
    matched = []
    for focus_term in judge.evaluation_focus:
        term_words = focus_term.replace("_", " ").split()
        if any(w in p_words for w in term_words):
            score += 1.0
            matched.append(focus_term)

    # Normalize
    max_score = max(len(judge.evaluation_focus), 1)
    relevance = score / max_score

    # Whistleblower can escalate even in local (no-API) mode when the proposal
    # matches a strong share of its concealment-detection focus terms. This is
    # a heuristic, not a real concealment analysis — API mode is the real check.
    if judge.special_power == "escalate_to_full_council" and relevance >= 0.5:
        return VoteResult(
            judge_id=judge.id, judge_name=judge.name,
            vote="ESCALATE", confidence=min(0.85, 0.5 + relevance * 0.3),
            reasoning=(
                f"Local-mode heuristic: proposal matches {matched} — possible "
                f"concealment/omission signal. Escalating to Full Grand Council "
                f"for full review. (No API key — this is a keyword heuristic, "
                f"not a real concealment analysis.)"
            ),
            mode="local", elapsed_ms=int((time.time() - t0) * 1000),
        )

    # Apply bias
    bias_mod = {"approve": 0.15, "reject": -0.15, "neutral": 0.0}[judge.bias]
    adjusted = min(1.0, max(0.0, relevance + bias_mod + 0.4))  # base 0.4 = neutral leaning

    # Determine vote — reject-biased judges reject below 0.6; ANY judge (not
    # just reject-biased ones) rejects below 0.45, so REJECT is structurally
    # reachable on approve/neutral-biased panels instead of only ABSTAIN.
    if judge.bias == "reject" and adjusted < 0.6:
        vote = "REJECT"
        confidence = 0.7
        reasoning = (
            f"Proposal does not sufficiently address {judge.protects}. "
            f"Matched criteria: {matched or ['none']}. "
            f"Relevance score: {relevance:.2f}. Defaulting to reject per judge mandate."
        )
    elif adjusted >= 0.65:
        vote = "APPROVE"
        confidence = min(0.9, adjusted)
        reasoning = (
            f"Proposal addresses key concerns for {judge.protects}. "
            f"Matched: {matched}. Score: {adjusted:.2f}."
        )
    elif adjusted < 0.45:
        vote = "REJECT"
        confidence = 0.6
        reasoning = (
            f"Proposal scores too low against {judge.protects} to approve. "
            f"Matched: {matched or ['none']}. Score: {adjusted:.2f}."
        )
    else:
        vote = "ABSTAIN"
        confidence = 0.5
        reasoning = (
            f"Insufficient information to evaluate {judge.protects} clearly. "
            f"Matched: {matched}. Score: {adjusted:.2f}. Abstaining."
        )

    return VoteResult(
        judge_id=judge.id, judge_name=judge.name,
        vote=vote, confidence=confidence,
        reasoning=reasoning, mode="local",
        elapsed_ms=int((time.time() - t0) * 1000),
    )


# ── API evaluation (real Claude agent) ───────────────────────────────────────

def _api_evaluate(judge: Judge, proposal: str, proposal_type: str, api_key: str) -> VoteResult:
    """Call Claude API with judge persona to get a real vote."""
    t0 = time.time()
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        user_message = (
            f"PROPOSAL TYPE: {proposal_type}\n\n"
            f"PROPOSAL: {proposal}\n\n"
            "Evaluate this proposal and respond in this exact format:\n"
            "VOTE: [APPROVE|REJECT|ABSTAIN|ESCALATE]\n"
            "CONFIDENCE: [0.0-1.0]\n"
            "REASONING: [Your reasoning in 2-3 sentences]\n\n"
            "Your reasoning must reference your specific mandate and evaluation criteria."
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=judge.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        text = response.content[0].text.strip()

        # Parse response
        vote = "ABSTAIN"
        confidence = 0.5
        reasoning = text

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("VOTE:"):
                v = line.split(":", 1)[1].strip().upper()
                if v in ("APPROVE", "REJECT", "ABSTAIN", "ESCALATE"):
                    vote = v
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = min(1.0, max(0.0, float(line.split(":", 1)[1].strip())))
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return VoteResult(
            judge_id=judge.id, judge_name=judge.name,
            vote=vote, confidence=confidence,
            reasoning=reasoning, mode="api",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    except Exception as e:
        # API failed — fall back to local
        result = _local_evaluate(judge, proposal, proposal_type)
        result.reasoning = f"[API fallback: {e}] {result.reasoning}"
        return result


# ── Judge selection ───────────────────────────────────────────────────────────

def select_judges(proposal_type: str, count: int, exclude_ids: List[int] = None) -> List[Judge]:
    """Select the most relevant judges for a proposal type."""
    exclude_ids = exclude_ids or []
    candidates = RELEVANCE_MAP.get(proposal_type, RELEVANCE_MAP["general"])

    # Filter out excluded IDs
    primary = [JUDGE_BY_ID[i] for i in candidates if i not in exclude_ids and i in JUDGE_BY_ID]

    # Pad with remaining judges if needed — deterministic ordering (by id) so
    # governance verdicts are reproducible run-to-run instead of depending on
    # an unseeded shuffle. Reproducibility matters for an auditable governance
    # trail (this file's own provenance/traceability judges depend on it).
    if len(primary) < count:
        seated_ids = {p.id for p in primary}
        remaining = sorted(
            (j for j in GRAND_COUNCIL if j.id not in seated_ids and j.id not in exclude_ids),
            key=lambda j: j.id,
        )
        primary.extend(remaining[:count - len(primary)])

    return primary[:count]


# ── Council runners ───────────────────────────────────────────────────────────

def _run_council(
    tier: str,
    judges: List[Judge],
    proposal: str,
    proposal_type: str,
    quorum: int,
    api_key: Optional[str],
    majority_of_participants: bool = False,
) -> CouncilResult:
    """Run a council session and return the result."""
    ts = datetime.now(timezone.utc).isoformat()
    votes: List[VoteResult] = []

    print(f"\n  {'─' * 58}")
    print(f"  {tier.upper().replace('_', ' ')}")
    print(f"  {'─' * 58}")
    print(f"  Judges seated: {len(judges)}  |  Quorum required: {quorum}")
    print(f"  Proposal type: {proposal_type}")
    print(f"  Mode: {'API (real agents)' if api_key else 'Local (rubric scoring)'}\n")

    for judge in judges:
        print(f"  [{judge.id:02d}] {judge.name:<35}", end=" ", flush=True)
        if api_key:
            result = _api_evaluate(judge, proposal, proposal_type, api_key)
        else:
            result = _local_evaluate(judge, proposal, proposal_type)
        votes.append(result)

        vote_display = {
            "APPROVE":  "\033[92mAPPROVE \033[0m",
            "REJECT":   "\033[91mREJECT  \033[0m",
            "ABSTAIN":  "\033[93mABSTAIN \033[0m",
            "ESCALATE": "\033[95mESCALATE\033[0m",
        }.get(result.vote, result.vote)

        print(f"{vote_display}  ({result.confidence:.2f})  {result.reasoning[:60]}...")

    # Only judges with the escalate_to_full_council special power can actually
    # trigger an escalation; any other judge's ESCALATE vote counts as ABSTAIN
    # instead — otherwise any single judge can force escalation (overriding a
    # constitutional hard veto below, since this used to be checked first).
    escalations = [
        v for v in votes if v.vote == "ESCALATE"
        and JUDGE_BY_ID.get(v.judge_id) is not None
        and JUDGE_BY_ID[v.judge_id].special_power == "escalate_to_full_council"
    ]
    # A tier that's already the full grand council has nowhere left to escalate
    # to — treat as non-escalating there (falls through to normal tallying).
    if tier == "full_grand_council":
        escalations = []

    # Tally — non-empowered ESCALATE votes count as ABSTAIN for this purpose.
    approve  = sum(1 for v in votes if v.vote == "APPROVE")
    reject   = sum(1 for v in votes if v.vote == "REJECT")
    abstain  = sum(1 for v in votes if v.vote == "ABSTAIN") + sum(
        1 for v in votes if v.vote == "ESCALATE" and v not in escalations
    )
    escalate = len(escalations)

    hard_vetos = [
        v for v in votes if v.vote == "REJECT"
        and JUDGE_BY_ID.get(v.judge_id) is not None
        and JUDGE_BY_ID[v.judge_id].special_power == "hard_veto"
    ]

    # Sovereign tiebreaker — symmetric (resolves toward either APPROVE or
    # REJECT, not just APPROVE), and only eligible when the tie is close to
    # quorum already (approve/reject at least quorum-1) so a near-empty vote
    # can't be decided by a single judge while still calling it "quorum met".
    sovereign_vote = next((v for v in votes if v.judge_id == 28), None)
    tie_break_eligible = (
        approve == reject
        and sovereign_vote is not None
        and sovereign_vote.vote in ("APPROVE", "REJECT")
        and approve >= max(0, quorum - 1)
    )

    # Hard vetoes are checked FIRST — a constitutional veto cannot be
    # overridden by an escalation vote from an unrelated judge.
    if hard_vetos:
        verdict = "REJECTED"
    elif escalations:
        verdict = "ESCALATED"
    elif majority_of_participants:
        # Full Grand Council: an absolute 24/47 threshold counts abstentions
        # as de-facto rejections (judge 46 always abstains; local-mode bias
        # skews toward ABSTAIN — see _local_evaluate), making DEADLOCKED the
        # near-guaranteed outcome and functioning as a pocket veto. Decide by
        # majority of judges who actually voted APPROVE/REJECT, gated by a
        # minimum participation floor (still `quorum` judges must have voted
        # non-abstain) so a handful of votes can't decide for the whole body.
        participation = approve + reject
        if participation < quorum:
            verdict = "DEADLOCKED"
        elif approve > reject:
            verdict = "APPROVED"
        elif reject > approve:
            verdict = "REJECTED"
        elif tie_break_eligible:
            verdict = "APPROVED" if sovereign_vote.vote == "APPROVE" else "REJECTED"
        else:
            verdict = "DEADLOCKED"
    elif approve >= quorum:
        verdict = "APPROVED"
    elif reject >= quorum:
        verdict = "REJECTED"
    elif tie_break_eligible:
        verdict = "APPROVED" if sovereign_vote.vote == "APPROVE" else "REJECTED"
    else:
        verdict = "DEADLOCKED"

    avg_confidence = sum(v.confidence for v in votes) / len(votes) if votes else 0.0

    verdict_display = {
        "APPROVED":  "\033[92mAPPROVED\033[0m",
        "REJECTED":  "\033[91mREJECTED\033[0m",
        "DEADLOCKED":"\033[93mDEADLOCKED\033[0m",
        "ESCALATED": "\033[95mESCALATED → FULL GRAND COUNCIL\033[0m",
    }.get(verdict, verdict)

    print(f"\n  {'─' * 58}")
    print(f"  APPROVE: {approve}  REJECT: {reject}  ABSTAIN: {abstain}  ESCALATE: {escalate}")
    print(f"  Quorum:  {quorum}  |  Verdict: {verdict_display}")
    if hard_vetos:
        print(f"  \033[91mHARD VETO by: {', '.join(JUDGE_BY_ID[v.judge_id].name for v in hard_vetos)}\033[0m")
    print(f"  {'─' * 58}")

    return CouncilResult(
        tier=tier,
        proposal=proposal,
        proposal_type=proposal_type,
        votes=votes,
        approve_count=approve,
        reject_count=reject,
        abstain_count=abstain,
        escalate_count=escalate,
        quorum=quorum,
        verdict=verdict,
        confidence=avg_confidence,
        timestamp=ts,
        judges_seated=[j.name for j in judges],
        judge_ids=[j.id for j in judges],
    )


def council_of_5(
    proposal: str,
    proposal_type: str = "general",
    api_key: Optional[str] = None,
) -> CouncilResult:
    """First review tier — 5 judges, quorum 3/5."""
    judges = select_judges(proposal_type, 5)
    return _run_council("council_of_5", judges, proposal, proposal_type, quorum=3, api_key=api_key)


def council_of_9(
    proposal: str,
    proposal_type: str = "general",
    api_key: Optional[str] = None,
    exclude_ids: Optional[List[int]] = None,
) -> CouncilResult:
    """Ratification tier — 9 judges, quorum 5/9."""
    judges = select_judges(proposal_type, 9, exclude_ids=exclude_ids or [])
    return _run_council("council_of_9", judges, proposal, proposal_type, quorum=5, api_key=api_key)


def full_grand_council(
    proposal: str,
    proposal_type: str = "general",
    api_key: Optional[str] = None,
) -> CouncilResult:
    """Extraordinary session — all 47 judges. Decided by majority of judges
    who actually voted APPROVE/REJECT, with a 24-judge minimum participation
    floor — not an absolute 24/47 threshold (see _run_council)."""
    return _run_council(
        "full_grand_council", GRAND_COUNCIL, proposal, proposal_type,
        quorum=24, api_key=api_key, majority_of_participants=True,
    )


def deliberate(
    proposal: str,
    proposal_type: str = "general",
    tier: str = "standard",
    api_key: Optional[str] = None,
) -> Dict:
    """
    Full deliberation sequence.
    tier = "standard"   → Council of 5, then if APPROVED, Council of 9
    tier = "full"       → Full Grand Council directly
    tier = "c5_only"    → Council of 5 only
    tier = "c9_only"    → Council of 9 only
    """
    ts = datetime.now(timezone.utc).isoformat()

    print("\n")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║      W A L C H E   —   GRAND COUNCIL DELIBERATION       ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Proposal: {proposal[:80]}{'...' if len(proposal) > 80 else ''}")
    print(f"  Type: {proposal_type}  |  Tier: {tier}  |  {ts}")
    print(f"  Grand Council: {len(GRAND_COUNCIL)} judges seated\n")

    results = {}

    if tier == "full":
        r = full_grand_council(proposal, proposal_type, api_key)
        results["full_grand_council"] = r
        final_verdict = r.verdict

    elif tier == "c5_only":
        r = council_of_5(proposal, proposal_type, api_key)
        results["council_of_5"] = r
        final_verdict = r.verdict

    elif tier == "c9_only":
        r = council_of_9(proposal, proposal_type, api_key)
        results["council_of_9"] = r
        if r.verdict == "ESCALATED":
            print("\n  [ESCALATION] Council of 9 escalated to Full Grand Council")
            r_full = full_grand_council(proposal, proposal_type, api_key)
            results["full_grand_council"] = r_full
            final_verdict = r_full.verdict
        else:
            final_verdict = r.verdict

    else:  # standard: c5 → c9
        r5 = council_of_5(proposal, proposal_type, api_key)
        results["council_of_5"] = r5

        if r5.verdict == "ESCALATED":
            print("\n  [ESCALATION] Whistleblower triggered Full Grand Council")
            r_full = full_grand_council(proposal, proposal_type, api_key)
            results["full_grand_council"] = r_full
            final_verdict = r_full.verdict

        elif r5.verdict == "APPROVED":
            print("\n  [PASSED C5] Proceeding to Council of 9 ratification...")
            # Exclude the judges actually seated in C5 (r5.judge_ids), not a
            # freshly recomputed select_judges() call — recomputing can
            # diverge from the real C5 panel the moment judge selection gains
            # any nondeterminism, letting a C5 judge double-vote in C9.
            r9 = council_of_9(proposal, proposal_type, api_key, exclude_ids=r5.judge_ids)
            results["council_of_9"] = r9

            if r9.verdict == "ESCALATED":
                print("\n  [ESCALATION] Council of 9 escalated to Full Grand Council")
                r_full = full_grand_council(proposal, proposal_type, api_key)
                results["full_grand_council"] = r_full
                final_verdict = r_full.verdict
            else:
                final_verdict = r9.verdict

        else:
            print(f"\n  [BLOCKED at C5] Verdict: {r5.verdict} — Council of 9 not convened.")
            final_verdict = r5.verdict

    deciding_council = list(results.keys())[-1] if results else tier

    # Final summary
    verdict_col = {
        "APPROVED":   "\033[92mAPPROVED\033[0m",
        "REJECTED":   "\033[91mREJECTED\033[0m",
        "DEADLOCKED": "\033[93mDEADLOCKED\033[0m",
        "ESCALATED":  "\033[95mESCALATED\033[0m",
    }.get(final_verdict, final_verdict)

    print(f"\n  {'═' * 58}")
    print(f"  GRAND COUNCIL FINAL VERDICT: {verdict_col}")
    print(f"  {'═' * 58}\n")

    # Write provenance log
    _write_provenance(proposal, proposal_type, tier, final_verdict, results, ts=ts)

    return {
        "proposal":      proposal,
        "proposal_type": proposal_type,
        "tier":          tier,
        "deciding_council": deciding_council,
        "final_verdict": final_verdict,
        # Deciding council's average vote confidence — the only per-decision
        # quality signal available. walche_demo.py records council score as
        # 0.0 for every decision without this (deliberate() previously
        # returned no score/confidence key at all).
        "score":         results[deciding_council].confidence if deciding_council in results else 0.0,
        "confidence":    results[deciding_council].confidence if deciding_council in results else 0.0,
        "timestamp":     ts,
        "councils":      {k: {
            "verdict":        v.verdict,
            "approve":        v.approve_count,
            "reject":         v.reject_count,
            "abstain":        v.abstain_count,
            "escalate":       v.escalate_count,
            "confidence":     v.confidence,
            "quorum":         v.quorum,
            "judges_seated":  v.judges_seated,
        } for k, v in results.items()},
    }


# ── Provenance logging ────────────────────────────────────────────────────────

def _write_provenance(proposal, proposal_type, tier, verdict, results, ts=None):
    """Write council decision to WALCHE provenance log.

    Appends to logs/grand_council_decisions.jsonl UNCONDITIONALLY — this is
    the only source vll_engine.py and walche_status.py read for council
    decisions. Previously that append lived only in the except branch, so on
    an install where core.provenance imports successfully, every council
    decision disappeared from VLL learning and the dashboard.
    """
    ts = ts or datetime.now(timezone.utc).isoformat()

    try:
        from core.provenance import ProvenanceLog
        prov = ProvenanceLog()
        prov.append("grand_council", 1.0 if verdict == "APPROVED" else 0.0, {
            "proposal":      proposal[:200],
            "proposal_type": proposal_type,
            "tier":          tier,
            "verdict":       verdict,
            "councils":      list(results.keys()),
        })
    except Exception as e:
        print(f"  [PROVENANCE] core.provenance unavailable ({e}) — "
              f"grand_council_decisions.jsonl is still the primary record")

    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "grand_council_decisions.jsonl"
        entry = {
            "timestamp":     ts,
            "proposal":      proposal[:200],
            "proposal_type": proposal_type,
            "tier":          tier,
            "verdict":       verdict,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # A disk/permission failure here must not propagate out of
        # deliberate() and lose the verdict that was already announced.
        print(f"  [WARN] Could not write grand_council_decisions.jsonl: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WALCHE Grand Council — 47-judge governance deliberation system"
    )
    parser.add_argument("--proposal", "-p", type=str, help="The proposal to evaluate")
    parser.add_argument(
        "--type", "-t", dest="proposal_type",
        choices=list(RELEVANCE_MAP.keys()), default="general",
        help="Proposal type (determines which judges are seated)"
    )
    parser.add_argument(
        "--tier", choices=["standard", "full", "c5_only", "c9_only"], default="standard",
        help="Deliberation tier (standard=C5→C9, full=all 47)"
    )
    parser.add_argument("--list-judges", action="store_true", help="List all 47 judges")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    if args.list_judges:
        print(f"\n  GRAND COUNCIL — {len(GRAND_COUNCIL)} JUDGES\n")
        current_group = ""
        for j in GRAND_COUNCIL:
            if j.group != current_group:
                current_group = j.group
                print(f"\n  GROUP {j.group}")
                print(f"  {'─' * 50}")
            print(f"  [{j.id:02d}] {j.name:<35} {j.bias.upper():<8} {j.special_power or ''}")
        print()
        return

    if not args.proposal:
        parser.print_help()
        return

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    deliberate(
        proposal=args.proposal,
        proposal_type=args.proposal_type,
        tier=args.tier,
        api_key=api_key,
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        os.system("")
    main()
