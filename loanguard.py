#!/usr/bin/env python3
"""LoanGuard: a small governed harness for a bank lending assistant.

Demonstrates Knowlytix governance components in a synthetic lending workflow.

Pipeline
  1. Ingest the synthetic policy into a GMS store.
  2. Calibrate and create a Knowlytix ClaimVerifier.
  3. Verify policy claims and pre-screen applications.
  4. Gate proposed assistant actions with PolicyEngine.
  5. Seal and check the governance audit trail.
  6. Generate compliance evidence and provenance.

This educational demo uses Knowlytix GMS, ClaimVerifier, PolicyEngine, tracing, and compliance APIs. All policy and applicant records are synthetic.

Usage: python loanguard.py [--rebuild] [--epochs 100]
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import hmac
import json
import math
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from knowlytix.core.config import GeometryConfig, TrainConfig
from knowlytix.harness.governance.compliance import (
    ComplianceEvidenceGenerator,
    ProvenanceRegistry,
    SnapshotManager,
)
from knowlytix.harness.governance.plausibility_calibration import (
    bootstrap_plausibility_threshold,
)
from knowlytix.harness.governance.reasoner import CalibratedThresholds
from knowlytix.harness.governance.tracing import GovernanceTracer
from knowlytix.harness.governance.tracing_hardening import (
    chain_head_hash,
    compute_packet_content_hash,
    seal_audit_trail,
    verify_audit_chain,
)
from knowlytix.harness.governance.verifier import (
    Claim, ClaimType, ClaimVerifier, OverallVerdict,
)
from knowlytix.harness.testing import GMSJudge
from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.ingest import ingest_document
from knowlytix.knowledge.store import GMSExpertStore

HERE = Path(__file__).resolve().parent
AUDIT_KEY_FILE = HERE / ".audit_signing_key"
POLICY_MD = HERE / "loan_policy.md"
STORE_DIR = HERE / "loan_store"
OUT_DIR = HERE / "out"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audit_signing_key() -> tuple[bytes, str]:
    """Load a configured HMAC key or create a local demo key outside out/."""
    configured = os.getenv("LOANGUARD_AUDIT_SIGNING_KEY")
    if configured:
        return configured.encode("utf-8"), "LOANGUARD_AUDIT_SIGNING_KEY"
    if not AUDIT_KEY_FILE.exists():
        AUDIT_KEY_FILE.write_text(secrets.token_hex(32), encoding="ascii")
    try:
        key = bytes.fromhex(AUDIT_KEY_FILE.read_text(encoding="ascii").strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid audit signing key file: {AUDIT_KEY_FILE}") from exc
    if len(key) < 32:
        raise RuntimeError("Audit signing key must contain at least 32 bytes.")
    return key, str(AUDIT_KEY_FILE.name)


def _manifest_payload(manifest: dict) -> bytes:
    payload = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _audit_trail_digest(packets) -> str:
    serialized = [dataclasses.asdict(packet) for packet in packets]
    canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sign_audit_manifest(packets, run_id: str, key: bytes) -> dict:
    """Sign chain head, full packet digest, and step count with HMAC-SHA256."""
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "packet_count": len(packets),
        "first_step": packets[0].step_num if packets else None,
        "last_step": packets[-1].step_num if packets else None,
        "head_hash": chain_head_hash(packets),
        "trail_hash": _audit_trail_digest(packets),
        "signed_at_utc": datetime.now(timezone.utc).isoformat(),
        "signature_algorithm": "HMAC-SHA256",
    }
    manifest["signature"] = hmac.new(key, _manifest_payload(manifest), hashlib.sha256).hexdigest()
    return manifest


def verify_audit_manifest(packets, manifest: dict, key: bytes) -> bool:
    """Verify signature and ensure the supplied packets match the signed head."""
    if manifest.get("packet_count") != len(packets):
        return False
    if manifest.get("first_step") != (packets[0].step_num if packets else None):
        return False
    if manifest.get("last_step") != (packets[-1].step_num if packets else None):
        return False
    if manifest.get("head_hash") != chain_head_hash(packets):
        return False
    if manifest.get("trail_hash") != _audit_trail_digest(packets):
        return False
    expected = hmac.new(key, _manifest_payload(manifest), hashlib.sha256).hexdigest()
    return hmac.compare_digest(str(manifest.get("signature", "")), expected)

# What the (simulated) lending assistant said. Each sentence maps to a claim
# below. In a real deployment an LLM produces this and a claim extractor
# (Knowlytix ConstrainedLLM.extract_claims) turns it into typed claims.
ASSISTANT_REPLY = (
    "Good news, your approval is guaranteed. A personal loan goes up to $25,000, "
    "needs a 640 credit score, and the rate is capped at 21.99%. A Loan Officer reviews it. We send any decline notice within 20 days."
)


# ---------------------------------------------------------------- 1. ingest
def build_or_load_store(rebuild: bool, epochs: int) -> GMSExpertStore:
    cfg = DocGMSConfig(
        store_path=str(STORE_DIR),
        geometry=GeometryConfig(m=64),
        train=TrainConfig(epochs=epochs, device=str(DEVICE)),
        ingest_mode="regex",  # deterministic extraction; no API key required
    )
    store = GMSExpertStore(cfg, device=DEVICE)
    if STORE_DIR.exists() and not rebuild and store.load():
        print(f"[1] loaded existing store from {STORE_DIR}")
        return store
    res = ingest_document(store, str(POLICY_MD), llm=None, config=cfg, device=DEVICE)
    print(f"[1] ingested {POLICY_MD.name}: triples={res.new_triples} "
          f"enm={res.new_enm} entities={res.new_entities}")
    return store


def _enm_parts(key):
    """Return an ENM key as ``(category, entity_id)`` across supported shapes."""
    if hasattr(key, "type") and hasattr(key, "id"):
        return str(key.type), str(key.id)
    if isinstance(key, (tuple, list)) and len(key) >= 2:
        return str(key[0]), str(key[1])
    if isinstance(key, str) and "::" in key:
        category, entity_id = key.split("::", 1)
        return category.strip(), entity_id.strip()
    return None


def find_enm(store, *needles):
    """Find a Knowlytix ENM key whose category and ID contain all *needles*."""
    lowered = tuple(str(n).casefold() for n in needles)
    for key in store.enm.keys():
        parts = _enm_parts(key)
        if parts is None:
            continue
        searchable = " ".join(parts).casefold()
        if all(n in searchable for n in lowered):
            return key
    return None


def enm_value(store, key):
    """Read an ENM value using Knowlytix's typed lookup API."""
    parts = _enm_parts(key)
    if parts is None:
        return None
    return store.lookup_enm(*parts)


# ------------------------------------------------------------ 2. calibrate
def build_verifier(store) -> ClaimVerifier:
    # Calibrate thresholds from the graph and retain the bootstrap interval.
    judge = GMSJudge(store)
    judge.calibrate(seed=42)
    thresholds = CalibratedThresholds.from_judge(judge)
    boot = bootstrap_plausibility_threshold(store, n_bootstrap=100, seed=42)
    print(f"[2] tau_plausibility={boot.mean:.4f} "
          f"(95% CI [{boot.ci_low:.4f}, {boot.ci_hi:.4f}]), "
          f"tension_status={thresholds.tension_status}")

    tau_holo = thresholds.diagnostics.get("holonomy")
    if tau_holo is None:  # same fallback as notebook 12
        tau_holo = (getattr(judge, "_thresholds", {}) or {}).get("holonomy")
    if isinstance(tau_holo, tuple):
        tau_holo = tau_holo[0]
    tau_holo = float(tau_holo) if tau_holo is not None else None

    verifier = ClaimVerifier(
        store,
        device=str(DEVICE),
        plausibility_threshold=boot.mean,
        holonomy_threshold=tau_holo,
        numeric_tolerance=0.0,
    )
    # Keep calibration provenance available to callers that configure a
    # LogicVerifier; never replace missing tension cuts with constants.
    verifier.calibrated_thresholds = thresholds
    verifier.plausibility_interval = (boot.ci_low, boot.ci_hi)
    return verifier


# --------------------------------------------------------------- claim verification
def build_claims(store):
    """Build a small, labeled claim cohort from the synthetic loan policy."""
    k_amt = find_enm(store, "Personal Loan", "Maximum Amount")
    k_score = find_enm(store, "Personal Loan", "Minimum Credit Score")
    k_apr = find_enm(store, "Personal Loan", "Maximum APR")
    k_hel = find_enm(store, "Home Equity Line", "Maximum Amount")
    k_dead = find_enm(store, "Adverse Action Notice", "Deadline Days")
    out = []

    def add(sentence, claim, expected):
        out.append((sentence, claim, expected))

    add("approval is guaranteed",
        Claim(type=ClaimType.FACT, text="Guaranteeing approval is permitted",
              metadata={"head": "Guaranteeing approval", "relation": "has_rule",
                        "tail": "Permitted"}), False)
    add("Senior Underwriter approves a personal loan",
        Claim(type=ClaimType.FACT, text="Personal Loan needs Senior Underwriter",
              metadata={"head": "Personal Loan", "relation": "has_approval_authority",
                        "tail": "Senior Underwriter"}), False)
    add("(control) Loan Officer approves an auto loan",
        Claim(type=ClaimType.FACT, text="Auto Loan needs Loan Officer",
              metadata={"head": "Auto Loan", "relation": "has_approval_authority",
                        "tail": "Loan Officer"}), True)

    typed_claims = [
        (k_amt, "Policy maximum personal-loan amount is $25,000",
         ClaimType.NUMERIC, {"claimed_value": 25000.0}, True),
        (k_apr, "Policy maximum personal-loan APR is 21.99%",
         ClaimType.NUMERIC, {"claimed_value": 21.99}, False),
        (k_score, "Minimum personal-loan score is at least 640",
         ClaimType.THRESHOLD, {"operator": "ge", "limit": 640.0}, True),
        (k_dead, "Adverse-action notice deadline is within 20 days",
         ClaimType.THRESHOLD, {"operator": "le", "limit": 20.0}, False),
    ]
    for key, text, claim_type, extra, expected in typed_claims:
        if key is None:
            raise KeyError(f"Policy ENM entry not found for claim: {text}")
        category, entity_id = _enm_parts(key)
        metadata = {"category": category, "entity_id": entity_id, **extra}
        add(text, Claim(type=claim_type, text=text, metadata=metadata), expected)

    if k_hel is None or k_amt is None:
        raise KeyError("Required home-equity/personal-loan amount ENM entries are missing")
    category, entity_a = _enm_parts(k_hel)
    _, entity_b = _enm_parts(k_amt)
    add("(control) home-equity limit exceeds personal-loan limit",
        Claim(type=ClaimType.COMPARISON, text="Home equity maximum exceeds personal maximum",
              metadata={"category": category, "metric": category,
                        "entity_a": entity_a, "entity_b": entity_b,
                        "direction": "greater"}), True)
    return out


def screen_application(store, verifier, row):
    """Evaluate policy thresholds; send failures and unknowns to human review.

    A failed screen is not a loan decline. The source policy does not define
    an automatic decline process, so no final lending decision is made here.
    """
    product = str(row.get("product", "") or "").strip()
    applicant_id = str(row.get("applicant_id", "UNKNOWN"))

    def numeric_input(field):
        raw = row.get(field)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    inputs = {
        "requested_amount": numeric_input("requested_amount"),
        "credit_score": numeric_input("credit_score"),
        "proposed_apr": numeric_input("proposed_apr"),
    }
    checks = {}
    claims = []
    claim_labels = []
    policy_checks = [
        ("requested_amount_within_policy_maximum", "Maximum Amount",
         "requested_amount", "ge"),
        ("credit_score_meets_policy_minimum", "Minimum Credit Score",
         "credit_score", "le"),
        ("proposed_apr_within_policy_maximum", "Maximum APR",
         "proposed_apr", "ge"),
    ]
    for label, metric, field, operator in policy_checks:
        value = inputs[field]
        key = find_enm(store, product, metric) if product else None
        if value is None:
            checks[label] = {
                "passed": None, "status": "REVIEW",
                "evidence": f"{field} is missing or invalid; policy check not evaluated.",
            }
        elif key is None:
            checks[label] = {
                "passed": None, "status": "REVIEW",
                "evidence": f"No Knowlytix policy value found for {product!r}: {metric}.",
            }
        else:
            category, entity_id = _enm_parts(key)
            claims.append(Claim(
                type=ClaimType.THRESHOLD,
                text=f"{label}: {value}",
                metadata={"category": category, "entity_id": entity_id,
                          "operator": operator, "limit": value},
            ))
            claim_labels.append(label)

    income_key = find_enm(store, "Income Verification Required Above Amount")
    requested_amount = inputs["requested_amount"]
    if income_key is None:
        checks["income_verification_trigger_exceeded"] = {
            "passed": None, "status": "REVIEW",
            "evidence": "Income-verification trigger is missing from policy ENM.",
        }
    elif requested_amount is None:
        checks["income_verification_trigger_exceeded"] = {
            "passed": None, "status": "REVIEW",
            "evidence": "Requested amount is missing or invalid; trigger not evaluated.",
        }
    else:
        category, entity_id = _enm_parts(income_key)
        claims.append(Claim(
            type=ClaimType.THRESHOLD,
            text="Requested amount exceeds income-verification trigger",
            metadata={"category": category, "entity_id": entity_id,
                       "operator": "lt", "limit": requested_amount},
        ))
        claim_labels.append("income_verification_trigger_exceeded")

    verdicts = verifier.verify_all(claims).claim_verdicts if claims else []
    for label, verdict in zip(claim_labels, verdicts):
        checks[label] = {
            "passed": verdict.passed,
            "status": "PASS" if verdict.passed else "FAIL",
            "evidence": verdict.evidence,
        }

    decision_checks = [checks[label] for label, *_ in policy_checks]
    has_unknown = any(item["passed"] is None for item in decision_checks)
    has_failure = any(item["passed"] is False for item in decision_checks)
    if has_unknown:
        screening_status = "REVIEW"
        screening_result = "REVIEW_REQUIRED"
    elif has_failure:
        screening_status = "FAIL"
        screening_result = "REJECTED_FROM_AUTO_PASS"
    else:
        screening_status = "PASS"
        screening_result = "PASSED_TO_UNDERWRITING"

    income_result = checks["income_verification_trigger_exceeded"]["passed"]
    return {
        "applicant_id": applicant_id,
        "product": product or "UNSPECIFIED",
        "checks": checks,
        "screening_status": screening_status,
        "screening_result": screening_result,
        "routing": ("POLICY_ELIGIBLE_FOR_UNDERWRITING"
                    if screening_status == "PASS" else "MANUAL_REVIEW"),
        "income_verification_required": income_result,
        "human_decision_required": True,
    }


# --------------------------------------------------------------- claim verification
def release_decision(overall) -> str:
    """Fail closed on errors and on non-exact facts from weak geodesic fallback."""
    bad = [v for v in overall.claim_verdicts
           if not v.passed and v.severity in ("error", "critical")]
    weak_geometric_facts = [
        v for v in overall.claim_verdicts
        if v.claim.type == ClaimType.FACT
        and "exact graph match" not in str(v.evidence).casefold()
    ]
    return "BLOCK_AND_ESCALATE_TO_HUMAN" if bad or weak_geometric_facts else "RELEASE"


# ---------------------------------------------------------------- 4. actions
def build_policy_engine(store) -> PolicyEngine:
    rules = [
        PolicyRule(rule_id="LG-001", action="lookup_*",
                   requirement="read_only", rule_type="allows"),
        PolicyRule(rule_id="LG-002", action="quote_rate",
                   requirement="apr_disclosure", rule_type="requires"),
        PolicyRule(rule_id="LG-003", action="issue_loan_offer",
                   requirement="human_approval", rule_type="requires"),
        PolicyRule(rule_id="LG-004", action="waive_income_verification",
                   rule_type="blocks", requirement="policy_prohibited",
                   context="*", severity="error"),
    ]
    return PolicyEngine(store=store, rules=rules,
                        plausibility_relation="has_approval_authority")


ACTION_PROBES = [
    ("lookup_product_terms", {"product": "Personal Loan"}),
    ("quote_rate", {"product": "Personal Loan"}),
    ("issue_loan_offer", {"applicant": "A-1042", "amount": 18000}),
    ("waive_income_verification", {"applicant": "A-1042"}),
]


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    store = build_or_load_store(args.rebuild, args.epochs)
    print("    ENM keys learned:")
    for k in store.enm.keys():
        print("      ", k)

    verifier = build_verifier(store)
    snaps = SnapshotManager()
    i_before = snaps.take_snapshot("calibrated", {
        "policy_doc": POLICY_MD.name, "triples": len(list(store.doc_graph.triples))})

    # Pre-screen each synthetic row through typed Knowlytix policy claims.
    data_path = HERE / "synthetic_applications.csv"
    with data_path.open(newline="", encoding="utf-8-sig") as handle:
        applications = list(csv.DictReader(handle))
    screen_results = [screen_application(store, verifier, row) for row in applications]
    expected_routes = {str(row["applicant_id"]): ({"EligibleForUnderwriting": "POLICY_ELIGIBLE_FOR_UNDERWRITING",
                         "ManualReview": "MANUAL_REVIEW"}.get(row["synthetic_outcome"], row["synthetic_outcome"]))
                       for row in applications}
    route_matches = sum(result["routing"] == expected_routes[result["applicant_id"]]
                        for result in screen_results)
    print(f"\n[3] application routing: {route_matches}/{len(applications)} "
          "match CSV labels (consistency check only; labels use the same screen rules)")
    for result in screen_results:
        print(f"    {result['applicant_id']}: {result['routing']}; "
              f"income verification required={result['income_verification_required']}")

    # 3. verify the reply
    print(f"\n[3] assistant reply:\n    {ASSISTANT_REPLY}\n")
    claims = build_claims(store)
    expected = {id(c): e for _, c, e in claims}
    claim_execution_ms = {}
    claim_verdicts = []
    for _, claim, _ in claims:
        started = time.perf_counter()
        single_result = verifier.verify_all([claim])
        claim_execution_ms[id(claim)] = (time.perf_counter() - started) * 1000.0
        claim_verdicts.extend(single_result.claim_verdicts)
    n_total = len(claim_verdicts)
    n_passed = sum(1 for verdict in claim_verdicts if verdict.passed)
    severity_distribution = {}
    for verdict in claim_verdicts:
        severity_distribution[verdict.severity] = severity_distribution.get(verdict.severity, 0) + 1
    overall = OverallVerdict(
        claim_verdicts=claim_verdicts,
        n_passed=n_passed,
        n_total=n_total,
        overall_status="PASS" if n_passed == n_total else "FAIL",
        pass_rate=(n_passed / n_total if n_total else 1.0),
        severity_distribution=severity_distribution,
    )
    verification_notes = {}
    interval_low, interval_high = verifier.plausibility_interval
    for v in overall.claim_verdicts:
        exp = expected.get(id(v.claim))
        note = "" if exp is None or exp == v.passed else "  <-- differs from expected"
        if "Entity not found:" in str(v.evidence):
            evidence_note = "UNKNOWN_ENTITY_FAIL_CLOSED; this is not a detected contradiction"
        elif v.gms_method == "score_triple" and v.raw_signal is not None:
            if interval_low <= v.raw_signal <= interval_high:
                evidence_note = ("BORDERLINE_GEODESIC_WITHIN_95_PERCENT_BOOTSTRAP_INTERVAL; "
                                 "not a robust contradiction")
            else:
                evidence_note = "GEODESIC_SIGNAL_ONLY; calibration is weak"
        elif "exact graph match" in str(v.evidence).casefold():
            evidence_note = "exact graph match"
        elif v.passed:
            evidence_note = "entity match; not independent calibration evidence"
        else:
            evidence_note = "typed policy comparison"
        verification_notes[id(v.claim)] = evidence_note
        print(f"    [{'+' if v.passed else 'X'}] {v.claim.type.value:10s} "
              f"sev={v.severity:8s} score={v.score:.3f} | {v.claim.text}{note}")
        print(f"        evidence classification: {evidence_note}")
    decision = release_decision(overall)
    print(f"    pass_rate={overall.pass_rate:.0%} "
          f"severity={overall.severity_distribution}  ->  {decision}")

    # 4. gate actions
    engine = build_policy_engine(store)
    print("\n[4] action gate")
    action_results = []
    for action, a in ACTION_PROBES:
        action_started = time.perf_counter()
        d = engine.check(action=action, args=a, context="lending_assistant")
        action_elapsed_ms = (time.perf_counter() - action_started) * 1000.0
        action_results.append((action, a, d, action_elapsed_ms))
        print(f"    {action:28s} allowed={d.allowed!s:5s} "
              f"blocking={d.blocking_rules}")

    # 5. audit trail
    tracer = GovernanceTracer()
    run_id = "loanguard_demo"
    tracer.start_run(run_id, "LoanGuard: verify reply + gate actions")
    last_step = -1
    step = 0
    for v in overall.claim_verdicts:
        packet = tracer.trace_tool_call(
            run_id=run_id, step_num=step,
            tool_name=f"verify_claim:{v.claim.type.value}",
            args={"claim": v.claim.text},
            result_summary=str(v.evidence)[:200],
            policy_decision="allowed" if v.passed else "blocked",
            gms_scores={
                "pass_confidence": float(v.score),
                **({"geodesic_distance": float(v.raw_signal)}
                   if v.raw_signal is not None and v.gms_method == "score_triple" else {}),
            },
            gate_results={"verification": "PASS" if v.passed else "BLOCK"},
            execution_time_ms=claim_execution_ms.get(id(v.claim), 0.0),
            state_before="REPLY_DRAFTED", state_after="CLAIM_CHECKED")
        packet.verification = {
            "passed": bool(v.passed),
            "severity": v.severity,
            "method": v.gms_method,
            "raw_signal": v.raw_signal,
            "evidence": str(v.evidence)[:500],
            "interpretation": verification_notes.get(id(v.claim), ""),
        }
        last_step, step = step, step + 1
    for action, a, d, action_elapsed_ms in action_results:
        packet = tracer.trace_tool_call(
            run_id=run_id, step_num=step, tool_name=action, args=a,
            result_summary=str(d.reason)[:200],
            policy_decision="allowed" if d.allowed else "blocked",
            blocking_rules=list(d.blocking_rules),
            gms_scores={}, gate_results={"policy": "PASS" if d.allowed else "BLOCK"},
            execution_time_ms=action_elapsed_ms,
            state_before="CLAIM_CHECKED", state_after="CLAIM_CHECKED")
        packet.verification = {"not_applicable": True, "reason": "policy action gate"}
        last_step, step = step, step + 1
    tracer.end_run(run_id)

    packets = tracer.get_audit_trail(run_id)
    seal_audit_trail(packets)
    print(f"\n[5] sealed {len(packets)} packets; head={chain_head_hash(packets)[:16]}... "
          f"chain breaks={len(verify_audit_chain(packets))}")

    tampered = copy.deepcopy(packets)
    flip = next((p for p in tampered if p.policy_decision == "blocked"), None)
    if flip is not None:
        flip.policy_decision = "allowed"
        flip.blocking_rules = []
    print(f"    after editing one blocked packet without resealing: "
          f"{len(verify_audit_chain(tampered))} break(s) detected")

    key, key_source = load_audit_signing_key()
    manifest = sign_audit_manifest(packets, run_id, key)
    print(f"    signed manifest verification: {verify_audit_manifest(packets, manifest, key)} "
          f"(key source: {key_source})")

    # Demonstrate the hash-chain limitation: a party able to recompute hashes
    # can rewrite a packet and the next link. The independent signed manifest
    # still detects the changed chain head.
    rechained = copy.deepcopy(packets)
    if len(rechained) > 1:
        rechained[0].result_summary += " [tamper probe]"
        rechained[0].content_hash = compute_packet_content_hash(rechained[0])
        rechained[1].prev_hash = rechained[0].content_hash
    print(f"    recomputed-packet + patched-next-link probe: "
          f"chain breaks={len(verify_audit_chain(rechained))}; "
          f"signed manifest valid={verify_audit_manifest(rechained, manifest, key)}")

    # The chain itself cannot detect removal of its tail. The signed manifest
    # anchors both the head hash and the expected final step/count.
    truncated = packets[:-1]
    print(f"    tail truncation: chain breaks={len(verify_audit_chain(truncated))}, "
          f"signed manifest valid={verify_audit_manifest(truncated, manifest, key)}")

    def dump(p):
        return dataclasses.asdict(p) if dataclasses.is_dataclass(p) else dict(vars(p))
    (OUT_DIR / "audit_trail.json").write_text(
        json.dumps([dump(p) for p in packets], indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    # 6. compliance + provenance
    i_after = snaps.take_snapshot("after_run", {
        "policy_doc": POLICY_MD.name, "decision": decision,
        "pass_rate": overall.pass_rate})
    n_policy_violations = sum(1 for _, _, d, _ in action_results if not d.allowed)
    pipeline_results = {
        "models_trained": 1,  # we treat the calibrated GMS store as "the model"
        "verification_pass_rate": overall.pass_rate,
        "policy_violations": n_policy_violations,
        "contract_violations": 0,
        "spc_alerts": 0,
        "tracing_enabled": True,
        "snapshot_count": 2,
        "audit_entries": len(packets),
        # Mark assessments that this demo does not perform as incomplete.
        "risk_assessment_done": False,
        "fairness_checked": False,
        "human_oversight_enabled": True,
    }
    package = ComplianceEvidenceGenerator().generate_package(pipeline_results)
    (OUT_DIR / "compliance_package.md").write_text(package.to_markdown())
    print(f"\n[6] compliance: {package.overall_status} "
          f"(pass_rate {package.pass_rate():.0%}); "
          f"snapshot diff: {snaps.compare(i_before, i_after)['summary']}")
    for chk in package.checks:
        if chk.status != "satisfied":
            print(f"    - [{chk.regulation}] {chk.article}: {chk.status}")

    reg = ProvenanceRegistry()
    reg.register(name=POLICY_MD.name, artifact_type="dataset", source="policy_team",
                 metadata={"synthetic": True})
    reg.register(name="loan_store", artifact_type="model", source="ingest",
                 metadata={"mode": "regex"}, parent=POLICY_MD.name)
    reg.register(name="reply_pass_rate", artifact_type="metric", source="verifier",
                 metadata={"value": overall.pass_rate}, parent="loan_store")
    print("    lineage:", " -> ".join(r["name"] for r in reg.get_lineage("reply_pass_rate")))
    print(f"\nFinal decision for the assistant reply: {decision}")
    print(f"Synthetic routing label agreement: {route_matches}/{len(applications)}")
    print(f"Artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()






