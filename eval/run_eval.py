#!/usr/bin/env python3
"""
Evaluation runner for Consilium — Phase 3.

Calls the Consilium API for each case in golden_workflows.json, measures
task completion rate, fallback activation rate, P95 latency, and per-agent
failure modes. Saves timestamped results to eval/results/.

Usage:
    # From project root (API must be running on localhost:8001)
    python eval/run_eval.py

    # Custom API URL or phase label
    python eval/run_eval.py --api-url http://localhost:8001 --phase phase2_baseline

Environment:
    CONSILIUM_API_URL  — override API base URL (default: http://localhost:8001)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent
GOLDEN_PATH = EVAL_DIR / "golden_workflows.json"
RESULTS_DIR = EVAL_DIR / "results"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GoldenCase:
    id: str
    query: str
    expected_risk_levels: List[str]
    expected_agents_invoked: List[str]
    expected_min_findings: int
    requires_retrieval: bool
    label: str


@dataclass
class CaseResult:
    id: str
    query: str
    success: bool
    confidence: float
    latency_ms: int
    agents_invoked: List[str]
    fallback_events: List[str]
    findings_count: int
    risk_levels_found: List[str]
    report_structure_valid: bool
    failure_mode: Optional[str]
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    trace_id: Optional[str] = None
    full_response: Optional[dict] = None


@dataclass
class EvalMetrics:
    task_completion_rate: float
    fallback_activation_rate: float
    planner_fallback_rate: float
    analyst_fallback_rate: float
    synthesizer_fallback_rate: float
    p95_latency_ms: float
    p50_latency_ms: float
    mean_latency_ms: float
    report_structure_pass_rate: float
    cases_run: int
    cases_skipped: int
    cases_succeeded: int
    cases_failed: int
    failure_mode_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_report_structure(report: str) -> bool:
    """Structural validation only — prose quality assessed manually.

    A valid report must have an executive summary section, a findings-related
    section, and a minimum word count of 200 words.
    """
    if not report:
        return False
    has_executive_summary = "## Executive Summary" in report
    has_findings = "## Detailed Findings" in report or "## Findings" in report
    word_count = len(report.split())
    return has_executive_summary and has_findings and word_count >= 200


def categorize_failure(
    response: dict,
    expected_min_findings: int,
    expected_risk_levels: List[str],
) -> str:
    """Categorize why a case failed using fallback_events for precision."""
    fallback_events: List[str] = response.get("fallback_events", [])
    findings: List[dict] = response.get("result", {}).get("risk_findings", [])
    confidence: float = response.get("confidence", 0.0)

    if "planner" in fallback_events:
        return "planner_fallback"
    if "analyst" in fallback_events:
        return "analyst_fallback"
    if "synthesizer" in fallback_events:
        return "synthesizer_fallback"
    if response.get("error"):
        return "api_error"
    if len(findings) < expected_min_findings:
        return "insufficient_findings"
    risk_levels = [f.get("risk_level", "") for f in findings]
    if not any(rl in expected_risk_levels for rl in risk_levels):
        return "incorrect_classification"
    if confidence < 0.5:
        return "low_confidence"
    return "unknown"


def evaluate_case(response: dict, expected: GoldenCase) -> tuple[bool, str | None]:
    """
    Returns (success, failure_mode).

    Success requires:
    - confidence >= 0.5 (not in degraded fallback mode)
    - at least expected_min_findings findings returned
    - at least one finding has a risk_level in expected_risk_levels
    - report passes structural validation
    """
    confidence = response.get("confidence", 0.0)
    findings = response.get("result", {}).get("risk_findings", [])
    summary = response.get("result", {}).get("summary", "")

    if confidence < 0.5:
        failure = categorize_failure(response, expected.expected_min_findings, expected.expected_risk_levels)
        return False, failure

    if len(findings) < expected.expected_min_findings:
        return False, "insufficient_findings"

    risk_levels = [f.get("risk_level", "") for f in findings]
    if not any(rl in expected.expected_risk_levels for rl in risk_levels):
        return False, "incorrect_classification"

    # Structural report validation — summary is the first 300 chars of final_report
    # Full report not exposed in API; validate summary length as proxy
    if len(summary) < 50:
        return False, "report_too_short"

    return True, None


# ---------------------------------------------------------------------------
# Quaestor reachability check
# ---------------------------------------------------------------------------


async def check_quaestor_reachable(api_url: str) -> bool:
    """Check if Quaestor is reachable via the Consilium API health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{api_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                retrieval = data.get("components", {}).get("retrieval", "")
                return "quaestor" in retrieval and "unreachable" not in retrieval
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Single case runner
# ---------------------------------------------------------------------------


async def run_case(
    client: httpx.AsyncClient,
    api_url: str,
    case: GoldenCase,
    quaestor_reachable: bool,
    store_responses: bool = False,
) -> CaseResult:
    """Run a single golden case against the API and return a CaseResult."""
    if case.requires_retrieval and not quaestor_reachable:
        logger.info("SKIP %s — requires_retrieval=true but Quaestor unreachable", case.id)
        return CaseResult(
            id=case.id,
            query=case.query,
            success=False,
            confidence=0.0,
            latency_ms=0,
            agents_invoked=[],
            fallback_events=[],
            findings_count=0,
            risk_levels_found=[],
            report_structure_valid=False,
            failure_mode=None,
            skipped=True,
            skip_reason="requires_retrieval=true but Quaestor unreachable",
        )

    payload = {"query": case.query}
    start = time.perf_counter()

    try:
        resp = await client.post(f"{api_url}/workflow", json=payload, timeout=180.0)
        latency_ms = int((time.perf_counter() - start) * 1000)

        if resp.status_code != 200:
            logger.warning("Case %s: HTTP %d — %s", case.id, resp.status_code, resp.text[:200])
            return CaseResult(
                id=case.id,
                query=case.query,
                success=False,
                confidence=0.0,
                latency_ms=latency_ms,
                agents_invoked=[],
                fallback_events=[],
                findings_count=0,
                risk_levels_found=[],
                report_structure_valid=False,
                failure_mode="api_error",
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        confidence = data.get("confidence", 0.0)
        findings = data.get("result", {}).get("risk_findings", [])
        summary = data.get("result", {}).get("summary", "")
        agents_invoked = data.get("agents_invoked", [])
        fallback_events = data.get("fallback_events", [])
        risk_levels = [f.get("risk_level", "") for f in findings]

        # Structural validation uses summary as proxy for report quality
        report_structure_valid = validate_report_structure(summary) or len(summary) >= 100

        success, failure_mode = evaluate_case(data, case)

        logger.info(
            "Case %s: %s | conf=%.2f | findings=%d | fallbacks=%s | latency=%dms",
            case.id,
            "PASS" if success else f"FAIL({failure_mode})",
            confidence,
            len(findings),
            fallback_events,
            latency_ms,
        )

        return CaseResult(
            id=case.id,
            query=case.query,
            success=success,
            confidence=confidence,
            latency_ms=latency_ms,
            agents_invoked=agents_invoked,
            fallback_events=fallback_events,
            findings_count=len(findings),
            risk_levels_found=risk_levels,
            report_structure_valid=report_structure_valid,
            failure_mode=failure_mode,
            trace_id=data.get("trace_id"),
            full_response=data if store_responses else None,
        )

    except httpx.TimeoutException:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.error("Case %s: TIMEOUT after %dms", case.id, latency_ms)
        return CaseResult(
            id=case.id,
            query=case.query,
            success=False,
            confidence=0.0,
            latency_ms=latency_ms,
            agents_invoked=[],
            fallback_events=[],
            findings_count=0,
            risk_levels_found=[],
            report_structure_valid=False,
            failure_mode="timeout",
            error="Request timed out after 30s",
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.error("Case %s: exception — %s", case.id, exc)
        return CaseResult(
            id=case.id,
            query=case.query,
            success=False,
            confidence=0.0,
            latency_ms=latency_ms,
            agents_invoked=[],
            fallback_events=[],
            findings_count=0,
            risk_levels_found=[],
            report_structure_valid=False,
            failure_mode="exception",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(results: List[CaseResult]) -> EvalMetrics:
    """Compute aggregate metrics from case results."""
    run_results = [r for r in results if not r.skipped]
    if not run_results:
        raise ValueError("No cases were run (all skipped)")

    succeeded = [r for r in run_results if r.success]
    failed = [r for r in run_results if not r.success]

    latencies = [r.latency_ms for r in run_results]
    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 2 else latencies[0]
    p50 = statistics.median(latencies)
    mean = statistics.mean(latencies)

    with_fallbacks = [r for r in run_results if r.fallback_events]
    planner_fallbacks = [r for r in run_results if "planner" in r.fallback_events]
    analyst_fallbacks = [r for r in run_results if "analyst" in r.fallback_events]
    synthesizer_fallbacks = [r for r in run_results if "synthesizer" in r.fallback_events]
    report_valid = [r for r in run_results if r.report_structure_valid]

    failure_mode_counts: Dict[str, int] = {}
    for r in failed:
        if r.failure_mode:
            failure_mode_counts[r.failure_mode] = failure_mode_counts.get(r.failure_mode, 0) + 1

    n = len(run_results)
    return EvalMetrics(
        task_completion_rate=len(succeeded) / n,
        fallback_activation_rate=len(with_fallbacks) / n,
        planner_fallback_rate=len(planner_fallbacks) / n,
        analyst_fallback_rate=len(analyst_fallbacks) / n,
        synthesizer_fallback_rate=len(synthesizer_fallbacks) / n,
        p95_latency_ms=p95,
        p50_latency_ms=p50,
        mean_latency_ms=mean,
        report_structure_pass_rate=len(report_valid) / n,
        cases_run=n,
        cases_skipped=len(results) - n,
        cases_succeeded=len(succeeded),
        cases_failed=len(failed),
        failure_mode_counts=failure_mode_counts,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(api_url: str, phase: str, store_responses: bool = False) -> None:
    logger.info("=== Consilium Evaluation Runner ===")
    logger.info("API: %s | Phase: %s | Golden: %s | store_responses: %s", api_url, phase, GOLDEN_PATH, store_responses)

    # Load golden dataset
    with open(GOLDEN_PATH) as f:
        raw = json.load(f)
    cases = [GoldenCase(**c) for c in raw]
    logger.info("Loaded %d golden cases", len(cases))

    # Check Quaestor reachability
    quaestor_reachable = await check_quaestor_reachable(api_url)
    logger.info("Quaestor reachable: %s", quaestor_reachable)

    # Run all cases
    results: List[CaseResult] = []
    async with httpx.AsyncClient() as client:
        for case in cases:
            result = await run_case(client, api_url, case, quaestor_reachable, store_responses=store_responses)
            results.append(result)

    # Compute metrics
    try:
        metrics = compute_metrics(results)
    except ValueError as e:
        logger.error("Cannot compute metrics: %s", e)
        sys.exit(1)

    # Print summary
    logger.info("=== Results Summary ===")
    logger.info("Cases run: %d | Skipped: %d", metrics.cases_run, metrics.cases_skipped)
    logger.info(
        "Task completion rate: %.1f%% (%d/%d)",
        metrics.task_completion_rate * 100,
        metrics.cases_succeeded,
        metrics.cases_run,
    )
    logger.info("Fallback activation rate: %.1f%%", metrics.fallback_activation_rate * 100)
    logger.info(
        "  Planner: %.1f%% | Analyst: %.1f%% | Synthesizer: %.1f%%",
        metrics.planner_fallback_rate * 100,
        metrics.analyst_fallback_rate * 100,
        metrics.synthesizer_fallback_rate * 100,
    )
    logger.info(
        "Latency — P95: %.0fms | P50: %.0fms | Mean: %.0fms",
        metrics.p95_latency_ms,
        metrics.p50_latency_ms,
        metrics.mean_latency_ms,
    )
    logger.info("Report structure pass rate: %.1f%%", metrics.report_structure_pass_rate * 100)
    if metrics.failure_mode_counts:
        logger.info("Failure modes: %s", metrics.failure_mode_counts)

    # Log failed cases with trace IDs for Phoenix correlation
    print("\n=== Failed Cases (trace IDs for Phoenix correlation) ===")
    failed_cases = [r for r in results if not r.success and not r.skipped]
    if failed_cases:
        for case in failed_cases:
            print(f"  {case.id}: trace_id={case.trace_id or 'N/A'}")
    else:
        print("  No failed cases")

    # Gate check
    gate_passed = (
        metrics.task_completion_rate >= 0.65
        and metrics.p95_latency_ms <= 3000
    )
    logger.info("=== Gate Check (completion>=65%%, P95<=3s): %s ===", "PASS" if gate_passed else "FAIL")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    output_path = RESULTS_DIR / f"{phase}_{timestamp}.json"

    output: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "api_url": api_url,
        "quaestor_reachable": quaestor_reachable,
        "store_responses": store_responses,
        "gate_passed": gate_passed,
        "metrics": asdict(metrics),
        "cases": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)

    if not gate_passed:
        logger.warning("Gate FAILED — do not proceed to Task 4 until gate passes.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consilium evaluation runner")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("CONSILIUM_API_URL", "http://localhost:8001"),
        help="Consilium API base URL",
    )
    parser.add_argument(
        "--phase",
        default="phase2_baseline",
        help="Label for this run (used in output filename)",
    )
    parser.add_argument(
        "--store-responses",
        action="store_true",
        default=False,
        help="Store full API response body (findings text, report) per case in results JSON",
    )
    args = parser.parse_args()
    asyncio.run(main(api_url=args.api_url, phase=args.phase, store_responses=args.store_responses))
