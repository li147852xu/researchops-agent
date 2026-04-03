#!/usr/bin/env python3
"""Verify research quality metrics for a run directory."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: verify_research_quality.py <run_dir>")
        return 1

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"FAIL: run_dir does not exist: {run_dir}")
        return 1

    failures: list[str] = []

    state_path = run_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        collect_rounds = state.get("collect_rounds", 1)
        max_rounds = 6
        if collect_rounds > max_rounds:
            failures.append(f"collect_rounds ({collect_rounds}) > max ({max_rounds})")

        rollback_history = state.get("rollback_history", [])
        if len(rollback_history) >= 3:
            hashes = [h.get("sources_hash", "") for h in rollback_history]
            for i in range(len(hashes) - 2):
                if hashes[i] == hashes[i + 1] == hashes[i + 2] and hashes[i]:
                    failures.append(f"Infinite progress loop: same sources_hash {hashes[i]} repeated 3 times")
                    break

    sources_path = run_dir / "sources.jsonl"
    if sources_path.exists():
        for line_num, line in enumerate(sources_path.read_text(encoding="utf-8").strip().splitlines(), 1):
            if not line.strip():
                continue
            try:
                src = json.loads(line)
            except json.JSONDecodeError:
                failures.append(f"sources.jsonl line {line_num}: invalid JSON")
                continue
            if not src.get("local_path"):
                failures.append(f"sources.jsonl line {line_num}: empty local_path for {src.get('source_id', '?')}")
            src_type = src.get("type", "")
            detail = src.get("source_type_detail", "")
            if src_type and detail and src_type == "pdf" and "html" in detail:
                    failures.append(f"sources.jsonl: type/detail mismatch: {src_type} vs {detail}")
    else:
        failures.append("sources.jsonl missing")

    notes_dir = run_dir / "notes"
    if notes_dir.exists():
        for f in notes_dir.iterdir():
            if f.suffix != ".json":
                failures.append(f"notes/ contains non-json file: {f.name}")
    else:
        failures.append("notes/ directory missing")

    eval_path = run_dir / "eval.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))

        bucket_cov = ev.get("bucket_coverage_rate", 0.0)
        if bucket_cov < 0.6:
            failures.append(f"bucket_coverage_rate ({bucket_cov:.2f}) below 0.6 threshold")

        relevance_avg = ev.get("relevance_avg", 0.0)
        if relevance_avg < 0.3:
            failures.append(f"relevance_avg ({relevance_avg:.2f}) below 0.3 threshold")

        unsupported = ev.get("unsupported_claim_rate", 0.0)
        if unsupported > 0.20:
            failures.append(f"unsupported_claim_rate ({unsupported:.2f}) above 0.20 threshold")
    else:
        failures.append("eval.json missing")

    report_path = run_dir / "report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        import re

        sections = re.split(r"^##\s+", report, flags=re.MULTILINE)
        for sec in sections[1:]:
            heading = sec.split("\n", 1)[0].strip()
            markers = re.findall(r"\[@(\w+)\]", sec)
            if len(markers) >= 3:
                from collections import Counter

                mc = Counter(markers).most_common(1)
                if mc:
                    dominant_id, count = mc[0]
                    ratio = count / len(markers)
                    if ratio > 0.35:
                        failures.append(f"Section '{heading}': source {dominant_id} dominates at {ratio:.0%}")
    else:
        failures.append("report.md missing")

    decisions_path = run_dir / "decisions.jsonl"
    if decisions_path.exists():
        for line_num, line in enumerate(decisions_path.read_text(encoding="utf-8").strip().splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                failures.append(f"decisions.jsonl line {line_num}: invalid JSON")
    # decisions.jsonl is optional (only created when rollbacks occur)

    if failures:
        print(f"FAIL: {len(failures)} quality issues found:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS: All research quality checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
