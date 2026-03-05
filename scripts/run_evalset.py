#!/usr/bin/env python3
"""Batch runner for evalset topics — produces aggregate_metrics.json."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    evalset_path = Path("evalset/topics.jsonl")
    if not evalset_path.exists():
        print("ERROR: evalset/topics.jsonl not found")
        sys.exit(1)

    topics = []
    for line in evalset_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            topics.append(json.loads(line))

    print(f"Running evalset: {len(topics)} topics")
    batch_dir = Path("runs_batch")
    batch_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failures = 0

    for i, entry in enumerate(topics):
        topic = entry["topic"]
        mode = entry.get("mode", "fast")
        sources = entry.get("sources", "demo")
        print(f"\n[{i + 1}/{len(topics)}] {topic} (mode={mode}, sources={sources})")

        cmd = [
            sys.executable, "-m", "researchops", "run", topic,
            "--mode", mode,
            "--allow-net", "false",
            "--llm", "none",
            "--sources", sources,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            failures += 1
            continue

        run_dirs = sorted(Path("runs").glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not run_dirs:
            print("  FAILED (no run dir created)")
            failures += 1
            continue

        latest = run_dirs[0]
        eval_path = latest / "eval.json"
        if eval_path.exists():
            eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
            eval_data["topic"] = topic
            eval_data["run_dir"] = str(latest)
            results.append(eval_data)
            cov = eval_data.get("citation_coverage", 0)
            print(f"  OK — citation_coverage={cov}")
        else:
            print("  FAILED (no eval.json)")
            failures += 1

    aggregate = _aggregate(results)
    aggregate["total_topics"] = len(topics)
    aggregate["successful_runs"] = len(results)
    aggregate["failed_runs"] = failures

    agg_path = batch_dir / "aggregate_metrics.json"
    agg_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nAggregate metrics written to {agg_path}")
    print(f"  Successful: {len(results)}/{len(topics)} | Failed: {failures}")

    if failures > 0:
        sys.exit(1)


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {"note": "no successful runs"}

    numeric_keys = [
        "citation_coverage", "reproduction_rate", "unsupported_claim_rate",
        "cache_hit_rate", "latency_sec", "tool_calls", "steps",
        "conflict_count", "artifacts_count", "papers_per_rq",
        "low_quality_source_rate", "section_nonempty_rate",
    ]
    agg: dict = {}
    for key in numeric_keys:
        values = [r.get(key, 0) for r in results if isinstance(r.get(key), (int, float))]
        if values:
            agg[key] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    return agg


if __name__ == "__main__":
    main()
