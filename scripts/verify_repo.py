#!/usr/bin/env python3
"""Verification script for ResearchOps Agent repo integrity (v1.0.0)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not ok:
        FAILURES.append(label)


def check_md_files() -> None:
    print("\n=== Markdown file constraint ===")
    md_files = []
    for f in REPO_ROOT.rglob("*.md"):
        rel = f.relative_to(REPO_ROOT)
        parts = rel.parts
        if any(p.startswith(".") for p in parts):
            continue
        if "runs" in parts or "__pycache__" in parts or "runs_batch" in parts:
            continue
        md_files.append(str(rel))
    allowed = {"README.md", "CHANGELOG.md"}
    extra = set(md_files) - allowed
    check("Only README.md and CHANGELOG.md exist", len(extra) == 0, f"extra: {extra}" if extra else "")


def check_cli_help() -> None:
    print("\n=== CLI help contains v1.0.0 parameters ===")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from researchops.cli import app; app(['run', '--help'])"],
            capture_output=True, text=True, timeout=10,
            cwd=str(REPO_ROOT),
        )
        help_text = result.stdout + result.stderr
    except Exception as e:
        check("CLI help accessible", False, str(e))
        return

    check("CLI help accessible", result.returncode == 0)
    for param in [
        "--llm-base-url", "--llm-provider-label", "--llm-headers",
        "--llm-api-key", "--seed", "--sources", "--retrieval", "--embedder",
    ]:
        check(f"CLI contains {param}", param in help_text)


def check_evalset() -> None:
    print("\n=== Evalset ===")
    evalset_path = REPO_ROOT / "evalset" / "topics.jsonl"
    check("evalset/topics.jsonl exists", evalset_path.exists())
    if evalset_path.exists():
        lines = [ln for ln in evalset_path.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        check("evalset has >= 20 topics", len(lines) >= 20, f"found {len(lines)}")


def check_tools_registered() -> None:
    print("\n=== Tool registration ===")
    try:
        from researchops.registry.builtin import register_builtin_tools
        from researchops.registry.manager import ToolRegistry

        reg = ToolRegistry()
        register_builtin_tools(reg)
        for tool_name in ["web_search", "fetch", "parse", "sandbox_exec", "cite", "arxiv_search", "arxiv_download_pdf"]:
            check(f"Tool '{tool_name}' registered", tool_name in reg._tools)
    except Exception as e:
        check("Tool registry initialization", False, str(e))


def check_runs_structure() -> None:
    print("\n=== Run artifact structure ===")
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        print("  (no runs/ directory found, skipping structure check)")
        return

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        print("  (no run directories found)")
        return

    latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
    print(f"  Checking latest run: {latest.name}")

    required = [
        "plan.json", "sources.jsonl", "report.md", "report_index.json",
        "trace.jsonl", "eval.json", "state.json",
    ]
    required_dirs = ["notes", "code", "artifacts", "downloads"]

    for fname in required:
        check(f"  {fname} exists", (latest / fname).exists())

    for dname in required_dirs:
        check(f"  {dname}/ exists", (latest / dname).is_dir())

    eval_path = latest / "eval.json"
    if eval_path.exists():
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
            v1_fields = [
                "conflict_count", "plan_refinement_count", "collect_rounds",
                "artifacts_count", "llm_provider_label",
                "papers_per_rq", "low_quality_source_rate", "section_nonempty_rate",
            ]
            for field in v1_fields:
                check(f"  eval.json has '{field}'", field in data)
        except Exception as e:
            check("  eval.json is valid JSON", False, str(e))

    qa_report = latest / "qa_report.json"
    check("  qa_report.json exists", qa_report.exists())

    qa_conflicts = latest / "qa_conflicts.json"
    check("  qa_conflicts.json exists", qa_conflicts.exists())


def main() -> int:
    print("=" * 60)
    print("ResearchOps Agent v1.0.0 — Repo Verification")
    print("=" * 60)

    check_md_files()
    check_cli_help()
    check_evalset()
    check_tools_registered()
    check_runs_structure()

    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  ✗ {f}")
        return 1
    else:
        print("ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
