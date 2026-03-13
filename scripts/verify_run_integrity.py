#!/usr/bin/env python3
"""Verify the data integrity of a completed run directory (v1.0.0)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_BOILERPLATE_RE = re.compile(
    r"doctype|meta charset|stylesheet|viewport|function\(|"
    r"padding\s*:|margin\s*:|rgb\(|\d+px;|addeventlistener|queryselector",
    re.IGNORECASE,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: verify_run_integrity.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"FAIL: run dir {run_dir} does not exist")
        sys.exit(1)

    ok = True

    sources_path = run_dir / "sources.jsonl"
    if sources_path.exists():
        for i, line in enumerate(sources_path.read_text(encoding="utf-8").strip().splitlines()):
            if not line.strip():
                continue
            src = json.loads(line)
            sid = src.get("source_id", f"line_{i}")
            lp = src.get("local_path", "")
            if not lp:
                print(f"FAIL: source {sid} has empty local_path")
                ok = False
            elif not Path(lp).exists():
                print(f"FAIL: source {sid} local_path does not exist: {lp}")
                ok = False
            if not src.get("hash"):
                print(f"FAIL: source {sid} has empty hash")
                ok = False

            if lp and Path(lp).exists():
                detected = _detect_type(Path(lp).read_bytes()[:512])
                src_type = src.get("type", "")
                if detected != "unknown" and detected != src_type:
                    print(f"WARN: source {sid} type={src_type} but magic-number says {detected}")
    else:
        print("WARN: sources.jsonl not found")

    notes_dir = run_dir / "notes"
    if notes_dir.exists():
        for f in notes_dir.iterdir():
            if f.suffix.lower() in (".html", ".htm", ".pdf"):
                print(f"FAIL: raw file in notes/: {f.name}")
                ok = False
            if f.is_file() and f.suffix.lower() != ".json":
                print(f"WARN: non-json file in notes/: {f.name}")

    report_path = run_dir / "report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        content_lines = [ln for ln in report.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        boilerplate_count = sum(1 for ln in content_lines if _BOILERPLATE_RE.search(ln))
        if boilerplate_count > 2:
            print(f"FAIL: report contains {boilerplate_count} lines with boilerplate/code keywords")
            ok = False
    else:
        print("WARN: report.md not found")

    failures_path = run_dir / "failures.json"
    if failures_path.exists():
        try:
            data = json.loads(failures_path.read_text(encoding="utf-8"))
            print(f"  INFO: failures.json contains {len(data)} failed sources")
        except Exception:
            print("WARN: failures.json is not valid JSON")

    qa_report = run_dir / "qa_report.json"
    if qa_report.exists():
        try:
            data = json.loads(qa_report.read_text(encoding="utf-8"))
            checks = data.get("checks", [])
            all_passed = data.get("all_passed", False)
            print(f"  INFO: qa_report.json has {len(checks)} checks, all_passed={all_passed}")
        except Exception:
            print("WARN: qa_report.json is not valid JSON")

    retrieval_idx = run_dir / "retrieval_index.json"
    if retrieval_idx.exists():
        print("  INFO: retrieval_index.json exists")
    else:
        print("  INFO: no retrieval_index.json (retrieval may not have been used)")

    if ok:
        print("PASS: run integrity check succeeded")
    else:
        print("\nFAIL: run integrity check failed")
        sys.exit(1)


def _detect_type(data: bytes) -> str:
    if data[:5] == b"%PDF-":
        return "pdf"
    head = data[:512].lstrip()
    if head[:1] == b"<" or b"<html" in head[:256].lower() or b"<!doctype" in head[:256].lower():
        return "html"
    return "unknown"


if __name__ == "__main__":
    main()
