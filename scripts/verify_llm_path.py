#!/usr/bin/env python3
"""Verify that LLM was actually invoked during a run by inspecting trace.jsonl and eval.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: verify_llm_path.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"FAIL: run dir {run_dir} does not exist")
        sys.exit(1)

    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        print(f"FAIL: {trace_path} not found")
        sys.exit(1)

    events = []
    for line in trace_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            events.append(json.loads(line))

    llm_calls = [e for e in events if e.get("action") == "llm.call"]
    llm_results = [e for e in events if e.get("action") == "llm.result"]
    llm_errors = [e for e in events if e.get("action") == "llm.error"]

    ok = True

    print(f"=== LLM Path Verification for {run_dir} ===")
    print(f"  Total trace events: {len(events)}")
    print(f"  llm.call events:    {len(llm_calls)}")
    print(f"  llm.result events:  {len(llm_results)}")
    print(f"  llm.error events:   {len(llm_errors)}")

    if not llm_calls:
        print("\nFAIL: No llm.call events found in trace.jsonl")
        print("  Possible causes:")
        print("  - Agents not calling ctx.reasoner.complete_text/complete_json")
        print("  - Reasoner not injected into RunContext")
        print("  - LLM key not resolved (check OPENAI_API_KEY / LLM_API_KEY / DEEPSEEK_API_KEY)")
        print("  - --llm set to 'none' or silently fell back to NoneReasoner")
        ok = False

    if llm_calls and not llm_results and not llm_errors:
        print("\nWARN: llm.call found but no llm.result or llm.error — possible trace logging gap")

    for call in llm_calls[:3]:
        meta = call.get("meta", {})
        print("\n  Sample llm.call:")
        print(f"    provider_label: {meta.get('provider_label', 'N/A')}")
        print(f"    base_url:       {meta.get('base_url', 'N/A')}")
        print(f"    model:          {meta.get('model', 'N/A')}")
        print(f"    input_chars:    {meta.get('input_chars', 'N/A')}")

    for result in llm_results[:3]:
        meta = result.get("meta", {})
        print("\n  Sample llm.result:")
        print(f"    output_chars:  {meta.get('output_chars', 'N/A')}")
        print(f"    tokens:        {meta.get('tokens', 'N/A')}")
        print(f"    latency_ms:    {meta.get('latency_ms', 'N/A')}")

    eval_path = run_dir / "eval.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        print("\n  eval.json:")
        print(f"    llm_enabled:          {eval_data.get('llm_enabled')}")
        print(f"    llm_provider_label:   {eval_data.get('llm_provider_label')}")
        print(f"    estimated_tokens:     {eval_data.get('estimated_tokens')}")
        print(f"    estimated_cost_usd:   {eval_data.get('estimated_cost_usd')}")
        print(f"    estimate_method:      {eval_data.get('estimate_method')}")

        if not eval_data.get("llm_enabled"):
            print("\n  WARN: eval.json shows llm_enabled=false")
    else:
        print(f"\n  WARN: {eval_path} not found")

    if ok:
        print("\nPASS: LLM path verification succeeded")
    else:
        print("\nFAIL: LLM path verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
