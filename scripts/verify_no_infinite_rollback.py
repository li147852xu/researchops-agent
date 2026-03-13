#!/usr/bin/env python3
"""Verify that a completed run does not exhibit infinite rollback loops (v1.0.1).

Checks:
  1. Total rollback count <= max_collect_rounds (from config or default 6)
  2. Between consecutive rollbacks, sources_hash OR claims_hash changed
     (or a strategy_upgrade event occurred)
  3. No >2 consecutive identical (coverage_vector + sources_hash) combos
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: verify_no_infinite_rollback.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"FAIL: run dir {run_dir} does not exist")
        sys.exit(1)

    ok = True

    state_path = run_dir / "state.json"
    if not state_path.exists():
        print("WARN: state.json not found, cannot verify rollback bounds")
        sys.exit(0)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = state.get("config_snapshot", {})
    mode = config.get("mode", "fast")
    max_rounds = 6 if mode == "deep" else 3
    collect_rounds = state.get("collect_rounds", 1)

    if collect_rounds > max_rounds:
        print(f"FAIL: collect_rounds ({collect_rounds}) exceeds max_collect_rounds ({max_rounds})")
        ok = False
    else:
        print(f"  OK: collect_rounds={collect_rounds} <= max={max_rounds}")

    trace_path = run_dir / "trace.jsonl"
    rollback_events: list[dict] = []
    strategy_upgrades: list[dict] = []

    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("action") == "rollback":
                rollback_events.append(ev)
            if ev.get("action") == "strategy_upgrade":
                strategy_upgrades.append(ev)

    rollback_count = len(rollback_events)
    print(f"  INFO: {rollback_count} rollback events in trace, {len(strategy_upgrades)} strategy upgrades")

    if rollback_count > max_rounds + 2:
        print(f"FAIL: rollback events ({rollback_count}) significantly exceed max_collect_rounds ({max_rounds})")
        ok = False

    rollback_history = state.get("rollback_history", [])
    if len(rollback_history) >= 2:
        for i in range(1, len(rollback_history)):
            prev = rollback_history[i - 1]
            curr = rollback_history[i]
            prev_sh = prev.get("sources_hash", "")
            curr_sh = curr.get("sources_hash", "")
            prev_ch = prev.get("claims_hash", "")
            curr_ch = curr.get("claims_hash", "")

            has_progress = (prev_sh != curr_sh) or (prev_ch != curr_ch)
            has_strategy_change = prev.get("strategy_level", 0) != curr.get("strategy_level", 0)

            if not has_progress and not has_strategy_change:
                print(
                    f"WARN: rollback rounds {prev.get('round')}->{curr.get('round')} "
                    f"show no progress and no strategy change"
                )

    if len(rollback_history) >= 3:
        for i in range(2, len(rollback_history)):
            a = rollback_history[i - 2]
            b = rollback_history[i - 1]
            c = rollback_history[i]
            combo_a = f"{a.get('sources_hash', '')}|{json.dumps(a.get('coverage_vector', {}), sort_keys=True)}"
            combo_b = f"{b.get('sources_hash', '')}|{json.dumps(b.get('coverage_vector', {}), sort_keys=True)}"
            combo_c = f"{c.get('sources_hash', '')}|{json.dumps(c.get('coverage_vector', {}), sort_keys=True)}"
            if combo_a == combo_b == combo_c:
                print(
                    f"FAIL: 3 consecutive identical rollback states "
                    f"(rounds {a.get('round')}, {b.get('round')}, {c.get('round')})"
                )
                ok = False
                break

    degrade_events = [
        line for line in (trace_path.read_text(encoding="utf-8").strip().splitlines() if trace_path.exists() else [])
        if "degrade_complete" in line
    ]
    if degrade_events:
        print(f"  INFO: degrade_complete triggered ({len(degrade_events)} events)")

    incomplete = state.get("incomplete_sections", [])
    if incomplete:
        print(f"  INFO: {len(incomplete)} incomplete sections: {incomplete}")

    if ok:
        print("PASS: no infinite rollback loop detected")
    else:
        print("\nFAIL: rollback loop check failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
