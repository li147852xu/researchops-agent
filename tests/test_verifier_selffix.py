from __future__ import annotations

import json
from pathlib import Path

from researchops.agents.verifier import VerifierAgent
from researchops.sandbox.proc import SubprocessSandbox


def test_selffix_import_error(tmp_run_dir: Path):
    """Verifier should fix a script with a bad import and succeed on retry."""
    code_dir = tmp_run_dir / "code"
    artifacts_dir = tmp_run_dir / "artifacts"
    notes_dir = tmp_run_dir / "notes"

    note = {
        "source_id": "s1",
        "claims": [
            {"claim_id": "s1_c0", "text": "Test claim", "evidence_spans": ["evidence"], "supports_rq": ["rq_test"]}
        ],
    }
    (notes_dir / "s1.json").write_text(json.dumps(note), encoding="utf-8")

    broken_script = code_dir / "verify_integrity_rq_test.py"
    broken_script.write_text(
        "import nonexistent_module_xyz\nprint('should not reach')\n",
        encoding="utf-8",
    )

    from researchops.agents.base import RunContext
    from researchops.config import RunConfig
    from researchops.models import StateSnapshot
    from researchops.reasoning.none import NoneReasoner
    from researchops.registry.builtin import register_builtin_tools
    from researchops.registry.manager import ToolRegistry
    from researchops.trace import TraceLogger

    config = RunConfig(topic="test", allow_net=False, run_dir=tmp_run_dir)
    registry = ToolRegistry()
    register_builtin_tools(registry)
    registry.grant_permissions({"sandbox"})
    trace = TraceLogger(tmp_run_dir / "trace.jsonl")
    sandbox = SubprocessSandbox()

    ctx = RunContext(
        run_dir=tmp_run_dir,
        config=config,
        state=StateSnapshot(run_id="test"),
        registry=registry,
        trace=trace,
        sandbox=sandbox,
        reasoner=NoneReasoner(),
    )

    verifier = VerifierAgent()
    result = verifier._run_verification(ctx, "rq_test", "integrity", code_dir, artifacts_dir)

    assert result["success"] is True
    assert result["attempts"] >= 1


def test_selffix_triggers_on_failure(tmp_run_dir: Path):
    """A deliberately broken script triggers the fix mechanism."""
    code_dir = tmp_run_dir / "code"
    script = code_dir / "broken.py"
    script.write_text("import bogus_module_999\nprint('hi')\n", encoding="utf-8")

    from researchops.trace import TraceLogger

    class FakeCtx:
        trace = TraceLogger(tmp_run_dir / "trace2.jsonl")

    verifier = VerifierAgent()
    verifier._fix_script(script, "ModuleNotFoundError: No module named 'bogus_module_999'", FakeCtx())

    fixed = script.read_text(encoding="utf-8")
    assert "bogus_module_999" not in fixed or "# removed" in fixed
