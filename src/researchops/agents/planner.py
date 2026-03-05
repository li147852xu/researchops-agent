from __future__ import annotations

import json
import re

from researchops.agents.base import AgentBase, RunContext
from researchops.models import (
    AgentResult,
    OutlineSection,
    PlanOutput,
    ResearchQuestion,
)


class PlannerAgent(AgentBase):
    name = "planner"

    def execute(self, ctx: RunContext) -> AgentResult:
        topic = ctx.config.topic
        ctx.trace.log(stage="PLAN", agent=self.name, action="start", input_summary=topic)

        if ctx.reasoner.is_llm:
            rqs = self._decompose_with_llm(topic, ctx)
        else:
            rqs = self._decompose_topic(topic, ctx.config.mode.value)
        outline = self._build_outline(rqs)

        plan = PlanOutput(
            topic=topic,
            research_questions=rqs,
            outline=outline,
            acceptance_threshold=0.7 if ctx.config.mode.value == "fast" else 0.85,
        )

        plan_path = ctx.run_dir / "plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        ctx.trace.log(
            stage="PLAN",
            agent=self.name,
            action="complete",
            output_summary=f"{len(rqs)} research questions, {len(outline)} sections",
        )
        return AgentResult(success=True, message=f"Plan created with {len(rqs)} RQs")

    def _decompose_with_llm(self, topic: str, ctx: RunContext) -> list[ResearchQuestion]:
        mode = ctx.config.mode.value
        num_rqs = 4 if mode == "deep" else 3
        prompt = (
            f"Generate {num_rqs} research questions for a literature survey on: {topic}\n"
            f"Return a JSON object with a single key 'questions' containing a list of objects, "
            f"each with: rq_id (string like rq_1, rq_2...), text (the question), "
            f"priority (int 1-3), needs_verification (bool, true for quantitative/empirical questions).\n"
            f"Make questions specific, diverse, and cover: overview/definition, current state, "
            f"challenges/limitations{', and future directions' if mode == 'deep' else ''}."
        )
        try:
            raw = ctx.reasoner.complete_text(prompt, trace=ctx.trace)
            data = json.loads(raw) if raw.strip().startswith("{") else json.loads("{" + raw.split("{", 1)[-1])
            questions = data.get("questions", [])
            rqs = []
            for q in questions[:num_rqs]:
                rqs.append(ResearchQuestion(
                    rq_id=q.get("rq_id", f"rq_{len(rqs)}"),
                    text=q.get("text", ""),
                    priority=q.get("priority", 1),
                    needs_verification=q.get("needs_verification", False),
                ))
            if rqs:
                return rqs
        except Exception:
            pass
        return self._decompose_topic(topic, mode)

    def _decompose_topic(self, topic: str, mode: str) -> list[ResearchQuestion]:
        words = re.split(r"\s+", topic.strip())
        core = " ".join(words[:6]) if len(words) > 6 else topic

        rqs = [
            ResearchQuestion(
                rq_id="rq_overview",
                text=f"What is {core} and what are its key components?",
                priority=1,
            ),
            ResearchQuestion(
                rq_id="rq_state",
                text=f"What is the current state of research on {core}?",
                priority=2,
                needs_verification=True,
            ),
            ResearchQuestion(
                rq_id="rq_challenges",
                text=f"What are the main challenges and limitations of {core}?",
                priority=2,
            ),
        ]

        if mode == "deep":
            rqs.append(
                ResearchQuestion(
                    rq_id="rq_future",
                    text=f"What are future directions for {core}?",
                    priority=3,
                    needs_verification=True,
                )
            )

        return rqs

    def _build_outline(self, rqs: list[ResearchQuestion]) -> list[OutlineSection]:
        sections = [
            OutlineSection(heading="Introduction", rq_refs=[]),
        ]
        for rq in rqs:
            sections.append(
                OutlineSection(
                    heading=rq.text.rstrip("?").split("What ")[-1].capitalize(),
                    rq_refs=[rq.rq_id],
                )
            )
        sections.append(OutlineSection(heading="Conclusion", rq_refs=[r.rq_id for r in rqs]))
        return sections
