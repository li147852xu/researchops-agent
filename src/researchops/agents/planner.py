from __future__ import annotations

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
