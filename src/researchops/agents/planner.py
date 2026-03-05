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

_GENERIC_BUCKETS = [
    {"bucket_id": "bkt_foundations", "bucket_name": "foundations", "description": "Core concepts, definitions, and theoretical underpinnings"},
    {"bucket_id": "bkt_methods", "bucket_name": "methods", "description": "Key methods, algorithms, and techniques"},
    {"bucket_id": "bkt_challenges", "bucket_name": "challenges", "description": "Challenges, limitations, and open problems"},
    {"bucket_id": "bkt_future", "bucket_name": "future", "description": "Future directions and emerging trends"},
]


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
        checklist = self._build_coverage_checklist(topic, ctx)

        plan = PlanOutput(
            topic=topic,
            research_questions=rqs,
            outline=outline,
            acceptance_threshold=0.7 if ctx.config.mode.value == "fast" else 0.85,
            coverage_checklist=checklist,
        )

        plan_path = ctx.run_dir / "plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        ctx.trace.log(
            stage="PLAN",
            agent=self.name,
            action="complete",
            output_summary=f"{len(rqs)} RQs, {len(checklist)} buckets, {len(outline)} sections",
        )
        return AgentResult(success=True, message=f"Plan created with {len(rqs)} RQs, {len(checklist)} buckets")

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

    def _build_coverage_checklist(self, topic: str, ctx: RunContext) -> list[dict]:
        is_deep = ctx.config.mode.value == "deep"
        min_sources = 3 if is_deep else 1
        min_claims = 5 if is_deep else 2

        if ctx.reasoner.is_llm:
            buckets = self._llm_buckets(topic, ctx)
            if buckets:
                for b in buckets:
                    b.setdefault("min_sources", min_sources)
                    b.setdefault("min_claims", min_claims)
                return buckets

        return self._rule_based_buckets(topic, min_sources, min_claims)

    def _llm_buckets(self, topic: str, ctx: RunContext) -> list[dict]:
        prompt = (
            f"Generate 5-8 topic-specific coverage buckets for a literature survey on: {topic}\n"
            f"Each bucket represents a sub-area that must be covered.\n"
            f"Return JSON: {{\"buckets\": [{{\"bucket_id\": \"bkt_xxx\", \"bucket_name\": \"short name\", "
            f"\"description\": \"what this bucket covers\"}}]}}\n"
            f"Make buckets specific to the topic, not generic."
        )
        try:
            raw = ctx.reasoner.complete_text(prompt, trace=ctx.trace)
            start = raw.find("{")
            data = json.loads(raw[start:]) if start >= 0 else json.loads(raw)
            buckets = data.get("buckets", [])
            if len(buckets) >= 3:
                return buckets[:8]
        except Exception:
            pass
        return []

    def _rule_based_buckets(
        self, topic: str, min_sources: int, min_claims: int,
    ) -> list[dict]:
        topic_lower = topic.lower()

        specialized: dict[str, list[dict]] = {
            "deep learning": [
                {"bucket_id": "bkt_architectures", "bucket_name": "architectures", "description": "Neural network architectures (CNN, RNN, Transformer, etc.)"},
                {"bucket_id": "bkt_optimization", "bucket_name": "optimization", "description": "Training methods, optimizers, and convergence"},
                {"bucket_id": "bkt_generalization", "bucket_name": "generalization", "description": "Generalization, regularization, and overfitting"},
                {"bucket_id": "bkt_robustness", "bucket_name": "robustness", "description": "Adversarial robustness and model reliability"},
                {"bucket_id": "bkt_scaling", "bucket_name": "scaling", "description": "Model scaling, efficiency, and compute trade-offs"},
                {"bucket_id": "bkt_applications", "bucket_name": "applications", "description": "Real-world applications and deployment"},
            ],
            "quantum computing": [
                {"bucket_id": "bkt_qubits", "bucket_name": "qubits", "description": "Qubit technologies and hardware implementations"},
                {"bucket_id": "bkt_algorithms", "bucket_name": "algorithms", "description": "Quantum algorithms and computational advantage"},
                {"bucket_id": "bkt_error_correction", "bucket_name": "error_correction", "description": "Quantum error correction and fault tolerance"},
                {"bucket_id": "bkt_applications", "bucket_name": "applications", "description": "Applications in cryptography, simulation, optimization"},
                {"bucket_id": "bkt_software", "bucket_name": "software", "description": "Quantum programming languages and SDKs"},
            ],
            "natural language processing": [
                {"bucket_id": "bkt_models", "bucket_name": "models", "description": "Language models and pre-training approaches"},
                {"bucket_id": "bkt_understanding", "bucket_name": "understanding", "description": "Text understanding and semantic analysis"},
                {"bucket_id": "bkt_generation", "bucket_name": "generation", "description": "Text generation and summarization"},
                {"bucket_id": "bkt_evaluation", "bucket_name": "evaluation", "description": "Benchmarks and evaluation metrics"},
                {"bucket_id": "bkt_ethics", "bucket_name": "ethics", "description": "Bias, fairness, and ethical considerations"},
            ],
        }

        for key, buckets in specialized.items():
            if key in topic_lower or all(w in topic_lower for w in key.split()):
                for b in buckets:
                    b["min_sources"] = min_sources
                    b["min_claims"] = min_claims
                return buckets

        result = []
        for b in _GENERIC_BUCKETS:
            entry = dict(b)
            entry["min_sources"] = min_sources
            entry["min_claims"] = min_claims
            entry["description"] = f"{entry['description']} related to {topic}"
            result.append(entry)
        return result
