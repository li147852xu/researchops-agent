"""Centralised prompt templates with system prompts, few-shot examples, and
output schemas for every LLM call site in the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptTemplate:
    """Typed container for a single prompt template."""

    name: str
    system: str
    user: str
    few_shot: list[dict[str, str]] = field(default_factory=list)

    def render(self, **kwargs: str) -> tuple[str, str]:
        """Return (system_message, user_message) with placeholders filled."""
        sys_msg = self.system
        user_parts: list[str] = []
        if self.few_shot:
            for ex in self.few_shot:
                user_parts.append(f"Example input:\n{ex['input']}\n\nExample output:\n{ex['output']}")
            user_parts.append("---\nNow handle the real input:\n")
        user_parts.append(self.user.format(**kwargs))
        return sys_msg, "\n\n".join(user_parts)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_RQS = PromptTemplate(
    name="planner_rqs",
    system=(
        "You are a research planning assistant. You decompose a broad research "
        "topic into specific, answerable research questions for a literature survey. "
        "Always return valid JSON."
    ),
    user=(
        "Generate {num_rqs} research questions for a literature survey on: {topic}\n\n"
        "Requirements:\n"
        "- Each question should target a distinct aspect (definition, methods, "
        "challenges, trends, applications)\n"
        "- Include at least one empirical/quantitative question (needs_verification=true)\n"
        "- Prioritise specificity over breadth\n\n"
        "Return JSON:\n"
        '{{"questions": [{{"rq_id": "rq_1", "text": "...", "priority": 1, '
        '"needs_verification": false}}, ...]}}'
    ),
    few_shot=[
        {
            "input": "Topic: transformer architectures, num_rqs: 3",
            "output": (
                '{"questions": ['
                '{"rq_id": "rq_1", "text": "What are the core architectural components of '
                'transformer models and how do they differ from RNN/CNN architectures?", '
                '"priority": 1, "needs_verification": false}, '
                '{"rq_id": "rq_2", "text": "How do recent efficient-transformer variants '
                '(e.g., FlashAttention, Mamba) compare in throughput and accuracy on standard '
                'benchmarks?", "priority": 2, "needs_verification": true}, '
                '{"rq_id": "rq_3", "text": "What are the main scalability challenges and '
                'open problems in training billion-parameter transformers?", '
                '"priority": 2, "needs_verification": false}]}'
            ),
        }
    ],
)

PLANNER_BUCKETS = PromptTemplate(
    name="planner_buckets",
    system=(
        "You are a research planning assistant. Generate topic-specific coverage "
        "buckets that define the sub-areas a survey must address. Return valid JSON."
    ),
    user=(
        "Generate 5-8 topic-specific coverage buckets for a literature survey on: {topic}\n\n"
        "Each bucket represents a mandatory sub-area.\n"
        "Return JSON: {{\"buckets\": [{{\"bucket_id\": \"bkt_xxx\", "
        "\"bucket_name\": \"short name\", "
        "\"description\": \"what this bucket covers\"}}]}}\n"
        "Make buckets specific to the topic, not generic."
    ),
    few_shot=[
        {
            "input": "Topic: reinforcement learning",
            "output": (
                '{"buckets": ['
                '{"bucket_id": "bkt_mdp", "bucket_name": "MDP foundations", '
                '"description": "Markov decision processes, Bellman equations, value functions"}, '
                '{"bucket_id": "bkt_policy", "bucket_name": "policy methods", '
                '"description": "Policy gradient, PPO, TRPO, actor-critic"}, '
                '{"bucket_id": "bkt_model_based", "bucket_name": "model-based RL", '
                '"description": "World models, Dreamer, MuZero"}, '
                '{"bucket_id": "bkt_exploration", "bucket_name": "exploration", '
                '"description": "Curiosity-driven, count-based, intrinsic motivation"}, '
                '{"bucket_id": "bkt_multi_agent", "bucket_name": "multi-agent", '
                '"description": "MARL, cooperative/competitive settings, communication"}, '
                '{"bucket_id": "bkt_applications", "bucket_name": "applications", '
                '"description": "Robotics, games, NLP, recommendation systems"}]}'
            ),
        }
    ],
)

PLANNER_HEADINGS = PromptTemplate(
    name="planner_headings",
    system=(
        "You are a research writing assistant. Convert research questions into "
        "concise section headings (4-8 words each) for an academic survey report. "
        "Always return valid JSON."
    ),
    user=(
        "Convert these research questions into short section headings (4-8 words each).\n\n"
        "Research questions:\n{rq_list}\n\n"
        "Return JSON: {{\"headings\": [{{\"rq_id\": \"...\", \"heading\": \"...\"}}]}}\n"
        "Headings should be noun phrases, NOT questions. Keep them concise."
    ),
    few_shot=[
        {
            "input": (
                "rq_1: What are the core architectural components of transformer models "
                "and how do they differ from RNN/CNN architectures?\n"
                "rq_2: How do recent efficient-transformer variants compare in throughput "
                "and accuracy on standard benchmarks?\n"
                "rq_3: What are the main scalability challenges in training "
                "billion-parameter transformers?"
            ),
            "output": (
                '{"headings": ['
                '{"rq_id": "rq_1", "heading": "Core Transformer Architecture"}, '
                '{"rq_id": "rq_2", "heading": "Efficient Variants and Benchmarks"}, '
                '{"rq_id": "rq_3", "heading": "Scalability Challenges"}]}'
            ),
        }
    ],
)

# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

COLLECTOR_QUERIES = PromptTemplate(
    name="collector_queries",
    system=(
        "You are a research librarian generating diverse academic search queries. "
        "Each query should use different terminology and angles to maximise recall."
    ),
    user=(
        "Generate 3 diverse academic search queries for:\n"
        "Research question: '{rq_text}'\n"
        "Topic: '{topic}'\n"
        "{neg_hint}"
        "Return exactly 3 lines, one query per line. No numbering, no bullets."
    ),
)

REACT_THOUGHT = PromptTemplate(
    name="react_thought",
    system=(
        "You are a research source collector using a ReAct (Reasoning + Acting) loop. "
        "Analyse the current collection state, reason about what's missing, then "
        "decide the next tool action. Available tools: arxiv_search, web_search, fetch."
    ),
    user=(
        "Topic: {topic}\n"
        "Gap research questions: {gap_rqs}\n"
        "Sources collected so far: {current_sources}/{target_sources}\n"
        "Buckets needing coverage: {uncovered_buckets}\n"
        "Previous actions: {history}\n\n"
        "Think step by step:\n"
        "1. What sub-topics or buckets still need sources?\n"
        "2. What search strategy would fill the gaps?\n"
        "3. Decide on ONE action.\n\n"
        "Return JSON:\n"
        '{{"thought": "your reasoning", '
        '"action": "arxiv_search"|"web_search"|"done", '
        '"query": "search query if action is a search", '
        '"reason": "why this action"}}'
    ),
    few_shot=[
        {
            "input": (
                "Topic: graph neural networks\n"
                "Gap RQs: rq_scalability\n"
                "Sources: 4/10\nUncovered: scalability, applications"
            ),
            "output": (
                '{"thought": "We have 4 sources but scalability and applications '
                'buckets are uncovered. I should search for scalability challenges '
                'in GNNs first since rq_scalability is a gap.", '
                '"action": "arxiv_search", '
                '"query": "graph neural network scalability large-scale training", '
                '"reason": "targeting uncovered scalability bucket and gap RQ"}'
            ),
        }
    ],
)

# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

READER_CLAIMS = PromptTemplate(
    name="reader_claims",
    system=(
        "You are a meticulous research paper reader. Extract structured claims "
        "from source text. Each claim must be a single, self-contained factual "
        "statement grounded in the text. Never fabricate information."
    ),
    user=(
        "Source title: {title}\n"
        "Text chunk:\n{chunk}\n\n"
        "Research questions:\n{rq_list}\n\n"
        "Extract 3-6 claims from this chunk. For each claim provide:\n"
        "- text: one concise sentence (max 220 chars)\n"
        "- claim_type: definition | method | result | limitation | trend | comparison\n"
        "- polarity: support | oppose | neutral\n"
        "- supports_rq: list of matching rq_ids from above\n"
        "- evidence_span: the key phrase from the text that supports this claim\n\n"
        "Return JSON: {{\"claims\": [{{\"text\": \"...\", \"claim_type\": \"...\", "
        "\"polarity\": \"...\", \"supports_rq\": [...], \"evidence_span\": \"...\"}}]}}"
    ),
    few_shot=[
        {
            "input": (
                "Title: Attention Is All You Need\n"
                "Chunk: The Transformer model relies entirely on self-attention "
                "mechanisms, dispensing with recurrence and convolution. It achieves "
                "state-of-the-art BLEU scores of 28.4 on the WMT 2014 English-to-German "
                "translation task.\nRQs: rq_arch, rq_perf"
            ),
            "output": (
                '{"claims": ['
                '{"text": "The Transformer architecture relies entirely on self-attention, '
                'eliminating recurrence and convolution", "claim_type": "method", '
                '"polarity": "support", "supports_rq": ["rq_arch"], '
                '"evidence_span": "relies entirely on self-attention mechanisms"}, '
                '{"text": "Transformer achieves 28.4 BLEU on WMT 2014 EN-DE, establishing '
                'a new state-of-the-art", "claim_type": "result", '
                '"polarity": "support", "supports_rq": ["rq_perf"], '
                '"evidence_span": "state-of-the-art BLEU scores of 28.4"}]}'
            ),
        }
    ],
)

READER_ENRICH = PromptTemplate(
    name="reader_enrich",
    system=(
        "You are a research analyst. Assign structured metadata to extracted claims."
    ),
    user=(
        "Given these claims, assign each a claim_type "
        "(definition/method/result/limitation/trend/comparison) "
        "and polarity (support/oppose/neutral).\n\n"
        "Claims:\n{claims_summary}\n\n"
        "Return JSON: {{\"claims\": [{{\"claim_id\": \"...\", "
        "\"claim_type\": \"...\", \"polarity\": \"...\"}}]}}"
    ),
)

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

WRITER_SECTION = PromptTemplate(
    name="writer_section",
    system=(
        "You are a scientific writer producing a literature survey section. "
        "You MUST cite every statement using [@source_id] markers from the "
        "provided evidence. Never make claims without citation. Write in "
        "a scholarly but accessible tone."
    ),
    user=(
        "Write a research report section titled '{heading}' in {lang}.\n\n"
        "Available evidence (each prefixed with [source_id]):\n{claims_block}\n\n"
        "Requirements:\n"
        "1. OVERVIEW paragraph (3-5 sentences): synthesise key findings. "
        "Every sentence MUST end with [@source_id].\n"
        "2. BULLETS (4-6 bullet points starting with '- '): each with [@source_id].\n"
        "{deep_instruction}\n"
        "Output format:\n"
        "OVERVIEW: <paragraph>\n"
        "BULLETS:\n- point1 [@source_id]\n- point2 [@source_id]\n"
        "TRENDS: <paragraph or empty>\n\n"
        "CRITICAL: Every sentence and bullet must contain at least one [@source_id]."
    ),
)

WRITER_SUMMARIZE = PromptTemplate(
    name="writer_summarize",
    system="You are a concise academic writer. Summarise research claims briefly.",
    user=(
        "Summarise the following research claim in one concise sentence (max 200 chars). "
        "Do NOT include citation markers. Just output the summary sentence.\n\n"
        "Claim: {claim_text}"
    ),
)

WRITER_CONCLUSION = PromptTemplate(
    name="writer_conclusion",
    system=(
        "You are a scientific writer producing the conclusion of a literature survey. "
        "Synthesise findings across all sections into a cohesive summary. "
        "You MUST cite every statement using [@source_id] markers."
    ),
    user=(
        "Write a conclusion for a research survey on '{topic}' in {lang}.\n\n"
        "The report has the following sections and key findings:\n{sections_summary}\n\n"
        "Requirements:\n"
        "1. Write 1-2 paragraphs that synthesise the key findings across ALL sections.\n"
        "2. Highlight common themes, notable gaps, and future directions.\n"
        "3. Every sentence MUST end with at least one [@source_id] citation.\n"
        "4. Do NOT simply repeat individual section content — synthesise and connect.\n\n"
        "Output the conclusion paragraphs directly (no 'OVERVIEW:' or 'BULLETS:' markers)."
    ),
)

# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_PLAN = PromptTemplate(
    name="supervisor_plan",
    system=(
        "You are a research strategy advisor. Analyse diagnostic signals from "
        "a literature survey pipeline and recommend concrete corrective actions."
    ),
    user=(
        "Topic: '{topic}'\n"
        "Detected issues: {reason_codes}\n"
        "Current coverage: {coverage_summary}\n\n"
        "Suggest:\n"
        "1. 3-5 search queries to fix coverage gaps\n"
        "2. 3-5 negative terms to exclude off-topic results\n"
        "3. 1-3 arXiv categories to search\n\n"
        "Return JSON: {{\"queries\": [...], \"negative_terms\": [...], \"categories\": [...]}}"
    ),
    few_shot=[
        {
            "input": (
                "Topic: federated learning\n"
                "Issues: ['coverage_gap', 'bucket_incomplete']\n"
                "Coverage: privacy bucket 0/3 sources, aggregation bucket 1/3"
            ),
            "output": (
                '{"queries": ["federated learning differential privacy", '
                '"secure aggregation protocols federated", '
                '"communication efficiency federated learning", '
                '"heterogeneous data federated learning"], '
                '"negative_terms": ["blockchain", "IoT sensor network", '
                '"wireless channel estimation"], '
                '"categories": ["cs.LG", "cs.CR", "cs.DC"]}'
            ),
        }
    ],
)

# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

VERIFIER_FIX = PromptTemplate(
    name="verifier_fix",
    system=(
        "You are a Python debugging assistant. Fix the script so it runs without "
        "errors. Return ONLY the corrected Python script, no explanations."
    ),
    user=(
        "Fix this Python script that produced the following error.\n\n"
        "Error:\n{error}\n\nScript:\n{code}"
    ),
)

# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------

JUDGE_FAITHFULNESS = PromptTemplate(
    name="judge_faithfulness",
    system=(
        "You are an impartial evaluator assessing the faithfulness of a research "
        "report. Faithfulness means every claim in the report is grounded in the "
        "provided source evidence."
    ),
    user=(
        "Report excerpt:\n{report}\n\n"
        "Source claims:\n{claims}\n\n"
        "Score the report's faithfulness from 0.0 to 1.0.\n"
        "- 1.0 = every statement is directly supported by source claims\n"
        "- 0.5 = roughly half the statements are supported\n"
        "- 0.0 = statements are fabricated\n\n"
        "Return JSON: {{\"score\": <float>, \"reasoning\": \"<1-2 sentences>\"}}"
    ),
)

JUDGE_COVERAGE = PromptTemplate(
    name="judge_coverage",
    system=(
        "You are an impartial evaluator assessing topic coverage of a research report."
    ),
    user=(
        "Topic: {topic}\n"
        "Research questions:\n{rq_list}\n\n"
        "Report:\n{report}\n\n"
        "Score coverage from 0.0 to 1.0 based on how well the report addresses "
        "each research question.\n\n"
        "Return JSON: {{\"score\": <float>, \"reasoning\": \"<1-2 sentences>\"}}"
    ),
)

JUDGE_COHERENCE = PromptTemplate(
    name="judge_coherence",
    system=(
        "You are an impartial evaluator assessing the coherence and readability "
        "of a research report."
    ),
    user=(
        "Report:\n{report}\n\n"
        "Score coherence from 0.0 to 1.0:\n"
        "- 1.0 = well-structured, logical flow, no redundancy\n"
        "- 0.5 = some disorganisation or repetition\n"
        "- 0.0 = incoherent or heavily redundant\n\n"
        "Return JSON: {{\"score\": <float>, \"reasoning\": \"<1-2 sentences>\"}}"
    ),
)

JUDGE_RELEVANCE = PromptTemplate(
    name="judge_relevance",
    system=(
        "You are an impartial evaluator assessing the relevance of a research "
        "report to its stated topic."
    ),
    user=(
        "Topic: {topic}\n\n"
        "Report:\n{report}\n\n"
        "Score relevance from 0.0 to 1.0:\n"
        "- 1.0 = entirely on-topic, every section addresses the topic directly\n"
        "- 0.5 = partially on-topic, some tangential sections\n"
        "- 0.0 = off-topic\n\n"
        "Return JSON: {{\"score\": <float>, \"reasoning\": \"<1-2 sentences>\"}}"
    ),
)
