"""Market Intelligence prompt templates for financial research pipeline."""

from __future__ import annotations

from researchops.apps.research.prompts import PromptTemplate

QUANT_PLANNER_RQS = PromptTemplate(
    name="quant_planner_rqs",
    system=(
        "You are a financial research analyst. Decompose a financial analysis "
        "question into specific, answerable research questions covering fundamentals, "
        "competitive landscape, risks, and market trends. Return valid JSON."
    ),
    user=(
        "Generate {num_rqs} research questions for financial analysis on: {topic}\n\n"
        "Requirements:\n"
        "- Cover at least these dimensions: fundamentals, competitive position, "
        "risks, and growth outlook\n"
        "- Include at least one question requiring numerical verification\n"
        "- Be specific to the company/sector mentioned\n\n"
        "Return JSON:\n"
        '{{"questions": [{{"rq_id": "rq_1", "text": "...", "priority": 1, '
        '"needs_verification": true, "dimension": "fundamentals"}}, ...]}}'
    ),
    few_shot=[
        {
            "input": "Topic: Analyze NVDA's competitive position in AI chips, num_rqs: 4",
            "output": (
                '{"questions": ['
                '{"rq_id": "rq_1", "text": "What is NVIDIA\'s current market share and revenue '
                'breakdown in the AI accelerator market?", "priority": 1, '
                '"needs_verification": true, "dimension": "fundamentals"}, '
                '{"rq_id": "rq_2", "text": "How does NVIDIA\'s GPU architecture compare to '
                'AMD and custom AI chips from Google and Amazon?", "priority": 1, '
                '"needs_verification": false, "dimension": "competitive"}, '
                '{"rq_id": "rq_3", "text": "What are the key risks to NVIDIA\'s dominance '
                'including supply chain, regulation, and technology disruption?", '
                '"priority": 2, "needs_verification": false, "dimension": "risk"}, '
                '{"rq_id": "rq_4", "text": "What is the projected growth trajectory for '
                'AI chip demand and NVIDIA\'s capacity to capture it?", "priority": 2, '
                '"needs_verification": true, "dimension": "growth"}]}'
            ),
        }
    ],
)

QUANT_PLANNER_BUCKETS = PromptTemplate(
    name="quant_planner_buckets",
    system=(
        "You are a financial research analyst. Generate topic-specific coverage "
        "buckets for a financial analysis report. Return valid JSON."
    ),
    user=(
        "Generate 4-6 coverage buckets for financial analysis on: {topic}\n\n"
        "Each bucket is a mandatory sub-area of the analysis.\n"
        "Return JSON: {{\"buckets\": [{{\"bucket_id\": \"bkt_xxx\", "
        "\"bucket_name\": \"short name\", "
        "\"description\": \"what this bucket covers\"}}]}}"
    ),
)

QUANT_PLANNER_HEADINGS = PromptTemplate(
    name="quant_planner_headings",
    system=(
        "You are a financial report writer. Convert research questions into "
        "concise section headings for a financial analysis report. Return valid JSON."
    ),
    user=(
        "Convert these research questions into short section headings (4-8 words).\n\n"
        "Research questions:\n{rq_list}\n\n"
        "Return JSON: {{\"headings\": [{{\"rq_id\": \"...\", \"heading\": \"...\"}}]}}"
    ),
)

QUANT_COLLECTOR_QUERIES = PromptTemplate(
    name="quant_collector_queries",
    system=(
        "You are a financial research analyst generating targeted search queries. "
        "Focus on financial news, SEC filings, earnings reports, and market data."
    ),
    user=(
        "Generate 3 targeted financial search queries for:\n"
        "Research question: '{rq_text}'\n"
        "Topic: '{topic}'\n"
        "{neg_hint}"
        "Focus on recent financial data, analyst reports, and market analysis.\n"
        "Return exactly 3 lines, one query per line."
    ),
)

QUANT_READER_CLAIMS = PromptTemplate(
    name="quant_reader_claims",
    system=(
        "You are a financial analyst extracting structured claims from market "
        "research. Focus on numerical data, metrics, comparisons, and factual "
        "financial statements. Never fabricate numbers."
    ),
    user=(
        "Source title: {title}\n"
        "Text chunk:\n{chunk}\n\n"
        "Research questions:\n{rq_list}\n\n"
        "Extract 3-6 financial claims. For each:\n"
        "- text: one concise statement (max 220 chars), preserve exact numbers\n"
        "- claim_type: metric | comparison | risk | outlook | fact\n"
        "- polarity: bullish | bearish | neutral\n"
        "- supports_rq: matching rq_ids\n"
        "- evidence_span: key phrase from text\n"
        "- has_numerical: true if contains specific numbers/percentages\n\n"
        "Return JSON: {{\"claims\": [{{\"text\": \"...\", \"claim_type\": \"...\", "
        "\"polarity\": \"...\", \"supports_rq\": [...], \"evidence_span\": \"...\", "
        "\"has_numerical\": true}}]}}"
    ),
)

QUANT_WRITER_SECTION = PromptTemplate(
    name="quant_writer_section",
    system=(
        "You are a financial analyst writing a research report section. "
        "Cite every statement with [@source_id] markers. Preserve exact numbers "
        "and metrics from source evidence. Write in professional analyst tone."
    ),
    user=(
        "Write a financial analysis section titled '{heading}' in {lang}.\n\n"
        "Available evidence:\n{claims_block}\n\n"
        "Requirements:\n"
        "1. OVERVIEW paragraph (3-5 sentences): synthesize key findings with citations.\n"
        "2. BULLETS (4-6 bullet points): key metrics and findings with [@source_id].\n"
        "{deep_instruction}\n"
        "CRITICAL: Every sentence must contain at least one [@source_id].\n"
        "Preserve exact numbers from the evidence."
    ),
)

QUANT_WRITER_CONCLUSION = PromptTemplate(
    name="quant_writer_conclusion",
    system=(
        "You are a financial analyst writing the conclusion of a research report. "
        "Synthesize findings into an investment thesis. Cite with [@source_id]."
    ),
    user=(
        "Write a conclusion for financial analysis on '{topic}' in {lang}.\n\n"
        "Section summaries:\n{sections_summary}\n\n"
        "Requirements:\n"
        "1. Synthesize key findings into 1-2 paragraphs.\n"
        "2. Highlight key metrics, risks, and outlook.\n"
        "3. Every sentence MUST have [@source_id] citation.\n"
    ),
)
