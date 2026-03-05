from __future__ import annotations

import json
import math
import re
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import (
    AgentResult,
    Claim,
    PlanOutput,
    Source,
    SourceNotes,
)
from researchops.tools.parse_doc import NOISE_PATTERNS

_CODE_TEMPLATE_RE = re.compile(
    r"function\s*\(|padding\s*:|margin\s*:|\.css|{color:|querySelector|"
    r"addEventListener|@media|@import|rgb\(|rgba\(|\d+px\s*;|"
    r"#[0-9a-fA-F]{3,8}\b|var\s+\w|const\s+\w|let\s+\w|=>\s*{|"
    r"\.style\.|\.className|window\.|document\.",
    re.IGNORECASE,
)

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "contribution": ["propose", "introduce", "present", "novel", "contribution", "develop", "design"],
    "method": ["method", "approach", "technique", "algorithm", "framework", "architecture", "implement"],
    "limitation": ["challenge", "limit", "problem", "issue", "drawback", "barrier", "difficult", "gap"],
    "finding": ["result", "show", "demonstrate", "find", "observe", "evidence", "significant", "suggest"],
}

_TYPE_KEYWORDS: dict[str, list[str]] = {
    "definition": ["define", "definition", "refers to", "is a", "known as", "called"],
    "method": ["method", "approach", "technique", "algorithm", "procedure"],
    "result": ["result", "found", "showed", "demonstrated", "observed", "measured"],
    "limitation": ["limitation", "drawback", "challenge", "problem", "issue", "barrier"],
    "trend": ["trend", "growing", "emerging", "increasing", "declining", "future"],
    "comparison": ["compared", "versus", "better", "worse", "outperform", "contrast", "differ"],
}

_OPPOSE_KEYWORDS = [
    "however", "but", "although", "despite", "contrary", "disagree",
    "challenge", "fail", "unable", "limitation", "drawback", "problem",
    "cannot", "insufficient", "inadequate", "not support",
]

_PREAMBLE_PATTERNS = [
    "in this article", "we'll look at", "we will explore", "let's dive",
    "click here", "read more", "this post", "this guide", "subscribe",
    "in this tutorial", "we'll discuss", "we will discuss", "this blog",
    "in this section", "table of contents", "skip to", "sign up",
    "you'll learn", "you will learn", "let us explore",
]

_MAX_CLAIMS_PER_SOURCE = 12
_MIN_SENTENCE_LEN = 40
_MAX_CLAIM_TEXT_LEN = 220
_PREFERRED_MIN = 80
_PREFERRED_MAX = 220
_MAX_PARA_LEN = 1500
_MIN_PARA_LEN = 80
_MIN_TEXT_LEN_FOR_QUALITY = 200


def _normalize_key(text: str) -> str:
    return re.sub(r"[\s\d\W]+", "", text.lower())


def _is_noisy(text: str) -> bool:
    lower = text.lower()
    return any(pat in lower for pat in NOISE_PATTERNS) or any(pat in lower for pat in _PREAMBLE_PATTERNS)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _text_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


class ReaderAgent(AgentBase):
    name = "reader"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="READ", agent=self.name, action="start")

        sources = self._load_sources(ctx.run_dir)
        plan = PlanOutput.model_validate(
            json.loads((ctx.run_dir / "plan.json").read_text(encoding="utf-8"))
        )
        rq_ids = [rq.rq_id for rq in plan.research_questions]
        rq_texts = {rq.rq_id: rq.text for rq in plan.research_questions}
        checklist = plan.coverage_checklist
        anchor_terms = self._build_anchor_terms(plan.topic, checklist)

        notes_dir = ctx.run_dir / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        total_claims = 0

        rq_claim_counts: dict[str, int] = {rq_id: 0 for rq_id in rq_ids}
        rq_source_counts: dict[str, set[str]] = {rq_id: set() for rq_id in rq_ids}

        low_quality_sources: list[str] = []
        for src in sources:
            is_arxiv_meta = src.source_type_detail == "arxiv_meta"

            if not src.local_path or not Path(src.local_path).exists():
                ctx.trace.log(
                    stage="READ", agent=self.name, action="parse.skipped",
                    meta={"source_id": src.source_id, "reason": "no local_path or file missing"},
                )
                continue

            if is_arxiv_meta:
                text = Path(src.local_path).read_text(encoding="utf-8", errors="replace")
                extraction_method = "arxiv_abstract"
            else:
                text = self._extract_text(src, ctx)
                extraction_method = "parse_doc"

            text_stripped = text.strip()
            if not text_stripped:
                ctx.trace.log(
                    stage="READ", agent=self.name, action="parse.low_quality",
                    meta={"source_id": src.source_id, "reason": "empty text after parse"},
                )
                low_quality_sources.append(src.source_id)
                continue

            if len(text_stripped) < _MIN_TEXT_LEN_FOR_QUALITY:
                ctx.trace.log(
                    stage="READ", agent=self.name, action="parse.low_quality",
                    meta={"source_id": src.source_id, "reason": f"text too short ({len(text_stripped)} chars)"},
                )
                low_quality_sources.append(src.source_id)
                continue

            entropy = _text_entropy(text_stripped[:500])
            if entropy > 6.5:
                ctx.trace.log(
                    stage="READ", agent=self.name, action="parse.low_quality",
                    meta={"source_id": src.source_id, "reason": f"high entropy ({entropy:.2f})"},
                )
                low_quality_sources.append(src.source_id)
                continue

            paragraphs = self._split_paragraphs(text)
            claims = self._extract_claims(src.source_id, paragraphs, rq_ids, rq_texts)

            if ctx.reasoner.is_llm:
                claims = self._llm_reading_cards(claims, text, src, ctx, rq_ids, rq_texts)

            if len(claims) < 2 and src.source_id not in low_quality_sources:
                ctx.trace.log(
                    stage="READ", agent=self.name, action="parse.low_quality",
                    meta={"source_id": src.source_id, "reason": f"only {len(claims)} claims"},
                )
                low_quality_sources.append(src.source_id)

            total_claims += len(claims)

            for claim in claims:
                for rq_id in claim.supports_rq:
                    if rq_id in rq_claim_counts:
                        rq_claim_counts[rq_id] += 1
                        rq_source_counts[rq_id].add(src.source_id)

            contribution = self._extract_section(text, "contribution")
            method = self._extract_section(text, "method")
            limitations = self._extract_section(text, "limitation")

            bibliographic = self._build_bibliographic(src)
            noise_flags = []
            if src.source_id in low_quality_sources:
                noise_flags.append("low_quality")
            text_len = len(text_stripped)
            readability = min(1.0, text_len / 3000)

            rel_score = self._compute_relevance_score(text_stripped, anchor_terms)
            b_hits = self._compute_bucket_hits(text_stripped, claims, checklist)
            if rel_score < ctx.config.relevance_threshold and src.source_id not in low_quality_sources:
                noise_flags.append("low_relevance")

            quality = {
                "readability_score": round(readability, 3),
                "noise_flags": noise_flags,
                "extraction_method": extraction_method,
                "text_length": text_len,
                "entropy": round(entropy, 3),
                "relevance_score": round(rel_score, 3),
            }

            notes = SourceNotes(
                source_id=src.source_id,
                claims=claims,
                contribution=contribution,
                method=method,
                limitations=limitations,
                bibliographic=bibliographic,
                quality=quality,
                relevance_score=round(rel_score, 3),
                bucket_hits=b_hits,
            )
            out_path = notes_dir / f"{src.source_id}.json"
            out_path.write_text(notes.model_dump_json(indent=2), encoding="utf-8")

        ctx.shared["low_quality_sources"] = low_quality_sources
        ctx.shared["rq_claim_counts"] = rq_claim_counts
        ctx.shared["rq_source_counts"] = {k: len(v) for k, v in rq_source_counts.items()}

        min_claims_target = 8 if ctx.config.mode.value == "deep" else 3
        underfilled_rqs = [
            rq_id for rq_id, count in rq_claim_counts.items()
            if count < min_claims_target
        ]
        if underfilled_rqs:
            ctx.trace.log(
                stage="READ", agent=self.name, action="claim_gap_detected",
                meta={
                    "underfilled_rqs": underfilled_rqs,
                    "rq_claim_counts": rq_claim_counts,
                    "min_target": min_claims_target,
                },
            )

        ctx.trace.log(
            stage="READ",
            agent=self.name,
            action="complete",
            output_summary=f"Extracted {total_claims} claims from {len(sources)} sources",
            meta={
                "low_quality_count": len(low_quality_sources),
                "rq_claim_counts": rq_claim_counts,
                "underfilled_rqs": underfilled_rqs,
            },
        )
        return AgentResult(
            success=True,
            message=f"Extracted {total_claims} claims from {len(sources)} sources",
        )

    def _llm_reading_cards(
        self, claims: list[Claim], text: str, src: Source,
        ctx: RunContext, rq_ids: list[str], rq_texts: dict[str, str],
    ) -> list[Claim]:
        text_snippet = text[:2000]
        rq_list = "\n".join(f"- {rid}: {rtxt}" for rid, rtxt in rq_texts.items())

        prompt = (
            f"You are a research paper reader. Given the following source text, "
            f"extract structured research claims.\n\n"
            f"Source title: {src.title}\n"
            f"Source text (first 2000 chars):\n{text_snippet}\n\n"
            f"Research questions:\n{rq_list}\n\n"
            f"For each claim, provide:\n"
            f"- text: one concise sentence (max 220 chars)\n"
            f"- claim_type: one of definition/method/result/limitation/trend/comparison\n"
            f"- polarity: support/oppose/neutral\n"
            f"- supports_rq: list of matching rq_ids\n"
            f"- evidence_span: the key phrase from source that supports this claim\n\n"
            f"Return JSON: {{\"claims\": [{{\"text\": \"...\", \"claim_type\": \"...\", "
            f"\"polarity\": \"...\", \"supports_rq\": [...], \"evidence_span\": \"...\"}}]}}\n"
            f"Extract 4-8 claims covering contribution, method, limitations, and trends."
        )

        try:
            raw = ctx.reasoner.complete_text(prompt, trace=ctx.trace)
            start = raw.find("{")
            data = json.loads(raw[start:]) if start >= 0 else json.loads(raw)

            llm_claims: list[Claim] = []
            existing_keys = {_normalize_key(c.text) for c in claims}

            for i, item in enumerate(data.get("claims", [])):
                claim_text = item.get("text", "").strip()
                if not claim_text or len(claim_text) < 20:
                    continue
                claim_text = _truncate(claim_text, _MAX_CLAIM_TEXT_LEN)
                norm = _normalize_key(claim_text)
                if norm in existing_keys:
                    continue
                existing_keys.add(norm)

                supports = item.get("supports_rq", [])
                if not supports or not any(r in rq_ids for r in supports):
                    supports = rq_ids[:1]

                evidence_span = item.get("evidence_span", "")
                if not evidence_span:
                    evidence_span = claim_text[:100]

                llm_claims.append(Claim(
                    claim_id=f"{src.source_id}_llm_c{i}",
                    text=claim_text,
                    evidence_spans=[_truncate(evidence_span, 150)],
                    supports_rq=[r for r in supports if r in rq_ids],
                    category="llm_extracted",
                    evidence_location="llm_reading_card",
                    claim_type=item.get("claim_type", "other"),
                    polarity=item.get("polarity", "neutral"),
                ))

            combined = claims + llm_claims
            return combined[:_MAX_CLAIMS_PER_SOURCE]

        except Exception:
            if claims:
                return self._enrich_claims_with_llm(claims, text, ctx)
            return claims

    def _enrich_claims_with_llm(
        self, claims: list[Claim], text: str, ctx: RunContext,
    ) -> list[Claim]:
        claims_summary = "\n".join(f"- [{c.claim_id}] {c.text[:150]}" for c in claims[:8])
        prompt = (
            f"Given these extracted claims, assign each a claim_type "
            f"(definition/method/result/limitation/trend/comparison) "
            f"and polarity (support/oppose/neutral). "
            f"Return JSON: {{\"claims\": [{{\"claim_id\": \"...\", \"claim_type\": \"...\", \"polarity\": \"...\"}}]}}\n\n"
            f"Claims:\n{claims_summary}"
        )
        try:
            raw = ctx.reasoner.complete_text(prompt, trace=ctx.trace)
            data = json.loads(raw) if raw.strip().startswith("{") else json.loads("{" + raw.split("{", 1)[-1])
            enrichments = {e["claim_id"]: e for e in data.get("claims", []) if "claim_id" in e}
            for claim in claims:
                if claim.claim_id in enrichments:
                    e = enrichments[claim.claim_id]
                    if e.get("claim_type"):
                        claim.claim_type = e["claim_type"]
                    if e.get("polarity"):
                        claim.polarity = e["polarity"]
        except Exception:
            pass
        return claims

    def _build_bibliographic(self, src: Source) -> dict:
        bib: dict = {}
        if src.source_type_detail in ("arxiv_meta", "arxiv_pdf"):
            arxiv_id = ""
            if "arxiv.org/abs/" in src.url:
                arxiv_id = src.url.split("/abs/")[-1]
            bib["paper_id"] = arxiv_id
            bib["title"] = src.title.replace(" [PDF]", "")
            bib["source"] = "arxiv"
        else:
            bib["title"] = src.title
            bib["domain"] = src.domain
        return bib

    def _load_sources(self, run_dir: Path) -> list[Source]:
        sources_path = run_dir / "sources.jsonl"
        if not sources_path.exists():
            return []
        sources = []
        for line in sources_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                sources.append(Source.model_validate(json.loads(line)))
        return sources

    def _extract_text(self, src: Source, ctx: RunContext) -> str:
        if not src.local_path or not Path(src.local_path).exists():
            return ""
        try:
            result = ctx.registry.invoke(
                "parse",
                {"file_path": src.local_path, "format": src.type.value},
                trace=ctx.trace,
            )
            if result.get("warning"):
                ctx.trace.log(
                    stage="READ", agent=self.name, action=f"parse.{result['warning']}",
                    meta={"source_id": src.source_id, "warning": result["warning"]},
                )
            return result.get("text", "")
        except Exception:
            return ""

    def _split_paragraphs(self, text: str) -> list[tuple[int, str]]:
        paragraphs = []
        for i, para in enumerate(re.split(r"\n\s*\n", text)):
            stripped = para.strip()
            if len(stripped) < _MIN_PARA_LEN:
                continue
            if _CODE_TEMPLATE_RE.search(stripped):
                continue
            if len(stripped) > _MAX_PARA_LEN:
                stripped = stripped[:_MAX_PARA_LEN]
            paragraphs.append((i, stripped))
        return paragraphs

    def _split_sentences(self, para: str) -> list[str]:
        parts = re.split(r"(?<=[.!?。！？；])\s*", para)
        sentences = []
        for s in parts:
            s = s.strip()
            if len(s) >= _MIN_SENTENCE_LEN and not _is_noisy(s):
                sentences.append(s)
        return sentences

    def _extract_claims(
        self,
        source_id: str,
        paragraphs: list[tuple[int, str]],
        rq_ids: list[str],
        rq_texts: dict[str, str],
    ) -> list[Claim]:
        candidates: list[tuple[str, int, str]] = []
        seen_keys: set[str] = set()

        for para_idx, para in paragraphs[:20]:
            if _is_noisy(para):
                continue

            for sentence in self._split_sentences(para):
                norm = _normalize_key(sentence)
                if norm in seen_keys:
                    continue
                seen_keys.add(norm)
                candidates.append((sentence, para_idx, sentence))

        preferred = [c for c in candidates if _PREFERRED_MIN <= len(c[0]) <= _PREFERRED_MAX]
        rest = [c for c in candidates if c not in preferred]
        ordered = preferred + rest

        claims: list[Claim] = []
        for sentence, para_idx, _raw in ordered[:_MAX_CLAIMS_PER_SOURCE]:
            category = self._classify_sentence(sentence)
            supports = self._match_rqs(sentence, rq_ids, rq_texts)
            evidence_loc = f"paragraph_{para_idx}"
            claim_type = self._classify_type(sentence)
            polarity = self._detect_polarity(sentence)

            claim_text = _truncate(sentence, _MAX_CLAIM_TEXT_LEN)
            evidence = _truncate(sentence, 150)

            claims.append(
                Claim(
                    claim_id=f"{source_id}_c{len(claims)}",
                    text=claim_text,
                    evidence_spans=[evidence],
                    supports_rq=supports,
                    category=category,
                    evidence_location=evidence_loc,
                    claim_type=claim_type,
                    polarity=polarity,
                )
            )
        return claims

    def _classify_sentence(self, sentence: str) -> str:
        lower = sentence.lower()
        scores: dict[str, int] = {}
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            scores[cat] = sum(1 for kw in keywords if kw in lower)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "other"

    def _classify_type(self, sentence: str) -> str:
        lower = sentence.lower()
        scores: dict[str, int] = {}
        for ctype, keywords in _TYPE_KEYWORDS.items():
            scores[ctype] = sum(1 for kw in keywords if kw in lower)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "other"

    def _detect_polarity(self, sentence: str) -> str:
        lower = sentence.lower()
        oppose_count = sum(1 for kw in _OPPOSE_KEYWORDS if kw in lower)
        if oppose_count >= 2:
            return "oppose"
        if oppose_count == 1 and any(kw in lower for kw in ("not", "fail", "cannot", "unable")):
            return "oppose"
        return "support" if len(sentence) > 50 else "neutral"

    def _match_rqs(self, sentence: str, rq_ids: list[str], rq_texts: dict[str, str]) -> list[str]:
        lower = sentence.lower()
        matched = []
        for rq_id in rq_ids:
            tag = rq_id.replace("rq_", "")
            if tag in lower:
                matched.append(rq_id)
        if not matched:
            if any(kw in lower for kw in ("current", "state", "research", "study")):
                matched = [r for r in rq_ids if "state" in r or "overview" in r]
            elif any(kw in lower for kw in ("challenge", "limit", "problem")):
                matched = [r for r in rq_ids if "challenge" in r]
            elif any(kw in lower for kw in ("future", "direction", "trend", "emerging")):
                matched = [r for r in rq_ids if "future" in r]

        if not matched:
            sent_words = set(re.findall(r"\w{3,}", lower))
            best_rq = ""
            best_overlap = 0
            for rq_id, rq_text in rq_texts.items():
                rq_words = set(re.findall(r"\w{3,}", rq_text.lower()))
                overlap = len(sent_words & rq_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_rq = rq_id
            if best_rq:
                matched = [best_rq]

        return matched or rq_ids[:1]

    def _extract_section(self, text: str, section_type: str) -> str:
        keywords = _CATEGORY_KEYWORDS.get(section_type, [])
        sentences = re.split(r"(?<=[.!?。！？])\s*", text)
        relevant = []
        for s in sentences:
            s = s.strip()
            if len(s) < 40 or _is_noisy(s):
                continue
            if any(kw in s.lower() for kw in keywords):
                relevant.append(_truncate(s, 200))
        return " ".join(relevant[:3])

    def _build_anchor_terms(self, topic: str, checklist: list[dict]) -> set[str]:
        terms: set[str] = set()
        for w in re.findall(r"\w{3,}", topic.lower()):
            terms.add(w)
        for bucket in checklist:
            desc = bucket.get("description", "") + " " + bucket.get("bucket_name", "")
            for w in re.findall(r"\w{3,}", desc.lower()):
                terms.add(w)
        return terms

    def _compute_relevance_score(self, text: str, anchor_terms: set[str]) -> float:
        if not anchor_terms:
            return 1.0
        text_words = set(re.findall(r"\w{3,}", text.lower()[:3000]))
        hits = len(anchor_terms & text_words)
        return hits / max(1, len(anchor_terms))

    def _compute_bucket_hits(
        self, text: str, claims: list[Claim], checklist: list[dict],
    ) -> list[str]:
        if not checklist:
            return []
        text_lower = text.lower()
        claim_text = " ".join(c.text.lower() for c in claims)
        combined = text_lower[:2000] + " " + claim_text

        hits: list[str] = []
        for bucket in checklist:
            bid = bucket.get("bucket_id", "")
            desc = bucket.get("description", "") + " " + bucket.get("bucket_name", "")
            keywords = set(re.findall(r"\w{3,}", desc.lower()))
            combined_words = set(re.findall(r"\w{3,}", combined))
            overlap = len(keywords & combined_words)
            if overlap >= max(1, len(keywords) // 3):
                hits.append(bid)
        return hits
