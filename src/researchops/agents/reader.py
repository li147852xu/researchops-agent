"""Reader agent — chunk-based claim extraction with LLM + embedding relevance."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, Claim, PlanOutput, Source, SourceNotes
from researchops.prompts import READER_CLAIMS, parse_json_response
from researchops.tools.parse_doc import NOISE_PATTERNS
from researchops.utils import chunk_text, load_sources, truncate

logger = logging.getLogger(__name__)

_CODE_TEMPLATE_RE = re.compile(
    r"function\s*\(|padding\s*:|margin\s*:|\.css|{color:|querySelector|"
    r"addEventListener|@media|@import|rgb\(|rgba\(|\d+px\s*;|"
    r"#[0-9a-fA-F]{3,8}\b|var\s+\w|const\s+\w|let\s+\w|=>\s*{|"
    r"\.style\.|\.className|window\.|document\.",
    re.IGNORECASE,
)

_PREAMBLE_PATTERNS = [
    "in this article", "we'll look at", "we will explore", "let's dive",
    "click here", "read more", "this post", "this guide", "subscribe",
    "in this tutorial", "we'll discuss", "we will discuss", "this blog",
    "in this section", "table of contents", "skip to", "sign up",
    "you'll learn", "you will learn", "let us explore",
]

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

_MAX_CLAIMS_PER_SOURCE = 12
_MIN_SENTENCE_LEN = 40
_MAX_CLAIM_TEXT_LEN = 220
_PREFERRED_MIN = 80
_PREFERRED_MAX = 220
_MIN_TEXT_LEN_FOR_QUALITY = 100
_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 60


def _normalize_key(text: str) -> str:
    return re.sub(r"[\s\d\W]+", "", text.lower())


def _is_noisy(text: str) -> bool:
    lower = text.lower()
    return any(pat in lower for pat in NOISE_PATTERNS) or any(pat in lower for pat in _PREAMBLE_PATTERNS)


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

        sources = load_sources(ctx.run_dir)
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
                ctx.trace.log(stage="READ", agent=self.name, action="parse.low_quality",
                              meta={"source_id": src.source_id, "reason": "empty text"})
                low_quality_sources.append(src.source_id)
                continue

            if len(text_stripped) < _MIN_TEXT_LEN_FOR_QUALITY:
                ctx.trace.log(stage="READ", agent=self.name, action="parse.low_quality",
                              meta={"source_id": src.source_id, "reason": f"too short ({len(text_stripped)})"})
                low_quality_sources.append(src.source_id)
                continue

            entropy = _text_entropy(text_stripped[:500])
            if entropy > 7.0:
                ctx.trace.log(stage="READ", agent=self.name, action="parse.low_quality",
                              meta={"source_id": src.source_id, "reason": f"high entropy ({entropy:.2f})"})
                low_quality_sources.append(src.source_id)
                continue

            # --- Chunk-based extraction ---
            if ctx.reasoner.is_llm:
                claims = self._chunk_extract_with_llm(
                    text_stripped, src, ctx, rq_ids, rq_texts,
                )
            else:
                paragraphs = self._split_paragraphs(text)
                claims = self._extract_claims_rule(src.source_id, paragraphs, rq_ids, rq_texts)

            if len(claims) < 1 and src.source_id not in low_quality_sources:
                ctx.trace.log(stage="READ", agent=self.name, action="parse.low_quality",
                              meta={"source_id": src.source_id, "reason": f"only {len(claims)} claims"})
                low_quality_sources.append(src.source_id)

            total_claims += len(claims)
            for claim in claims:
                for rq_id in claim.supports_rq:
                    if rq_id in rq_claim_counts:
                        rq_claim_counts[rq_id] += 1
                        rq_source_counts[rq_id].add(src.source_id)

            rel_score = self._compute_relevance(text_stripped, anchor_terms, ctx)
            b_hits = self._compute_bucket_hits(text_stripped, claims, checklist)

            noise_flags: list[str] = []
            if src.source_id in low_quality_sources:
                noise_flags.append("low_quality")
            if rel_score < ctx.config.relevance_threshold:
                noise_flags.append("low_relevance")

            notes = SourceNotes(
                source_id=src.source_id,
                claims=claims,
                contribution=self._extract_section(text, "contribution"),
                method=self._extract_section(text, "method"),
                limitations=self._extract_section(text, "limitation"),
                bibliographic=self._build_bibliographic(src),
                quality={
                    "readability_score": round(min(1.0, len(text_stripped) / 3000), 3),
                    "noise_flags": noise_flags,
                    "extraction_method": extraction_method,
                    "text_length": len(text_stripped),
                    "entropy": round(entropy, 3),
                    "relevance_score": round(rel_score, 3),
                },
                relevance_score=round(rel_score, 3),
                bucket_hits=b_hits,
            )
            out_path = notes_dir / f"{src.source_id}.json"
            out_path.write_text(notes.model_dump_json(indent=2), encoding="utf-8")

        ctx.shared["low_quality_sources"] = low_quality_sources
        ctx.shared["rq_claim_counts"] = rq_claim_counts
        ctx.shared["rq_source_counts"] = {k: len(v) for k, v in rq_source_counts.items()}

        ctx.trace.log(
            stage="READ", agent=self.name, action="complete",
            output_summary=f"Extracted {total_claims} claims from {len(sources)} sources",
            meta={
                "low_quality_count": len(low_quality_sources),
                "rq_claim_counts": rq_claim_counts,
            },
        )
        return AgentResult(
            success=True,
            message=f"Extracted {total_claims} claims from {len(sources)} sources",
        )

    # ------------------------------------------------------------------
    # Chunk-based LLM claim extraction
    # ------------------------------------------------------------------

    def _chunk_extract_with_llm(
        self, text: str, src: Source, ctx: RunContext,
        rq_ids: list[str], rq_texts: dict[str, str],
    ) -> list[Claim]:
        chunks = chunk_text(text, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)
        if not chunks:
            return []

        rq_list = "\n".join(f"- {rid}: {rtxt}" for rid, rtxt in rq_texts.items())
        all_claims: list[Claim] = []
        seen_keys: set[str] = set()

        for ci, chunk in enumerate(chunks[:6]):
            if _is_noisy(chunk) or _CODE_TEMPLATE_RE.search(chunk):
                continue

            sys_msg, user_msg = READER_CLAIMS.render(
                title=src.title,
                chunk=chunk[:1500],
                rq_list=rq_list,
            )
            try:
                raw = ctx.reasoner.complete_text(
                    user_msg, context=sys_msg, trace=ctx.trace,
                )
                data = parse_json_response(raw)
                for i, item in enumerate(data.get("claims", [])):
                    claim_text = item.get("text", "").strip()
                    if not claim_text or len(claim_text) < 20:
                        continue
                    claim_text = truncate(claim_text, _MAX_CLAIM_TEXT_LEN)
                    norm = _normalize_key(claim_text)
                    if norm in seen_keys:
                        continue
                    seen_keys.add(norm)

                    supports = item.get("supports_rq", [])
                    if not supports or not any(r in rq_ids for r in supports):
                        supports = rq_ids[:1]

                    evidence_span = item.get("evidence_span", claim_text[:100])

                    all_claims.append(Claim(
                        claim_id=f"{src.source_id}_c{ci}_{i}",
                        text=claim_text,
                        evidence_spans=[truncate(evidence_span, 150)],
                        supports_rq=[r for r in supports if r in rq_ids],
                        category="llm_extracted",
                        evidence_location=f"chunk_{ci}",
                        claim_type=item.get("claim_type", "other"),
                        polarity=item.get("polarity", "neutral"),
                    ))
            except Exception as exc:
                logger.warning("LLM chunk extraction failed for %s chunk %d: %s",
                               src.source_id, ci, exc)
                continue

        if not all_claims:
            paragraphs = self._split_paragraphs(text)
            all_claims = self._extract_claims_rule(src.source_id, paragraphs, rq_ids, rq_texts)

        return all_claims[:_MAX_CLAIMS_PER_SOURCE]

    # ------------------------------------------------------------------
    # Rule-based claim extraction (fallback)
    # ------------------------------------------------------------------

    def _extract_claims_rule(
        self, source_id: str, paragraphs: list[tuple[int, str]],
        rq_ids: list[str], rq_texts: dict[str, str],
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
            claim_type = self._classify_type(sentence)
            polarity = self._detect_polarity(sentence)

            claims.append(Claim(
                claim_id=f"{source_id}_c{len(claims)}",
                text=truncate(sentence, _MAX_CLAIM_TEXT_LEN),
                evidence_spans=[truncate(sentence, 150)],
                supports_rq=supports,
                category=category,
                evidence_location=f"paragraph_{para_idx}",
                claim_type=claim_type,
                polarity=polarity,
            ))
        return claims

    # ------------------------------------------------------------------
    # Relevance scoring (embedding-based when available, keyword fallback)
    # ------------------------------------------------------------------

    def _compute_relevance(self, text: str, anchor_terms: set[str], ctx: RunContext) -> float:
        retriever = ctx.shared.get("retriever")
        if retriever is not None and hasattr(retriever, "embedding"):
            try:
                emb_retriever = retriever.embedding
                topic_emb = emb_retriever.encode_query(ctx.config.topic)
                text_emb = emb_retriever.encode_texts([text[:2000]])[0]
                import numpy as np
                score = float(np.dot(topic_emb, text_emb))
                return max(0.0, min(1.0, score))
            except Exception:
                pass
        return self._compute_relevance_keyword(text, anchor_terms)

    def _compute_relevance_keyword(self, text: str, anchor_terms: set[str]) -> float:
        if not anchor_terms:
            return 1.0
        text_words = set(re.findall(r"\w{3,}", text.lower()[:3000]))
        hits = len(anchor_terms & text_words)
        return hits / max(1, len(anchor_terms))

    # ------------------------------------------------------------------
    # Helpers (kept from original)
    # ------------------------------------------------------------------

    def _build_anchor_terms(self, topic: str, checklist: list[dict]) -> set[str]:
        terms: set[str] = set()
        for w in re.findall(r"\w{3,}", topic.lower()):
            terms.add(w)
        for bucket in checklist:
            desc = bucket.get("description", "") + " " + bucket.get("bucket_name", "")
            for w in re.findall(r"\w{3,}", desc.lower()):
                terms.add(w)
        return terms

    def _compute_bucket_hits(self, text: str, claims: list[Claim], checklist: list[dict]) -> list[str]:
        if not checklist:
            return []
        text_lower = text.lower()
        claim_text = " ".join(c.text.lower() for c in claims)
        combined = text_lower[:2000] + " " + claim_text
        combined_words = set(re.findall(r"\w{3,}", combined))

        hits: list[str] = []
        for bucket in checklist:
            bid = bucket.get("bucket_id", "")
            desc = bucket.get("description", "") + " " + bucket.get("bucket_name", "")
            keywords = set(re.findall(r"\w{3,}", desc.lower()))
            overlap = len(keywords & combined_words)
            if overlap >= max(1, len(keywords) // 3):
                hits.append(bid)
        return hits

    def _build_bibliographic(self, src: Source) -> dict:
        bib: dict[str, Any] = {}
        if src.source_type_detail in ("arxiv_meta", "arxiv_pdf"):
            arxiv_id = src.url.split("/abs/")[-1] if "arxiv.org/abs/" in src.url else ""
            bib["paper_id"] = arxiv_id
            bib["title"] = src.title.replace(" [PDF]", "")
            bib["source"] = "arxiv"
        else:
            bib["title"] = src.title
            bib["domain"] = src.domain
        return bib

    def _extract_text(self, src: Source, ctx: RunContext) -> str:
        if not src.local_path or not Path(src.local_path).exists():
            return ""
        try:
            result = ctx.registry.invoke(
                "parse", {"file_path": src.local_path, "format": src.type.value},
                trace=ctx.trace,
            )
            return result.get("text", "")
        except Exception:
            return ""

    def _split_paragraphs(self, text: str) -> list[tuple[int, str]]:
        paragraphs = []
        for i, para in enumerate(re.split(r"\n\s*\n", text)):
            stripped = para.strip()
            if len(stripped) < 80 or _CODE_TEMPLATE_RE.search(stripped):
                continue
            if len(stripped) > 1500:
                stripped = stripped[:1500]
            paragraphs.append((i, stripped))
        return paragraphs

    def _split_sentences(self, para: str) -> list[str]:
        parts = re.split(r"(?<=[.!?。！？；])\s*", para)
        return [s.strip() for s in parts if len(s.strip()) >= _MIN_SENTENCE_LEN and not _is_noisy(s.strip())]

    def _classify_sentence(self, sentence: str) -> str:
        lower = sentence.lower()
        scores = {cat: sum(1 for kw in kws if kw in lower) for cat, kws in _CATEGORY_KEYWORDS.items()}
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "other"

    def _classify_type(self, sentence: str) -> str:
        lower = sentence.lower()
        scores = {ct: sum(1 for kw in kws if kw in lower) for ct, kws in _TYPE_KEYWORDS.items()}
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
            sent_words = set(re.findall(r"\w{3,}", lower))
            best_rq, best_overlap = "", 0
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
                relevant.append(truncate(s, 200))
        return " ".join(relevant[:3])
