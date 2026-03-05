from __future__ import annotations

import json
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

_MAX_CLAIMS_PER_SOURCE = 8
_MIN_SENTENCE_LEN = 40
_MAX_CLAIM_TEXT_LEN = 260
_PREFERRED_MIN = 80
_PREFERRED_MAX = 220
_MAX_PARA_LEN = 1500
_MIN_PARA_LEN = 80


def _normalize_key(text: str) -> str:
    return re.sub(r"[\s\d\W]+", "", text.lower())


def _is_noisy(text: str) -> bool:
    lower = text.lower()
    return any(pat in lower for pat in NOISE_PATTERNS) or any(pat in lower for pat in _PREAMBLE_PATTERNS)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


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

        notes_dir = ctx.run_dir / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        total_claims = 0

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

            if not text.strip():
                ctx.trace.log(
                    stage="READ", agent=self.name, action="parse.low_quality",
                    meta={"source_id": src.source_id, "reason": "empty text after parse"},
                )
                low_quality_sources.append(src.source_id)
                continue

            paragraphs = self._split_paragraphs(text)
            claims = self._extract_claims(src.source_id, paragraphs, rq_ids, rq_texts)

            if ctx.reasoner.is_llm and claims:
                claims = self._enrich_claims_with_llm(claims, text, ctx)

            if len(claims) < 2:
                ctx.trace.log(
                    stage="READ", agent=self.name, action="parse.low_quality",
                    meta={"source_id": src.source_id, "reason": f"only {len(claims)} claims"},
                )
                low_quality_sources.append(src.source_id)

            total_claims += len(claims)

            contribution = self._extract_section(text, "contribution")
            method = self._extract_section(text, "method")
            limitations = self._extract_section(text, "limitation")

            bibliographic = self._build_bibliographic(src)
            noise_flags = []
            if src.source_id in low_quality_sources:
                noise_flags.append("low_quality")
            text_len = len(text.strip())
            readability = min(1.0, text_len / 3000)
            quality = {
                "readability_score": round(readability, 3),
                "noise_flags": noise_flags,
                "extraction_method": extraction_method,
            }

            notes = SourceNotes(
                source_id=src.source_id,
                claims=claims,
                contribution=contribution,
                method=method,
                limitations=limitations,
                bibliographic=bibliographic,
                quality=quality,
            )
            out_path = notes_dir / f"{src.source_id}.json"
            out_path.write_text(notes.model_dump_json(indent=2), encoding="utf-8")

        ctx.shared["low_quality_sources"] = low_quality_sources
        ctx.trace.log(
            stage="READ",
            agent=self.name,
            action="complete",
            output_summary=f"Extracted {total_claims} claims from {len(sources)} sources",
            meta={"low_quality_count": len(low_quality_sources)},
        )
        return AgentResult(
            success=True,
            message=f"Extracted {total_claims} claims from {len(sources)} sources",
        )

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

    def _enrich_claims_with_llm(
        self, claims: list[Claim], text: str, ctx: RunContext,
    ) -> list[Claim]:
        claims_summary = "\n".join(f"- [{c.claim_id}] {c.text[:150]}" for c in claims[:6])
        prompt = (
            f"Given these extracted claims, assign each a claim_type "
            f"(definition/method/result/limitation/trend/comparison) "
            f"and polarity (support/oppose/neutral). "
            f"Return JSON: {{\"claims\": [{{\"claim_id\": \"...\", \"claim_type\": \"...\", \"polarity\": \"...\"}}]}}\n\n"
            f"Claims:\n{claims_summary}"
        )
        try:
            raw = ctx.reasoner.complete_text(prompt, trace=ctx.trace)
            import json as _json
            data = _json.loads(raw) if raw.strip().startswith("{") else _json.loads("{" + raw.split("{", 1)[-1])
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
