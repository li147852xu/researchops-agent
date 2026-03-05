from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from researchops.agents.base import AgentBase, RunContext
from researchops.models import AgentResult, PlanOutput, Source, SourceNotes, Stage
from researchops.tools.cite import reset_citations
from researchops.tools.parse_doc import NOISE_PATTERNS

_PREAMBLE_PATTERNS = [
    "in this article", "we'll look at", "we will explore", "let's dive",
    "click here", "read more", "this post", "this guide",
    "in this tutorial", "we'll discuss", "this blog",
    "you'll learn", "you will learn", "let us explore",
]

_EN_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "about", "between", "after", "before", "during",
    "and", "or", "but", "if", "so", "than", "that", "this", "these",
    "those", "it", "its", "they", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "not", "no", "also", "more", "such",
    "very", "just", "all", "each", "both", "few", "many", "much",
    "other", "some", "any", "what", "which", "who", "how", "when", "where",
}

_CN_STOP = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
}

_HEADING_CN_MAP: list[tuple[list[str], str]] = [
    (["introduction", "intro"], "引言"),
    (["key components", "overview", "概述"], "概述与关键组成"),
    (["current state", "state of research", "研究现状"], "研究现状"),
    (["challenges", "limitations", "挑战", "局限"], "主要挑战与局限"),
    (["future", "directions", "未来", "方向"], "未来方向"),
    (["conclusion", "summary", "总结"], "总结"),
    (["references", "参考"], "参考来源"),
    (["disagreements", "conflicts", "分歧"], "分歧与争议"),
    (["note", "说明"], "说明"),
]

_TEMPLATES_EN = [
    "Research indicates that {phrase}.",
    "Key factors include {phrase}.",
    "A notable aspect is {phrase}.",
    "Studies suggest that {phrase}.",
    "In terms of implementation, {phrase}.",
    "Evidence points to {phrase}.",
    "Analysis reveals that {phrase}.",
]

_TEMPLATES_CN = [
    "研究表明，{phrase}。",
    "关键要素包括{phrase}。",
    "值得注意的是，{phrase}。",
    "相关研究指出，{phrase}。",
    "在实现层面，{phrase}。",
    "证据表明，{phrase}。",
    "分析显示，{phrase}。",
]

_MAX_SENTENCE_LEN = 260
_MAX_PARA_LEN = 600
_MAX_CLAIMS_PER_SECTION = 6
_SOURCE_CAP_RATIO = 0.4


def _is_cjk_topic(topic: str) -> bool:
    if not topic:
        return False
    cjk_count = sum(1 for c in topic if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    return cjk_count / max(1, len(topic)) >= 0.3


def _localize_heading(heading: str, is_cjk: bool) -> str:
    if not is_cjk:
        cleaned = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf]+", "", heading).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned if cleaned else heading

    lower = heading.lower()
    for keywords, cn_heading in _HEADING_CN_MAP:
        if any(kw in lower for kw in keywords):
            return cn_heading

    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", heading))
    has_latin = bool(re.search(r"[a-zA-Z]{3,}", heading))
    if has_cjk and has_latin:
        cjk_only = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf\s]", "", heading).strip()
        if len(cjk_only) >= 2:
            return cjk_only
        for keywords, cn_heading in _HEADING_CN_MAP:
            if any(kw in lower for kw in keywords):
                return cn_heading
        return heading

    return heading


def _extract_key_phrases(text: str, max_words: int = 15) -> str:
    tokens = re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]+|[a-zA-Z]{3,}", text)
    filtered = []
    for t in tokens:
        tl = t.lower()
        if tl in _EN_STOP or tl in _CN_STOP:
            continue
        filtered.append(t)
        if len(filtered) >= max_words:
            break
    return " ".join(filtered) if filtered else text[:60]


_DISCARD_PATTERNS = [
    "not a research claim", "not a claim", "this is css", "this is javascript",
    "corrupted", "encoded data", "not meaningful", "cannot extract",
    "padding:", "margin:", "function(", "queryselector", "addeventlistener",
    "stylesheet", "doctype", "viewport", "meta charset",
]


def _is_noisy_claim(text: str) -> bool:
    lower = text.lower()
    if any(p in lower for p in NOISE_PATTERNS):
        return True
    if any(p in lower for p in _PREAMBLE_PATTERNS):
        return True
    return any(p in lower for p in _DISCARD_PATTERNS)


def _normalize(text: str) -> str:
    return re.sub(r"[\s\d\W]+", "", text.lower())


def _truncate_sentence(sentence: str, max_len: int = _MAX_SENTENCE_LEN) -> str:
    if len(sentence) <= max_len:
        return sentence
    marker_match = re.search(r"\s*\[@\w+\]\s*$", sentence)
    marker = marker_match.group(0) if marker_match else ""
    body = sentence[: len(sentence) - len(marker)] if marker else sentence
    budget = max_len - len(marker) - 3
    return body[:budget].rstrip() + "..." + marker


class WriterAgent(AgentBase):
    name = "writer"

    def execute(self, ctx: RunContext) -> AgentResult:
        ctx.trace.log(stage="WRITE", agent=self.name, action="start")
        reset_citations()

        plan = PlanOutput.model_validate(
            json.loads((ctx.run_dir / "plan.json").read_text(encoding="utf-8"))
        )
        sources = self._load_sources(ctx.run_dir)
        all_notes = self._load_all_notes(ctx.run_dir)
        is_cjk = _is_cjk_topic(plan.topic)
        templates = _TEMPLATES_CN if is_cjk else _TEMPLATES_EN

        retriever = ctx.shared.get("retriever")

        lines: list[str] = []
        report_index: list[dict] = []
        template_idx = 0
        unsupported_sections = 0

        lines.append(f"# {plan.topic}\n")
        gen_label = "*ResearchOps Agent 自动生成*" if is_cjk else "*Generated by ResearchOps Agent*"
        lines.append(f"{gen_label}\n")

        for section in plan.outline:
            heading = _localize_heading(section.heading, is_cjk)
            lines.append(f"\n## {heading}\n")

            if retriever and section.rq_refs:
                retrieved_claims = []
                for rq_ref in section.rq_refs:
                    retrieved_claims.extend(retriever.retrieve_for_rq(rq_ref, top_k=6))
                raw_claims = []
                seen_ids: set[str] = set()
                for rc in retrieved_claims:
                    cid = rc.get("claim_id", "")
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    raw_claims.append((rc.get("text", ""), rc.get("source_id", ""), cid))
            else:
                raw_claims = self._claims_for_section(section.rq_refs, all_notes)

            filtered: list[tuple[str, str, str]] = []
            seen_norms: set[str] = set()
            for claim_text, source_id, claim_id in raw_claims:
                if _is_noisy_claim(claim_text):
                    continue
                norm = _normalize(claim_text)
                if norm in seen_norms:
                    continue
                seen_norms.add(norm)
                filtered.append((claim_text, source_id, claim_id))

            source_counts: dict[str, int] = {}
            max_per_source = max(1, int(_MAX_CLAIMS_PER_SECTION * _SOURCE_CAP_RATIO))
            selected: list[tuple[str, str, str]] = []
            for claim_text, source_id, claim_id in filtered:
                if len(selected) >= _MAX_CLAIMS_PER_SECTION:
                    break
                cnt = source_counts.get(source_id, 0)
                if cnt >= max_per_source:
                    continue
                source_counts[source_id] = cnt + 1
                selected.append((claim_text, source_id, claim_id))

            if not selected and filtered:
                selected = filtered[:_MAX_CLAIMS_PER_SECTION]

            if selected:
                para_sentences: list[str] = []
                for claim_text, source_id, claim_id in selected:
                    summary = self._summarize_claim(claim_text, ctx, templates, template_idx)
                    template_idx += 1
                    marker = f" [@{source_id}]"
                    sentence = summary + marker
                    sentence = _truncate_sentence(sentence)
                    para_sentences.append(sentence)

                    report_index.append({
                        "hash": hashlib.sha256(sentence.encode()).hexdigest()[:12],
                        "text": summary[:200],
                        "source_ids": [source_id],
                        "claim_ids": [claim_id],
                    })

                paragraphs = self._group_into_paragraphs(para_sentences)
                for para in paragraphs:
                    lines.append(f"{para}\n")
                lines.append("")
            else:
                unsupported_sections += 1
                if is_cjk:
                    fallback = "该部分证据不足，需要进一步收集研究资料。"
                else:
                    fallback = "Evidence insufficient for this section — further collection needed."
                lines.append(f"{fallback}\n")
                report_index.append({
                    "hash": hashlib.sha256(fallback.encode()).hexdigest()[:12],
                    "text": fallback,
                    "source_ids": [],
                    "claim_ids": [],
                })
                ctx.trace.log(
                    stage="WRITE", agent=self.name, action="evidence_insufficient",
                    meta={"section": heading, "rq_refs": section.rq_refs},
                )

        if ctx.shared.get("evidence_limited"):
            note_heading = "说明" if is_cjk else "Note"
            lines.append(f"\n## {note_heading}\n")
            if is_cjk:
                lines.append("*由于离线模式（allow_net=false），证据收集受限。部分研究问题可能覆盖不足。*\n")
            else:
                lines.append("*Evidence collection was limited due to offline mode (allow_net=false). "
                             "Some research questions may have insufficient coverage.*\n")

        if ctx.shared.get("has_conflicts"):
            conf_heading = "分歧与争议" if is_cjk else "Disagreements and Conflicts"
            lines.append(f"\n## {conf_heading}\n")
            conflicts_path = ctx.run_dir / "qa_conflicts.json"
            if conflicts_path.exists():
                try:
                    cdata = json.loads(conflicts_path.read_text(encoding="utf-8"))
                    for conflict in cdata.get("conflicts", [])[:3]:
                        rq = conflict.get("rq_id", "unknown")
                        lines.append(f"- **{rq}**: {conflict.get('support_count', 0)} supporting vs "
                                     f"{conflict.get('oppose_count', 0)} opposing claims\n")
                except Exception:
                    lines.append("- Conflicting evidence was detected across sources.\n")
            else:
                lines.append("- Conflicting evidence was detected across sources.\n")

        ref_heading = "参考来源" if is_cjk else "References"
        lines.append(f"\n## {ref_heading}\n")
        seen_refs: set[str] = set()
        for src in sources:
            ref_key = f"{src.title}|{src.domain}"
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            lines.append(f"- [@{src.source_id}] {src.title or src.url} ({src.domain})")

        report_path = ctx.run_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        index_path = ctx.run_dir / "report_index.json"
        index_path.write_text(
            json.dumps({"entries": report_index}, indent=2), encoding="utf-8"
        )

        if unsupported_sections > 0 and ctx.config.allow_net:
            ctx.trace.log(
                stage="WRITE", agent=self.name, action="evidence_warning",
                output_summary=f"{unsupported_sections} sections had insufficient evidence",
            )
            return AgentResult(
                success=False,
                message=f"Report written but {unsupported_sections} sections lack evidence",
                rollback_to=Stage.COLLECT,
            )

        ctx.trace.log(
            stage="WRITE",
            agent=self.name,
            action="complete",
            output_summary=f"Report: {len(lines)} lines, {len(report_index)} indexed entries",
        )
        return AgentResult(success=True, message="Report written with traceability index")

    def _summarize_claim(
        self, claim_text: str, ctx: RunContext, templates: list[str], template_idx: int,
    ) -> str:
        if ctx.reasoner.is_llm:
            try:
                prompt = (
                    f"Summarize the following research claim in one concise sentence (max 200 chars). "
                    f"Do NOT include citation markers. Just output the summary sentence.\n\n"
                    f"Claim: {claim_text[:500]}"
                )
                result = ctx.reasoner.complete_text(prompt, trace=ctx.trace)
                result = result.strip().strip('"').strip("'")
                if 20 < len(result) < 260 and not _is_noisy_claim(result):
                    result_lower = result.lower()
                    if any(dp in result_lower for dp in _DISCARD_PATTERNS):
                        pass
                    else:
                        return result
            except Exception:
                pass
        phrase = _extract_key_phrases(claim_text)
        tpl = templates[template_idx % len(templates)]
        return tpl.format(phrase=phrase)

    def _group_into_paragraphs(self, sentences: list[str]) -> list[str]:
        paragraphs: list[str] = []
        current: list[str] = []
        current_len = 0

        for s in sentences:
            if current_len + len(s) + 1 > _MAX_PARA_LEN and current:
                paragraphs.append(" ".join(current))
                current = [s]
                current_len = len(s)
            else:
                current.append(s)
                current_len += len(s) + 1

            if len(current) >= 4:
                paragraphs.append(" ".join(current))
                current = []
                current_len = 0

        if current:
            paragraphs.append(" ".join(current))

        return [p for p in paragraphs if p.strip()]

    def _load_sources(self, run_dir: Path) -> list[Source]:
        path = run_dir / "sources.jsonl"
        if not path.exists():
            return []
        sources = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                sources.append(Source.model_validate(json.loads(line)))
        return sources

    def _load_all_notes(self, run_dir: Path) -> dict[str, SourceNotes]:
        notes_dir = run_dir / "notes"
        result: dict[str, SourceNotes] = {}
        if not notes_dir.exists():
            return result
        for f in sorted(notes_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                notes = SourceNotes.model_validate(data)
                result[notes.source_id] = notes
            except Exception:
                continue
        return result

    def _claims_for_section(
        self, rq_refs: list[str], all_notes: dict[str, SourceNotes]
    ) -> list[tuple[str, str, str]]:
        results: list[tuple[str, str, str]] = []
        for src_id, notes in all_notes.items():
            for claim in notes.claims:
                if not rq_refs or any(r in claim.supports_rq for r in rq_refs):
                    results.append((claim.text, src_id, claim.claim_id))
        return results
