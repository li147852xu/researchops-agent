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
    (["evidence gap", "证据缺口"], "证据缺口"),
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
_MAX_CLAIMS_PER_SECTION = 8
_SOURCE_CAP_RATIO = 0.35


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


def _clean_rq_heading(heading: str) -> str:
    heading = re.sub(r"^(Is|Are|What|How|Why|Does|Do|Can|Could|Should)\s+", "", heading, flags=re.IGNORECASE)
    heading = re.sub(r"\s+and\s+what\s+are\s+", " — ", heading, flags=re.IGNORECASE)
    heading = heading.strip().rstrip("?")
    if heading:
        heading = heading[0].upper() + heading[1:]
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
        is_deep = ctx.config.mode.value == "deep"

        retriever = ctx.shared.get("retriever")
        incomplete_sections = list(ctx.state.incomplete_sections)
        at_max_rounds = ctx.state.collect_rounds >= ctx.config.max_collect_rounds

        evidence_map = self._build_evidence_map(plan, retriever, all_notes, ctx)
        emap_path = ctx.run_dir / "evidence_map.json"
        emap_path.write_text(json.dumps(evidence_map, indent=2), encoding="utf-8")

        lines: list[str] = []
        report_index: list[dict] = []
        template_idx = 0
        unsupported_sections = 0
        evidence_gap_sections: list[str] = []

        lines.append(f"# {plan.topic}\n")
        gen_label = "*ResearchOps Agent 自动生成*" if is_cjk else "*Generated by ResearchOps Agent*"
        lines.append(f"{gen_label}\n")

        for section in plan.outline:
            heading = _localize_heading(section.heading, is_cjk)
            heading = _clean_rq_heading(heading)
            lines.append(f"\n## {heading}\n")

            emap_entry = evidence_map.get(section.heading, {})
            selected = self._gather_claims_for_section(
                section.rq_refs, retriever, all_notes,
            )

            if selected and (emap_entry.get("source_count", 0) > 0 or not emap_entry):
                overview, bullets, trend_block, idx_entries, new_idx = self._write_section(
                    selected, ctx, templates, template_idx, heading, is_cjk, is_deep,
                )
                template_idx = new_idx

                if overview:
                    lines.append(f"{overview}\n")
                if bullets:
                    lines.append("")
                    for b in bullets:
                        lines.append(b)
                    lines.append("")
                if trend_block:
                    lines.append(f"{trend_block}\n")

                report_index.extend(idx_entries)
            else:
                unsupported_sections += 1
                evidence_gap_sections.append(heading)

                if at_max_rounds or heading in [_localize_heading(s, is_cjk) for s in incomplete_sections]:
                    gap_text = self._write_evidence_gap(heading, section.rq_refs, is_cjk)
                    lines.append(f"{gap_text}\n")
                    report_index.append({
                        "hash": hashlib.sha256(gap_text.encode()).hexdigest()[:12],
                        "text": gap_text[:200],
                        "source_ids": [],
                        "claim_ids": [],
                        "evidence_gap": True,
                    })
                else:
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
                    meta={"section": heading, "rq_refs": section.rq_refs, "at_max_rounds": at_max_rounds},
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

        if unsupported_sections > 0 and ctx.config.allow_net and not at_max_rounds:
            ctx.state.incomplete_sections = evidence_gap_sections
            ctx.trace.log(
                stage="WRITE", agent=self.name, action="evidence_warning",
                output_summary=f"{unsupported_sections} sections had insufficient evidence",
                meta={"evidence_gap_sections": evidence_gap_sections},
            )
            return AgentResult(
                success=False,
                message=f"Report written but {unsupported_sections} sections lack evidence",
                rollback_to=Stage.COLLECT,
            )

        if evidence_gap_sections:
            ctx.state.incomplete_sections = evidence_gap_sections

        ctx.trace.log(
            stage="WRITE",
            agent=self.name,
            action="complete",
            output_summary=f"Report: {len(lines)} lines, {len(report_index)} indexed entries",
            meta={"evidence_gap_sections": evidence_gap_sections},
        )
        return AgentResult(success=True, message="Report written with traceability index")

    def _gather_claims_for_section(
        self, rq_refs: list[str], retriever, all_notes: dict[str, SourceNotes],
    ) -> list[tuple[str, str, str]]:
        if retriever and rq_refs:
            retrieved_claims = []
            for rq_ref in rq_refs:
                retrieved_claims.extend(retriever.retrieve_for_rq(rq_ref, top_k=8))
            raw_claims = []
            seen_ids: set[str] = set()
            for rc in retrieved_claims:
                cid = rc.get("claim_id", "")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                raw_claims.append((rc.get("text", ""), rc.get("source_id", ""), cid))
        else:
            raw_claims = self._claims_for_section(rq_refs, all_notes)

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

        return selected

    def _write_section(
        self,
        claims: list[tuple[str, str, str]],
        ctx: RunContext,
        templates: list[str],
        template_idx: int,
        heading: str,
        is_cjk: bool,
        is_deep: bool,
    ) -> tuple[str, list[str], str, list[dict], int]:
        idx_entries: list[dict] = []

        overview_claims = claims[:3]
        bullet_claims = claims[:6]
        trend_claims = [c for c in claims if len(c[0]) > 60][:3] if is_deep else []

        if ctx.reasoner.is_llm:
            overview, bullets, trend_block, idx_entries, template_idx = self._llm_write_section(
                claims, ctx, heading, is_cjk, is_deep, template_idx,
            )
            if overview or bullets:
                return overview, bullets, trend_block, idx_entries, template_idx

        overview_sentences: list[str] = []
        for claim_text, source_id, claim_id in overview_claims:
            summary = self._summarize_claim(claim_text, ctx, templates, template_idx)
            template_idx += 1
            sentence = _truncate_sentence(f"{summary} [@{source_id}]")
            overview_sentences.append(sentence)
            idx_entries.append({
                "hash": hashlib.sha256(sentence.encode()).hexdigest()[:12],
                "text": summary[:200],
                "source_ids": [source_id],
                "claim_ids": [claim_id],
            })

        overview = " ".join(overview_sentences)

        bullets: list[str] = []
        for claim_text, source_id, claim_id in bullet_claims:
            phrase = _extract_key_phrases(claim_text, max_words=12)
            bullet = f"- {phrase} [@{source_id}]"
            bullet = _truncate_sentence(bullet)
            bullets.append(bullet)
            idx_entries.append({
                "hash": hashlib.sha256(bullet.encode()).hexdigest()[:12],
                "text": phrase[:200],
                "source_ids": [source_id],
                "claim_ids": [claim_id],
            })

        trend_block = ""
        if is_deep and trend_claims:
            trend_parts: list[str] = []
            for claim_text, source_id, _claim_id in trend_claims:
                summary = self._summarize_claim(claim_text, ctx, templates, template_idx)
                template_idx += 1
                trend_parts.append(_truncate_sentence(f"{summary} [@{source_id}]"))
            trend_block = " ".join(trend_parts)

        return overview, bullets, trend_block, idx_entries, template_idx

    def _llm_write_section(
        self,
        claims: list[tuple[str, str, str]],
        ctx: RunContext,
        heading: str,
        is_cjk: bool,
        is_deep: bool,
        template_idx: int,
    ) -> tuple[str, list[str], str, list[dict], int]:
        idx_entries: list[dict] = []

        claims_block = "\n".join(
            f"[{sid}] {text[:200]}" for text, sid, cid in claims
        )
        lang = "Chinese" if is_cjk else "English"
        deep_instruction = ""
        if is_deep:
            deep_instruction = (
                "Also write a 'Trends and Limitations' paragraph (2-3 sentences) "
                "comparing approaches or noting limitations. Every sentence must end with [@source_id]."
            )

        prompt = (
            f"Write a research report section titled '{heading}' in {lang}.\n"
            f"Available evidence (each prefixed with [source_id]):\n{claims_block}\n\n"
            f"Requirements:\n"
            f"1. Write an overview paragraph (2-4 sentences) synthesizing the key findings. "
            f"Every sentence MUST end with a citation marker [@source_id] from the evidence above.\n"
            f"2. Write 3-6 bullet points (each starting with '- '), each with a [@source_id] citation.\n"
            f"{deep_instruction}\n"
            f"Output format:\n"
            f"OVERVIEW: <paragraph>\n"
            f"BULLETS:\n- point1 [@source_id]\n- point2 [@source_id]\n"
            f"TRENDS: <paragraph or empty>\n"
            f"CRITICAL: Every single sentence and bullet must contain at least one [@source_id] marker."
        )

        try:
            raw = ctx.reasoner.complete_text(prompt, trace=ctx.trace)
            overview, bullets, trend_block = self._parse_llm_section_output(raw, claims)

            if not re.search(r"\[@\w+\]", overview) and claims:
                overview = ""

            valid_bullets = [b for b in bullets if re.search(r"\[@\w+\]", b)]
            if not valid_bullets and not overview:
                return "", [], "", [], template_idx

            for sentence in re.split(r"(?<=[.!?。！？])\s*", overview):
                sentence = sentence.strip()
                if len(sentence) > 20:
                    marker_match = re.search(r"\[@(\w+)\]", sentence)
                    sid = marker_match.group(1) if marker_match else ""
                    cid = ""
                    for _, s, c in claims:
                        if s == sid:
                            cid = c
                            break
                    idx_entries.append({
                        "hash": hashlib.sha256(sentence.encode()).hexdigest()[:12],
                        "text": sentence[:200],
                        "source_ids": [sid] if sid else [],
                        "claim_ids": [cid] if cid else [],
                    })

            for b in valid_bullets:
                marker_match = re.search(r"\[@(\w+)\]", b)
                sid = marker_match.group(1) if marker_match else ""
                cid = ""
                for _, s, c in claims:
                    if s == sid:
                        cid = c
                        break
                idx_entries.append({
                    "hash": hashlib.sha256(b.encode()).hexdigest()[:12],
                    "text": b[:200],
                    "source_ids": [sid] if sid else [],
                    "claim_ids": [cid] if cid else [],
                })

            return overview, valid_bullets, trend_block, idx_entries, template_idx

        except Exception:
            return "", [], "", [], template_idx

    def _parse_llm_section_output(
        self, raw: str, claims: list[tuple[str, str, str]],
    ) -> tuple[str, list[str], str]:
        overview = ""
        bullets: list[str] = []
        trend_block = ""

        if "OVERVIEW:" in raw:
            parts = raw.split("OVERVIEW:", 1)[1]
            if "BULLETS:" in parts:
                overview = parts.split("BULLETS:", 1)[0].strip()
                rest = parts.split("BULLETS:", 1)[1]
                if "TRENDS:" in rest:
                    bullet_text = rest.split("TRENDS:", 1)[0]
                    trend_block = rest.split("TRENDS:", 1)[1].strip()
                else:
                    bullet_text = rest
                for line in bullet_text.strip().splitlines():
                    line = line.strip()
                    if line.startswith("- "):
                        bullets.append(line)
            else:
                overview = parts.strip()
        else:
            lines = raw.strip().splitlines()
            para_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("- "):
                    bullets.append(line)
                elif len(line) > 30:
                    para_lines.append(line)
            overview = " ".join(para_lines[:4])

        return overview, bullets, trend_block

    def _write_evidence_gap(self, heading: str, rq_refs: list[str], is_cjk: bool) -> str:
        if is_cjk:
            rq_list = "、".join(rq_refs) if rq_refs else "未知"
            return (
                f"**证据缺口**：本节（{heading}）相关的研究问题（{rq_list}）"
                f"在当前检索范围内证据不足。建议补充该领域的综述论文、实验报告或最新会议论文。"
            )
        rq_list = ", ".join(rq_refs) if rq_refs else "unknown"
        return (
            f"**Evidence Gap**: This section ({heading}) covering research questions ({rq_list}) "
            f"has insufficient evidence in the current search scope. "
            f"Recommended: add survey papers, experimental reports, or recent conference papers in this area."
        )

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
                    if not any(dp in result_lower for dp in _DISCARD_PATTERNS):
                        return result
            except Exception:
                pass
        phrase = _extract_key_phrases(claim_text)
        tpl = templates[template_idx % len(templates)]
        return tpl.format(phrase=phrase)

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

    def _build_evidence_map(
        self,
        plan: PlanOutput,
        retriever,
        all_notes: dict[str, SourceNotes],
        ctx: RunContext,
    ) -> dict:
        evidence_map: dict = {}
        checklist = plan.coverage_checklist
        bucket_map = {}
        for bucket in checklist:
            bid = bucket.get("bucket_id", "")
            bucket_map[bid] = bucket

        for section in plan.outline:
            heading = section.heading
            rq_refs = section.rq_refs
            claims_data = self._gather_claims_for_section(rq_refs, retriever, all_notes)

            matched_buckets: list[str] = []
            for bucket in checklist:
                bid = bucket.get("bucket_id", "")
                bname = bucket.get("bucket_name", "").lower()
                if bname and bname in heading.lower():
                    matched_buckets.append(bid)
                    continue
                if rq_refs:
                    for _sid, notes in all_notes.items():
                        if bid in notes.bucket_hits and bid not in matched_buckets and any(
                            c.supports_rq and any(r in c.supports_rq for r in rq_refs) for c in notes.claims
                        ):
                            matched_buckets.append(bid)

            source_ids = list({sid for _, sid, _ in claims_data})
            unique_domains = set()
            for src_id in source_ids:
                for note in all_notes.values():
                    if note.source_id == src_id:
                        domain = note.bibliographic.get("domain", "")
                        if domain:
                            unique_domains.add(domain)

            target = ctx.config.target_sources_per_bucket
            evidence_map[heading] = {
                "bucket_ids": matched_buckets,
                "claims": [
                    {"claim_id": cid, "source_id": sid, "text": text[:200]}
                    for text, sid, cid in claims_data[:_MAX_CLAIMS_PER_SECTION]
                ],
                "source_count": len(source_ids),
                "diversity_ok": len(unique_domains) >= max(1, target // 2) if source_ids else False,
            }

        return evidence_map

    def _claims_for_section(
        self, rq_refs: list[str], all_notes: dict[str, SourceNotes]
    ) -> list[tuple[str, str, str]]:
        results: list[tuple[str, str, str]] = []
        for src_id, notes in all_notes.items():
            for claim in notes.claims:
                if not rq_refs or any(r in claim.supports_rq for r in rq_refs):
                    results.append((claim.text, src_id, claim.claim_id))
        return results
