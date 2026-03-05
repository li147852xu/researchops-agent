"""Tests for v0.3.3 fetch integrity and data pipeline fixes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchops.tools.fetch_page import _detect_type, fetch_page
from researchops.tools.parse_doc import parse_doc


@pytest.fixture()
def tmp_dest(tmp_path: Path) -> Path:
    d = tmp_path / "downloads"
    d.mkdir()
    return d


# ── B1: Magic-number detection ──────────────────────────────────────


def test_magic_number_pdf_detected():
    data = b"%PDF-1.4 some pdf content" + b"\x00" * 3000
    assert _detect_type(data) == "pdf"


def test_magic_number_html_detected():
    data = b"<html><head></head><body>Hello world</body></html>"
    assert _detect_type(data) == "html"


def test_magic_number_doctype_html():
    data = b"<!DOCTYPE html><html><body>content</body></html>"
    assert _detect_type(data) == "html"


def test_magic_number_unknown():
    data = b"just some plain text content here nothing special"
    assert _detect_type(data) == "unknown"


# ── B1: PDF URL + HTML content = rejected ───────────────────────────


def test_pdf_url_html_content_rejected(tmp_dest: Path):
    html_body = b"<html><body>" + b"A" * 3000 + b"</body></html>"
    with patch("researchops.tools.fetch_page.urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = html_body
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        result = fetch_page("https://example.com/paper.pdf", str(tmp_dest))

    assert result["status"] == "failed"
    assert "captcha" in result["error"].lower() or "html" in result["error"].lower()
    assert result["local_path"] == ""


# ── B1: Successful fetch has local_path and hash ────────────────────


def test_fetch_success_has_local_path(tmp_dest: Path):
    content = b"<html><body>" + b"Real content here. " * 200 + b"</body></html>"
    with patch("researchops.tools.fetch_page.urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = content
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        result = fetch_page("https://example.com/article", str(tmp_dest))

    assert result["status"] == "success"
    assert result["local_path"] != ""
    assert Path(result["local_path"]).exists()
    assert result["content_hash"] != ""
    assert result["detected_type"] == "html"


# ── B1: Small response rejected ─────────────────────────────────────


def test_small_response_rejected(tmp_dest: Path):
    content = b"<html><body>tiny</body></html>"
    with patch("researchops.tools.fetch_page.urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = content
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        result = fetch_page("https://example.com/page", str(tmp_dest))

    assert result["status"] == "failed"
    assert "small" in result["error"].lower() or "bytes" in result["error"].lower()


# ── B4: parse_doc rejects fake PDF ──────────────────────────────────


def test_parse_rejects_fake_pdf(tmp_path: Path):
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"<html><body>This is not a PDF</body></html>")

    result = parse_doc(str(fake_pdf), format="pdf")
    assert result["text"] == ""
    assert "warning" in result or "error" in result


# ── B4: parse_doc quality gate: code-heavy content ──────────────────


def test_parse_code_heavy_rejected(tmp_path: Path):
    code_lines = "\n".join(
        f"function test{i}() {{ padding: {i}px; margin: 10px; color: rgb({i},{i},{i}); }}"
        for i in range(50)
    )
    html_file = tmp_path / "code.html"
    html_file.write_text(f"<html><body><pre>{code_lines}</pre></body></html>")

    result = parse_doc(str(html_file))
    assert result["text"] == "" or result.get("warning") in ("code_heavy", "low_quality")


# ── B4: parse_doc quality gate: too short ────────────────────────────


def test_parse_low_quality_short(tmp_path: Path):
    html_file = tmp_path / "short.html"
    html_file.write_text("<html><body><p>Short.</p></body></html>")

    result = parse_doc(str(html_file))
    assert result["text"] == "" or result.get("warning") == "low_quality"


# ── B6: QA code garbage detector ────────────────────────────────────


def test_qa_code_garbage_rollback():
    from researchops.agents.qa import QAAgent

    qa = QAAgent()
    report = "\n".join([
        "# Test Report",
        "function() { padding: 10px; margin: 5px; }",
        "querySelector('.test') addEventListener('click')",
        "stylesheet meta charset viewport",
        "Normal paragraph about research.",
    ])
    issues = qa._code_garbage_detector(report)
    assert len(issues) > 0
    assert "code_garbage" in issues[0].lower()


# ── B6: QA source availability check ────────────────────────────────


def test_qa_source_availability_fail(tmp_path: Path):
    from researchops.agents.qa import QAAgent
    from researchops.models import Source, SourceType

    qa = QAAgent()
    sources = [
        Source(source_id="s1", type=SourceType.HTML, local_path=str(tmp_path / "nonexistent.html")),
    ]
    issues = qa._source_availability_check(sources, "fast")
    assert len(issues) > 0
    assert "source_availability" in issues[0]
