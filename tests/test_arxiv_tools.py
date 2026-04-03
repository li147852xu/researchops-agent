from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

SAMPLE_ARXIV_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Quantum Computing Survey</title>
    <published>2023-01-15T00:00:00Z</published>
    <summary>This paper surveys recent advances in quantum computing.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <arxiv:primary_category term="quant-ph"/>
    <link title="pdf" href="http://arxiv.org/pdf/2301.00001v1" rel="related"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00002v1</id>
    <title>Quantum Error Correction</title>
    <published>2023-02-20T00:00:00Z</published>
    <summary>A study on error correction methods for quantum systems.</summary>
    <author><name>Charlie Brown</name></author>
    <arxiv:primary_category term="quant-ph"/>
    <link title="pdf" href="http://arxiv.org/pdf/2301.00002v1" rel="related"/>
  </entry>
</feed>"""


def test_arxiv_search_parses_xml():
    """arxiv_search should parse arXiv Atom XML into structured paper list."""
    from researchops.apps.research.adapters import arxiv_search

    mock_resp = MagicMock()
    mock_resp.read.return_value = SAMPLE_ARXIV_XML
    mock_resp.status = 200
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = arxiv_search("quantum computing", max_results=5)

    assert len(results) == 2
    assert results[0]["arxiv_id"] == "2301.00001v1"
    assert results[0]["title"] == "Quantum Computing Survey"
    assert "Alice Smith" in results[0]["authors"]
    assert results[0]["categories"] == ["quant-ph"]
    assert results[0]["abstract"].startswith("This paper surveys")
    assert results[0]["pdf_url"].endswith("2301.00001v1")


def test_arxiv_search_handles_error():
    """arxiv_search should return error dict on network failure."""
    from researchops.apps.research.adapters import arxiv_search

    with patch("urllib.request.urlopen", side_effect=Exception("connection failed")):
        results = arxiv_search("test query")

    assert len(results) == 1
    assert "error" in results[0]


def test_arxiv_download_pdf_magic_number():
    """arxiv_download_pdf should reject non-PDF responses."""
    from researchops.apps.research.adapters import arxiv_download_pdf

    html_resp = MagicMock()
    html_resp.read.return_value = b"<html><body>Not a PDF</body></html>"
    html_resp.status = 200
    html_resp.__enter__ = lambda s: s
    html_resp.__exit__ = MagicMock(return_value=False)

    with tempfile.TemporaryDirectory() as tmpdir, patch("urllib.request.urlopen", return_value=html_resp):
        result = arxiv_download_pdf("http://arxiv.org/pdf/test", tmpdir)

    assert result["status"] == "failed"
    assert "not a PDF" in result["error"]


def test_arxiv_download_pdf_success():
    """arxiv_download_pdf should save valid PDFs."""
    from researchops.apps.research.adapters import arxiv_download_pdf

    pdf_data = b"%PDF-1.4 " + b"x" * 5000

    pdf_resp = MagicMock()
    pdf_resp.read.return_value = pdf_data
    pdf_resp.status = 200
    pdf_resp.__enter__ = lambda s: s
    pdf_resp.__exit__ = MagicMock(return_value=False)

    with tempfile.TemporaryDirectory() as tmpdir, patch("urllib.request.urlopen", return_value=pdf_resp):
        result = arxiv_download_pdf("http://arxiv.org/pdf/test", tmpdir)
        assert result["status"] == "success"
        assert result["detected_type"] == "pdf"
        assert result["bytes"] > 4096
        assert Path(result["local_path"]).exists()
