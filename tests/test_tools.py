"""Tests for tool behavior, SSRF mitigation, timeouts, and error handling."""
import pytest
from unittest.mock import patch, MagicMock
from app.tools.web_search import WebSearchTool
from app.tools.fetch_page import PageFetchTool
from app.utils.errors import SSRFViolationError


def test_web_search_missing_key_graceful():
    tool = WebSearchTool(api_key="")
    results, record = tool.search("test query")
    assert results == []
    assert record.status == "FAILURE"
    assert "not configured" in record.error_message


def test_web_search_success_mock():
    tool = WebSearchTool(api_key="mock_key")
    mock_payload = {
        "results": [
            {"title": "Doc A", "url": "https://example.com/a", "content": "Sample snippet", "score": 0.9},
            {"title": "Doc A duplicate", "url": "https://example.com/a", "content": "Duplicate", "score": 0.8}
        ]
    }
    with patch("tavily.TavilyClient.search", return_value=mock_payload):
        results, record = tool.search("test query")
        assert len(results) == 1
        assert results[0].url == "https://example.com/a"
        assert record.status == "SUCCESS"


def test_fetch_page_ssrf_rejections():
    fetcher = PageFetchTool()
    with pytest.raises(SSRFViolationError):
        fetcher.fetch("http://localhost:8080/admin")
    with pytest.raises(SSRFViolationError):
        fetcher.fetch("http://127.0.0.1/secret")
    with pytest.raises(SSRFViolationError):
        fetcher.fetch("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(SSRFViolationError):
        fetcher.fetch("ftp://ftp.example.com/file.txt")


def test_fetch_page_success_mock():
    fetcher = PageFetchTool()
    html_content = b"<html><head><title>Vector DBs</title></head><body><h1>Guide</h1><p>Qdrant is fast.</p></body></html>"

    mock_resp = MagicMock()
    mock_resp.content = html_content
    mock_resp.is_success = True
    mock_resp.status_code = 200
    mock_resp.url = "https://example.com/vector-db"

    with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 80))]):
        with patch("httpx.Client.get", return_value=mock_resp):
            page, record = fetcher.fetch("https://example.com/vector-db")
            assert record.status == "SUCCESS"
            assert page.title == "Vector DBs"
            assert "Qdrant is fast." in page.extracted_text
