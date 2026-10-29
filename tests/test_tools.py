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

    # Verify validator raises SSRFViolationError directly
    with pytest.raises(SSRFViolationError):
        fetcher._validate_ssrf_safety("http://localhost:8080/admin")
    with pytest.raises(SSRFViolationError):
        fetcher._validate_ssrf_safety("http://127.0.0.1/secret")
    with pytest.raises(SSRFViolationError):
        fetcher._validate_ssrf_safety("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(SSRFViolationError):
        fetcher._validate_ssrf_safety("ftp://ftp.example.com/file.txt")

    # Verify public fetch() absorbs the error gracefully without crashing
    page, record = fetcher.fetch("http://localhost:8080/admin")
    assert record.status == "FAILURE"
    assert "SSRFViolationError" in record.error_message