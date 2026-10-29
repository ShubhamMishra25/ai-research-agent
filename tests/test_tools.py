"""Tests for tool behavior, SSRF mitigation, timeouts, and error handling."""
import pytest
from unittest.mock import patch, MagicMock
import httpx
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


def _stream_response(status_code=200, headers=None, chunks=None):
    response = httpx.Response(
        status_code,
        headers=headers or {},
        request=httpx.Request("GET", "https://public.example"),
    )
    response.iter_bytes = MagicMock(return_value=iter(chunks or [b"<html><title>OK</title>content</html>"]))
    return response


def test_fetch_page_success_and_stream_limit():
    fetcher = PageFetchTool()
    fetcher.settings.max_page_content_bytes = 10
    response = _stream_response(chunks=[b"1234567890", b"this must not be read"])
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.stream.return_value.__enter__.return_value = response
    client.stream.return_value.__exit__.return_value = None

    with patch.object(fetcher, "_validate_ssrf_safety"), patch(
        "app.tools.fetch_page.httpx.Client", return_value=client
    ):
        page, record = fetcher.fetch("https://public.example")

    assert record.status == "SUCCESS"
    assert page.extracted_text == "1234567890"


def test_fetch_page_rejects_private_redirect():
    fetcher = PageFetchTool()
    redirect = _stream_response(
        status_code=302,
        headers={"location": "http://127.0.0.1/admin"},
        chunks=[],
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.stream.return_value.__enter__.return_value = redirect
    client.stream.return_value.__exit__.return_value = None

    with patch.object(
        fetcher,
        "_validate_ssrf_safety",
        side_effect=[None, None, SSRFViolationError("private redirect")],
    ) as validate, patch(
        "app.tools.fetch_page.httpx.Client", return_value=client
    ):
        fetcher.fetch("https://public.example")

    assert validate.call_count == 3
    validate.assert_called_with("http://127.0.0.1/admin")