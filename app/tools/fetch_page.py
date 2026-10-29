"""Production page fetcher with SSRF mitigation, timeouts, and HTML sanitization."""
import ipaddress
import socket
import time
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup

from app.config import get_settings
from app.schemas import FetchedPage, ToolCallRecord
from app.utils.errors import SSRFViolationError
from app.utils.logging_config import setup_logger

logger = setup_logger("tools.fetch_page")


class PageFetchTool:
    def __init__(self):
        self.settings = get_settings()

    def _validate_ssrf_safety(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            raise SSRFViolationError(f"Unsupported URI scheme: {parsed.scheme}")

        hostname = parsed.hostname
        if not hostname:
            raise SSRFViolationError("Missing host in URL")

        if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
            raise SSRFViolationError(f"Rejected access to localhost address: {hostname}")

        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip_obj = ipaddress.ip_address(ip_str)

                if (
                    ip_obj.is_private
                    or ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_multicast
                    or ip_obj.is_reserved
                    or ip_str in ("169.254.169.254", "0.0.0.0")
                ):
                    raise SSRFViolationError(f"Disallowed destination IP: {ip_str} for host: {hostname}")
        except socket.gaierror as e:
            raise SSRFViolationError(f"DNS resolution failure for host '{hostname}': {str(e)}")

    def fetch(self, url: str) -> tuple[FetchedPage, ToolCallRecord]:
        start = time.perf_counter()

        try:
            self._validate_ssrf_safety(url)

            headers = {
                "User-Agent": "AIResearchAgent/1.0 (Production Research Bot; Evaluation)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }

            with httpx.Client(
                timeout=self.settings.request_timeout_seconds,
                follow_redirects=False,
                headers=headers,
                max_redirects=0,
                trust_env=False
            ) as client:
                current_url = url
                response = None
                for _ in range(4):
                    # Validate each hop immediately before connecting. This
                    # prevents redirects from bypassing the initial check.
                    self._validate_ssrf_safety(current_url)
                    response = client.get(current_url)
                    if response.status_code not in (301, 302, 303, 307, 308):
                        break
                    location = response.headers.get("location")
                    if not location:
                        break
                    current_url = urljoin(current_url, location)
                else:
                    raise httpx.TooManyRedirects(
                        "Exceeded maximum redirect count",
                        request=client.build_request("GET", current_url)
                    )

                if response is None:
                    raise httpx.RequestError("No response received")

                if len(response.content) > self.settings.max_page_content_bytes:
                    content_bytes = response.content[:self.settings.max_page_content_bytes]
                else:
                    content_bytes = response.content

                soup = BeautifulSoup(content_bytes, "html.parser")

                for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
                    element.decompose()

                title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
                text = " ".join(soup.stripped_strings)

                if len(text) > self.settings.max_extracted_text_chars:
                    text = text[:self.settings.max_extracted_text_chars] + "... [TRUNCATED]"

                latency = (time.perf_counter() - start) * 1000
                record = ToolCallRecord(
                    tool_name="fetch_page",
                    query_or_url=url,
                    status="SUCCESS" if response.is_success else "FAILURE",
                    latency_ms=round(latency, 2),
                    error_message=None if response.is_success else f"HTTP {response.status_code}"
                )

                page = FetchedPage(
                    url=url,
                    canonical_url=str(response.url),
                    title=title,
                    extracted_text=text,
                    http_status=response.status_code,
                    char_count=len(text)
                )
                return page, record

        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            err_msg = f"Failed to fetch {url}: {type(exc).__name__} - {str(exc)}"
            logger.warning(err_msg)

            record = ToolCallRecord(
                tool_name="fetch_page",
                query_or_url=url,
                status="FAILURE",
                latency_ms=round(latency, 2),
                error_message=err_msg
            )
            page = FetchedPage(
                url=url,
                title="Error Loading Page",
                extracted_text="",
                http_status=500,
                char_count=0,
                error=err_msg
            )
            return page, record
