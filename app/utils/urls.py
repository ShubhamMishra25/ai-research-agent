"""URL canonicalization used for deduplication (not for citation display)."""
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

DIRECT_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


def canonicalize_url(url: str) -> str:
    """Return a stable dedupe key while preserving the original URL elsewhere."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS
    ]
    return urlunparse((scheme, netloc, path, "", urlencode(query_pairs), ""))


def extract_direct_urls(text: str) -> list[str]:
    return [match.rstrip(".,;") for match in DIRECT_URL_RE.findall(text or "")]
