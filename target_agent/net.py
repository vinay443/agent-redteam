"""Network egress allowlist.

The target agent has **no network tool** in this first pass (see SAFETY.md).
This module exists anyway, and is enforced today, because:

* the agent process itself makes exactly one kind of outbound call — to the
  Anthropic API — and :func:`assert_url_allowed` gates the ``base_url`` at
  startup, so a tampered ``ANTHROPIC_BASE_URL`` cannot redirect traffic;
* when a ``web_fetch`` tool is eventually added it must reuse this function
  rather than growing its own, weaker check.

The rules are deliberately strict — this is an allowlist, not a filter:

* ``https`` only (no ``http``, ``file``, ``gopher``, ``data`` …);
* no credentials in the URL (``https://user:pass@host``);
* host must match an allowlisted domain exactly, or be a subdomain of one;
* no bare IP literals — an IP cannot be allowlisted by name and is the usual
  shape of an SSRF payload;
* default port only.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import ParseResult, urlsplit

__all__ = ["EgressViolation", "assert_url_allowed", "is_url_allowed", "normalise_host"]


class EgressViolation(Exception):
    """Raised when a URL is not permitted by the egress allowlist."""

    def __init__(self, message: str, *, url: str, host: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.url = url
        self.host = host

    def as_dict(self) -> dict[str, str | None]:
        return {"message": self.message, "url": self.url, "host": self.host}


def normalise_host(host: str) -> str:
    """Lowercase, strip a trailing dot and any surrounding IPv6 brackets."""
    cleaned = host.strip().lower().rstrip(".")
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    return cleaned


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _host_matches(host: str, allowed: str) -> bool:
    allowed = normalise_host(allowed)
    return host == allowed or host.endswith("." + allowed)


def assert_url_allowed(url: str, allowlist: tuple[str, ...] | list[str]) -> ParseResult:
    """Validate ``url`` against ``allowlist``; return the parsed URL or raise.

    Raises:
        EgressViolation: for any URL that is not unambiguously permitted.
    """
    if not isinstance(url, str) or not url.strip():
        raise EgressViolation("url must be a non-empty string", url=str(url))

    parts = urlsplit(url.strip())

    if parts.scheme.lower() != "https":
        raise EgressViolation(
            f"scheme {parts.scheme!r} is not permitted; https only",
            url=url,
        )

    if parts.username or parts.password:
        raise EgressViolation("credentials in the URL are not permitted", url=url)

    host = normalise_host(parts.hostname or "")
    if not host:
        raise EgressViolation("url has no host", url=url)

    if _is_ip_literal(host):
        raise EgressViolation(
            "bare IP addresses are not permitted; allowlist entries are hostnames",
            url=url,
            host=host,
        )

    try:
        port = parts.port
    except ValueError as exc:
        raise EgressViolation(f"invalid port: {exc}", url=url, host=host) from exc
    if port not in (None, 443):
        raise EgressViolation(f"port {port} is not permitted; 443 only", url=url, host=host)

    if not allowlist:
        raise EgressViolation("egress allowlist is empty; all requests denied", url=url, host=host)

    if not any(_host_matches(host, entry) for entry in allowlist):
        raise EgressViolation(
            f"host {host!r} is not in the egress allowlist {tuple(allowlist)!r}",
            url=url,
            host=host,
        )

    # Re-parse with urlparse-compatible result for callers that want it.
    return ParseResult(
        scheme=parts.scheme,
        netloc=parts.netloc,
        path=parts.path,
        params="",
        query=parts.query,
        fragment=parts.fragment,
    )


def is_url_allowed(url: str, allowlist: tuple[str, ...] | list[str]) -> bool:
    """Boolean form of :func:`assert_url_allowed`."""
    try:
        assert_url_allowed(url, allowlist)
    except EgressViolation:
        return False
    return True
