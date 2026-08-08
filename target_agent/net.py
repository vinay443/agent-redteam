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

## The one exception: the local model endpoint

An allowlist entry may carry an explicit port (``host.docker.internal:11434``).
Such an entry permits **exactly** that host on **exactly** that port, over http
or https, and nothing else — no subdomains, no other ports, no other host.

It exists for a single case: the container running against a model server on the
host machine (Ollama). ``localhost`` inside the container is the container, so
the host is reached through the ``host.docker.internal`` alias that
``docker-compose.yml`` maps to the host gateway. That endpoint is plaintext http
on a non-default port, which the general rules forbid — hence a narrow, named
exception rather than relaxing the general rules.

To keep it narrow, only hosts in :data:`LOCAL_MODEL_HOST_ALIASES` may appear in
a ``host:port`` entry. ``evil.example.com:80`` is not a misconfiguration that
silently grants plaintext egress to the internet — it raises.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import ParseResult, urlsplit

__all__ = [
    "EgressViolation",
    "LOCAL_MODEL_HOST_ALIASES",
    "assert_url_allowed",
    "is_url_allowed",
    "normalise_host",
    "split_allowlist",
]

#: Hostnames that may carry an explicit port in an allowlist entry. These are
#: container→host aliases, not routable internet names: Docker resolves
#: ``host.docker.internal`` to the host gateway of the container's own bridge
#: network (see ``extra_hosts: host-gateway`` in docker/docker-compose.yml).
LOCAL_MODEL_HOST_ALIASES: frozenset[str] = frozenset({"host.docker.internal"})


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


def split_allowlist(
    allowlist: tuple[str, ...] | list[str], *, url: str = ""
) -> tuple[tuple[str, ...], frozenset[tuple[str, int]]]:
    """Split allowlist entries into plain hosts and explicit ``(host, port)`` endpoints.

    A ``host:port`` entry is only accepted for a host in
    :data:`LOCAL_MODEL_HOST_ALIASES`; anything else raises, so a typo or an
    over-broad entry fails loudly instead of widening egress.

    Raises:
        EgressViolation: for a malformed or non-local ``host:port`` entry.
    """
    hosts: list[str] = []
    endpoints: set[tuple[str, int]] = set()

    for raw in allowlist:
        entry = normalise_host(str(raw))
        if not entry:
            continue
        host, sep, port_text = entry.rpartition(":")
        if not sep:
            hosts.append(entry)
            continue
        try:
            port = int(port_text)
        except ValueError as exc:
            raise EgressViolation(
                f"malformed allowlist entry {raw!r}: {port_text!r} is not a port",
                url=url,
            ) from exc
        if host not in LOCAL_MODEL_HOST_ALIASES:
            raise EgressViolation(
                f"allowlist entry {raw!r} names an explicit port, which is only permitted "
                f"for a local model endpoint {tuple(sorted(LOCAL_MODEL_HOST_ALIASES))!r}. "
                f"Remote hosts are reached over https on 443 or not at all.",
                url=url,
                host=host,
            )
        if not 1 <= port <= 65535:
            raise EgressViolation(f"allowlist entry {raw!r} has an out-of-range port", url=url)
        endpoints.add((host, port))

    return tuple(hosts), frozenset(endpoints)


def assert_url_allowed(url: str, allowlist: tuple[str, ...] | list[str]) -> ParseResult:
    """Validate ``url`` against ``allowlist``; return the parsed URL or raise.

    Raises:
        EgressViolation: for any URL that is not unambiguously permitted.
    """
    if not isinstance(url, str) or not url.strip():
        raise EgressViolation("url must be a non-empty string", url=str(url))

    parts = urlsplit(url.strip())

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise EgressViolation(
            f"scheme {parts.scheme!r} is not permitted; https only",
            url=url,
        )

    if parts.username or parts.password:
        raise EgressViolation("credentials in the URL are not permitted", url=url)

    host = normalise_host(parts.hostname or "")
    if not host:
        raise EgressViolation("url has no host", url=url)

    try:
        port = parts.port
    except ValueError as exc:
        raise EgressViolation(f"invalid port: {exc}", url=url, host=host) from exc

    if not allowlist:
        raise EgressViolation("egress allowlist is empty; all requests denied", url=url, host=host)

    hosts, endpoints = split_allowlist(allowlist, url=url)

    # The narrow exception, checked first: an exact host:port match against an
    # allowlisted local model endpoint. Exact host — no subdomain match — and
    # exact port, so this grants precisely one destination and nothing near it.
    if port is not None and (host, port) in endpoints:
        return _to_parse_result(parts)

    # Everything else goes through the strict general rules, unchanged.
    if scheme != "https":
        raise EgressViolation(
            f"scheme {parts.scheme!r} is not permitted; https only (the sole exception is "
            f"an allowlisted local model endpoint, e.g. host.docker.internal:11434)",
            url=url,
            host=host,
        )

    if _is_ip_literal(host):
        raise EgressViolation(
            "bare IP addresses are not permitted; allowlist entries are hostnames",
            url=url,
            host=host,
        )

    if port not in (None, 443):
        raise EgressViolation(f"port {port} is not permitted; 443 only", url=url, host=host)

    if not any(_host_matches(host, entry) for entry in hosts):
        raise EgressViolation(
            f"host {host!r} is not in the egress allowlist {tuple(allowlist)!r}",
            url=url,
            host=host,
        )

    return _to_parse_result(parts)


def _to_parse_result(parts: ParseResult) -> ParseResult:
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
