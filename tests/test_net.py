"""The egress allowlist must be strict — allowlist, not blocklist."""

from __future__ import annotations

import pytest

from target_agent.net import (
    EgressViolation,
    assert_url_allowed,
    is_url_allowed,
    split_allowlist,
)

ALLOW = ("api.anthropic.com",)

# What docker/docker-compose.yml puts in the container's environment.
CONTAINER_ALLOW = ("api.anthropic.com", "host.docker.internal:11434")


@pytest.mark.parametrize(
    "url",
    [
        "https://api.anthropic.com",
        "https://api.anthropic.com/v1/messages",
        "https://api.anthropic.com:443/v1/messages",
        "https://sub.api.anthropic.com/x",  # subdomain of an allowed host
    ],
)
def test_allowed_urls(url):
    assert is_url_allowed(url, ALLOW)


@pytest.mark.parametrize(
    "url",
    [
        "http://api.anthropic.com/v1",           # not https
        "https://evil.com/v1",                    # not allowlisted
        "https://api.anthropic.com.evil.com/",    # suffix trick
        "https://anthropic.com/",                 # parent, not the allowed host
        "https://user:pass@api.anthropic.com/",   # credentials in URL
        "https://api.anthropic.com:8443/",        # non-default port
        "https://93.184.216.34/",                 # bare IP
        "file:///etc/passwd",                     # non-http scheme
        "https:///nohost",                        # no host
        "ftp://api.anthropic.com/",               # wrong scheme
    ],
)
def test_denied_urls(url):
    assert not is_url_allowed(url, ALLOW)
    with pytest.raises(EgressViolation):
        assert_url_allowed(url, ALLOW)


def test_empty_allowlist_denies_everything():
    with pytest.raises(EgressViolation):
        assert_url_allowed("https://api.anthropic.com/", ())


def test_case_and_trailing_dot_normalised():
    assert is_url_allowed("https://API.Anthropic.COM./v1", ALLOW)


# --- the local model endpoint (container -> host Ollama) ---------------------
#
# A `host:port` entry is the ONE exception to https-and-443-only. It must grant
# exactly that host on exactly that port, and nothing adjacent to it.


@pytest.mark.parametrize(
    "url",
    [
        "http://host.docker.internal:11434",
        "http://host.docker.internal:11434/api/chat",
        "https://host.docker.internal:11434/api/tags",  # same endpoint over TLS
        "http://HOST.Docker.Internal:11434/api/tags",  # case-normalised
    ],
)
def test_local_model_endpoint_allowed_when_listed(url):
    assert is_url_allowed(url, CONTAINER_ALLOW)


@pytest.mark.parametrize(
    "url",
    [
        "http://host.docker.internal:11434",  # right endpoint, not allowlisted
        "http://host.docker.internal/api/tags",  # no port -> http still denied
    ],
)
def test_local_model_endpoint_denied_without_the_entry(url):
    assert not is_url_allowed(url, ALLOW)


@pytest.mark.parametrize(
    "url",
    [
        "http://host.docker.internal:11435/api/tags",  # neighbouring port
        "http://host.docker.internal:22/",  # host's SSH
        "http://host.docker.internal:80/",  # host's web server
        "http://evil.host.docker.internal:11434/",  # subdomain, not exact host
        "http://host.docker.internal.evil.com:11434/",  # suffix trick
        "http://other.internal:11434/",  # different host, same port
        "http://user:pass@host.docker.internal:11434/",  # credentials
        "http://172.17.0.1:11434/",  # the gateway by IP
        "ftp://host.docker.internal:11434/",  # non-http scheme
    ],
)
def test_local_model_entry_grants_nothing_else(url):
    assert not is_url_allowed(url, CONTAINER_ALLOW)
    with pytest.raises(EgressViolation):
        assert_url_allowed(url, CONTAINER_ALLOW)


def test_anthropic_rules_unchanged_by_the_local_entry():
    # The extra entry must not relax anything on the hosted path.
    assert is_url_allowed("https://api.anthropic.com/v1/messages", CONTAINER_ALLOW)
    for url in (
        "http://api.anthropic.com/v1",  # plaintext
        "https://api.anthropic.com:11434/",  # the local port, wrong host
        "https://evil.com/",
    ):
        assert not is_url_allowed(url, CONTAINER_ALLOW)


@pytest.mark.parametrize(
    "entry",
    [
        "evil.example.com:80",  # arbitrary host + port: refused outright
        "api.anthropic.com:443",  # even a benign-looking one
        "host.docker.internal:notaport",
        "host.docker.internal:0",
        "host.docker.internal:70000",
    ],
)
def test_only_local_aliases_may_carry_a_port(entry):
    with pytest.raises(EgressViolation):
        split_allowlist((entry,))
    # A URL check with such an allowlist fails closed rather than silently
    # ignoring the bad entry.
    with pytest.raises(EgressViolation):
        assert_url_allowed("https://api.anthropic.com/", ("api.anthropic.com", entry))


def test_split_allowlist_shape():
    hosts, endpoints = split_allowlist(CONTAINER_ALLOW)
    assert hosts == ("api.anthropic.com",)
    assert endpoints == frozenset({("host.docker.internal", 11434)})
