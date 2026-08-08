"""The egress allowlist must be strict — allowlist, not blocklist."""

from __future__ import annotations

import pytest

from target_agent.net import EgressViolation, assert_url_allowed, is_url_allowed

ALLOW = ("api.anthropic.com",)


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
