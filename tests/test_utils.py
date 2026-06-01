"""Tests for shared utilities."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import (
    extract_keywords,
    detect_error_pattern,
    short_hash,
    now_ts,
)


class TestExtractKeywords:
    def test_english_identifiers(self):
        kw = extract_keywords("JWT authentication middleware for Express")
        assert "jwt" in kw or "authentication" in kw

    def test_camel_case(self):
        kw = extract_keywords("UserAuthService handles login")
        assert "user" in kw or "auth" in kw or "service" in kw

    def test_snake_case(self):
        kw = extract_keywords("user_auth_service handles login")
        assert "user" in kw or "auth" in kw or "service" in kw

    def test_stop_words_filtered(self):
        kw = extract_keywords("the and for this that with from")
        assert all(w not in kw for w in ["the", "and", "for"])

    def test_short_words_filtered(self):
        kw = extract_keywords("a b c ab cd ef")
        assert all(len(w) > 1 for w in kw)

    def test_empty_string(self):
        kw = extract_keywords("")
        assert kw == []

    def test_max_keywords(self):
        # Long text with many identifiers
        text = " ".join([f"module_{i}_component" for i in range(50)])
        kw = extract_keywords(text, max_kw=8)
        assert len(kw) <= 8


class TestDetectErrorPattern:
    def test_syntax_error(self):
        assert detect_error_pattern("SyntaxError: unexpected token") == "syntax_error"

    def test_type_error(self):
        assert detect_error_pattern("TypeError: cannot read property of null") == "type_error"

    def test_import_error(self):
        assert detect_error_pattern("ModuleNotFoundError: No module named 'foo'") == "import_error"

    def test_auth_error(self):
        assert detect_error_pattern("HTTP 401 Unauthorized") == "auth_error"
        assert detect_error_pattern("403 Forbidden") == "auth_error"

    def test_timeout(self):
        assert detect_error_pattern("Connection timed out after 30s") == "timeout"

    def test_null_ref(self):
        assert detect_error_pattern("Cannot read property 'x' of null") == "null_ref"

    def test_oom(self):
        assert detect_error_pattern("Out of memory error") == "oom"

    def test_no_match(self):
        assert detect_error_pattern("Everything is fine") is None

    def test_case_insensitive(self):
        assert detect_error_pattern("SYNTAXERROR: BAD TOKEN") == "syntax_error"


class TestShortHash:
    def test_length(self):
        h = short_hash("hello world")
        assert len(h) == 8

    def test_deterministic(self):
        assert short_hash("test") == short_hash("test")

    def test_different_input(self):
        assert short_hash("a") != short_hash("b")


class TestNowTs:
    def test_positive(self):
        assert now_ts() > 0

    def test_monotonic(self):
        a = now_ts()
        b = now_ts()
        assert b >= a
