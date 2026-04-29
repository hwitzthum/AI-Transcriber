"""Tests for redact_secrets — used to scrub credentials before any
exception text reaches the Streamlit UI."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transcriber.cloud_engine import redact_secrets


def test_bearer_token_is_redacted():
    msg = "AuthenticationError: Invalid header 'Authorization: Bearer sk-abcd1234XYZ' on request"
    out = redact_secrets(msg)
    assert "sk-abcd1234XYZ" not in out
    assert "Bearer sk-abcd1234XYZ" not in out
    assert "[REDACTED]" in out


def test_openai_style_key_is_redacted():
    msg = "openai.AuthenticationError: api_key='sk-proj-1234567890ABCDEFGHIJ' rejected"
    out = redact_secrets(msg)
    assert "sk-proj-1234567890ABCDEFGHIJ" not in out
    assert "[REDACTED]" in out


def test_groq_and_deepgram_style_keys_redacted():
    msg = "Authorization failed: gsk_abcdef1234567890zyxwvuts and dg_token1234567890ABCDEF"
    out = redact_secrets(msg)
    assert "gsk_abcdef1234567890zyxwvuts" not in out
    assert "dg_token1234567890ABCDEF" not in out


def test_authorization_header_value_redacted():
    msg = 'Headers: {"Authorization": "MyVerySecretToken1234567890"}'
    out = redact_secrets(msg)
    assert "MyVerySecretToken1234567890" not in out
    assert "[REDACTED]" in out


def test_normal_error_text_preserved():
    """No false positives on benign error messages — short numbers like
    HTTP status codes must not be mistaken for keys."""
    msg = "Connection error: 502 Bad Gateway from upstream"
    out = redact_secrets(msg)
    assert out == msg


def test_empty_input():
    assert redact_secrets("") == ""
    assert redact_secrets(None) is None