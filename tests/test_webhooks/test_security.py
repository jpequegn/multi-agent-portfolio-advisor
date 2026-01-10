"""Tests for webhook security module."""

import time

import pytest

from src.webhooks.security import (
    SIGNATURE_HEADER,
    SIGNATURE_VALIDITY_SECONDS,
    TIMESTAMP_HEADER,
    create_signature_headers,
    generate_signature,
    verify_from_headers,
    verify_signature,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_payload():
    """Sample webhook payload."""
    return {
        "id": "evt_123",
        "type": "analysis.started",
        "data": {"trace_id": "trace-456"},
    }


@pytest.fixture
def sample_secret():
    """Sample webhook secret."""
    return "whsec_test_secret_key"


# ============================================================================
# generate_signature Tests
# ============================================================================


class TestGenerateSignature:
    """Tests for generate_signature function."""

    def test_generate_signature_dict_payload(self, sample_payload, sample_secret):
        """Test generating signature from dict payload."""
        signature, timestamp = generate_signature(sample_payload, sample_secret)

        assert signature is not None
        assert len(signature) == 64  # SHA256 hex is 64 chars
        assert timestamp > 0

    def test_generate_signature_string_payload(self, sample_secret):
        """Test generating signature from string payload."""
        payload_str = '{"test":"value"}'
        signature, timestamp = generate_signature(payload_str, sample_secret)

        assert signature is not None
        assert len(signature) == 64

    def test_generate_signature_with_custom_timestamp(self, sample_payload, sample_secret):
        """Test generating signature with custom timestamp."""
        custom_timestamp = 1704067200  # 2024-01-01 00:00:00 UTC
        signature, returned_timestamp = generate_signature(
            sample_payload, sample_secret, timestamp=custom_timestamp
        )

        assert returned_timestamp == custom_timestamp

    def test_generate_signature_deterministic(self, sample_payload, sample_secret):
        """Test that same input produces same signature."""
        timestamp = int(time.time())
        sig1, _ = generate_signature(sample_payload, sample_secret, timestamp=timestamp)
        sig2, _ = generate_signature(sample_payload, sample_secret, timestamp=timestamp)

        assert sig1 == sig2

    def test_generate_signature_different_secrets(self, sample_payload):
        """Test that different secrets produce different signatures."""
        timestamp = int(time.time())
        sig1, _ = generate_signature(sample_payload, "secret1", timestamp=timestamp)
        sig2, _ = generate_signature(sample_payload, "secret2", timestamp=timestamp)

        assert sig1 != sig2

    def test_generate_signature_different_payloads(self, sample_secret):
        """Test that different payloads produce different signatures."""
        timestamp = int(time.time())
        sig1, _ = generate_signature({"a": 1}, sample_secret, timestamp=timestamp)
        sig2, _ = generate_signature({"a": 2}, sample_secret, timestamp=timestamp)

        assert sig1 != sig2


# ============================================================================
# verify_signature Tests
# ============================================================================


class TestVerifySignature:
    """Tests for verify_signature function."""

    def test_verify_valid_signature(self, sample_payload, sample_secret):
        """Test verifying a valid signature."""
        signature, timestamp = generate_signature(sample_payload, sample_secret)

        is_valid = verify_signature(sample_payload, signature, sample_secret, timestamp)

        assert is_valid is True

    def test_verify_invalid_signature(self, sample_payload, sample_secret):
        """Test rejecting an invalid signature."""
        timestamp = int(time.time())

        is_valid = verify_signature(
            sample_payload, "invalid_signature", sample_secret, timestamp
        )

        assert is_valid is False

    def test_verify_wrong_secret(self, sample_payload, sample_secret):
        """Test rejecting signature with wrong secret."""
        signature, timestamp = generate_signature(sample_payload, sample_secret)

        is_valid = verify_signature(
            sample_payload, signature, "wrong_secret", timestamp
        )

        assert is_valid is False

    def test_verify_modified_payload(self, sample_payload, sample_secret):
        """Test rejecting signature for modified payload."""
        signature, timestamp = generate_signature(sample_payload, sample_secret)

        # Modify payload
        modified_payload = dict(sample_payload)
        modified_payload["data"] = {"trace_id": "modified"}

        is_valid = verify_signature(
            modified_payload, signature, sample_secret, timestamp
        )

        assert is_valid is False

    def test_verify_expired_signature(self, sample_payload, sample_secret):
        """Test rejecting expired signature."""
        # Create signature with old timestamp
        old_timestamp = int(time.time()) - SIGNATURE_VALIDITY_SECONDS - 60
        signature, _ = generate_signature(
            sample_payload, sample_secret, timestamp=old_timestamp
        )

        is_valid = verify_signature(
            sample_payload, signature, sample_secret, old_timestamp
        )

        assert is_valid is False

    def test_verify_future_timestamp(self, sample_payload, sample_secret):
        """Test rejecting future timestamp."""
        # Create signature with future timestamp
        future_timestamp = int(time.time()) + SIGNATURE_VALIDITY_SECONDS + 60
        signature, _ = generate_signature(
            sample_payload, sample_secret, timestamp=future_timestamp
        )

        is_valid = verify_signature(
            sample_payload, signature, sample_secret, future_timestamp
        )

        assert is_valid is False

    def test_verify_with_custom_max_age(self, sample_payload, sample_secret):
        """Test custom max age for signature."""
        # Create signature 10 seconds ago
        old_timestamp = int(time.time()) - 10
        signature, _ = generate_signature(
            sample_payload, sample_secret, timestamp=old_timestamp
        )

        # Should fail with 5 second max age
        is_valid_short = verify_signature(
            sample_payload, signature, sample_secret, old_timestamp, max_age_seconds=5
        )
        assert is_valid_short is False

        # Should pass with 60 second max age
        is_valid_long = verify_signature(
            sample_payload, signature, sample_secret, old_timestamp, max_age_seconds=60
        )
        assert is_valid_long is True


# ============================================================================
# create_signature_headers Tests
# ============================================================================


class TestCreateSignatureHeaders:
    """Tests for create_signature_headers function."""

    def test_create_headers(self, sample_payload, sample_secret):
        """Test creating signature headers."""
        headers = create_signature_headers(sample_payload, sample_secret)

        assert SIGNATURE_HEADER in headers
        assert TIMESTAMP_HEADER in headers
        assert len(headers[SIGNATURE_HEADER]) == 64
        assert int(headers[TIMESTAMP_HEADER]) > 0

    def test_headers_are_verifiable(self, sample_payload, sample_secret):
        """Test that created headers can be verified."""
        headers = create_signature_headers(sample_payload, sample_secret)

        signature = headers[SIGNATURE_HEADER]
        timestamp = int(headers[TIMESTAMP_HEADER])

        is_valid = verify_signature(sample_payload, signature, sample_secret, timestamp)

        assert is_valid is True


# ============================================================================
# verify_from_headers Tests
# ============================================================================


class TestVerifyFromHeaders:
    """Tests for verify_from_headers function."""

    def test_verify_valid_headers(self, sample_payload, sample_secret):
        """Test verifying valid headers."""
        headers = create_signature_headers(sample_payload, sample_secret)

        is_valid = verify_from_headers(sample_payload, headers, sample_secret)

        assert is_valid is True

    def test_verify_missing_signature_header(self, sample_payload, sample_secret):
        """Test error when signature header is missing."""
        headers = {TIMESTAMP_HEADER: str(int(time.time()))}

        with pytest.raises(ValueError, match=f"Missing {SIGNATURE_HEADER}"):
            verify_from_headers(sample_payload, headers, sample_secret)

    def test_verify_missing_timestamp_header(self, sample_payload, sample_secret):
        """Test error when timestamp header is missing."""
        headers = {SIGNATURE_HEADER: "some_signature"}

        with pytest.raises(ValueError, match=f"Missing {TIMESTAMP_HEADER}"):
            verify_from_headers(sample_payload, headers, sample_secret)

    def test_verify_invalid_timestamp(self, sample_payload, sample_secret):
        """Test error when timestamp is not an integer."""
        headers = {
            SIGNATURE_HEADER: "some_signature",
            TIMESTAMP_HEADER: "not_a_number",
        }

        with pytest.raises(ValueError, match="must be integer"):
            verify_from_headers(sample_payload, headers, sample_secret)

    def test_verify_string_payload(self, sample_secret):
        """Test verifying with string payload."""
        payload_str = '{"test":"value"}'
        headers = create_signature_headers({"test": "value"}, sample_secret)

        # Should work with string payload too
        is_valid = verify_from_headers(payload_str, headers, sample_secret)

        assert is_valid is True
