"""Webhook security utilities.

Provides HMAC signature generation and verification for webhook payloads
to ensure authenticity and prevent tampering.
"""

import hashlib
import hmac
import json
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default signature header name
SIGNATURE_HEADER = "X-Webhook-Signature"
TIMESTAMP_HEADER = "X-Webhook-Timestamp"

# Signature validity window (5 minutes)
SIGNATURE_VALIDITY_SECONDS = 300


def generate_signature(
    payload: dict[str, Any] | str,
    secret: str,
    *,
    timestamp: int | None = None,
) -> tuple[str, int]:
    """Generate HMAC-SHA256 signature for webhook payload.

    The signature is computed as:
    HMAC-SHA256(secret, timestamp + "." + payload_json)

    This format prevents timing attacks and replay attacks by including
    a timestamp in the signed data.

    Args:
        payload: Webhook payload (dict or JSON string).
        secret: Webhook secret key.
        timestamp: Optional Unix timestamp (defaults to current time).

    Returns:
        Tuple of (signature_hex, timestamp_used).
    """
    if timestamp is None:
        timestamp = int(time.time())

    # Convert payload to canonical JSON string if needed
    if isinstance(payload, dict):
        payload_str = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    else:
        payload_str = payload

    # Create signed payload with timestamp
    signed_payload = f"{timestamp}.{payload_str}"

    # Generate HMAC-SHA256 signature
    signature = hmac.new(
        secret.encode("utf-8"),
        signed_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    logger.debug(
        "webhook_signature_generated",
        timestamp=timestamp,
        payload_length=len(payload_str),
    )

    return signature, timestamp


def verify_signature(
    payload: dict[str, Any] | str,
    signature: str,
    secret: str,
    timestamp: int,
    *,
    max_age_seconds: int = SIGNATURE_VALIDITY_SECONDS,
) -> bool:
    """Verify HMAC-SHA256 signature of webhook payload.

    Checks both the cryptographic signature and that the timestamp
    is within the validity window to prevent replay attacks.

    Args:
        payload: Webhook payload (dict or JSON string).
        signature: Claimed signature to verify.
        secret: Webhook secret key.
        timestamp: Timestamp from request.
        max_age_seconds: Maximum age of valid signature.

    Returns:
        True if signature is valid, False otherwise.
    """
    # Check timestamp freshness
    current_time = int(time.time())
    age = abs(current_time - timestamp)

    if age > max_age_seconds:
        logger.warning(
            "webhook_signature_expired",
            timestamp=timestamp,
            age_seconds=age,
            max_age=max_age_seconds,
        )
        return False

    # Generate expected signature
    expected_signature, _ = generate_signature(payload, secret, timestamp=timestamp)

    # Use constant-time comparison to prevent timing attacks
    is_valid = hmac.compare_digest(signature, expected_signature)

    if not is_valid:
        logger.warning(
            "webhook_signature_invalid",
            timestamp=timestamp,
        )
    else:
        logger.debug(
            "webhook_signature_verified",
            timestamp=timestamp,
        )

    return is_valid


def create_signature_headers(
    payload: dict[str, Any],
    secret: str,
) -> dict[str, str]:
    """Create HTTP headers with signature for webhook delivery.

    Args:
        payload: Webhook payload.
        secret: Webhook secret key.

    Returns:
        Dictionary of headers to include in request.
    """
    signature, timestamp = generate_signature(payload, secret)

    return {
        SIGNATURE_HEADER: signature,
        TIMESTAMP_HEADER: str(timestamp),
    }


def verify_from_headers(
    payload: dict[str, Any] | str,
    headers: dict[str, str],
    secret: str,
) -> bool:
    """Verify webhook signature from request headers.

    Args:
        payload: Received webhook payload.
        headers: Request headers.
        secret: Webhook secret key.

    Returns:
        True if signature is valid.

    Raises:
        ValueError: If required headers are missing.
    """
    signature = headers.get(SIGNATURE_HEADER)
    timestamp_str = headers.get(TIMESTAMP_HEADER)

    if not signature:
        raise ValueError(f"Missing {SIGNATURE_HEADER} header")

    if not timestamp_str:
        raise ValueError(f"Missing {TIMESTAMP_HEADER} header")

    try:
        timestamp = int(timestamp_str)
    except ValueError as e:
        raise ValueError(f"Invalid {TIMESTAMP_HEADER} header: must be integer") from e

    return verify_signature(payload, signature, secret, timestamp)
