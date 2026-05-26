"""Storage and token configuration helpers."""

from __future__ import annotations

import os
from typing import Any


def _first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def get_storage_config() -> dict[str, str]:
    """Build storage config with backward-compatible MinIO env names."""
    return {
        "aws_access_key_id": _first_env(
            "AWS_ACCESS_KEY_ID",
            "S3_ACCESS_KEY",
            "S3_ACCESS_KEY_ID",
            default="minioadmin",
        ),
        "aws_secret_access_key": _first_env(
            "AWS_SECRET_ACCESS_KEY",
            "S3_SECRET_KEY",
            "S3_SECRET_ACCESS_KEY",
            default="minioadmin",
        ),
        "bucket_name": _first_env("S3_BUCKET_NAME", default="aitest"),
        "region": _first_env("AWS_REGION", default="us-east-1"),
        "endpoint_url": _first_env("S3_ENDPOINT_URL", "S3_ENDPOINT"),
        "cdn_domain": _first_env("CDN_DOMAIN"),
    }


def get_token_config() -> dict[str, Any]:
    """Build token config from environment variables."""
    return {
        "jwt_secret": os.getenv("JWT_SECRET", "your-secret-key-here"),
        "token_expire_hours": int(os.getenv("TOKEN_EXPIRE_HOURS", "24")),
        "max_access_count": int(os.getenv("MAX_ACCESS_COUNT", "100")),
    }


STORAGE_CONFIG = get_storage_config()
TOKEN_CONFIG = get_token_config()
