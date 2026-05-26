"""Version 1 API blueprint and route registration helpers."""

from __future__ import annotations

from flask import Blueprint

bp = Blueprint("v1", __name__, url_prefix="/v1")

_ROUTES_REGISTERED = False


def register_routes() -> None:
    """Import controller modules once so they can register their routes."""
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return

    from . import analysis_tasks, face_align, index, infer, measure, mobile, mobile_chat, tongue_segment  # noqa: F401

    _ROUTES_REGISTERED = True


__all__ = ["bp", "register_routes"]
