from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Any

from flask import Flask, Blueprint, jsonify, g, request

logger = logging.getLogger("SyntropyGovernor")


@dataclass
class GovernorConfig:
    rate_limit_per_minute: int = 60
    max_input_chars: int = 4000
    audit_buffer_size: int = 200


class InMemoryRateLimiter:
    """Simple thread-safe fixed-window rate limiter by client IP."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[int, int]] = {}

    def allow(self, client_id: str) -> tuple[bool, int]:
        now = int(time.time())
        window = now // 60
        with self._lock:
            current_window, count = self._buckets.get(client_id, (window, 0))
            if current_window != window:
                current_window, count = window, 0
            if count >= self.requests_per_minute:
                retry_after = 60 - (now % 60)
                self._buckets[client_id] = (current_window, count)
                return False, retry_after

            count += 1
            self._buckets[client_id] = (current_window, count)
            return True, 0


class AuditLogStore:
    """In-memory audit ring buffer plus append-only file sink."""

    def __init__(self, maxlen: int, log_path: str):
        self._events: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=True)
        with self._lock:
            self._events.append(event)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._events)[-limit:]


REQUEST_COUNTER = defaultdict(int)


def _build_blueprint(config: GovernorConfig, audit: AuditLogStore, limiter: InMemoryRateLimiter) -> Blueprint:
    bp = Blueprint("governor", __name__, url_prefix="/api/governor")

    @bp.route("/status", methods=["GET"])
    def governor_status():
        uptime = max(0.0, time.time() - request.environ.get("governor_started_at", time.time()))
        return jsonify(
            {
                "status": "healthy",
                "component": "governor",
                "config": asdict(config),
                "request_count": sum(REQUEST_COUNTER.values()),
                "rate_limited_clients": len(limiter._buckets),
                "audit_events_buffered": len(audit.recent(config.audit_buffer_size)),
                "uptime_seconds": round(uptime, 2),
            }
        )

    @bp.route("/audit", methods=["GET"])
    def governor_audit():
        limit_raw = request.args.get("limit", "50")
        try:
            limit = max(1, min(200, int(limit_raw)))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400

        return jsonify(
            {
                "events": audit.recent(limit),
                "count": limit,
            }
        )

    @bp.route("/dashboard", methods=["GET"])
    def governor_dashboard():
        return jsonify(
            {
                "title": "Syntropy Governor Dashboard",
                "security": {
                    "rate_limiting": True,
                    "input_validation": True,
                    "security_headers": True,
                    "audit_logging": True,
                },
                "endpoints": [
                    "/api/governor/status",
                    "/api/governor/audit?limit=50",
                    "/health",
                    "/api/atlantean/query",
                ],
            }
        )

    return bp


def _client_id() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.remote_addr or "unknown"


def _request_payload_size() -> int:
    if request.content_length is not None:
        return int(request.content_length)
    raw = request.get_data(cache=True, as_text=False) or b""
    return len(raw)


def init_governor(app: Flask) -> None:
    """Attach governor middleware and monitoring endpoints to a Flask app."""

    cfg = GovernorConfig(
        rate_limit_per_minute=int(os.getenv("GOVERNOR_RATE_LIMIT_PER_MINUTE", "60")),
        max_input_chars=int(os.getenv("GOVERNOR_MAX_INPUT_CHARS", "4000")),
        audit_buffer_size=int(os.getenv("GOVERNOR_AUDIT_BUFFER_SIZE", "200")),
    )
    audit_path = os.getenv("GOVERNOR_AUDIT_LOG", os.path.join("governor", "audit.log"))

    limiter = InMemoryRateLimiter(cfg.rate_limit_per_minute)
    audit = AuditLogStore(maxlen=cfg.audit_buffer_size, log_path=audit_path)
    started_at = time.time()

    @app.before_request
    def governor_before_request():
        request.environ["governor_started_at"] = started_at
        g.request_started_at = time.time()

        allow, retry_after = limiter.allow(_client_id())
        if not allow:
            return (
                jsonify({"error": "rate limit exceeded", "retry_after_seconds": retry_after}),
                429,
                {"Retry-After": str(retry_after)},
            )

        if request.method in {"POST", "PUT", "PATCH"}:
            size_bytes = _request_payload_size()
            if size_bytes > 1_000_000:
                return jsonify({"error": "payload too large"}), 413

        if request.path == "/api/atlantean/query" and request.method == "POST":
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify({"error": "invalid JSON body"}), 400

            value = payload.get("input")
            if not isinstance(value, str):
                return jsonify({"error": "input must be a string"}), 400

            if not value.strip():
                return jsonify({"error": "input must not be empty"}), 400

            if len(value) > cfg.max_input_chars:
                return jsonify({"error": f"input too long (max {cfg.max_input_chars})"}), 400

    @app.after_request
    def governor_after_request(response):
        REQUEST_COUNTER[request.path] += 1

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"

        if request.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"

        event = {
            "timestamp": time.time(),
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "client": _client_id(),
            "duration_ms": round((time.time() - g.get("request_started_at", time.time())) * 1000, 2),
            "content_length": _request_payload_size() if request.method != "GET" else 0,
            "user_agent": request.headers.get("User-Agent", ""),
        }
        audit.append(event)

        return response

    app.register_blueprint(_build_blueprint(cfg, audit, limiter))
    logger.info("Governor initialized (rate_limit=%s/min, max_input_chars=%s)", cfg.rate_limit_per_minute, cfg.max_input_chars)
