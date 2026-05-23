# Governor Layer

This module adds baseline governance controls to the unified backend:

- Request validation for critical endpoints
- Per-client rate limiting
- Security response headers
- Audit logging (in-memory ring buffer + append-only log file)
- Status/dashboard API routes

## Endpoints

- `GET /api/governor/status`
- `GET /api/governor/dashboard`
- `GET /api/governor/audit?limit=50`

## Environment Variables

- `GOVERNOR_RATE_LIMIT_PER_MINUTE` (default: `60`)
- `GOVERNOR_MAX_INPUT_CHARS` (default: `4000`)
- `GOVERNOR_AUDIT_BUFFER_SIZE` (default: `200`)
- `GOVERNOR_AUDIT_LOG` (default: `governor/audit.log`)
