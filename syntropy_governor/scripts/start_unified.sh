#!/bin/bash
# Syntropy Governor - Unified Startup Script
# Starts: Governor (security) → Core Brain (Syntropy) → Consciousness (UI) → Atlantean backend

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REDIS_CONTAINER_NAME="syntropy-governor-redis"

is_redis_ready() {
	python3 - <<'PY'
import socket
import sys

s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", 6379))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
}

start_redis_if_needed() {
	if is_redis_ready; then
		echo "✅ Redis already available on localhost:6379"
		return 0
	fi

	echo ""
	echo "🧱 Redis not detected on localhost:6379. Starting Redis..."

	if command -v redis-server >/dev/null 2>&1; then
		redis-server --daemonize yes >/dev/null 2>&1 || true
	elif command -v docker >/dev/null 2>&1; then
		if docker ps -a --format '{{.Names}}' | grep -qx "$REDIS_CONTAINER_NAME"; then
			docker start "$REDIS_CONTAINER_NAME" >/dev/null
		else
			docker run -d --name "$REDIS_CONTAINER_NAME" -p 6379:6379 redis:7-alpine >/dev/null
		fi
	else
		echo "⚠️  Could not start Redis automatically (no redis-server or docker found)."
		return 1
	fi

	for _ in {1..15}; do
		if is_redis_ready; then
			echo "✅ Redis running on localhost:6379"
			return 0
		fi
		sleep 1
	done

	echo "⚠️  Redis startup attempted, but localhost:6379 is still unreachable."
	return 1
}

echo "=================================================="
echo "  SYNTROPY GOVERNOR - UNIFIED INTELLIGENCE"
echo "  Consciousness + Core Brain + Governor"
echo "=================================================="

start_redis_if_needed || true

# 1. Start Atlantean + Syntropy Backend
echo ""
echo "🧠 Starting Unified Backend (Atlantean + Syntropy Bridge) on port 5001..."
cd "$ROOT_DIR/unified_backend"
python3 -m flask --app server run --host=0.0.0.0 --port=5001 &
BACKEND_PID=$!
sleep 3

echo "✅ Backend running (PID: $BACKEND_PID)"

# 2. Governor is integrated as middleware/endpoints in unified backend

# 3. Start Frontend (Consciousness)
echo ""
echo "🌐 Starting Consciousness UI (React) on port 3000..."
FRONTEND_PID=""
if [[ -f "$ROOT_DIR/consciousness/studious-enigma-main/package.json" ]]; then
	cd "$ROOT_DIR/consciousness/studious-enigma-main"
	npm run dev &
	FRONTEND_PID=$!
elif [[ -f "$ROOT_DIR/consciousness/package.json" ]]; then
	cd "$ROOT_DIR/consciousness"
	npm run dev &
	FRONTEND_PID=$!
else
	echo "⚠️  No frontend package.json found; skipping UI startup."
fi

echo ""
echo "=================================================="
echo "  READY"
echo "=================================================="
echo "  Backend (Syntropy + Atlantean): http://localhost:5001"
echo "  Consciousness UI:               http://localhost:3000"
echo "  Governor Status:                http://localhost:5001/api/governor/status"
echo "  Governor Dashboard:             http://localhost:5001/api/governor/dashboard"
echo "=================================================="
echo ""
echo "Press Ctrl+C to stop everything."

if [[ -n "$FRONTEND_PID" ]]; then
	wait $BACKEND_PID $FRONTEND_PID
else
	wait $BACKEND_PID
fi