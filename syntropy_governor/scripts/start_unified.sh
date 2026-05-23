#!/bin/bash
# Syntropy Governor - Unified Startup Script
# Starts: Governor (security) → Core Brain (Syntropy) → Consciousness (UI) → Atlantean backend

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=================================================="
echo "  SYNTROPY GOVERNOR - UNIFIED INTELLIGENCE"
echo "  Consciousness + Core Brain + Governor"
echo "=================================================="

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