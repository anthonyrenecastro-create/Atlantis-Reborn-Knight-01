#!/usr/bin/env python3
"""
Syntropy Governor Unified Backend Server
========================================
Flask server that exposes the Atlantean + Syntropy bridge via HTTP.

Endpoints:
- GET  /health
- GET  /api/atlantean/status
- POST /api/atlantean/query
- POST /api/atlantean/learning-event
- GET  /api/atlantean/fields
- GET  /api/atlantean/simulations

This replaces the old Gemini-dependent backend.
Fully local. No external APIs.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
from pathlib import Path
from atlantean_syntropy_bridge import AtlanteanSyntropyBridge
import time

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from governor import init_governor

app = Flask(__name__)
CORS(app)  # Allow React frontend to connect
init_governor(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyntropyGovernorServer")

# Global bridge instance
bridge = None

def get_bridge():
    global bridge
    if bridge is None:
        bridge = AtlanteanSyntropyBridge()
    return bridge

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "Syntropy Governor Unified Backend",
        "version": "1.0.0",
        "timestamp": time.time()
    })

@app.route("/api/atlantean/status")
def status():
    b = get_bridge()
    return jsonify(b.get_status())

@app.route("/api/atlantean/query", methods=["POST"])
def query():
    b = get_bridge()
    data = request.get_json() or {}
    user_input = data.get("input", "Hello")
    
    try:
        result = b.query(user_input)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({
            "response": "The fields are currently in flux. Please try again.",
            "error": str(e),
            "field_state": b.get_status()["field_state"]
        }), 500

@app.route("/api/atlantean/learning-event", methods=["POST"])
def learning_event():
    b = get_bridge()
    data = request.get_json() or {}
    event = data.get("event", "user_confirmation")
    intensity = data.get("intensity", 0.5)
    
    b.trigger_learning_event(event, intensity)
    return jsonify({
        "status": "learning_signal_applied",
        "event": event,
        "new_field_state": b.get_status()["field_state"]
    })

@app.route("/api/atlantean/fields")
def fields():
    b = get_bridge()
    status = b.get_status()
    return jsonify({
        "phi1": status["field_state"]["phi1_mean"],
        "phi5": status["field_state"]["phi5_mean"],
        "Phi": status["field_state"]["Phi"],
        "learning_capacity": status["learning_capacity"],
        "version": status["version"]
    })


@app.route("/api/atlantean/simulations")
def simulations():
    b = get_bridge()
    query = request.args.get("q", "")
    limit_raw = request.args.get("limit", "100")
    try:
        limit = int(limit_raw)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    return jsonify(
        {
            "simulations": b.get_simulations(limit=limit, query=query),
            "query": query,
            "count": max(1, min(500, limit)),
        }
    )

if __name__ == "__main__":
    print("🚀 Starting Syntropy Governor Unified Backend on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=False)