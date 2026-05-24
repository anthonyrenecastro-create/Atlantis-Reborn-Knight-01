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

Unified backend for local Syntropy cognition with optional Gemini mediation.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
from pathlib import Path
from atlantean_syntropy_bridge import AtlanteanSyntropyBridge
import time
import threading
import subprocess
import uuid
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load environment for unified runtime (project root and consciousness UI env).
load_dotenv(BASE_DIR / ".env.local")
load_dotenv(BASE_DIR / "consciousness" / ".env.local")

from governor import init_governor

app = Flask(__name__)
CORS(app)  # Allow React frontend to connect
init_governor(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyntropyGovernorServer")

# Global bridge instance
bridge = None
training_jobs = {}
training_jobs_lock = threading.Lock()
active_training_job_id = None
MAX_TRAINING_JOBS = 25

def get_bridge():
    global bridge
    if bridge is None:
        bridge = AtlanteanSyntropyBridge()
    return bridge


def _now() -> float:
    return time.time()


def _trim_training_jobs():
    if len(training_jobs) <= MAX_TRAINING_JOBS:
        return
    oldest_ids = sorted(training_jobs.keys(), key=lambda job_id: training_jobs[job_id].get("created_at", 0.0))
    for job_id in oldest_ids[: max(0, len(training_jobs) - MAX_TRAINING_JOBS)]:
        training_jobs.pop(job_id, None)


def _resolve_path(raw_path: str, fallback: Path) -> Path:
    if not raw_path:
        return fallback
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


def _run_training_job(job_id: str):
    global active_training_job_id

    b = get_bridge()
    with training_jobs_lock:
        job = training_jobs.get(job_id)
        if not job:
            active_training_job_id = None
            return
        job["status"] = "running"
        job["started_at"] = _now()
        job["updated_at"] = _now()

    params = job["params"]
    try:
        export_result = b.export_training_dataset(limit=params["export_limit"])
        export_path_raw = export_result.get("export_path")
        if not export_path_raw:
            raise RuntimeError("Training export returned no dataset path.")

        dataset_path = Path(export_path_raw).resolve()
        checkpoint_path = _resolve_path(
            params.get("checkpoint_path") or (b.active_model_path or ""),
            BASE_DIR / "core_brain" / "shakespeare_model.pt",
        )

        output_path = _resolve_path(
            params.get("output_path")
            or str(BASE_DIR / "core_brain" / f"shakespeare_model_sovereign_{int(_now())}.pt"),
            BASE_DIR / "core_brain" / f"shakespeare_model_sovereign_{int(_now())}.pt",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trainer_script = BASE_DIR / "core_brain" / "train_on_dataset.py"
        cmd = [
            sys.executable,
            str(trainer_script),
            "--dataset",
            str(dataset_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--epochs",
            str(params["epochs"]),
            "--batch-size",
            str(params["batch_size"]),
            "--seq-len",
            str(params["seq_len"]),
            "--lr",
            str(params["lr"]),
            "--device",
            params["device"],
        ]

        if params.get("max_rows", 0) > 0:
            cmd.extend(["--max-rows", str(params["max_rows"])])

        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )

        combined_log = (proc.stdout or "")
        if proc.stderr:
            combined_log += "\n" + proc.stderr
        combined_log = combined_log[-12000:]

        with training_jobs_lock:
            job = training_jobs.get(job_id)
            if not job:
                active_training_job_id = None
                return
            job["dataset_path"] = str(dataset_path)
            job["output_checkpoint"] = str(output_path)
            job["process_exit_code"] = proc.returncode
            job["log_tail"] = combined_log
            job["updated_at"] = _now()

        if proc.returncode != 0:
            raise RuntimeError(f"Training script failed with exit code {proc.returncode}.")

        reload_result = {
            "status": "skipped",
            "active_model_path": b.active_model_path,
        }
        if params.get("auto_hot_swap", True):
            reload_result = b.reload_model(str(output_path))
            if reload_result.get("status") != "reloaded":
                raise RuntimeError(
                    f"Training succeeded, but hot-swap failed: {reload_result.get('error', 'unknown error')}"
                )

        with training_jobs_lock:
            job = training_jobs.get(job_id)
            if job:
                job["status"] = "completed"
                job["reload"] = reload_result
                job["finished_at"] = _now()
                job["updated_at"] = _now()
    except Exception as exc:
        logger.error(f"Training job {job_id} failed: {exc}")
        with training_jobs_lock:
            job = training_jobs.get(job_id)
            if job:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = _now()
                job["updated_at"] = _now()
    finally:
        with training_jobs_lock:
            active_training_job_id = None

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
    # Accept multiple client payload shapes to avoid brittle frontend/backend coupling.
    user_input = data.get("input")
    if user_input is None:
        user_input = data.get("prompt")
    if user_input is None:
        user_input = data.get("message")
    if not isinstance(user_input, str):
        user_input = str(user_input or "")
    user_input = user_input.strip() or "Hello"
    llm_provider = str(data.get("llm_provider", "auto") or "auto")
    api_key_override = data.get("api_key")
    model_override = data.get("model")
    
    try:
        result = b.query(
            user_input,
            llm_provider=llm_provider,
            api_key_override=api_key_override,
            model_override=model_override,
        )
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
    interaction_id = data.get("interaction_id")
    correction = data.get("correction")
    
    b.trigger_learning_event(event, intensity, interaction_id=interaction_id, correction=correction)
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


@app.route("/api/atlantean/training/export")
def training_export():
    b = get_bridge()
    limit_raw = request.args.get("limit", "1000")
    try:
        limit = int(limit_raw)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    result = b.export_training_dataset(limit=limit)
    return jsonify(result)


@app.route("/api/atlantean/training/jobs", methods=["POST"])
def training_jobs_start():
    global active_training_job_id

    data = request.get_json() or {}
    with training_jobs_lock:
        if active_training_job_id is not None:
            current = training_jobs.get(active_training_job_id)
            if current and current.get("status") in {"queued", "running"}:
                return jsonify(
                    {
                        "error": "A training job is already running.",
                        "active_job_id": active_training_job_id,
                    }
                ), 409

        try:
            params = {
                "export_limit": max(10, min(10000, int(data.get("export_limit", 1500)))),
                "epochs": max(1, min(20, int(data.get("epochs", 2)))),
                "batch_size": max(1, min(64, int(data.get("batch_size", 8)))),
                "seq_len": max(16, min(512, int(data.get("seq_len", 128)))),
                "lr": float(data.get("lr", 3e-4)),
                "max_rows": max(0, min(200000, int(data.get("max_rows", 0)))),
                "device": str(data.get("device", "cpu")),
                "checkpoint_path": data.get("checkpoint_path"),
                "output_path": data.get("output_path"),
                "auto_hot_swap": bool(data.get("auto_hot_swap", True)),
            }
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid training job parameters."}), 400

        job_id = f"train_{uuid.uuid4().hex[:12]}"
        training_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "started_at": None,
            "finished_at": None,
            "params": params,
            "dataset_path": None,
            "output_checkpoint": None,
            "process_exit_code": None,
            "reload": None,
            "error": None,
            "log_tail": None,
        }
        _trim_training_jobs()
        active_training_job_id = job_id

    thread = threading.Thread(target=_run_training_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/atlantean/training/jobs")
def training_jobs_list():
    with training_jobs_lock:
        rows = list(training_jobs.values())
    rows.sort(key=lambda item: item.get("created_at", 0.0), reverse=True)
    return jsonify({"jobs": rows[:20]})


@app.route("/api/atlantean/training/jobs/<job_id>")
def training_job_get(job_id: str):
    with training_jobs_lock:
        job = training_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Training job not found."}), 404
    return jsonify(job)


@app.route("/api/atlantean/model/reload", methods=["POST"])
def model_reload():
    b = get_bridge()
    data = request.get_json() or {}
    model_path = data.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        return jsonify({"error": "model_path is required."}), 400

    result = b.reload_model(model_path)
    if result.get("status") != "reloaded":
        return jsonify(result), 500
    return jsonify(result)

if __name__ == "__main__":
    print("🚀 Starting Syntropy Governor Unified Backend on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=False)