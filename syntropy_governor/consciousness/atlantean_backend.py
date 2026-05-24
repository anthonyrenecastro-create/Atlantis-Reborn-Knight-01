"""
Atlantean Backend Server

HTTP server that exposes Atlantean Intelligence Core to the Quadra-Seer frontend.
This is the Phase 1 integration - a standalone backend that Quadra-Seer can call.

LEGACY NOTICE:
- This file is retained for historical reference and compatibility experiments.
- Canonical runtime backend for this repository is `unified_backend/server.py`.
- Preferred startup path is `./scripts/start_unified.sh` from repository root.

Run with: python atlantean_backend.py
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
import sys
import os
import io
import pickle
import uuid
import time
import hashlib
import json
from importlib import import_module
import redis
import numpy as np
import torch
from typing import Any, Dict, List
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# Import the bridge module
from atlantean_quadra_bridge import AtlanteanQuadraBridge, QuadraLearningEvent
from hrm import HRMAdapter, HRMState

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('VITE_GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_API_KEY != 'PLACEHOLDER_API_KEY':
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"✅ Gemini API configured")
else:
    client = None
    print("⚠️  No Gemini API key found - using mock responses")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Try to connect to Redis, fallback to in-memory storage if unavailable
try:
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_client = redis.Redis(host=redis_host, port=6379, socket_connect_timeout=2)
    redis_client.ping()  # Test connection
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_REDIS'] = redis_client
    Session(app)
    print(f"✅ Redis connected on {redis_host}:6379")
    REDIS_AVAILABLE = True
except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
    print(f"⚠️  Redis unavailable ({type(e).__name__}) - using in-memory storage")
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)
    REDIS_AVAILABLE = False
    
    # In-memory fallback for Redis operations
    class InMemoryRedis:
        def __init__(self):
            self.data = {}
            self.lists = {}
        
        def get(self, key):
            return self.data.get(key, None)
        
        def set(self, key, value):
            self.data[key] = value
        
        def incr(self, key):
            current = int(self.data.get(key, 0))
            self.data[key] = str(current + 1)
            return current + 1
        
        def rpush(self, key, value):
            if key not in self.lists:
                self.lists[key] = []
            self.lists[key].append(value)

        def lpush(self, key, value):
            if key not in self.lists:
                self.lists[key] = []
            self.lists[key].insert(0, value)
        
        def lrange(self, key, start, end):
            if key not in self.lists:
                return []
            items = self.lists[key]
            if end == -1:
                return items[start:]
            return items[start:end+1]

        def ltrim(self, key, start, end):
            if key not in self.lists:
                return
            items = self.lists[key]
            if end == -1:
                self.lists[key] = items[start:]
            else:
                self.lists[key] = items[start:end+1]
        
        def delete(self, key):
            if key in self.data:
                del self.data[key]
            if key in self.lists:
                del self.lists[key]
    
    redis_client = InMemoryRedis()

# In-process bridge cache
bridges = {}
hrm_adapters = {}

HOT_MEMORY_KEY_PREFIX = 'hot_memory:'
SNAPSHOT_INDEX_PREFIX = 'snapshot_index:'
HRM_STATE_KEY_PREFIX = 'hrm_state:'
EVENT_LOG_KEY_PREFIX = 'event_log:'
EVENT_SEQ_KEY_PREFIX = 'event_seq:'
EVENT_HEAD_HASH_PREFIX = 'event_head_hash:'
CHECKPOINT_INDEX_PREFIX = 'checkpoint_index:'
CHECKPOINT_INTERVAL = 20
SNAPSHOT_DATA_PREFIX = 'snapshot_data:'


def _canonical_json(data: Any) -> str:
    """Canonical JSON representation for deterministic hashing/signing."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _compute_hot_state_hash(bridge: 'AtlanteanQuadraBridge') -> str:
    """Compute deterministic hash of the authoritative hot-memory state."""
    state = {
        'phi1': bridge.hot_memory.phi1.tolist(),
        'phi5': bridge.hot_memory.phi5.tolist(),
        'Phi': bridge.hot_memory.Phi.tolist(),
        'Theta': bridge.hot_memory.Theta,
        'version': int(bridge.hot_memory.version),
        'device_id': bridge.hot_memory.device_id,
        'identity_fingerprint': bridge.hot_memory.identity_fingerprint,
    }
    return _sha256_hex(_canonical_json(state))


def _sign_event_hash_hex(bridge: 'AtlanteanQuadraBridge', event_hash: str) -> str | None:
    """Sign event hash with the active local identity when available."""
    if not bridge.identity:
        return None
    try:
        return bridge.identity.sign(event_hash.encode('utf-8')).hex()
    except Exception:
        return None


def _verify_event_signature(bridge: 'AtlanteanQuadraBridge', event_hash: str, signature_hex: str | None) -> bool:
    """Verify event signature when identity and signature are available."""
    if not bridge.identity:
        return signature_hex is None
    if not signature_hex:
        return False
    try:
        return bridge.identity.verify(event_hash.encode('utf-8'), bytes.fromhex(signature_hex))
    except Exception:
        return False


def _compute_merkle_root(event_hashes: List[str]) -> str:
    """Compute a simple merkle root over event hashes."""
    if not event_hashes:
        return _sha256_hex('')
    level = event_hashes[:]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: List[str] = []
        for i in range(0, len(level), 2):
            next_level.append(_sha256_hex(level[i] + level[i + 1]))
        level = next_level
    return level[0]


def _append_signed_event(
    session_id: str,
    bridge: 'AtlanteanQuadraBridge',
    event_type: str,
    payload: Dict[str, Any],
    hrm_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Append an immutable, hash-linked, signed event to the session ledger."""
    seq = int(redis_client.incr(f'{EVENT_SEQ_KEY_PREFIX}{session_id}'))
    prev_hash_raw = redis_client.get(f'{EVENT_HEAD_HASH_PREFIX}{session_id}')
    prev_hash = prev_hash_raw.decode('utf-8') if isinstance(prev_hash_raw, bytes) else (prev_hash_raw or '')

    unsigned_event = {
        'schema_version': 1,
        'session_id': session_id,
        'seq': seq,
        'timestamp': int(time.time() * 1000),
        'event_type': event_type,
        'payload': payload,
        'prev_hash': prev_hash,
        'hot_version': int(bridge.hot_memory.version),
        'state_hash': _compute_hot_state_hash(bridge),
        'hrm': hrm_snapshot or {},
        'identity_fingerprint': bridge.identity.fingerprint() if bridge.identity else None,
    }
    event_hash = _sha256_hex(_canonical_json(unsigned_event))
    signature_hex = _sign_event_hash_hex(bridge, event_hash)

    event = {
        **unsigned_event,
        'event_hash': event_hash,
        'signature': signature_hex,
    }
    redis_client.rpush(f'{EVENT_LOG_KEY_PREFIX}{session_id}', _canonical_json(event))
    redis_client.set(f'{EVENT_HEAD_HASH_PREFIX}{session_id}', event_hash)

    if seq % CHECKPOINT_INTERVAL == 0:
        _create_signed_checkpoint(
            session_id=session_id,
            bridge=bridge,
            label=f'Auto checkpoint #{seq}',
            source='auto_interval',
            hrm_snapshot=hrm_snapshot or {},
        )

    return event


def _create_signed_checkpoint(
    session_id: str,
    bridge: 'AtlanteanQuadraBridge',
    label: str | None,
    source: str,
    hrm_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a signed checkpoint record bound to the current event log state."""
    events_raw = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    event_hashes: List[str] = []
    for row in events_raw:
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            event_hashes.append(json.loads(decoded).get('event_hash', ''))
        except Exception:
            continue
    merkle_root = _compute_merkle_root([h for h in event_hashes if h])

    snapshot_payload = bridge.create_snapshot(label)
    checkpoint_body = {
        'schema_version': 1,
        'id': str(uuid.uuid4()),
        'session_id': session_id,
        'label': snapshot_payload.get('label'),
        'source': source,
        'created_at': int(time.time() * 1000),
        'seq': int(redis_client.get(f'{EVENT_SEQ_KEY_PREFIX}{session_id}') or 0),
        'state_hash': _compute_hot_state_hash(bridge),
        'merkle_root': merkle_root,
        'snapshot': snapshot_payload,
        'hrm': hrm_snapshot or {},
        'identity_fingerprint': bridge.identity.fingerprint() if bridge.identity else None,
    }
    checkpoint_hash = _sha256_hex(_canonical_json(checkpoint_body))
    checkpoint_sig = _sign_event_hash_hex(bridge, checkpoint_hash)
    checkpoint = {
        **checkpoint_body,
        'checkpoint_hash': checkpoint_hash,
        'signature': checkpoint_sig,
    }
    redis_client.lpush(f'{CHECKPOINT_INDEX_PREFIX}{session_id}', _canonical_json(checkpoint))
    redis_client.ltrim(f'{CHECKPOINT_INDEX_PREFIX}{session_id}', 0, 199)
    return checkpoint


def _verify_ledger_integrity(session_id: str, bridge: 'AtlanteanQuadraBridge') -> Dict[str, Any]:
    """Verify hash chain, event hashes, and signatures for a session ledger."""
    rows = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    prev = ''
    issues: List[Dict[str, Any]] = []
    verified = 0

    for idx, row in enumerate(rows):
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            event = json.loads(decoded)
        except Exception:
            issues.append({'index': idx, 'type': 'decode_error'})
            continue

        event_hash = event.get('event_hash', '')
        event_sig = event.get('signature')

        unsigned = dict(event)
        unsigned.pop('event_hash', None)
        unsigned.pop('signature', None)
        recomputed = _sha256_hex(_canonical_json(unsigned))

        if event.get('prev_hash', '') != prev:
            issues.append({'index': idx, 'type': 'chain_mismatch'})
        if event_hash != recomputed:
            issues.append({'index': idx, 'type': 'hash_mismatch'})
        if not _verify_event_signature(bridge, event_hash, event_sig):
            issues.append({'index': idx, 'type': 'signature_invalid'})

        prev = event_hash
        verified += 1

    checkpoints_rows = redis_client.lrange(f'{CHECKPOINT_INDEX_PREFIX}{session_id}', 0, -1)
    checkpoint_count = len(checkpoints_rows)
    return {
        'valid': len(issues) == 0,
        'session_id': session_id,
        'events_verified': verified,
        'checkpoint_count': checkpoint_count,
        'issues': issues,
        'head_hash': prev,
    }


def _deterministic_replay_proof(session_id: str, bridge: 'AtlanteanQuadraBridge') -> Dict[str, Any]:
    """
    Build a one-shot deterministic replay proof from the signed event log.

    Replay model:
    - Recompute each event hash from canonical unsigned payload.
    - Validate hash-chain linkage and signatures.
    - Use the last verified event state_hash as replay-derived expected state hash.
    - Compare with current live hot-memory state hash.
    """
    rows = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    prev_hash = ''
    issues: List[Dict[str, Any]] = []
    verified_events: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            event = json.loads(decoded)
        except Exception:
            issues.append({'index': idx, 'type': 'decode_error'})
            continue

        unsigned = dict(event)
        event_hash = unsigned.pop('event_hash', '')
        signature_hex = unsigned.pop('signature', None)
        recomputed = _sha256_hex(_canonical_json(unsigned))

        event_issues = []
        if unsigned.get('prev_hash', '') != prev_hash:
            event_issues.append('chain_mismatch')
        if event_hash != recomputed:
            event_issues.append('hash_mismatch')
        if not _verify_event_signature(bridge, event_hash, signature_hex):
            event_issues.append('signature_invalid')

        if event_issues:
            issues.append({'index': idx, 'type': 'event_invalid', 'details': event_issues})
            break

        verified_events.append(event)
        prev_hash = event_hash

    replay_state_hash = verified_events[-1].get('state_hash') if verified_events else None
    live_state_hash = _compute_hot_state_hash(bridge)

    return {
        'session_id': session_id,
        'events_total': len(rows),
        'events_verified': len(verified_events),
        'replay_state_hash': replay_state_hash,
        'live_state_hash': live_state_hash,
        'match': replay_state_hash == live_state_hash if replay_state_hash else False,
        'valid': len(issues) == 0,
        'issues': issues,
        'replay_head_hash': prev_hash,
        'verified_up_to_seq': verified_events[-1].get('seq') if verified_events else None,
    }


def _apply_hrm_field_coupling(bridge: 'AtlanteanQuadraBridge', hrm_snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Delegate to the bridge's own isomorphic coupling method.

    Stores the snapshot on the bridge first so that any subsequent internal
    hot-memory update (e.g. inside recall_chat_memory / recall_simulations)
    also uses the current HRM topology, not stale defaults.
    """
    bridge.set_hrm_snapshot(hrm_snapshot)
    return bridge._apply_hrm_field_coupling()


def _save_bridge_to_redis(session_id: str, bridge: 'AtlanteanQuadraBridge'):
    """Persist hot memory state to Redis (backend-persistent, not per-file)."""
    try:
        buf = io.BytesIO()
        if bridge.identity:
            bridge.hot_memory.save(buf, identity=bridge.identity)
        else:
            bridge.hot_memory.save(buf)
        redis_client.set(f'{HOT_MEMORY_KEY_PREFIX}{session_id}', buf.getvalue())
    except Exception as e:
        print(f"⚠️  Failed to persist hot memory to Redis for {session_id}: {e}")


def _load_bridge_from_redis(session_id: str, bridge: 'AtlanteanQuadraBridge') -> bool:
    """Load hot memory state from Redis. Returns True if state was found."""
    try:
        data = redis_client.get(f'{HOT_MEMORY_KEY_PREFIX}{session_id}')
        if data is None:
            return False
        buf = io.BytesIO(data)
        try:
            AtlanteanHotMemory = import_module('atlantean_core.hot_memory').AtlanteanHotMemory
        except Exception:
            AtlanteanHotMemory = import_module('hot_memory').AtlanteanHotMemory
        verify = bridge.identity if bridge.identity else None
        bridge.hot_memory = AtlanteanHotMemory.load(buf, verify_identity=verify)
        bridge.bridge.hot = bridge.hot_memory
        return True
    except Exception as e:
        print(f"⚠️  Failed to load hot memory from Redis for {session_id}: {e}")
        return False


def _save_hrm_to_redis(session_id: str, adapter: HRMAdapter):
    """Persist HRM state so phase/topology continue across requests."""
    try:
        s = adapter.state
        payload = {
            't': float(s.t),
            'theta': float(s.theta),
            'channel': int(s.channel),
            'domain': int(s.domain),
            'layer': int(s.layer),
            'phi': s.phi.tolist(),
            'S': s.S.tolist(),
            'guna': s.guna.tolist(),
            'energy': float(s.energy),
            'coherence': float(s.coherence),
        }
        redis_client.set(f'{HRM_STATE_KEY_PREFIX}{session_id}', json.dumps(payload))
    except Exception as e:
        print(f"⚠️  Failed to persist HRM state for {session_id}: {e}")


def _serialize_hrm_state(adapter: HRMAdapter) -> Dict[str, Any]:
    """Serialize full HRM state for deterministic snapshot restore."""
    s = adapter.state
    return {
        't': float(s.t),
        'theta': float(s.theta),
        'channel': int(s.channel),
        'domain': int(s.domain),
        'layer': int(s.layer),
        'phi': s.phi.tolist(),
        'S': s.S.tolist(),
        'guna': s.guna.tolist(),
        'energy': float(s.energy),
        'coherence': float(s.coherence),
    }


def _restore_hrm_state_from_snapshot(adapter: HRMAdapter, snapshot: Dict[str, Any]) -> bool:
    """Restore HRM adapter state from snapshot payload. Returns False on legacy/incomplete payload."""
    required = ('phi', 'S', 'guna', 'theta', 'channel', 'domain', 'layer', 't')
    if not all(key in snapshot for key in required):
        return False

    adapter.state = HRMState(
        t=float(snapshot['t']),
        theta=float(snapshot['theta']),
        channel=int(snapshot['channel']),
        domain=int(snapshot['domain']),
        layer=int(snapshot['layer']),
        phi=np.array(snapshot['phi'], dtype=float),
        S=np.array(snapshot['S'], dtype=float),
        guna=np.array(snapshot['guna'], dtype=float),
        energy=float(snapshot.get('energy', 0.0)),
        coherence=float(snapshot.get('coherence', 0.0)),
    )
    return True


def _load_hrm_from_redis(session_id: str, adapter: HRMAdapter) -> bool:
    """Load HRM state from Redis if available."""
    try:
        raw = redis_client.get(f'{HRM_STATE_KEY_PREFIX}{session_id}')
        if not raw:
            return False
        data = json.loads(raw)
        adapter.state = HRMState(
            t=float(data['t']),
            theta=float(data['theta']),
            channel=int(data['channel']),
            domain=int(data['domain']),
            layer=int(data['layer']),
            phi=np.array(data['phi'], dtype=float),
            S=np.array(data['S'], dtype=float),
            guna=np.array(data['guna'], dtype=float),
            energy=float(data.get('energy', 0.0)),
            coherence=float(data.get('coherence', 0.0)),
        )
        return True
    except Exception as e:
        print(f"⚠️  Failed to load HRM state for {session_id}: {e}")
        return False


def get_bridge_for_session(session_id: str) -> 'AtlanteanQuadraBridge':
    """Get or create the bridge instance for a session, backed by Redis."""
    if session_id not in bridges:
        bridges[session_id] = AtlanteanQuadraBridge(
            grid_size=(32, 32),
            device_id=f"quadra-seer-{session_id}"
        )
        if _load_bridge_from_redis(session_id, bridges[session_id]):
            print(f"✅ Loaded hot memory from Redis for session {session_id}")
        else:
            print(f"✨ Initialized new intelligence state for session {session_id}")
    return bridges[session_id]


def get_hrm_for_session(session_id: str) -> HRMAdapter:
    """Get or create per-session HRM adapter with persisted state."""
    if session_id not in hrm_adapters:
        hrm_adapters[session_id] = HRMAdapter()
        if _load_hrm_from_redis(session_id, hrm_adapters[session_id]):
            print(f"✅ Loaded HRM state from Redis for session {session_id}")
        else:
            print(f"✨ Initialized new HRM state for session {session_id}")
    return hrm_adapters[session_id]


def get_bridge() -> 'AtlanteanQuadraBridge':
    """Get the default (global) bridge instance."""
    return get_bridge_for_session('default')


def _resolve_session_id(explicit_session_id: str | None = None) -> str:
    """Resolve a stable session id without assuming Flask-Session internals."""
    if explicit_session_id:
        return explicit_session_id
    sid = getattr(session, 'sid', None)
    if sid:
        return str(sid)
    return 'default'


def _index_phase_snapshot(
    session_id: str,
    bridge: 'AtlanteanQuadraBridge',
    label: str | None = None,
    hrm_snapshot: dict | None = None,
    source: str = 'manual',
) -> dict:
    """Create and index a phase-locked snapshot record in Redis."""
    snapshot = bridge.create_snapshot(label)
    snapshot_id = str(uuid.uuid4())
    snapshot_data_key = f'{SNAPSHOT_DATA_PREFIX}{session_id}:{snapshot_id}'
    snapshot_record = {
        'id': snapshot_id,
        'session_id': session_id,
        'label': snapshot.get('label'),
        'created_at': snapshot.get('created_at'),
        'version': snapshot.get('version'),
        'source': source,
        'phase_lock': {
            'phi1_mean': float(bridge.hot_memory.phi1.mean()),
            'phi5_mean': float(bridge.hot_memory.phi5.mean()),
            'Phi': float(bridge.hot_memory.Phi.item()),
        },
        'hrm': hrm_snapshot or {},
        'snapshot_data_key': snapshot_data_key,
    }
    try:
        key = f'{SNAPSHOT_INDEX_PREFIX}{session_id}'
        redis_client.set(snapshot_data_key, json.dumps(snapshot))
        redis_client.lpush(key, json.dumps(snapshot_record))
        redis_client.ltrim(key, 0, 199)
    except Exception as e:
        print(f"⚠️  Failed to index snapshot for {session_id}: {e}")

    snapshot['id'] = snapshot_id
    snapshot['session_id'] = session_id
    snapshot['source'] = source
    if hrm_snapshot:
        snapshot['hrm'] = hrm_snapshot
    return snapshot


def _decode_json_row(row: Any) -> Dict[str, Any] | None:
    """Decode a Redis list row that stores JSON objects."""
    try:
        decoded = row.decode('utf-8') if isinstance(row, bytes) else row
        parsed = json.loads(decoded)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _load_snapshot_records(session_id: str) -> List[Dict[str, Any]]:
    """Load all snapshot records for a session from Redis."""
    key = f'{SNAPSHOT_INDEX_PREFIX}{session_id}'
    rows = redis_client.lrange(key, 0, -1)
    records: List[Dict[str, Any]] = []
    for row in rows:
        parsed = _decode_json_row(row)
        if parsed:
            records.append(parsed)
    return records


def _save_snapshot_records(session_id: str, records: List[Dict[str, Any]]) -> None:
    """Replace the snapshot list for a session with provided records."""
    key = f'{SNAPSHOT_INDEX_PREFIX}{session_id}'
    redis_client.delete(key)
    for record in records:
        redis_client.rpush(key, json.dumps(record))


# ========== Core API Endpoints ==========

@app.route('/api/atlantean/status', methods=['GET'])
def status():
    """Get current intelligence status."""
    session_id = request.args.get('session_id') or session.sid or 'default'
    b = get_bridge_for_session(session_id)
    return jsonify(b.get_status())


@app.route('/api/atlantean/query', methods=['POST'])
def query():
    """
    Process user query through Atlantean-powered LLM.
    
    Request body:
    {
        "input": "user message",
        "llm_provider": "gemini" | "edenai" | "mock",
        "api_key": "optional_api_key_override"
    }
    """
    data = request.json
    user_input = data.get('input', '')
    llm_provider = data.get('llm_provider', 'gemini')
    api_key = data.get('api_key') or GEMINI_API_KEY
    session_id = data.get('session_id') or session.sid or 'default'
    history = data.get('history', [])
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    hrm_step = hrm.step(steps=1)
    hrm_snapshot = hrm_step.get('snapshot', {})
    hrm_coupling = _apply_hrm_field_coupling(b, hrm_snapshot)
    
    # Build context from hot memory
    status = b.get_status()
    field_stats = status['field_stats']
    learning_capacity = status['learning_capacity']

    # Recall cross-chat/domain memory from cold storage and feed it into generation.
    recalled_memories = b.recall_chat_memory(
        query=user_input,
        limit=8,
        session_id=session_id,
        include_simulations=True,
    )
    recalled_facts = b.extract_recalled_facts(recalled_memories, limit=4)
    user_input_norm = user_input.strip().lower()
    filtered_facts = []
    seen_facts = set()
    for fact in recalled_facts:
        norm = fact.strip().lower()
        if not norm or norm == user_input_norm or norm in seen_facts:
            continue
        seen_facts.add(norm)
        filtered_facts.append(fact)
    recalled_facts = filtered_facts
    
    # Encode intelligence state as natural language context
    context_parts = []
    
    if field_stats['phi1_mean'] > 0.5:
        context_parts.append("Operating with high confidence and decisiveness.")
    elif field_stats['phi1_mean'] < -0.5:
        context_parts.append("In exploratory mode, considering multiple possibilities.")
    else:
        context_parts.append("Balanced state, ready to adapt.")
    
    if field_stats['phi5_mean'] > 0.5:
        context_parts.append("High learning capacity - actively adapting to new patterns.")
    elif field_stats['phi5_mean'] < 0.3:
        context_parts.append("Stable patterns established - operating from experience.")
    
    if learning_capacity > 0.7:
        context_parts.append("Strong potential for growth and adaptation.")
    elif learning_capacity < 0.3:
        context_parts.append("Patterns consolidating - reinforcing core knowledge.")
    
    intelligence_context = " ".join(context_parts)
    hrm_context = (
        f"HRM phase theta={hrm_snapshot.get('theta', 0):.3f}, "
        f"coherence={hrm_snapshot.get('coherence', 0):.3f}, "
        f"energy={hrm_snapshot.get('energy', 0):.3f}, "
        f"channel={hrm_snapshot.get('channel', 0)}, "
        f"domain={hrm_snapshot.get('domain', 0)}, "
        f"layer={hrm_snapshot.get('layer', 0)}."
    )
    
    # Call LLM (stateless - context from fields, not history)
    response_text = None
    model_used = None
    generation_mode = 'mock'
    
    try:
        if llm_provider == 'gemini' and api_key and api_key != 'PLACEHOLDER_API_KEY':
            # Create client with provided key
            api_client = genai.Client(api_key=api_key)
            
            # Build prompt with Atlantean context
            system_prompt = f"""You are Quadra Seer Intelligence, powered by Atlantean Core.

Your current intelligence state: {intelligence_context}

Current HRM state: {hrm_context}

You are a brilliant predictive intelligence entity, expert in data forecasting, complex systems, and technical analysis.

Respond naturally and helpfully to the user's query. Your intelligence fields are evolving based on user interactions."""
            
            history_lines = []
            for turn in history[-12:]:
                role = str(turn.get('role', '')).lower()
                content = str(turn.get('content', '')).strip()
                if not content:
                    continue
                if role == 'user':
                    history_lines.append(f"User: {content}")
                else:
                    history_lines.append(f"Assistant: {content}")

            conversation_context = "\n".join(history_lines)

            memory_lines = []
            for mem in recalled_memories:
                mem_type = mem.get('type', 'memory')
                source_session = mem.get('session_id', 'unknown')
                content = str(mem.get('content_raw', ''))
                # Strip legacy provenance/debug tails from older stored turns.
                if '\n\n[Provenance]' in content:
                    content = content.split('\n\n[Provenance]', 1)[0].strip()
                if len(content) > 260:
                    content = content[:260] + '...'
                memory_lines.append(f"[{mem_type} | session={source_session}] {content}")
            memory_context = "\n".join(memory_lines)

            facts_context = "\n".join([f"- {fact}" for fact in recalled_facts])

            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation so far:\n{conversation_context}\n\n"
                f"High-confidence recalled user facts:\n{facts_context}\n\n"
                f"Recalled memory from prior chats/domains:\n{memory_context}\n\n"
                f"User: {user_input}\n"
                f"Assistant:"
            )
            
            # Generate response using a robust model fallback chain.
            # Google model IDs evolve frequently; this avoids hard failures when one is retired.
            configured_model = os.getenv('GEMINI_MODEL') or os.getenv('ATLANTEAN_GEMINI_MODEL')
            candidate_models = [m for m in [
                configured_model,
                'gemini-2.5-flash',
                'gemini-2.0-flash',
                'gemini-1.5-flash',
            ] if m]

            last_model_error = None
            for model_name in candidate_models:
                try:
                    response = api_client.models.generate_content(
                        model=model_name,
                        contents=full_prompt
                    )
                    response_text = response.text
                    model_used = model_name
                    generation_mode = 'gemini'
                    break
                except Exception as model_err:
                    last_model_error = model_err

            if response_text is None and last_model_error:
                raise last_model_error
            
        else:
            memory_snippets = []
            for mem in recalled_memories[:3]:
                snippet = str(mem.get('content_raw', ''))
                if len(snippet) > 120:
                    snippet = snippet[:120] + '...'
                memory_snippets.append(f"- {snippet}")
            memory_section = "\n".join(memory_snippets) if memory_snippets else "- No prior recalled memory"

            # Mock response for testing
            response_text = f"""🧠 **Atlantean Intelligence Active**

Intelligence State: {intelligence_context}

Recalled Memory ({len(recalled_memories)}):
{memory_section}

Processing your query: "{user_input}"

This is a demo response. Configure a real Gemini API key to enable full LLM capabilities.

Current field stats:
- Decision field (φ₁): {field_stats['phi1_mean']:.3f}
- Learning field (φ₅): {field_stats['phi5_mean']:.3f}
- Global coherence (Φ): {field_stats['Phi']:.3f}
- Learning capacity: {learning_capacity:.1%}"""
            generation_mode = 'mock'
    
    except Exception as e:
        # Fallback to mock on error
        response_text = f"🧠 Atlantean Intelligence (Error: {str(e)})\n\nIntelligence State: {intelligence_context}\n\nQuery processed, but LLM call failed. Using fallback response."
        generation_mode = 'fallback'

    # Build retrieval/generation confidence and integrity metadata.
    retrieval_confidence = 0.0
    if recalled_memories:
        avg_score = sum(float(m.get('score', 0.0)) for m in recalled_memories) / len(recalled_memories)
        retrieval_confidence = min(max(avg_score, 0.0), 1.0)

    generation_confidence = {
        'gemini': 0.82,
        'mock': 0.45,
        'fallback': 0.35,
    }.get(generation_mode, 0.4)

    overall_confidence = round((retrieval_confidence * 0.4) + (generation_confidence * 0.6), 3)

    recalled_items = []
    for mem in recalled_memories[:5]:
        snippet = str(mem.get('content_raw', ''))
        if len(snippet) > 140:
            snippet = snippet[:140] + '...'
        recalled_items.append({
            'type': mem.get('type'),
            'session_id': mem.get('session_id'),
            'domain': mem.get('domain'),
            'score': round(float(mem.get('score', 0.0)), 3),
            'timestamp': mem.get('timestamp'),
            'snippet': snippet,
        })

    provenance_payload = {
        'session_id': session_id,
        'query': user_input,
        'model_used': model_used,
        'generation_mode': generation_mode,
        'recalled_items': recalled_items,
        'recalled_count': len(recalled_memories),
    }
    provenance_hash = hashlib.sha256(
        json.dumps(provenance_payload, sort_keys=True).encode('utf-8')
    ).hexdigest()

    # Keep provenance in structured API metadata only; do not leak internal
    # retrieval/generation traces into the user-visible assistant text.
    if not response_text:
        response_text = "I processed your request, but no response text was generated."

    # Persist both sides of this interaction into cold memory.
    try:
        b.store_chat_turn(
            role='user',
            content=user_input,
            session_id=session_id,
            domain='chat',
            relevance=0.45,
        )
        if response_text:
            assistant_memory = response_text
            if len(assistant_memory) > 320:
                assistant_memory = assistant_memory[:320] + '...'
            b.store_chat_turn(
                role='assistant',
                content=assistant_memory,
                session_id=session_id,
                domain='chat',
                relevance=0.3,
            )
    except Exception as memory_err:
        print(f"⚠️  Failed to persist chat memory for {session_id}: {memory_err}")
    
    # Auto-save after each query
    _save_bridge_to_redis(session_id, b)
    _save_hrm_to_redis(session_id, hrm)
    auto_snapshot = _index_phase_snapshot(
        session_id=session_id,
        bridge=b,
        label=f"Auto phase lock @ t={hrm_snapshot.get('t', 0):.2f}",
        hrm_snapshot=hrm_snapshot,
        source='query_auto',
    )
    ledger_event = _append_signed_event(
        session_id=session_id,
        bridge=b,
        event_type='query_processed',
        payload={
            'query': user_input,
            'generation_mode': generation_mode,
            'model_used': model_used,
            'recalled_count': len(recalled_memories),
            'confidence_overall': overall_confidence,
            'hrm_coupling': hrm_coupling,
        },
        hrm_snapshot=hrm_snapshot,
    )
    
    return jsonify({
        'response': response_text,
        'status': b.get_status(),
        'intelligence_context': intelligence_context,
        'hrm': hrm_snapshot,
        'auto_snapshot': {
            'id': auto_snapshot.get('id'),
            'label': auto_snapshot.get('label'),
            'source': auto_snapshot.get('source'),
        },
        'recalled_memory_count': len(recalled_memories),
        'recalled_facts': recalled_facts,
        'session_id': session_id,
        'retrieval': {
            'count': len(recalled_memories),
            'items': recalled_items,
        },
        'generation': {
            'mode': generation_mode,
            'model_used': model_used,
        },
        'confidence': {
            'overall': overall_confidence,
            'retrieval': round(retrieval_confidence, 3),
            'generation': generation_confidence,
        },
        'hrm_coupling': hrm_coupling,
        'ledger': {
            'seq': ledger_event.get('seq'),
            'event_hash': ledger_event.get('event_hash'),
            'signed': bool(ledger_event.get('signature')),
        },
        'integrity': {
            'provenance_hash': provenance_hash,
            'timestamp': int(time.time() * 1000),
        },
    })


@app.route('/api/atlantean/fields', methods=['GET'])
def get_fields():
    """Get field visualization data."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    b = get_bridge_for_session(session_id)
    return jsonify(b.get_field_visualization_data())


@app.route('/api/atlantean/chat/history', methods=['GET'])
def get_chat_history():
    """Get recent persisted chat turns for a session."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    limit = int(request.args.get('limit', 80))
    b = get_bridge_for_session(session_id)
    messages = b.list_chat_history(session_id=session_id, limit=limit)
    return jsonify({
        'session_id': session_id,
        'messages': messages,
    })


@app.route('/api/atlantean/learning-event', methods=['POST'])
def learning_event():
    """
    Trigger a learning event.
    
    Request body:
    {
        "session_id": "optional session id",
        "event": "user_confirmation" | "user_correction" | etc.,
        "data": { ... event-specific data ... }
    }
    """
    data = request.json
    session_id = data.get('session_id') or session.sid or 'default'
    event = data.get('event')
    event_data = data.get('data', {})
    
    if not event:
        return jsonify({'error': 'No event type provided'}), 400
    
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    
    try:
        b.on_event(event, **event_data)
        # Capture HRM snapshot AFTER event application, which modifies hot memory
        hrm_snapshot = hrm.snapshot()
        hrm_coupling = _apply_hrm_field_coupling(b, hrm_snapshot)
        _save_bridge_to_redis(session_id, b)
        _append_signed_event(
            session_id=session_id,
            bridge=b,
            event_type='learning_event',
            payload={
                'event': event,
                'data': event_data,
                'hrm_coupling': hrm_coupling,
            },
            hrm_snapshot=hrm_snapshot,
        )
        
        return jsonify({
            'success': True,
            'status': b.get_status(),
            'hrm_coupling': hrm_coupling,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== Simulation Endpoints ==========

@app.route('/api/atlantean/simulation/store', methods=['POST'])
def store_simulation():
    """Store simulation in cold memory."""
    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    simulation = data.get('simulation')
    confidence = data.get('confidence', 0.5)
    
    if not simulation:
        return jsonify({'error': 'No simulation data'}), 400
    
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    b.store_simulation(simulation, confidence, session_id=session_id)
    # Capture HRM snapshot AFTER store operation, which modifies hot memory
    hrm_snapshot = hrm.snapshot()
    _save_bridge_to_redis(session_id, b)
    _append_signed_event(
        session_id=session_id,
        bridge=b,
        event_type='simulation_store',
        payload={
            'scenario': simulation.get('scenario', 'unknown') if isinstance(simulation, dict) else 'unknown',
            'confidence': float(confidence),
        },
        hrm_snapshot=hrm_snapshot,
    )
    
    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/atlantean/simulation/recall', methods=['POST'])
def recall_simulations():
    """Recall past simulations."""
    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    query = data.get('query', '')
    limit = data.get('limit', 10)

    b = get_bridge_for_session(session_id)
    simulations = b.recall_simulations(query, limit)

    return jsonify({'session_id': session_id, 'simulations': simulations})


@app.route('/api/atlantean/simulation/list', methods=['GET'])
def list_simulations():
    """List all stored simulations directly from Redis (no embedding needed)."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    limit = int(request.args.get('limit', 50))
    b = get_bridge_for_session(session_id)
    simulations = b.list_all_simulations(limit, session_id=session_id)
    return jsonify({'session_id': session_id, 'simulations': simulations})


@app.route('/api/atlantean/cold/manifests', methods=['GET'])
def list_cold_manifests():
    """List detachable cold-memory manifests for a session."""
    session_id = request.args.get('session_id') or session.sid or 'default'
    include_detached = str(request.args.get('include_detached', 'false')).lower() == 'true'
    b = get_bridge_for_session(session_id)
    manifests = b.list_cold_manifests(include_detached=include_detached)
    return jsonify({'session_id': session_id, 'manifests': manifests})


@app.route('/api/atlantean/cold/manifest/export', methods=['GET'])
def export_cold_manifest():
    """Export a detachable cold-memory manifest."""
    session_id = request.args.get('session_id') or session.sid or 'default'
    manifest_id = request.args.get('manifest_id', 'default')
    b = get_bridge_for_session(session_id)
    exported = b.export_cold_manifest(manifest_id)
    return jsonify({'session_id': session_id, 'manifest': exported})


@app.route('/api/atlantean/cold/manifest/import', methods=['POST'])
def import_cold_manifest():
    """Import a detached cold-memory manifest into a session."""
    data = request.json or {}
    session_id = data.get('session_id') or session.sid or 'default'
    manifest = data.get('manifest')
    if not manifest:
        return jsonify({'error': 'No manifest provided'}), 400

    b = get_bridge_for_session(session_id)
    result = b.import_cold_manifest(manifest)
    return jsonify({'session_id': session_id, 'result': result})


@app.route('/api/atlantean/cold/tombstone', methods=['POST'])
def tombstone_cold_item():
    """Tombstone a cold-memory item instead of hard deleting it."""
    data = request.json or {}
    session_id = data.get('session_id') or session.sid or 'default'
    item_id = data.get('item_id')
    reason = data.get('reason', 'manual')
    if not item_id:
        return jsonify({'error': 'No item_id provided'}), 400

    b = get_bridge_for_session(session_id)
    b.tombstone_cold_item(item_id, reason=reason)
    return jsonify({'success': True, 'session_id': session_id, 'item_id': item_id, 'reason': reason})


# ========== Archive Endpoints ==========

@app.route('/api/atlantean/snapshot', methods=['POST'])
def create_snapshot():
    """Create a labeled snapshot."""
    data = request.json or {}
    label = data.get('label')
    session_id = data.get('session_id') or session.sid or 'default'
    
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    hrm_snapshot = _serialize_hrm_state(hrm)
    snapshot = _index_phase_snapshot(
        session_id=session_id,
        bridge=b,
        label=label,
        hrm_snapshot=hrm_snapshot,
        source='manual',
    )
    checkpoint = _create_signed_checkpoint(
        session_id=session_id,
        bridge=b,
        label=label,
        source='manual_snapshot',
        hrm_snapshot=hrm_snapshot,
    )
    
    return jsonify({'snapshot': snapshot, 'checkpoint': {
        'id': checkpoint.get('id'),
        'seq': checkpoint.get('seq'),
        'checkpoint_hash': checkpoint.get('checkpoint_hash'),
        'signed': bool(checkpoint.get('signature')),
    }})


@app.route('/api/atlantean/snapshots', methods=['GET'])
def list_snapshots():
    """List indexed snapshots (phase-locked memory points) for a session."""
    session_id = request.args.get('session_id') or session.sid or 'default'
    limit = int(request.args.get('limit', 50))
    key = f'{SNAPSHOT_INDEX_PREFIX}{session_id}'
    rows = redis_client.lrange(key, 0, max(limit - 1, 0))
    snapshots = []
    for row in rows:
        parsed = _decode_json_row(row)
        if parsed:
            snapshots.append(parsed)
    return jsonify({'session_id': session_id, 'snapshots': snapshots})


@app.route('/api/atlantean/snapshot/restore', methods=['POST'])
def restore_snapshot():
    """Restore hot-memory state from a previously indexed snapshot."""
    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    snapshot_id = data.get('snapshot_id')

    if not snapshot_id:
        return jsonify({'error': 'No snapshot_id provided'}), 400

    records = _load_snapshot_records(session_id)
    target = next((record for record in records if record.get('id') == snapshot_id), None)
    if not target:
        return jsonify({'error': 'Snapshot not found', 'session_id': session_id, 'snapshot_id': snapshot_id}), 404

    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)

    # Rehydrate deterministic field state from stored snapshot payload.
    try:
        source = None
        snapshot_data_key = target.get('snapshot_data_key')
        if snapshot_data_key:
            raw_source = redis_client.get(snapshot_data_key)
            if raw_source:
                decoded = raw_source.decode('utf-8') if isinstance(raw_source, bytes) else raw_source
                source = json.loads(decoded)
        if not isinstance(source, dict):
            return jsonify({'error': 'Snapshot payload is unavailable for restore'}), 400

        b.hot_memory.phi1 = torch.tensor(source['phi1'], dtype=b.hot_memory.phi1.dtype)
        b.hot_memory.phi5 = torch.tensor(source['phi5'], dtype=b.hot_memory.phi5.dtype)
        b.hot_memory.Phi = torch.tensor([float(source['Phi'])], dtype=b.hot_memory.Phi.dtype)
        b.hot_memory.Theta = source.get('Theta', {}) or {}
        b.hot_memory.version = int(source.get('version', b.hot_memory.version))
        snapshot_ts = source.get('timestamp')
        if isinstance(snapshot_ts, (int, float)):
            b.hot_memory.last_update = float(snapshot_ts)
        b.bridge.hot = b.hot_memory

        # Restore HRM state when available for phase-consistent post-restore behavior.
        restored_hrm_payload = target.get('hrm')
        if isinstance(restored_hrm_payload, dict):
            restored = _restore_hrm_state_from_snapshot(hrm, restored_hrm_payload)
            if not restored:
                print(f"ℹ️  Snapshot {snapshot_id} has legacy HRM payload; keeping current HRM state")

        b.set_hrm_snapshot(hrm.snapshot())
    except Exception as e:
        return jsonify({'error': f'Failed to restore snapshot: {e}'}), 400

    _save_bridge_to_redis(session_id, b)
    _save_hrm_to_redis(session_id, hrm)
    _append_signed_event(
        session_id=session_id,
        bridge=b,
        event_type='snapshot_restore',
        payload={
            'snapshot_id': snapshot_id,
            'snapshot_label': target.get('label'),
        },
        hrm_snapshot=hrm.snapshot(),
    )

    return jsonify({
        'success': True,
        'session_id': session_id,
        'snapshot_id': snapshot_id,
        'status': b.get_status(),
    })


@app.route('/api/atlantean/snapshot/delete', methods=['POST'])
def delete_snapshot():
    """Delete a snapshot record from the archive index."""
    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    snapshot_id = data.get('snapshot_id')

    if not snapshot_id:
        return jsonify({'error': 'No snapshot_id provided'}), 400

    records = _load_snapshot_records(session_id)
    filtered = [record for record in records if record.get('id') != snapshot_id]
    if len(filtered) == len(records):
        return jsonify({'error': 'Snapshot not found', 'session_id': session_id, 'snapshot_id': snapshot_id}), 404

    _save_snapshot_records(session_id, filtered)
    snapshot_data_key = next(
        (record.get('snapshot_data_key') for record in records if record.get('id') == snapshot_id),
        None,
    )
    if snapshot_data_key:
        redis_client.delete(snapshot_data_key)

    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    _append_signed_event(
        session_id=session_id,
        bridge=b,
        event_type='snapshot_delete',
        payload={
            'snapshot_id': snapshot_id,
        },
        hrm_snapshot=hrm.snapshot(),
    )

    return jsonify({
        'success': True,
        'session_id': session_id,
        'snapshot_id': snapshot_id,
        'remaining': len(filtered),
    })


@app.route('/api/atlantean/checkpoint', methods=['POST'])
def create_checkpoint():
    """Create a signed checkpoint tied to the session event ledger."""
    data = request.json or {}
    session_id = data.get('session_id') or session.sid or 'default'
    label = data.get('label')

    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    checkpoint = _create_signed_checkpoint(
        session_id=session_id,
        bridge=b,
        label=label,
        source='manual_checkpoint',
        hrm_snapshot=hrm.snapshot(),
    )
    return jsonify({'checkpoint': checkpoint})


@app.route('/api/atlantean/integrity/verify', methods=['GET'])
def verify_integrity():
    """Verify the signed event/checkpoint chain for a session."""
    session_id = request.args.get('session_id') or session.sid or 'default'
    b = get_bridge_for_session(session_id)
    return jsonify(_verify_ledger_integrity(session_id, b))


@app.route('/api/atlantean/integrity/replay', methods=['GET'])
def replay_integrity_proof():
    """Deterministic one-shot replay proof: event-log state hash vs current live state hash."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    b = get_bridge_for_session(session_id)
    return jsonify(_deterministic_replay_proof(session_id, b))


@app.route('/api/atlantean/export', methods=['GET'])
def export_intelligence_bundle():
    """Export signed events/checkpoints + current state for user-owned portability."""
    session_id = request.args.get('session_id') or session.sid or 'default'
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)

    event_rows = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    checkpoint_rows = redis_client.lrange(f'{CHECKPOINT_INDEX_PREFIX}{session_id}', 0, -1)
    events = []
    checkpoints = []

    for row in event_rows:
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            events.append(json.loads(decoded))
        except Exception:
            continue

    for row in checkpoint_rows:
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            checkpoints.append(json.loads(decoded))
        except Exception:
            continue

    bundle = {
        'schema_version': 1,
        'exported_at': int(time.time() * 1000),
        'session_id': session_id,
        'identity': b.identity.to_dict() if b.identity else None,
        'events': events,
        'checkpoints': checkpoints,
        'hot_state_hash': _compute_hot_state_hash(b),
        'hot_state': b.create_snapshot('Export state'),
        'hrm_state': hrm.snapshot(),
    }
    return jsonify({'bundle': bundle})


# ========== Sync Endpoints ==========

@app.route('/api/atlantean/sync/prepare', methods=['GET'])
def prepare_sync():
    """Prepare sync package for multi-device."""
    b = get_bridge()
    
    try:
        package = b.prepare_sync_package()
        return jsonify({'package': package})
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/atlantean/sync/merge', methods=['POST'])
def merge_sync():
    """Merge sync package from another device."""
    data = request.json
    package = data.get('package')
    
    if not package:
        return jsonify({'error': 'No sync package'}), 400
    
    b = get_bridge()
    
    try:
        b.merge_from_device(package)
        _save_bridge_to_redis('default', b)
        
        return jsonify({
            'success': True,
            'status': b.get_status()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== Admin Endpoints ==========

@app.route('/api/atlantean/reset', methods=['POST'])
def reset():
    """Reset intelligence state (for testing)."""
    bridges.pop('default', None)
    hrm_adapters.pop('default', None)
    
    # Delete state from Redis
    try:
        redis_client.delete(f'{HOT_MEMORY_KEY_PREFIX}default')
        redis_client.delete(f'{HRM_STATE_KEY_PREFIX}default')
        redis_client.delete(f'{EVENT_SEQ_KEY_PREFIX}default')
        redis_client.delete(f'{EVENT_HEAD_HASH_PREFIX}default')
        redis_client.delete(f'{EVENT_LOG_KEY_PREFIX}default')
        redis_client.delete(f'{CHECKPOINT_INDEX_PREFIX}default')
    except Exception as e:
        print(f"⚠️  Failed to delete Redis key: {e}")
    
    # Re-initialize
    get_bridge()
    
    return jsonify({
        'success': True,
        'message': 'Intelligence state reset'
    })


@app.route('/api/atlantean/save', methods=['POST'])
def save():
    """Manually save state."""
    b = get_bridge()
    _save_bridge_to_redis('default', b)
    return jsonify({'success': True})


@app.route('/api/atlantean/load', methods=['POST'])
def load():
    """Manually load state."""
    b = get_bridge()
    
    try:
        if not _load_bridge_from_redis('default', b):
            return jsonify({'error': 'No saved state found in backend'}), 404
        return jsonify({
            'success': True,
            'status': b.get_status()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== Health Check ==========

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Atlantean Backend',
        'version': '1.0.0'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 ATLANTEAN INTELLIGENCE BACKEND")
    print("="*60)
    print("\nStarting server on http://localhost:5001")
    print("\nAPI Endpoints:")
    print("  GET  /health                        - Health check")
    print("  GET  /api/atlantean/status          - Get intelligence status")
    print("  POST /api/atlantean/query           - Process user query")
    print("  GET  /api/atlantean/fields          - Get field visualization")
    print("  POST /api/atlantean/learning-event  - Trigger learning")
    print("  POST /api/atlantean/simulation/*    - Simulation storage")
    print("  POST /api/atlantean/snapshot        - Create snapshot")
    print("  GET  /api/atlantean/sync/*          - Multi-device sync")
    print("  POST /api/atlantean/reset           - Reset state")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Initialize on startup (loads from Redis if available)
    get_bridge()
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False
    )
