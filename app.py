import json
import os
import queue
import threading
import uuid
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS

from agent import OperationPlan, OperationStep, WikiAgent
from wiki_client import WikiClient

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'wikigen-dev-secret')
CORS(app)

DATA_DIR = Path(os.environ.get('DATA_DIR', '.'))
CONNECTIONS_FILE = DATA_DIR / 'connections.json'
HISTORY_FILE = DATA_DIR / 'history.json'
PLANS_DIR = DATA_DIR / 'plans'
PLANS_DIR.mkdir(exist_ok=True)
ARCHIVED_PLANS_FILE = DATA_DIR / 'archived_plans.json'

anthropic_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))

_plans: dict[str, OperationPlan] = {}
_wiki_clients: dict[str, WikiClient] = {}
_job_queues: dict[str, queue.Queue] = {}      # plan phase 1 SSE queues
_exec_queues: dict[str, queue.Queue] = {}     # execute phase 2 SSE queues
_cancel_events: dict[str, threading.Event] = {}


# ─── CONNECTIONS ──────────────────────────────────────────────────────────────

def _load_connections() -> dict:
    if CONNECTIONS_FILE.exists():
        return json.loads(CONNECTIONS_FILE.read_text())
    default = {
        'connections': [],
        'active_connection_id': None,
    }
    if os.environ.get('WIKI_URL'):
        conn_id = str(uuid.uuid4())
        default['connections'].append({
            'id': conn_id,
            'name': os.environ.get('WIKI_NAME', 'Default Wiki'),
            'wiki_url': os.environ['WIKI_URL'],
            'username': os.environ.get('WIKI_USERNAME', ''),
            'password': os.environ.get('WIKI_PASSWORD', ''),
            'system_prompt': '',
            'chips': [],
        })
        default['active_connection_id'] = conn_id
        _save_connections(default)
    return default


def _save_connections(data: dict):
    CONNECTIONS_FILE.write_text(json.dumps(data, indent=2))


def _get_active_connection() -> dict | None:
    data = _load_connections()
    active_id = data.get('active_connection_id')
    for conn in data.get('connections', []):
        if conn['id'] == active_id:
            return conn
    if data.get('connections'):
        return data['connections'][0]
    return None


def _get_connection_by_id(connection_id: str) -> dict | None:
    data = _load_connections()
    for conn in data.get('connections', []):
        if conn['id'] == connection_id:
            return conn
    return None


def get_wiki_client(connection_id: str) -> WikiClient | None:
    if connection_id in _wiki_clients:
        client = _wiki_clients[connection_id]
        if not client._connected:
            client.connect()
        return client
    conn = _get_connection_by_id(connection_id)
    if not conn:
        return None
    client = WikiClient(conn['wiki_url'], conn['username'], conn['password'])
    client.connect()
    _wiki_clients[connection_id] = client
    return client


def _resolve_connection(request_data: dict | None = None) -> dict | None:
    conn_id = None
    if request_data:
        conn_id = request_data.get('connection_id')
    if not conn_id:
        conn_id = request.args.get('connection_id')
    if conn_id:
        return _get_connection_by_id(conn_id)
    return _get_active_connection()


def _save_plan_to_disk(plan: OperationPlan):
    try:
        path = PLANS_DIR / f'{plan.id}.json'
        path.write_text(json.dumps(plan.to_dict(), indent=2))
    except Exception:
        pass


def _append_history(plan: OperationPlan, results: list[dict]):
    try:
        history = []
        if HISTORY_FILE.exists():
            history = json.loads(HISTORY_FILE.read_text())
        done = sum(1 for r in results if r.get('success'))
        history.append({
            'plan_id': plan.id,
            'operation_type': plan.operation_type,
            'description': plan.description,
            'step_count': len(plan.steps),
            'success_count': done,
            'completed_at': datetime.now(timezone.utc).isoformat(),
        })
        HISTORY_FILE.write_text(json.dumps(history[-200:], indent=2))
    except Exception:
        pass


# ─── SSE ──────────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f'data: {json.dumps(data)}\n\n'


def _plan_from_disk(plan_id: str) -> OperationPlan | None:
    path = PLANS_DIR / f'{plan_id}.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    plan = OperationPlan(
        id=data['id'],
        operation_type=data.get('operation_type', ''),
        description=data.get('description', ''),
        connection_id=data.get('connection_id', ''),
        created_at=data.get('created_at', ''),
        status=data.get('status', 'pending'),
    )
    for s in data.get('steps', []):
        plan.steps.append(OperationStep(
            id=s['id'],
            type=s.get('type', 'write'),
            title=s.get('title', ''),
            from_title=s.get('from_title'),
            content=s.get('content'),
            old_content=s.get('old_content'),
            summary=s.get('summary', ''),
            description=s.get('description', ''),
            depends_on=s.get('depends_on', []),
            status=s.get('status', 'pending'),
            error=s.get('error'),
            links_to=s.get('links_to', []),
            diff=s.get('diff'),
            image_file=s.get('image_file'),
            commons_url=s.get('commons_url'),
        ))
    return plan


# ─── CONNECTION MANAGER ROUTES ────────────────────────────────────────────────

@app.route('/api/connections', methods=['GET'])
def list_connections():
    return jsonify(_load_connections())


@app.route('/api/connections', methods=['POST'])
def add_connection():
    body = request.json or {}
    data = _load_connections()
    conn = {
        'id': str(uuid.uuid4()),
        'name': body.get('name', 'New Wiki'),
        'wiki_url': body.get('wiki_url', ''),
        'username': body.get('username', ''),
        'password': body.get('password', ''),
        'system_prompt': body.get('system_prompt', ''),
        'chips': body.get('chips', []),
    }
    data['connections'].append(conn)
    if not data.get('active_connection_id'):
        data['active_connection_id'] = conn['id']
    _save_connections(data)
    return jsonify({'success': True, 'connection': conn})


@app.route('/api/connections/<conn_id>', methods=['PUT'])
def update_connection(conn_id: str):
    body = request.json or {}
    data = _load_connections()
    for conn in data['connections']:
        if conn['id'] == conn_id:
            conn.update({k: v for k, v in body.items() if k != 'id' and not (k == 'password' and not v)})
            _wiki_clients.pop(conn_id, None)
            _save_connections(data)
            return jsonify({'success': True, 'connection': conn})
    return jsonify({'error': 'Not found'}), 404


@app.route('/api/connections/<conn_id>', methods=['DELETE'])
def delete_connection(conn_id: str):
    data = _load_connections()
    data['connections'] = [c for c in data['connections'] if c['id'] != conn_id]
    if data.get('active_connection_id') == conn_id:
        data['active_connection_id'] = data['connections'][0]['id'] if data['connections'] else None
    _wiki_clients.pop(conn_id, None)
    _save_connections(data)
    return jsonify({'success': True})


@app.route('/api/connections/<conn_id>/activate', methods=['POST'])
def activate_connection(conn_id: str):
    data = _load_connections()
    if not any(c['id'] == conn_id for c in data['connections']):
        return jsonify({'error': 'Not found'}), 404
    data['active_connection_id'] = conn_id
    _save_connections(data)
    return jsonify({'success': True})


@app.route('/api/connections/<conn_id>/test', methods=['POST'])
def test_connection(conn_id: str):
    _wiki_clients.pop(conn_id, None)
    client = get_wiki_client(conn_id)
    if client and client._connected:
        return jsonify({'connected': True})
    return jsonify({'connected': False, 'error': 'Authentication failed'})


# ─── LEGACY CHECK CONNECTION ──────────────────────────────────────────────────

@app.route('/api/check_connection')
def check_connection():
    conn = _get_active_connection()
    if not conn:
        return jsonify({'connected': False, 'error': 'No connections configured'})
    client = get_wiki_client(conn['id'])
    if client and client._connected:
        return jsonify({'connected': True, 'wiki_url': conn['wiki_url'], 'name': conn.get('name', '')})
    return jsonify({'connected': False, 'error': 'Wiki connection failed'})


# ─── LEGACY PUBLISH ───────────────────────────────────────────────────────────

@app.route('/api/publish', methods=['POST'])
def publish():
    body = request.json or {}
    conn = _resolve_connection(body)
    if not conn:
        return jsonify({'error': 'No wiki connection configured'}), 400
    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500
    result = client.write_page(body['title'], body['content'], body.get('summary', ''))
    return jsonify(result)


# ─── WIKI READ ROUTES ─────────────────────────────────────────────────────────

@app.route('/api/wiki/page')
def wiki_page():
    conn_id = request.args.get('connection_id')
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({'error': 'title required'}), 400
    conn = _get_connection_by_id(conn_id) if conn_id else _get_active_connection()
    if not conn:
        return jsonify({'error': 'No connection'}), 400
    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500
    return jsonify(client.get_page(title))


@app.route('/api/wiki/search')
def wiki_search():
    conn_id = request.args.get('connection_id')
    term = request.args.get('term', '').strip()
    if not term:
        return jsonify({'error': 'term required'}), 400
    conn = _get_connection_by_id(conn_id) if conn_id else _get_active_connection()
    if not conn:
        return jsonify({'error': 'No connection'}), 400
    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500
    return jsonify({'pages': client.search(term)})


@app.route('/api/wiki/all_pages')
def wiki_all_pages():
    conn_id = request.args.get('connection_id')
    conn = _get_connection_by_id(conn_id) if conn_id else _get_active_connection()
    if not conn:
        return jsonify({'error': 'No connection'}), 400
    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500
    return jsonify({'titles': client.get_all_pages()})


# ─── AGENT PLAN ROUTES ────────────────────────────────────────────────────────

@app.route('/api/agent/plan', methods=['POST'])
def agent_plan():
    body = request.json or {}
    instruction = body.get('instruction', '').strip()
    if not instruction:
        return jsonify({'error': 'instruction required'}), 400

    conn = _resolve_connection(body)
    if not conn:
        return jsonify({'error': 'No wiki connection configured'}), 400

    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500

    context_pages = body.get('context_pages', [])

    plan_id = str(uuid.uuid4())
    job_q: queue.Queue = queue.Queue()
    _job_queues[plan_id] = job_q

    cancel_ev = threading.Event()
    _cancel_events[plan_id] = cancel_ev

    placeholder = OperationPlan(
        id=plan_id,
        operation_type='auto',
        description='Planning in progress…',
        connection_id=conn['id'],
        status='running',
    )
    _plans[plan_id] = placeholder

    def _worker():
        try:
            site_index = client.get_pages_with_categories()
        except Exception:
            site_index = {}

        agent = WikiAgent(
            client, anthropic_client, conn.get('system_prompt', ''), conn['id'],
            site_index=site_index, context_pages=context_pages,
        )
        agent._stream_callback = lambda evt: job_q.put(evt)
        agent.cancel_event = cancel_ev

        try:
            plan = agent.generate_plan(instruction)
            plan.id = plan_id
            _plans[plan_id] = plan
            _save_plan_to_disk(plan)
        except Exception as e:
            job_q.put({'type': 'error', 'error': str(e)})

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({'plan_id': plan_id, 'status': 'running'})


@app.route('/api/agent/plan/stream/<plan_id>')
def agent_plan_stream(plan_id: str):
    def generate():
        job_q = _job_queues.get(plan_id)
        if not job_q:
            yield _sse({'type': 'error', 'error': 'Job not found'})
            return

        while True:
            try:
                event = job_q.get(timeout=60)
            except queue.Empty:
                yield ': keepalive\n\n'
                continue

            yield _sse(event)

            if event.get('type') in ('done', 'error'):
                _job_queues.pop(plan_id, None)
                plan = _plans.get(plan_id)
                if plan and event.get('type') == 'done':
                    yield _sse({
                        'type': 'plan_complete',
                        'plan_id': plan_id,
                        'description': plan.description,
                        'steps': [s.to_dict() for s in plan.steps],
                    })
                break

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    )


@app.route('/api/agent/plan/<plan_id>/cancel', methods=['POST'])
def cancel_plan(plan_id: str):
    if plan_id not in _cancel_events:
        _cancel_events[plan_id] = threading.Event()
    _cancel_events[plan_id].set()
    plan = _plans.get(plan_id)
    if plan:
        plan.status = 'failed'
        _save_plan_to_disk(plan)
    return jsonify({'success': True})


@app.route('/api/agent/plan/<plan_id>', methods=['GET'])
def get_plan(plan_id: str):
    plan = _plans.get(plan_id)
    if plan:
        return jsonify(plan.to_dict())
    loaded = _plan_from_disk(plan_id)
    if loaded:
        return jsonify(loaded.to_dict())
    return jsonify({'error': 'Plan not found'}), 404


def _load_archived_ids() -> set:
    if ARCHIVED_PLANS_FILE.exists():
        try:
            return set(json.loads(ARCHIVED_PLANS_FILE.read_text()))
        except Exception:
            pass
    return set()


def _save_archived_ids(ids: set):
    ARCHIVED_PLANS_FILE.write_text(json.dumps(sorted(ids), indent=2))


@app.route('/api/agent/plans', methods=['GET'])
def list_plans():
    show_archived = request.args.get('show_archived', '0') == '1'
    archived_ids = _load_archived_ids()
    plans = []
    if PLANS_DIR.exists():
        for path in PLANS_DIR.glob('*.json'):
            try:
                data = json.loads(path.read_text())
                plan_id = data.get('id', path.stem)
                is_archived = plan_id in archived_ids
                if not show_archived and is_archived:
                    continue
                plans.append({
                    'id': plan_id,
                    'operation_type': data.get('operation_type', ''),
                    'description': data.get('description', ''),
                    'status': data.get('status', 'pending'),
                    'step_count': len(data.get('steps', [])),
                    'created_at': data.get('created_at', ''),
                    'connection_id': data.get('connection_id', ''),
                    'archived': is_archived,
                })
            except Exception:
                continue
    plans.sort(key=lambda p: p.get('created_at') or '', reverse=True)
    return jsonify({'plans': plans})


@app.route('/api/agent/plans/<plan_id>/archive', methods=['POST'])
def archive_plan(plan_id: str):
    ids = _load_archived_ids()
    ids.add(plan_id)
    _save_archived_ids(ids)
    return jsonify({'success': True})


# ─── EXECUTE ROUTES ───────────────────────────────────────────────────────────

@app.route('/api/agent/step/preview', methods=['POST'])
def step_preview_route():
    body = request.json or {}
    plan_id = body.get('plan_id')
    step_id = body.get('step_id')

    plan = _plans.get(plan_id) or _plan_from_disk(plan_id)
    if not plan:
        return jsonify({'error': 'Plan not found'}), 404

    step = next((s for s in plan.steps if s.id == step_id), None)
    if not step:
        return jsonify({'error': 'Step not found'}), 404

    conn = _get_connection_by_id(plan.connection_id)
    if not conn:
        return jsonify({'error': 'Connection not found'}), 400

    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500

    try:
        site_index = client.get_pages_with_categories()
    except Exception:
        site_index = {}

    agent = WikiAgent(
        client, anthropic_client, conn.get('system_prompt', ''), conn['id'],
        site_index=site_index,
    )
    try:
        agent.generate_step_preview(step)
        _plans[plan_id] = plan
        _save_plan_to_disk(plan)
        return jsonify({'success': True, 'step': step.to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'step': step.to_dict()})


@app.route('/api/agent/execute_step', methods=['POST'])
def execute_step_route():
    body = request.json or {}
    plan_id = body.get('plan_id')
    step_id = body.get('step_id')

    plan = _plans.get(plan_id) or _plan_from_disk(plan_id)
    if not plan:
        return jsonify({'error': 'Plan not found'}), 404

    step = next((s for s in plan.steps if s.id == step_id), None)
    if not step:
        return jsonify({'error': 'Step not found'}), 404

    conn = _get_connection_by_id(plan.connection_id)
    if not conn:
        return jsonify({'error': 'Connection not found'}), 400

    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500

    step.status = 'approved'
    agent = WikiAgent(client, anthropic_client, conn.get('system_prompt', ''), conn['id'])
    result = agent.execute_step(step)
    _plans[plan_id] = plan
    _save_plan_to_disk(plan)
    return jsonify({'success': result.get('success', False), 'step': step.to_dict(), **result})


@app.route('/api/agent/execute_plan', methods=['POST'])
def execute_plan_route():
    body = request.json or {}
    plan_id = body.get('plan_id')
    approved_ids = set(body.get('approved_step_ids', []))

    plan = _plans.get(plan_id) or _plan_from_disk(plan_id)
    if not plan:
        return jsonify({'error': 'Plan not found'}), 404

    conn = _get_connection_by_id(plan.connection_id)
    if not conn:
        return jsonify({'error': 'Connection not found'}), 400

    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500

    # Mark approved steps
    for step in plan.steps:
        if step.id in approved_ids:
            step.status = 'approved'

    # Apply user-edited content overrides (pre-generated or edited in the side editor)
    step_contents = body.get('step_contents', {})
    for step in plan.steps:
        if step.id in step_contents:
            step.content = step_contents[step.id]

    _plans[plan_id] = plan

    exec_q: queue.Queue = queue.Queue()
    _exec_queues[plan_id] = exec_q

    cancel_ev = _cancel_events.get(plan_id, threading.Event())
    _cancel_events[plan_id] = cancel_ev

    def _exec_worker():
        try:
            site_index = client.get_pages_with_categories()
        except Exception:
            site_index = {}

        agent = WikiAgent(
            client, anthropic_client, conn.get('system_prompt', ''), conn['id'],
            site_index=site_index,
        )
        agent._stream_callback = lambda evt: exec_q.put(evt)
        agent.cancel_event = cancel_ev

        results = agent.execute_plan(plan)
        _save_plan_to_disk(plan)
        _append_history(plan, results)

    threading.Thread(target=_exec_worker, daemon=True).start()
    return jsonify({'plan_id': plan_id, 'status': 'running'})


@app.route('/api/agent/execute/stream/<plan_id>')
def agent_execute_stream(plan_id: str):
    def generate():
        exec_q = _exec_queues.get(plan_id)
        if not exec_q:
            yield _sse({'type': 'error', 'error': 'Execution not found'})
            return

        while True:
            try:
                event = exec_q.get(timeout=60)
            except queue.Empty:
                yield ': keepalive\n\n'
                continue

            yield _sse(event)

            if event.get('type') in ('done', 'error'):
                _exec_queues.pop(plan_id, None)
                plan = _plans.get(plan_id)
                if plan and event.get('type') == 'done':
                    done_count = sum(1 for s in plan.steps if s.status == 'done')
                    yield _sse({
                        'type': 'execute_complete',
                        'plan_id': plan_id,
                        'plan_status': plan.status,
                        'steps': [s.to_dict() for s in plan.steps],
                        'success_count': done_count,
                    })
                break

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    )


@app.route('/api/history')
def get_history():
    if not HISTORY_FILE.exists():
        return jsonify({'history': []})
    return jsonify({'history': json.loads(HISTORY_FILE.read_text())})


# ─── MAIN ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5055))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)
