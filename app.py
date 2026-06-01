import io
import json
import mimetypes
import os
import queue
import re
import threading
import uuid
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS

from agent import OperationPlan, OperationStep, WikiAgent, _make_diff
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
UPLOADS_DIR = DATA_DIR / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)

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
            images=s.get('images', []),
            source_url=s.get('source_url'),
            upload_id=s.get('upload_id'),
        ))
    return plan


def _attach_uploads_to_plan(plan: OperationPlan, context_pages: list):
    """Link upload_file steps to the actual uploaded file bytes.

    Each context document with an upload_id is matched to the corresponding
    upload_file step. Matching tries exact filename, then stem similarity, then
    falls back to the sole doc if there's only one unmatched. Any source_url the
    planner may have grabbed from inside a document's text is cleared — the
    attached file always takes precedence.
    """
    docs = [p for p in (context_pages or []) if p.get('upload_id')]
    if not docs:
        return
    unmatched_docs = list(docs)
    for step in plan.steps:
        if step.type != 'upload_file':
            continue
        step_fname = step.title.removeprefix('File:').strip()
        step_stem = Path(step_fname).stem.lower().replace('-', ' ').replace('_', ' ')

        # 1. Exact filename match (case-insensitive)
        match = next((d for d in unmatched_docs
                      if d.get('filename', '').lower() == step_fname.lower()), None)
        # 2. Stem similarity: "report" matches "report.pdf"
        if not match:
            match = next((d for d in unmatched_docs
                          if Path(d.get('filename', '')).stem.lower().replace('-', ' ').replace('_', ' ')
                          == step_stem), None)
        # 3. Only one remaining unmatched doc — use it
        if not match and len(unmatched_docs) == 1:
            match = unmatched_docs[0]

        if match:
            step.upload_id = match['upload_id']
            step.source_url = None
            unmatched_docs = [d for d in unmatched_docs if d is not match]


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


@app.route('/api/wiki/rewrite', methods=['POST'])
def wiki_rewrite():
    body = request.json or {}
    title = body.get('title', '')
    content = body.get('content', '')
    instruction = body.get('instruction', '').strip()
    if not instruction:
        return jsonify({'error': 'instruction required'}), 400
    use_plan_scope = body.get('use_plan_scope', False)
    plan_id = body.get('plan_id')

    conn = _resolve_connection(body)
    if not conn:
        return jsonify({'error': 'No wiki connection configured'}), 400
    client = get_wiki_client(conn['id'])
    if not client:
        return jsonify({'error': 'Wiki connection failed'}), 500

    if use_plan_scope and plan_id and plan_id in _plans:
        plan = _plans[plan_id]
        site_index = {step.title: [] for step in plan.steps}
    else:
        try:
            site_index = client.get_pages_with_categories()
        except Exception:
            site_index = {}

    agent = WikiAgent(client, anthropic_client, conn.get('system_prompt', ''), conn['id'], site_index=site_index)
    new_content = agent._edit_page_content(title, content, instruction)
    diff = _make_diff(content, new_content)
    return jsonify({'content': new_content, 'diff': diff})


# ─── DOCUMENT UPLOAD & URL FETCH ─────────────────────────────────────────────

def _extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in ('.txt', '.md'):
        return file_bytes.decode('utf-8', errors='replace')
    if ext in ('.html', '.htm'):
        from bs4 import BeautifulSoup
        return BeautifulSoup(file_bytes, 'lxml').get_text(separator='\n')
    if ext == '.pdf':
        import pymupdf
        doc = pymupdf.open(stream=file_bytes, filetype='pdf')
        return '\n\n'.join(page.get_text() for page in doc)
    if ext == '.docx':
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError(f'Unsupported file type: {ext}')


def _fetch_url_text(url: str, timeout: int = 10, max_bytes: int = 1_048_576) -> tuple[str, str, bytes, str]:
    """Fetch URL and return (title, text, raw_bytes, content_type). Raises on error."""
    import requests as req_lib
    resp = req_lib.get(url, timeout=timeout, stream=True,
                       headers={'User-Agent': 'WikiGen/3.0'})
    resp.raise_for_status()
    raw = b''
    for chunk in resp.iter_content(65536):
        raw += chunk
        if len(raw) >= max_bytes:
            raw = raw[:max_bytes]
            break
    content_type = resp.headers.get('Content-Type', '')
    if 'html' in content_type:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, 'lxml')
        title_tag = soup.find('title')
        page_title = title_tag.get_text(strip=True) if title_tag else url
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose()
        content_el = soup.find('main') or soup.find('article') or soup.find('body') or soup
        text = content_el.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
    elif 'pdf' in content_type:
        import pymupdf
        doc = pymupdf.open(stream=raw, filetype='pdf')
        text = '\n\n'.join(page.get_text() for page in doc)
        page_title = url.rsplit('/', 1)[-1] or url
    else:
        text = raw.decode('utf-8', errors='replace')
        page_title = url.rsplit('/', 1)[-1] or url
    return page_title, text, raw, content_type


@app.route('/api/upload_document', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    filename = f.filename or 'document'
    allowed_exts = {'.txt', '.md', '.html', '.htm', '.pdf', '.docx'}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_exts:
        return jsonify({'error': f'Unsupported file type: {ext}. Allowed: {", ".join(sorted(allowed_exts))}'}), 400
    file_bytes = f.read()
    if len(file_bytes) > 10_485_760:
        return jsonify({'error': 'File too large (max 10 MB)'}), 400
    try:
        text = _extract_text_from_file(filename, file_bytes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    text = text.strip()[:50000]
    title = Path(filename).stem.replace('-', ' ').replace('_', ' ')
    # Persist the original bytes so the file itself (not just its text) can be
    # uploaded to the wiki later if an upload_file step references it.
    upload_id = uuid.uuid4().hex
    (UPLOADS_DIR / upload_id).write_bytes(file_bytes)
    return jsonify({'title': title, 'content': text, 'filename': filename,
                    'upload_id': upload_id, 'is_document': True})


@app.route('/api/fetch_url', methods=['POST'])
def fetch_url():
    body = request.json or {}
    url = body.get('url', '').strip()
    if not url:
        return jsonify({'error': 'url required'}), 400
    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'URL must start with http:// or https://'}), 400
    try:
        title, text, raw_bytes, content_type = _fetch_url_text(url)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch URL: {e}'}), 500
    text = text.strip()[:50000]
    # Derive a filename from the URL path for wiki upload purposes
    url_filename = url.rsplit('/', 1)[-1].split('?')[0] or 'document'
    if not Path(url_filename).suffix:
        url_filename += '.html'
    # Save raw bytes so the file can be wiki-uploaded later (e.g. PDF from URL)
    upload_id = uuid.uuid4().hex
    (UPLOADS_DIR / upload_id).write_bytes(raw_bytes)
    return jsonify({'title': title, 'content': text, 'source_url': url,
                    'filename': url_filename, 'upload_id': upload_id, 'is_document': True})


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
            _attach_uploads_to_plan(plan, context_pages)
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
    agent = WikiAgent(client, anthropic_client, conn.get('system_prompt', ''), conn['id'],
                      uploads_dir=UPLOADS_DIR)
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

    cancel_ev = threading.Event()  # always fresh — clears any prior cancel
    _cancel_events[plan_id] = cancel_ev

    def _exec_worker():
        try:
            site_index = client.get_pages_with_categories()
        except Exception:
            site_index = {}

        agent = WikiAgent(
            client, anthropic_client, conn.get('system_prompt', ''), conn['id'],
            site_index=site_index, uploads_dir=UPLOADS_DIR,
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
