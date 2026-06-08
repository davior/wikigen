"""Frozen, connection-scoped site index for WikiGen.

The planner needs to know every page that exists in the wiki (and ideally a
little about each one) on every operation. Rebuilding that from the MediaWiki
``allpages`` API on each call costs ~N/50 sequential round-trips — and, more
importantly, every change to the index text *invalidates the Anthropic prompt
cache* for the index block and everything after it.

To keep that cache warm across a long content-generation session, the index is
**frozen**: it is built once (auto-populated on first use), stored on the
connection, and reused byte-for-byte on every operation. It only changes when
the user explicitly refreshes it. Pages WikiGen creates mid-session are NOT
folded into the frozen index; instead they are tracked separately as a small
"pending" list that the planner receives in the uncached tail of its prompt
(see ``get_recent_pages``) — so the model still knows they exist without
disturbing the cached block.

Storage per connection:
  * an in-process cache,
  * a local sidecar file (``site_index_<id>.json`` in the storage dir) so the
    index survives restarts without touching the wiki,
  * an optional mirror to a JSON page in the wiki (default
    ``User:<bot>/wikigen-index.json``) written on (re)build, so the index is
    also visible inside the wiki.

Each cache entry has the shape::

    {'pages': {title: {'c': [categories], 'd': 'description', 't': ''}},
     'pending': {title: {'c': [...], 'd': '...'}},
     'latest_rcid': int,
     'generated': isoformat str}

``get_index`` returns the frozen ``pages`` mapping; ``get_recent_pages`` returns
the ``pending`` mapping.
"""

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

_cache: dict[str, dict] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_storage_dir: Path | None = None

_CAT_RE = re.compile(r'\[\[Category:([^|\]]+)', re.IGNORECASE)
_IMG_RE = re.compile(r'\[\[(?:File|Image):', re.IGNORECASE)


def set_storage_dir(path) -> None:
    """Configure where per-connection sidecar files are stored."""
    global _storage_dir
    _storage_dir = Path(path)


def _lock_for(connection_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(connection_id)
        if lock is None:
            lock = threading.Lock()
            _locks[connection_id] = lock
        return lock


def _default_page_title(client) -> str:
    user = (getattr(client, '_username', '') or '').split('@')[0].strip()
    return f'User:{user}/wikigen-index.json' if user else 'User:WikiGen/wikigen-index.json'


def _summarize(content: str) -> tuple[str, list[str]]:
    """Extract a one-line description and category list from page wikitext."""
    if not content:
        return '', []
    cats = [c.strip() for c in _CAT_RE.findall(content)]
    desc = ''
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(('=', '{', '#', '*', '|', '!', '[[Category', '[[File', '[[Image')):
            continue
        line = re.sub(r"\[\[[^|\]]*\|", '', line)       # piped link -> label
        line = re.sub(r"'''?|\[\[|\]\]", '', line)       # bold/italic and bare links
        line = line.strip()
        if line:
            desc = line
            break
    if desc:
        m = re.search(r'^(.+?[.!?])(\s|$)', desc)
        desc = (m.group(1) if m else desc).strip()[:200]
    return desc, cats


def _safe_rc_head(client) -> int | None:
    try:
        return client.get_recentchanges_head()
    except Exception:
        return None


def _full_rebuild(client) -> dict:
    # Take the high-water mark *before* the (multi-call) scan so any edit that
    # lands mid-rebuild shows up as "stale" on the next status check rather than
    # being silently assumed present.
    head = _safe_rc_head(client) or 0
    raw = client.get_pages_with_categories()  # {title: {'cats': [...], 'has_image': bool}}
    pages = {
        title: {'c': info['cats'], 'd': '', 't': '', 'i': info['has_image']}
        for title, info in raw.items()
    }
    return {
        'pages': pages,
        'pending': {},
        'latest_rcid': head,
        'generated': datetime.now(timezone.utc).isoformat(),
    }


# ── persistence: local sidecar ────────────────────────────────────────────────

def _sidecar_path(connection_id: str) -> Path | None:
    if _storage_dir is None:
        return None
    return _storage_dir / f'site_index_{connection_id}.json'


def _read_sidecar(connection_id: str) -> dict | None:
    path = _sidecar_path(connection_id)
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if data.get('version') != SCHEMA_VERSION or not isinstance(data.get('pages'), dict):
        return None
    return {
        'pages': data['pages'],
        'pending': data.get('pending') or {},
        'latest_rcid': data.get('latest_rcid', 0),
        'generated': data.get('generated', ''),
    }


def _write_sidecar(connection_id: str, entry: dict) -> None:
    path = _sidecar_path(connection_id)
    if not path:
        return
    payload = {
        'version': SCHEMA_VERSION,
        'generated': entry.get('generated', ''),
        'latest_rcid': entry.get('latest_rcid', 0),
        'page_count': len(entry['pages']),
        'pages': entry['pages'],
        'pending': entry.get('pending', {}),
    }
    try:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        tmp.replace(path)
    except Exception:
        pass


# ── persistence: optional wiki-page mirror ────────────────────────────────────

def _read_page(client, title: str) -> dict | None:
    try:
        page = client.get_page(title)
    except Exception:
        return None
    if not page.get('exists') or not page.get('content'):
        return None
    try:
        data = json.loads(page['content'])
    except Exception:
        return None
    if data.get('version') != SCHEMA_VERSION or not isinstance(data.get('pages'), dict):
        return None
    return {
        'pages': data['pages'],
        'pending': {},
        'latest_rcid': data.get('latest_rcid', 0),
        'generated': data.get('generated', ''),
    }


def _write_page(client, title: str, entry: dict) -> None:
    payload = {
        'version': SCHEMA_VERSION,
        'generated': entry.get('generated', ''),
        'latest_rcid': entry.get('latest_rcid', 0),
        'page_count': len(entry['pages']),
        'pages': entry['pages'],
    }
    try:
        client.write_page(
            title,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            'WikiGen site index refresh',
        )
    except Exception:
        pass


def _load_entry(client, connection_id: str) -> dict | None:
    """Return the cached entry, hydrating from sidecar then wiki page if needed."""
    entry = _cache.get(connection_id)
    if entry is not None:
        return entry
    entry = _read_sidecar(connection_id)
    if entry is not None:
        _cache[connection_id] = entry
        return entry
    title = _default_page_title(client)
    entry = _read_page(client, title)
    if entry is not None:
        _cache[connection_id] = entry
        _write_sidecar(connection_id, entry)  # cache it locally for next time
        return entry
    return None


# ── public API ────────────────────────────────────────────────────────────────

def get_index(client, connection_id: str, page_title: str | None = None) -> dict:
    """Return the FROZEN {title: {'c': [...], 'd': '...'}} index.

    Served from cache/sidecar/wiki without ever auto-refreshing — so the planner's
    cached prompt block stays byte-identical between explicit refreshes. Only the
    very first use (cold, with nothing stored anywhere) pays for a full scan to
    auto-populate the index.
    """
    title = page_title or _default_page_title(client)
    with _lock_for(connection_id):
        entry = _load_entry(client, connection_id)
        if entry is None:
            # First run: auto-populate once, then freeze.
            entry = _full_rebuild(client)
            _cache[connection_id] = entry
            _write_sidecar(connection_id, entry)
            _write_page(client, title, entry)
        return dict(entry['pages'])


def get_recent_pages(connection_id: str) -> dict:
    """Return pages created/updated this session that aren't in the frozen index.

    Shape ``{title: {'c': [...], 'd': '...'}}``. Fed to the planner in the
    uncached tail of its prompt so it knows these pages exist (and won't recreate
    them) without invalidating the cached site-index block.
    """
    with _lock_for(connection_id):
        entry = _cache.get(connection_id)
        return dict(entry['pending']) if entry else {}


def record_session_changes(connection_id: str, changes: dict) -> None:
    """Record WikiGen's own completed steps as pending, WITHOUT touching the
    frozen index (keeps the prompt cache warm).

    ``changes`` = {'created': [{title, content}], 'edited': [{title, content}],
    'deleted': [title], 'moved': [{from, to}]}. Created/edited pages are added to
    the pending list (which also accrues a one-line description); deleted/moved-
    away titles are dropped from pending. The frozen ``pages`` mapping is only
    reconciled on an explicit ``refresh_index``.
    """
    with _lock_for(connection_id):
        entry = _cache.get(connection_id)
        if entry is None:
            return  # index not built yet; next get_index() builds it fresh
        pending = entry.setdefault('pending', {})

        for title in changes.get('deleted', []):
            pending.pop(title, None)
        for mv in changes.get('moved', []):
            if mv.get('from'):
                pending.pop(mv['from'], None)

        considered = list(changes.get('created', [])) + list(changes.get('edited', []))
        for mv in changes.get('moved', []):
            if mv.get('to'):
                considered.append({'title': mv['to'], 'content': ''})
        for item in considered:
            title = item.get('title')
            if not title:
                continue
            content = item.get('content') or ''
            desc, cats = _summarize(content)
            has_image = bool(_IMG_RE.search(content)) if content else False
            prev = pending.get(title) or {}
            pending[title] = {
                'c': cats or prev.get('c', []),
                'd': desc or prev.get('d', ''),
                'i': has_image if content else prev.get('i', False),
            }

        _write_sidecar(connection_id, entry)


def refresh_index(client, connection_id: str, page_title: str | None = None) -> dict:
    """Force a full rebuild, clear the pending list, and persist everywhere.

    This is the ONLY path (besides first-run auto-populate) that changes the
    frozen index. Returns ``{'page_count': int}``.
    """
    title = page_title or _default_page_title(client)
    with _lock_for(connection_id):
        entry = _full_rebuild(client)
        _cache[connection_id] = entry
        _write_sidecar(connection_id, entry)
        _write_page(client, title, entry)
        return {'page_count': len(entry['pages'])}


def index_status(client, connection_id: str, page_title: str | None = None) -> dict:
    """Cheap, cache-safe staleness hint for the UI (one recentchanges call).

    Never mutates the frozen index. Returns ``{exists, page_count, pending_count,
    generated, stale}`` where ``stale`` means the wiki has changed since the index
    was last (re)built.
    """
    with _lock_for(connection_id):
        entry = _load_entry(client, connection_id)
        if entry is None:
            return {'exists': False}
        head = _safe_rc_head(client)
        stale = head is not None and head != entry.get('latest_rcid', 0)
        return {
            'exists': True,
            'page_count': len(entry['pages']),
            'pending_count': len(entry.get('pending', {})),
            'generated': entry.get('generated', ''),
            'stale': stale,
        }
