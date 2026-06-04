"""Persistent, cached site index for WikiGen.

The planner needs to know every page that exists in the wiki (and ideally a
little about each one) on every operation. Rebuilding that from the MediaWiki
``allpages`` API on each call costs ~N/50 sequential round-trips. This module
maintains the index once and reuses it:

  * an in-process cache per connection,
  * a single ``recentchanges`` high-water-mark check to detect staleness cheaply,
  * incremental refresh of only the changed pages (full rebuild as a fallback),
  * persistence to a JSON page in the wiki (default ``User:<bot>/wikigen-index.json``)
    so the index survives restarts and can be read in one call.

The index is also enriched over time: when WikiGen creates or edits a page it
records a one-line description and the page's categories via
``apply_local_changes`` — giving the planner deeper whole-site context without an
expensive content scan.

Each cache entry has the shape::

    {'pages': {title: {'c': [categories], 'd': 'description', 't': ''}},
     'latest_rcid': int,
     'rc_checked_at': float}

``get_index`` returns the ``pages`` mapping; ``WikiAgent`` reads its keys as the
set of existing titles and passes it to ``_format_site_index``.
"""

import json
import re
import threading
import time
from datetime import datetime, timezone

SCHEMA_VERSION = 1
# Minimum seconds between recentchanges head checks for a warm cache. Within this
# window a cached index is served without touching the wiki at all.
RC_CHECK_THROTTLE = 30.0

_cache: dict[str, dict] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

_CAT_RE = re.compile(r'\[\[Category:([^|\]]+)', re.IGNORECASE)


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
    # lands mid-rebuild is caught on the next freshness check rather than lost.
    head = _safe_rc_head(client) or 0
    raw = client.get_pages_with_categories()  # {title: [categories]}
    pages = {title: {'c': cats, 'd': '', 't': ''} for title, cats in raw.items()}
    return {'pages': pages, 'latest_rcid': head}


def _apply_delta(client, entry: dict, head: int) -> None:
    """Refresh only the pages that changed since entry['latest_rcid']."""
    try:
        changed = client.get_changes_since(entry['latest_rcid'])
    except Exception:
        changed = None
    if changed is None:  # gap too large (or error) — rebuild from scratch
        rebuilt = _full_rebuild(client)
        entry['pages'] = rebuilt['pages']
        entry['latest_rcid'] = rebuilt['latest_rcid']
        return
    if changed:
        try:
            cats = client.get_categories_for_titles(changed)
        except Exception:
            cats = {}
        pages = entry['pages']
        for title in changed:
            if title not in cats:
                continue
            new_cats = cats[title]
            if new_cats is None:  # deleted or moved away
                pages.pop(title, None)
                continue
            prev = pages.get(title) or {}
            if isinstance(prev, list):
                prev = {'c': prev}
            pages[title] = {'c': new_cats, 'd': prev.get('d', ''), 't': prev.get('t', '')}
    entry['latest_rcid'] = head


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
    return {'pages': data['pages'], 'latest_rcid': data.get('latest_rcid', 0)}


def _write_page(client, title: str, entry: dict) -> None:
    payload = {
        'version': SCHEMA_VERSION,
        'generated': datetime.now(timezone.utc).isoformat(),
        'latest_rcid': entry['latest_rcid'],
        'page_count': len(entry['pages']),
        'pages': entry['pages'],
    }
    try:
        client.write_page(
            title,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            'WikiGen site index update',
        )
    except Exception:
        pass


def get_index(client, connection_id: str, page_title: str | None = None) -> dict:
    """Return {title: {'c': [...], 'd': '...'}} for all main-namespace pages.

    Serves a warm in-memory cache when possible, falls back to the persisted
    wiki page, and only does a full ``allpages`` scan when neither is usable.
    """
    title = page_title or _default_page_title(client)
    with _lock_for(connection_id):
        now = time.time()
        entry = _cache.get(connection_id)

        if entry is not None:
            if now - entry.get('rc_checked_at', 0) < RC_CHECK_THROTTLE:
                return dict(entry['pages'])
            head = _safe_rc_head(client)
            entry['rc_checked_at'] = now
            if head is None or head == entry['latest_rcid']:
                return dict(entry['pages'])
            if entry['latest_rcid']:
                _apply_delta(client, entry, head)
                _write_page(client, title, entry)
                return dict(entry['pages'])
            # No stored rcid to delta from — fall through to rebuild.

        # Cold cache: try the persisted page before paying for a full scan.
        page_entry = _read_page(client, title)
        if page_entry is not None:
            head = _safe_rc_head(client)
            page_entry['rc_checked_at'] = now
            if head is None or head == page_entry['latest_rcid']:
                _cache[connection_id] = page_entry
                return dict(page_entry['pages'])
            if page_entry['latest_rcid']:
                _apply_delta(client, page_entry, head)
                _cache[connection_id] = page_entry
                _write_page(client, title, page_entry)
                return dict(page_entry['pages'])

        entry = _full_rebuild(client)
        entry['rc_checked_at'] = now
        _cache[connection_id] = entry
        _write_page(client, title, entry)
        return dict(entry['pages'])


def apply_local_changes(connection_id: str, client, page_title: str | None,
                        changes: dict) -> None:
    """Update the cached/persisted index from WikiGen's own completed steps.

    ``changes`` = {'created': [{title, content}], 'edited': [{title, content}],
    'deleted': [title], 'moved': [{from, to}]}. This keeps the index current
    immediately after execution without waiting for the next recentchanges check.
    """
    with _lock_for(connection_id):
        entry = _cache.get(connection_id)
        if entry is None:
            return  # nothing cached yet; the next get_index() will build it fresh
        pages = entry['pages']

        for title in changes.get('deleted', []):
            pages.pop(title, None)

        for mv in changes.get('moved', []):
            dst = mv.get('to')
            if not dst:
                continue
            prev = pages.pop(mv.get('from'), None) if mv.get('from') else None
            if isinstance(prev, list):
                prev = {'c': prev}
            prev = prev or {}
            pages[dst] = {'c': prev.get('c', []), 'd': prev.get('d', ''), 't': ''}

        for item in list(changes.get('created', [])) + list(changes.get('edited', [])):
            title = item.get('title')
            if not title:
                continue
            desc, cats = _summarize(item.get('content') or '')
            prev = pages.get(title)
            if isinstance(prev, list):
                prev = {'c': prev}
            prev = prev or {}
            pages[title] = {
                'c': cats or prev.get('c', []),
                'd': desc or prev.get('d', ''),
                't': '',
            }

        head = _safe_rc_head(client)
        if head is not None:
            entry['latest_rcid'] = head
        entry['rc_checked_at'] = time.time()
        _write_page(client, page_title or _default_page_title(client), entry)
