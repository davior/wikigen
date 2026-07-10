"""Primary image discovery via imagescout (DuckDuckGo engine only).

Wraps imagescout.ImageSearcher behind one fail-soft function. This is
deliberately not a generic multi-provider abstraction — it only replaces
what the old Bing-backed search used to do (raw, not-yet-uploaded search
hits). The Wikipedia/Commons fallback returns a different kind of result
(already Commons-hosted, no download/upload needed) and stays where it is
in wiki_client.py / agent.py.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from imagescout import ImageSearcher, SearchConfig, ImageResult

logger = logging.getLogger(__name__)

# Hardcoded per product decision: DuckDuckGo only (no Playwright/browser
# dependency), default license_filter="any". If ops-level tuning is ever
# needed, this is the one place to add env-var overrides.
_ENGINES = ["duckduckgo"]

_searcher: Optional[ImageSearcher] = None
_searcher_lock = threading.Lock()

# imagescout's per-engine "politeness" delays are calibrated for a single
# sequential caller. wikigen runs as one process but can execute several
# plan steps concurrently (ThreadPoolExecutor), so every search is
# serialized process-wide through this lock — concurrent steps queue up
# and run their image searches one at a time instead of bursting requests
# at DuckDuckGo simultaneously.
_search_lock = threading.Lock()


def _get_searcher() -> ImageSearcher:
    """Lazily build and cache a process-wide ImageSearcher."""
    global _searcher
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                _searcher = ImageSearcher(engines=_ENGINES, config=SearchConfig())
    return _searcher


def _shorten_queries(query: str) -> list[str]:
    """Return a list of progressively shorter variants of a search query.

    Narrow/long queries often have no indexed images; dropping trailing
    words often finds results when the exact phrase doesn't.
    e.g. "DARPA N3 Programme" -> ["DARPA N3 Programme", "DARPA N3", "DARPA"]
    Always includes at least the original query.
    """
    words = query.split()
    queries = [query]
    while len(words) > 1:
        words = words[:-1]
        shorter = ' '.join(words)
        if shorter not in queries:
            queries.append(shorter)
    return queries


def search_images(query: str, *, limit: int = 5, shorten: bool = True) -> list[ImageResult]:
    """Search imagescout (DuckDuckGo) for `query`.

    Retries with progressively shorter query variants until one returns
    results or all variants are exhausted. Returns [] if nothing is found
    or anything goes wrong — this never raises, so callers need no
    try/except around it.
    """
    try:
        searcher = _get_searcher()
    except Exception:
        logger.exception('imagescout: failed to initialize ImageSearcher')
        return []

    with _search_lock:
        for attempt_query in (_shorten_queries(query) if shorten else [query]):
            try:
                outcome = searcher.search(attempt_query, max_results=limit, per_engine=limit)
            except Exception:
                logger.exception('imagescout: search() raised for query %r', attempt_query)
                continue
            if outcome.ok:
                return outcome.results
            if outcome.failures:
                logger.debug('imagescout: no results for %r (failures: %s)',
                              attempt_query, [(f.engine, f.error) for f in outcome.failures])
    return []
