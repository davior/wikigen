import json
import logging
import re
import time
import requests

log = logging.getLogger(__name__)


def _short_reason(exc: Exception) -> str:
    """Pull the innermost cause out of a requests/urllib3 exception chain."""
    msg = str(exc)
    for marker in ('Failed to resolve', 'Name or service not known', 'Temporary failure in name resolution',
                   'Connection refused', 'No route to host', 'Network is unreachable',
                   'CERTIFICATE_VERIFY_FAILED'):
        if marker in msg:
            return marker
    return msg[:200]


class _TimeoutSession(requests.Session):
    """Session that applies a default timeout, so a wiki that stops answering
    raises instead of hanging the calling thread forever."""

    def __init__(self, timeout: float = 60):
        super().__init__()
        self._default_timeout = timeout

    def request(self, *args, **kwargs):
        kwargs.setdefault('timeout', self._default_timeout)
        return super().request(*args, **kwargs)


class WikiClient:
    def __init__(self, wiki_url: str, username: str, password: str, api_token: str | None = None):
        self._url = wiki_url
        self._username = username
        self._password = password
        self._session = _TimeoutSession()
        self._session.headers.update({
            'User-Agent': 'WikiGen/3.0 (wiki management bot; https://github.com/davior/wikigen)',
        })
        if api_token:
            self._session.headers.update({'X-Wikigen-Token': api_token})
        self._csrf_token = None
        self._connected = False
        self._last_write_time = 0.0
        self.last_error: str | None = None

    def connect(self) -> bool:
        self.last_error = self._login()
        self._connected = self.last_error is None
        if self.last_error:
            log.warning('Wiki connection to %s failed: %s', self._url, self.last_error)
        return self._connected

    def _login(self) -> str | None:
        """Log in and fetch a CSRF token. Returns None on success, else a readable reason."""
        try:
            r = self._session.get(self._url, params={
                'action': 'query', 'meta': 'tokens', 'type': 'login', 'format': 'json'
            }, timeout=15)
            r.raise_for_status()
            login_token = r.json()['query']['tokens']['logintoken']

            r2 = self._session.post(self._url, data={
                'action': 'login',
                'lgname': self._username,
                'lgpassword': self._password,
                'lgtoken': login_token,
                'format': 'json',
            }, timeout=15)
            r2.raise_for_status()
            login = r2.json()['login']
            if login.get('result') != 'Success':
                reason = login.get('reason')
                if isinstance(reason, dict):
                    reason = reason.get('text') or reason.get('code')
                return f"Login failed: {login.get('result')}" + (f' — {reason}' if reason else '')

            self._csrf_token = self._fetch_csrf()
            if not self._csrf_token:
                return 'Logged in but could not fetch CSRF token'
            return None
        except requests.exceptions.Timeout:
            return f'Timed out reaching {self._url}'
        except requests.exceptions.ConnectionError as e:
            return f'Cannot reach {self._url}: {_short_reason(e)}'
        except requests.exceptions.HTTPError as e:
            return f'HTTP {e.response.status_code} from {self._url}'
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return f'Unexpected response from {self._url} (not a MediaWiki api.php?)'
        except requests.exceptions.RequestException as e:
            return f'Request to {self._url} failed: {_short_reason(e)}'

    def _fetch_csrf(self) -> str | None:
        try:
            r = self._session.get(self._url, params={
                'action': 'query', 'meta': 'tokens', 'format': 'json'
            }, timeout=15)
            r.raise_for_status()
            return r.json()['query']['tokens']['csrftoken']
        except Exception:
            return None

    def _ensure_csrf(self):
        if not self._csrf_token:
            self._csrf_token = self._fetch_csrf()

    def _ensure_connected(self):
        """Re-authenticate if the session has expired."""
        if not self._connected:
            self.connect()

    def _rate_limit(self):
        elapsed = time.time() - self._last_write_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_write_time = time.time()

    def get_page(self, title: str) -> dict:
        self._ensure_connected()
        r = self._session.get(self._url, params={
            'action': 'query',
            'titles': title,
            'prop': 'revisions|info',
            'rvprop': 'content|timestamp',
            'inprop': 'url',
            'format': 'json',
        })
        r.raise_for_status()
        data = r.json()
        pages = data.get('query', {}).get('pages') or {}
        page = next(iter(pages.values()), {})
        exists = page.get('pageid', -1) != -1 and str(page.get('pageid', -1)) != '-1'
        content = None
        last_modified = None
        if exists and 'revisions' in page:
            rev = page['revisions'][0]
            content = rev.get('*') or rev.get('content')
            last_modified = rev.get('timestamp')

        # Build view URL from api URL
        view_url = self._url.replace('api.php', f'index.php?title={requests.utils.quote(title, safe="")}')

        return {
            'title': title,
            'content': content,
            'exists': exists,
            'url': view_url,
            'last_modified': last_modified,
        }

    def page_exists(self, title: str) -> bool:
        self._ensure_connected()
        r = self._session.get(self._url, params={
            'action': 'query', 'titles': title, 'format': 'json'
        })
        r.raise_for_status()
        pages = r.json().get('query', {}).get('pages') or {}
        page = next(iter(pages.values()), {})
        page_id = str(page.get('pageid', -1))
        return page_id != '-1' and int(page_id) > 0

    def get_all_pages(self) -> list[str]:
        self._ensure_connected()
        titles = []
        params = {
            'action': 'query', 'list': 'allpages',
            'aplimit': 500, 'format': 'json'
        }
        while True:
            r = self._session.get(self._url, params=params)
            if r.status_code == 403:
                raise PermissionError(
                    f'Wiki denied access to allpages (HTTP 403). '
                    f'Check that your bot account has read permissions, '
                    f'or try a different operation type that does not require listing all pages.'
                )
            r.raise_for_status()
            data = r.json()
            titles.extend(p['title'] for p in data['query']['allpages'])
            if 'continue' not in data:
                break
            params['apcontinue'] = data['continue']['apcontinue']
        return titles

    def get_pages_with_categories(self) -> dict[str, dict]:
        """Return {title: {'cats': [category_names], 'has_image': bool}} for all main-namespace pages."""
        self._ensure_connected()
        result: dict[str, dict] = {}
        params = {
            'action': 'query',
            'generator': 'allpages',
            'gapnamespace': 0,
            'gaplimit': 50,
            'prop': 'categories|images',
            'cllimit': 50,
            'clshow': '!hidden',
            'imlimit': 1,
            'format': 'json',
        }
        while True:
            r = self._session.get(self._url, params=params)
            if r.status_code == 403:
                raise PermissionError(
                    'Wiki denied access to allpages (HTTP 403). '
                    'Check bot account read permissions.'
                )
            r.raise_for_status()
            data = r.json()
            for page in data.get('query', {}).get('pages', {}).values():
                title = page['title']
                cats = [c['title'].replace('Category:', '') for c in page.get('categories', [])]
                entry = result.setdefault(title, {'cats': [], 'has_image': False})
                entry['cats'].extend(cats)
                if page.get('images'):
                    entry['has_image'] = True
            if 'continue' not in data:
                break
            params.update(data['continue'])
        return result

    def get_recentchanges_head(self) -> int:
        """Return the rcid of the most recent main-namespace change (0 if none).

        This is a single cheap call used as a high-water mark to decide whether a
        cached site index is still fresh, instead of re-listing every page.
        """
        self._ensure_connected()
        r = self._session.get(self._url, params={
            'action': 'query', 'list': 'recentchanges',
            'rcnamespace': 0, 'rclimit': 1, 'rcprop': 'ids',
            'rctype': 'edit|new|log', 'format': 'json',
        })
        r.raise_for_status()
        changes = r.json().get('query', {}).get('recentchanges', [])
        return changes[0].get('rcid', 0) if changes else 0

    def search(self, term: str, limit: int = 50) -> list[dict]:
        self._ensure_connected()
        r = self._session.get(self._url, params={
            'action': 'query', 'list': 'search',
            'srsearch': term, 'srwhat': 'text',
            'srlimit': limit, 'srprop': 'snippet|titlesnippet',
            'format': 'json',
        })
        r.raise_for_status()
        results = r.json().get('query', {}).get('search', [])
        return [{'title': r['title'], 'snippet': _strip_html(r.get('snippet', ''))} for r in results]

    def _reconnect(self):
        self._connected = False
        self._csrf_token = None
        self.connect()

    def write_page(self, title: str, content: str, summary: str = '') -> dict:
        for _attempt in range(2):
            self._ensure_csrf()
            self._rate_limit()
            r = self._session.post(self._url, data={
                'action': 'edit',
                'title': title,
                'text': content,
                'summary': summary,
                'bot': '1',
                'token': self._csrf_token,
                'format': 'json',
            })
            r.raise_for_status()
            data = r.json()
            if data.get('edit', {}).get('result') == 'Success':
                return {'success': True}
            err = str(data)
            if 'badtoken' in err:
                self._csrf_token = self._fetch_csrf()
                continue
            if 'permissiondenied' in err:
                self._reconnect()
                continue
            return {'success': False, 'error': err}
        return {'success': False, 'error': 'write_page failed after reconnect'}

    def move_page(self, from_title: str, to_title: str, reason: str = '') -> dict:
        for _attempt in range(2):
            self._ensure_csrf()
            self._rate_limit()
            r = self._session.post(self._url, data={
                'action': 'move',
                'from': from_title,
                'to': to_title,
                'reason': reason,
                'movetalk': '1',
                'token': self._csrf_token,
                'format': 'json',
            })
            r.raise_for_status()
            data = r.json()
            if 'move' in data:
                return {'success': True}
            err = str(data)
            if 'badtoken' in err:
                self._csrf_token = self._fetch_csrf()
                continue
            if 'permissiondenied' in err:
                self._reconnect()
                continue
            return {'success': False, 'error': err}
        return {'success': False, 'error': 'move_page failed after reconnect'}

    def delete_page(self, title: str, reason: str = '') -> dict:
        for _attempt in range(2):
            self._ensure_csrf()
            self._rate_limit()
            r = self._session.post(self._url, data={
                'action': 'delete',
                'title': title,
                'reason': reason,
                'token': self._csrf_token,
                'format': 'json',
            })
            r.raise_for_status()
            data = r.json()
            if 'delete' in data:
                return {'success': True}
            err = str(data)
            if 'badtoken' in err:
                self._csrf_token = self._fetch_csrf()
                continue
            if 'permissiondenied' in err:
                self._reconnect()
                continue
            return {'success': False, 'error': err}
        return {'success': False, 'error': 'delete_page failed after reconnect'}

    def upload_file(self, filename: str, file_data: bytes, mime_type: str = 'application/octet-stream',
                    description: str = '', comment: str | None = None) -> dict:
        """Upload a file. `description` is the file page text; `comment` the log entry (defaults to it)."""
        for _attempt in range(2):
            self._ensure_csrf()
            self._rate_limit()
            r = self._session.post(self._url, data={
                'action': 'upload',
                'filename': filename,
                'comment': comment or description,
                'text': description,
                'token': self._csrf_token,
                'format': 'json',
                'ignorewarnings': '1',
            }, files={'file': (filename, file_data, mime_type)})
            r.raise_for_status()
            data = r.json()
            if data.get('upload', {}).get('result') in ('Success', 'Warning'):
                return {'success': True, 'filename': filename}
            err = str(data)
            if 'badtoken' in err:
                self._csrf_token = self._fetch_csrf()
                continue
            if 'permissiondenied' in err:
                self._reconnect()
                continue
            return {'success': False, 'error': err}
        return {'success': False, 'error': 'upload_file failed after reconnect'}

    def upload_file_from_url(self, filename: str, url: str, description: str = '') -> dict:
        for _attempt in range(2):
            self._ensure_csrf()
            self._rate_limit()
            r = self._session.post(self._url, data={
                'action': 'upload',
                'filename': filename,
                'url': url,
                'comment': description,
                'text': description,
                'token': self._csrf_token,
                'format': 'json',
                'ignorewarnings': '1',
            })
            r.raise_for_status()
            data = r.json()
            if data.get('upload', {}).get('result') in ('Success', 'Warning'):
                return {'success': True, 'filename': filename}
            err = str(data)
            if 'badtoken' in err:
                self._csrf_token = self._fetch_csrf()
                continue
            if 'permissiondenied' in err:
                self._reconnect()
                continue
            # MediaWiki may disallow remote URL uploads; return clear error
            return {'success': False, 'error': err}
        return {'success': False, 'error': 'upload_file_from_url failed after reconnect'}

    def get_links_from_page(self, title: str) -> list[str]:
        """Get main-namespace [[wikilinks]] from a saved page via the API."""
        links = []
        params = {
            'action': 'query', 'titles': title,
            'prop': 'links', 'pllimit': 500, 'plnamespace': 0, 'format': 'json'
        }
        while True:
            r = self._session.get(self._url, params=params)
            r.raise_for_status()
            data = r.json()
            pages = data.get('query', {}).get('pages') or {}
            page = next(iter(pages.values()), {})
            links.extend(l['title'] for l in page.get('links', []))
            if 'continue' not in data:
                break
            params['plcontinue'] = data['continue']['plcontinue']
        return links

    def extract_links_from_content(self, content: str) -> list[str]:
        """Extract [[wikilinks]] from raw wikitext (for unsaved pages)."""
        raw = re.findall(r'\[\[(?!Category:|File:|Image:|Special:|Template:)([^\]|#]+)', content)
        seen = set()
        result = []
        for link in raw:
            link = link.strip()
            if link and link not in seen:
                seen.add(link)
                result.append(link)
        return result

    def check_local_files_exist(self, filenames: list[str]) -> set[str]:
        """Return the subset of filenames that exist in this wiki's File namespace."""
        if not filenames:
            return set()
        self._ensure_csrf()
        existing = set()
        for i in range(0, len(filenames), 50):
            batch = filenames[i:i + 50]
            r = self._session.get(self._url, params={
                'action': 'query',
                'titles': '|'.join(f'File:{f}' for f in batch),
                'prop': 'imageinfo',
                'format': 'json',
            }, timeout=10)
            r.raise_for_status()
            for page in r.json().get('query', {}).get('pages', {}).values():
                pid = page.get('pageid', -1)
                if pid != -1 and int(pid) > 0 and 'imageinfo' in page:
                    existing.add(page['title'][5:])  # strip "File:"
        return existing

    @staticmethod
    def check_commons_files_exist(filenames: list[str]) -> set[str]:
        """Return the subset of filenames that actually exist on Wikimedia Commons."""
        if not filenames:
            return set()
        headers = {'User-Agent': 'WikiGen/3.0 (wiki management bot; https://github.com/davior/wikigen)'}
        api = 'https://commons.wikimedia.org/w/api.php'
        existing = set()
        for i in range(0, len(filenames), 50):
            batch = filenames[i:i + 50]
            r = requests.get(api, params={
                'action': 'query',
                'titles': '|'.join(f'File:{f}' for f in batch),
                'prop': 'imageinfo',
                'format': 'json',
            }, headers=headers, timeout=10)
            r.raise_for_status()
            for page in r.json().get('query', {}).get('pages', {}).values():
                pid = page.get('pageid', -1)
                if pid != -1 and int(pid) > 0 and 'imageinfo' in page:
                    existing.add(page['title'][5:])  # strip "File:"
        return existing


def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)


def html_to_wikitext(html_content: str, source_url: str = None) -> str:
    """Convert HTML to MediaWiki wikitext format, preserving structure.

    Args:
        html_content: Raw HTML string
        source_url: Optional URL to add as attribution

    Returns:
        MediaWiki wikitext format string
    """
    import html2text

    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = False
    converter.body_width = 0
    converter.unicode_snob = True

    markdown = converter.handle(html_content)

    lines = markdown.split('\n')
    wikitext_lines = []

    for line in lines:
        line = line.rstrip()

        if line.startswith('# '):
            wikitext_lines.append('= ' + line[2:] + ' =')
        elif line.startswith('## '):
            wikitext_lines.append('== ' + line[3:] + ' ==')
        elif line.startswith('### '):
            wikitext_lines.append('=== ' + line[4:] + ' ===')
        elif line.startswith('#### '):
            wikitext_lines.append('==== ' + line[5:] + ' ====')
        elif line.startswith('##### '):
            wikitext_lines.append('===== ' + line[6:] + ' =====')
        elif line.startswith('###### '):
            wikitext_lines.append('====== ' + line[7:] + ' ======')
        elif line.startswith('  * '):
            wikitext_lines.append('** ' + line[4:])
        elif line.startswith('    * '):
            wikitext_lines.append('*** ' + line[6:])
        elif line.startswith('* '):
            wikitext_lines.append('* ' + line[2:])
        elif line.startswith('  1. '):
            wikitext_lines.append('## ' + line[5:])
        elif line.startswith('    1. '):
            wikitext_lines.append('### ' + line[7:])
        elif line.startswith('1. '):
            wikitext_lines.append('# ' + line[3:])
        elif line.startswith('> '):
            wikitext_lines.append(': ' + line[2:])
        else:
            converted_line = line
            converted_line = re.sub(r'\*\*([^*]+)\*\*', r"'''\1'''", converted_line)
            converted_line = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r"''\1''", converted_line)
            converted_line = re.sub(r'_([^_]+)_', r"''\1''", converted_line)
            converted_line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'[\2 \1]', converted_line)
            wikitext_lines.append(converted_line)

    wikitext = '\n'.join(wikitext_lines).strip()

    if source_url:
        attribution = f"''Extracted from [{source_url}]''\n\n"
        wikitext = attribution + wikitext

    return wikitext
