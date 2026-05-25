import re
import time
import requests


class WikiClient:
    def __init__(self, wiki_url: str, username: str, password: str):
        self._url = wiki_url
        self._username = username
        self._password = password
        self._session = requests.Session()
        self._csrf_token = None
        self._connected = False
        self._last_write_time = 0.0

    def connect(self) -> bool:
        try:
            r = self._session.get(self._url, params={
                'action': 'query', 'meta': 'tokens', 'type': 'login', 'format': 'json'
            })
            r.raise_for_status()
            login_token = r.json()['query']['tokens']['logintoken']

            r2 = self._session.post(self._url, data={
                'action': 'login',
                'lgname': self._username,
                'lgpassword': self._password,
                'lgtoken': login_token,
                'format': 'json',
            })
            r2.raise_for_status()
            result = r2.json()['login']['result']
            if result != 'Success':
                return False

            self._csrf_token = self._fetch_csrf()
            self._connected = bool(self._csrf_token)
            return self._connected
        except Exception:
            return False

    def _fetch_csrf(self) -> str | None:
        try:
            r = self._session.get(self._url, params={
                'action': 'query', 'meta': 'tokens', 'format': 'json'
            })
            r.raise_for_status()
            return r.json()['query']['tokens']['csrftoken']
        except Exception:
            return None

    def _ensure_csrf(self):
        if not self._csrf_token:
            self._csrf_token = self._fetch_csrf()

    def _rate_limit(self):
        elapsed = time.time() - self._last_write_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_write_time = time.time()

    def get_page(self, title: str) -> dict:
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
        page = next(iter(data['query']['pages'].values()))
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
        r = self._session.get(self._url, params={
            'action': 'query', 'titles': title, 'format': 'json'
        })
        r.raise_for_status()
        page = next(iter(r.json()['query']['pages'].values()))
        page_id = str(page.get('pageid', -1))
        return page_id != '-1' and int(page_id) > 0

    def get_all_pages(self) -> list[str]:
        titles = []
        params = {
            'action': 'query', 'list': 'allpages',
            'aplimit': 500, 'format': 'json'
        }
        while True:
            r = self._session.get(self._url, params=params)
            r.raise_for_status()
            data = r.json()
            titles.extend(p['title'] for p in data['query']['allpages'])
            if 'continue' not in data:
                break
            params['apcontinue'] = data['continue']['apcontinue']
        return titles

    def search(self, term: str, limit: int = 50) -> list[dict]:
        r = self._session.get(self._url, params={
            'action': 'query', 'list': 'search',
            'srsearch': term, 'srwhat': 'text',
            'srlimit': limit, 'srprop': 'snippet|titlesnippet',
            'format': 'json',
        })
        r.raise_for_status()
        results = r.json().get('query', {}).get('search', [])
        return [{'title': r['title'], 'snippet': _strip_html(r.get('snippet', ''))} for r in results]

    def write_page(self, title: str, content: str, summary: str = '') -> dict:
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
        if 'badtoken' in str(data):
            self._csrf_token = self._fetch_csrf()
            return self.write_page(title, content, summary)
        return {'success': False, 'error': str(data)}

    def move_page(self, from_title: str, to_title: str, reason: str = '') -> dict:
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
        if 'badtoken' in str(data):
            self._csrf_token = self._fetch_csrf()
            return self.move_page(from_title, to_title, reason)
        return {'success': False, 'error': str(data)}

    def delete_page(self, title: str, reason: str = '') -> dict:
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
        if 'badtoken' in str(data):
            self._csrf_token = self._fetch_csrf()
            return self.delete_page(title, reason)
        return {'success': False, 'error': str(data)}

    def get_links_from_page(self, title: str) -> list[str]:
        """Get [[wikilinks]] from a saved page via the API."""
        links = []
        params = {
            'action': 'query', 'titles': title,
            'prop': 'links', 'pllimit': 500, 'format': 'json'
        }
        while True:
            r = self._session.get(self._url, params=params)
            r.raise_for_status()
            data = r.json()
            page = next(iter(data['query']['pages'].values()))
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


def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)
