import re
import time
import requests


class WikiClient:
    def __init__(self, wiki_url: str, username: str, password: str):
        self._url = wiki_url
        self._username = username
        self._password = password
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'WikiGen/3.0 (wiki management bot; https://github.com/davior/wikigen)',
        })
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
        self._ensure_connected()
        r = self._session.get(self._url, params={
            'action': 'query', 'titles': title, 'format': 'json'
        })
        r.raise_for_status()
        page = next(iter(r.json()['query']['pages'].values()))
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

    def get_pages_with_categories(self) -> dict[str, list[str]]:
        """Return {title: [category_names]} for all main-namespace pages."""
        self._ensure_connected()
        result: dict[str, list[str]] = {}
        params = {
            'action': 'query',
            'generator': 'allpages',
            'gapnamespace': 0,
            'gaplimit': 50,
            'prop': 'categories',
            'cllimit': 50,
            'clshow': '!hidden',
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
                if title in result:
                    result[title].extend(cats)
                else:
                    result[title] = cats
            if 'continue' not in data:
                break
            params.update(data['continue'])
        return result

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
                    description: str = '') -> dict:
        for _attempt in range(2):
            self._ensure_csrf()
            self._rate_limit()
            r = self._session.post(self._url, data={
                'action': 'upload',
                'filename': filename,
                'comment': description,
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

    @staticmethod
    def search_commons_images(query: str, limit: int = 5) -> list[dict]:
        """Search Wikimedia Commons for images. Returns list of {filename, thumb_url, commons_url}."""
        headers = {'User-Agent': 'WikiGen/3.0 (wiki management bot; https://github.com/davior/wikigen)'}
        api = 'https://commons.wikimedia.org/w/api.php'
        r = requests.get(api, params={
            'action': 'query',
            'list': 'search',
            'srnamespace': 6,
            'srsearch': query,
            'srlimit': limit,
            'format': 'json',
        }, headers=headers, timeout=10)
        r.raise_for_status()
        file_titles = [s['title'] for s in r.json().get('query', {}).get('search', [])]
        if not file_titles:
            return []

        ir = requests.get(api, params={
            'action': 'query',
            'titles': '|'.join(file_titles),
            'prop': 'imageinfo',
            'iiprop': 'url|mime',
            'iiurlwidth': 300,
            'format': 'json',
        }, headers=headers, timeout=10)
        ir.raise_for_status()

        images = []
        for page in ir.json().get('query', {}).get('pages', {}).values():
            if 'imageinfo' not in page:
                continue
            info = page['imageinfo'][0]
            if not info.get('mime', '').startswith('image/'):
                continue
            filename = page['title'][5:]  # strip "File:"
            images.append({
                'filename': filename,
                'thumb_url': info.get('thumburl', ''),
                'commons_url': f"https://commons.wikimedia.org/wiki/File:{requests.utils.quote(filename, safe='')}",
            })
        return images


def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)
