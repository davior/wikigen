"""AI image generation via fal.ai, and the local store of generated images.

Images are generated while a step's content is previewed and kept in
DATA_DIR/generated/ until the page referencing them is written to the wiki:
upload_generated_refs() uploads them then, so a step that is never executed
leaves nothing on the wiki.

Every prompt ends with the connection's house style guide, so all of a wiki's
images share one look while their subjects follow the page content.
"""

import base64
import json
import logging
import mimetypes
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

FAL_QUEUE_URL = 'https://queue.fal.run'
DEFAULT_MODEL = 'fal-ai/flux/dev'
DEFAULT_PARAMS = {'image_size': 'landscape_4_3'}
DEFAULT_IMAGES_PER_PAGE = 3
MAX_IMAGES_PER_PAGE = 4
GENERATION_TIMEOUT = 180  # seconds to wait for one image
POLL_INTERVAL = 1.5
IMAGE_CATEGORY = 'AI-generated images'

STYLE_GUIDE_PROMPT = """\
Write the house illustration style for this wiki: a style guide that will be appended \
to every AI image prompt so that all of the wiki's images look like one consistent set, \
whatever they depict.

Base it on the wiki's subject matter and tone (see the system prompt and the SITE INDEX). \
In 60-120 words of plain, comma-separated prose, cover:
- medium and technique (e.g. "detailed ink and watercolour illustration", \
"muted documentary photograph", "technical cutaway diagram")
- colour palette
- lighting and mood
- composition and framing
- level of detail

End with: "No text, lettering, captions, logos, watermarks or borders."

Describe only the style, never a subject. Return the style guide text only, with no \
heading, quotes or explanation."""

_EXTENSIONS = {'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'image/gif': 'gif'}
_FILE_REF_RE = re.compile(r'\[\[\s*(?:File|Image)\s*:([^|\]]+)', re.IGNORECASE)

_storage_dir = Path('generated')
_style_locks: dict[str, threading.Lock] = {}
_style_locks_guard = threading.Lock()


class ImageGenError(Exception):
    pass


# ─── STORAGE ─────────────────────────────────────────────────────────────────

def set_storage_dir(data_dir) -> None:
    global _storage_dir
    # Absolute, because Flask's send_from_directory resolves relative paths against the app root.
    _storage_dir = Path(data_dir).resolve() / 'generated'
    _storage_dir.mkdir(parents=True, exist_ok=True)


def storage_dir() -> Path:
    return _storage_dir


def canonical_filename(name: str) -> str:
    """MediaWiki's form of a file name: spaces for underscores, first letter upper-case."""
    name = re.sub(r'[\s_]+', ' ', name or '').strip()
    return name[:1].upper() + name[1:]


def _slug(text: str, limit: int) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '-', text or '').strip('-')[:limit].strip('-')


def find_generated(name: str) -> str | None:
    """Return the stored file name for a generated image, or None if `name` isn't one."""
    name = (name or '').strip()
    if not name or '/' in name or '\\' in name or name.startswith('.') or name.endswith('.json'):
        return None
    for candidate in (name, name[:1].upper() + name[1:]):
        if (_storage_dir / candidate).is_file():
            return candidate
    return None


def read_meta(filename: str) -> dict:
    try:
        return json.loads((_storage_dir / f'{filename}.json').read_text())
    except Exception:
        return {}


def save_generated(data: bytes, content_type: str, page_title: str, caption: str, meta: dict) -> str:
    """Store a generated image and its metadata; return its wiki file name.

    Names are ASCII words joined by hyphens with a capitalised first letter,
    which is already MediaWiki's canonical form, so the name written into the
    page is exactly the name the wiki reports back.
    """
    ext = _EXTENSIONS.get(content_type) or (mimetypes.guess_extension(content_type) or '.png').lstrip('.')
    stem = '-'.join(p for p in (_slug(page_title, 40), _slug(caption, 40)) if p) or 'Image'
    filename = f'{stem[:1].upper()}{stem[1:]}-{uuid.uuid4().hex[:8]}.{ext}'
    _storage_dir.mkdir(parents=True, exist_ok=True)
    (_storage_dir / filename).write_bytes(data)
    (_storage_dir / f'{filename}.json').write_text(json.dumps({
        **meta,
        'page': page_title,
        'caption': caption,
        'content_type': content_type,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return filename


def generated_refs(content: str) -> list[str]:
    """Generated images referenced by [[File:...]] links in `content`, in page order."""
    names: list[str] = []
    for m in _FILE_REF_RE.finditer(content or ''):
        name = find_generated(m.group(1))
        if name and name not in names:
            names.append(name)
    return names


def _description_page(meta: dict) -> str:
    page = meta.get('page', '')
    model = meta.get('model', '')
    lines = [meta.get('caption') or page, '']
    lines.append('AI-generated illustration'
                 + (f' for [[{page}]]' if page else '')
                 + ', made with fal.ai' + (f' ({model})' if model else '') + '.')
    if meta.get('prompt'):
        prompt = meta['prompt'].replace('</nowiki>', '')
        lines += ['', f"'''Prompt:''' <nowiki>{prompt}</nowiki>"]
    lines += ['', f'[[Category:{IMAGE_CATEGORY}]]']
    return '\n'.join(lines)


def upload_generated_refs(wiki, content: str) -> list[str]:
    """Upload the generated images `content` references that the wiki doesn't have yet.

    Call this just before writing the page. Returns the names uploaded; raises
    ImageGenError if an upload fails, so the page isn't written with a red link.
    """
    names = generated_refs(content)
    if not names:
        return []
    existing = {canonical_filename(n) for n in wiki.check_local_files_exist(names)}
    uploaded = []
    for name in names:
        if canonical_filename(name) in existing:
            continue
        meta = read_meta(name)
        mime = meta.get('content_type') or mimetypes.guess_type(name)[0] or 'image/png'
        page = meta.get('page')
        comment = f'AI-generated illustration for [[{page}]]' if page else 'AI-generated illustration'
        result = wiki.upload_file(name, (_storage_dir / name).read_bytes(), mime,
                                  _description_page(meta), comment=comment)
        if not result.get('success'):
            raise ImageGenError(f'Uploading "{name}" to the wiki failed: {result.get("error")}')
        uploaded.append(name)
    return uploaded


# ─── STYLE ───────────────────────────────────────────────────────────────────

def build_prompt(scene: str, style: str) -> str:
    scene = scene.strip().rstrip('.')
    return f'{scene}.\n\nStyle: {style.strip()}' if style and style.strip() else f'{scene}.'


def style_lock(key: str) -> threading.Lock:
    """One lock per connection, so parallel steps don't each draft a different style."""
    with _style_locks_guard:
        return _style_locks.setdefault(key, threading.Lock())


# ─── FAL.AI ──────────────────────────────────────────────────────────────────

def parse_params(value) -> dict:
    """Extra model inputs from the connection: a dict, a JSON string, or unset (defaults)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return dict(DEFAULT_PARAMS)
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        log.warning('Ignoring invalid image_params %r', value)
        return dict(DEFAULT_PARAMS)
    return parsed if isinstance(parsed, dict) else dict(DEFAULT_PARAMS)


def _images_per_page(value) -> int:
    try:
        return max(0, min(MAX_IMAGES_PER_PAGE, int(value)))
    except (TypeError, ValueError):
        return DEFAULT_IMAGES_PER_PAGE


def from_connection(conn: dict) -> 'FalImageGenerator | None':
    """The connection's image generator, or None when no fal.ai key is configured."""
    key = (conn.get('fal_key') or os.environ.get('FAL_KEY') or '').strip()
    if not key:
        return None
    return FalImageGenerator(
        key,
        model=conn.get('image_model'),
        params=parse_params(conn.get('image_params')),
        style=conn.get('image_style') or '',
        images_per_page=_images_per_page(conn.get('images_per_page', DEFAULT_IMAGES_PER_PAGE)),
    )


def _fal_error(resp: requests.Response, stage: str) -> ImageGenError:
    try:
        body = resp.json()
    except ValueError:
        body = resp.text[:300]
    detail = body
    if isinstance(body, dict):
        detail = body.get('detail') or body.get('error') or body.get('message') or body
    if isinstance(detail, list):  # validation errors: [{loc, msg, type}]
        detail = '; '.join(
            f"{'.'.join(str(p) for p in d.get('loc', [])[1:]) or 'input'}: {d.get('msg')}"
            if isinstance(d, dict) else str(d) for d in detail)
    hint = ''
    if resp.status_code in (401, 403):
        hint = ' (check the fal.ai key in the connection settings)'
    elif resp.status_code == 404:
        hint = ' (check the model ID in the connection settings)'
    return ImageGenError(f'fal.ai {stage} failed ({resp.status_code}): {str(detail)[:300]}{hint}')


class FalImageGenerator:
    def __init__(self, api_key: str, model: str | None = None, params: dict | None = None,
                 style: str = '', images_per_page: int = DEFAULT_IMAGES_PER_PAGE):
        self.api_key = api_key
        self.model = (model or DEFAULT_MODEL).strip().strip('/')
        self.params = dict(DEFAULT_PARAMS) if params is None else dict(params)
        self.style = (style or '').strip()
        self.images_per_page = images_per_page

    def generate(self, prompt: str) -> tuple[bytes, str]:
        """Generate one image through fal's queue API. Returns (bytes, content_type)."""
        headers = {'Authorization': f'Key {self.api_key}'}
        payload = {'num_images': 1, **self.params, 'prompt': prompt}
        try:
            r = requests.post(f'{FAL_QUEUE_URL}/{self.model}', json=payload, headers=headers, timeout=30)
            if not r.ok:
                raise _fal_error(r, 'request')
            job = r.json()
            status_url, response_url = job.get('status_url'), job.get('response_url')
            if not status_url or not response_url:
                raise ImageGenError(f'fal.ai returned no request URLs: {str(job)[:200]}')

            deadline = time.monotonic() + GENERATION_TIMEOUT
            while True:
                s = requests.get(status_url, headers=headers, timeout=30)
                if not s.ok:
                    raise _fal_error(s, 'status check')
                status = s.json()
                if status.get('error'):
                    raise ImageGenError(f'fal.ai generation failed: {str(status["error"])[:300]}')
                if status.get('status') == 'COMPLETED':
                    break
                if time.monotonic() > deadline:
                    if job.get('cancel_url'):
                        try:
                            requests.put(job['cancel_url'], headers=headers, timeout=10)
                        except requests.RequestException:
                            pass
                    raise ImageGenError(f'fal.ai took longer than {GENERATION_TIMEOUT}s to generate an image')
                time.sleep(POLL_INTERVAL)

            res = requests.get(response_url, headers=headers, timeout=60)
            if not res.ok:
                raise _fal_error(res, 'generation')
            return self._image_from_result(res.json())
        except requests.RequestException as e:
            raise ImageGenError(f'Could not reach fal.ai: {e}') from e

    @staticmethod
    def _image_from_result(data: dict) -> tuple[bytes, str]:
        images = data.get('images') or ([data['image']] if data.get('image') else [])
        if not images:
            raise ImageGenError(f'fal.ai returned no image: {str(data)[:200]}')
        nsfw = data.get('has_nsfw_concepts') or []
        if nsfw and nsfw[0]:
            raise ImageGenError("fal.ai's safety checker blocked the image")
        img = images[0]
        url = img.get('url', '') if isinstance(img, dict) else str(img)
        content_type = (img.get('content_type') if isinstance(img, dict) else '') or ''
        if url.startswith('data:'):
            header, _, encoded = url.partition(',')
            content_type = content_type or header[5:].split(';')[0]
            return base64.b64decode(encoded), content_type or 'image/png'
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        content_type = content_type or r.headers.get('Content-Type', '').split(';')[0].strip() or 'image/jpeg'
        if not content_type.startswith('image/'):
            raise ImageGenError(f'fal.ai returned {content_type}, not an image')
        return r.content, content_type
