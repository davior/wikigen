import concurrent.futures
import json
import queue
import re
import requests
import threading
import time
import uuid
import difflib
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

from wiki_client import WikiClient


# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

PLANNER_PREFIX = """You are a wiki management agent for a MediaWiki wiki.

You receive:
1. A full index of all current wiki pages grouped by category (in the system context)
2. Optionally, contents of specific wiki pages selected for reference (in the system context)
3. A natural language instruction (in the user message)

Analyse the instruction relative to the existing wiki structure and return a structured plan.

CRITICAL RULE: The SITE INDEX lists EVERY page that currently exists in this wiki. You MUST NOT produce a 'create' step for any title that already appears in the SITE INDEX. When the user asks for "missing pages", only create steps for pages that are explicitly listed as not existing.

If the user requests creation of a page that already exists in the SITE INDEX:
- Do NOT create a 'create' step for it
- Suggest an 'edit' step instead if the user wants to modify the existing page
- In your description, note which requested pages already exist so the user understands why they were not included

Step types you may produce:
- create: Create a new page from scratch (ONLY for titles NOT in the SITE INDEX)
- edit: Modify an existing page (must exist in the index)
- delete: Delete a page (must exist in the index)
- move: Rename a page; title = destination, from_title = source
- find_replace: Bulk text replacement across the wiki; title = "*"; describe find/replace pairs in description
- ensure_disambig: Create a redirect or disambiguation page for an abbreviation
- add_image: Source an image from Wikimedia Commons and embed it into an existing page
- upload_file: Upload a document or file to the wiki file namespace; title = "File:<exact filename>" using the filename shown in [wiki filename: File:...] next to the document in the UPLOADED DOCUMENTS section. Do NOT set source_url for attached documents — the bytes are already saved. Only set source_url when the user explicitly provides a remote URL to fetch (not a URL found inside the document's content). After uploading, reference the file in page content as [[File:filename.ext]].
- When the user has MULTIPLE documents and asks to upload all of them, create one upload_file step per document, each using the exact filename from its [wiki filename: ...] label.
- When the user wants to find existing pages that should reference an uploaded file, scan the SITE INDEX for topically related pages and create edit steps for them that add [[File:filename.ext]] where appropriate.
- When the user imports HTML content or web pages as wiki articles (not file uploads), create a normal create step (not upload_file). The HTML has been pre-converted to MediaWiki wikitext format and is available as document text in context — use that content directly in the created page.
- When the user wants a wiki page CREATED FROM the content of an uploaded document or URL, create a normal create step (not upload_file) and describe the desired page content in the step description; the AI generating that step will have the document text in context to draw from.
- Use depends_on to ensure upload_file steps complete before any create/edit steps that reference the uploaded file.

When describing content to create or edit, be specific. The description will be used as the sole instruction for content generation, so include:
- Topic scope and angle
- Key sections to include
- Links to related pages that should be included
- Any stylistic or structural requirements

Wiki markup conventions:
- Lead paragraph summarising the topic
- == Section == headings
- [[wikilinks]] to related topics
- [[Category:Name]] tags at the end

When returning JSON, return ONLY valid JSON with no markdown fences, no preamble, no explanation.
When returning wikitext, return raw wikitext only with no explanation.
"""

PLAN_SCHEMA = """\
Return a JSON object with this exact structure:
{
  "description": "one-sentence summary of the overall plan",
  "steps": [
    {
      "id": "s1",
      "type": "create|edit|delete|move|find_replace|ensure_disambig|add_image|upload_file",
      "title": "Page Title",
      "from_title": "Source Title",
      "source_url": "https://example.com/file.pdf",
      "description": "detailed instructions for this step",
      "summary": "short one-line description shown to the user",
      "depends_on": []
    }
  ]
}
Notes:
- id must be unique within the plan (e.g. s1, s2, s3)
- depends_on lists step ids that must fully complete before this step starts
- For move: title = destination, from_title = source
- For find_replace: title = "*", description = the pairs to find and replace
- For delete: description = reason for deletion
- For upload_file: title = "File:<exact filename from context>", description = file caption; do NOT set source_url for attached docs; after uploading use [[File:filename.ext]] to reference it
- For multiple documents: one upload_file step per document, each with its own exact filename
- Always use depends_on so create/edit steps that reference [[File:...]] wait for the upload_file step to finish
- NEVER produce a create step for a title that appears in the SITE INDEX — it already exists
- Use depends_on when order matters (e.g. upload before creating a page that references it)"""

DISAMBIG_TEMPLATE = "#REDIRECT [[{target}]]\n\n{{{{Redirect}}}}"

DISAMBIG_LISTING_TEMPLATE = """'''{abbr}''' may refer to:

{items}

[[Category:Disambiguation pages]]"""


# ─── DATACLASSES ─────────────────────────────────────────────────────────────

@dataclass
class OperationStep:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = 'write'
    title: str = ''
    from_title: Optional[str] = None
    content: Optional[str] = None
    old_content: Optional[str] = None
    summary: str = ''
    description: str = ''
    depends_on: list = field(default_factory=list)
    status: str = 'pending'
    error: Optional[str] = None
    links_to: list = field(default_factory=list)
    diff: Optional[str] = None
    image_file: Optional[str] = None
    commons_url: Optional[str] = None
    images: list = field(default_factory=list)  # [{filename, commons_url, caption, section}]
    source_url: Optional[str] = None
    upload_id: Optional[str] = None

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class OperationPlan:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = ''
    description: str = ''
    steps: list = field(default_factory=list)
    connection_id: str = ''
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = 'pending'

    def to_dict(self):
        d = dataclasses.asdict(self)
        d['steps'] = [s.to_dict() if isinstance(s, OperationStep) else s for s in self.steps]
        return d


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    # Skip any preamble before the JSON object/array
    for ch in ('{', '['):
        pos = text.find(ch)
        if pos != -1:
            text = text[pos:]
            break
    if not text:
        raise ValueError('AI returned an empty response — no JSON found')
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        snippet = text[:120].replace('\n', ' ')
        raise ValueError(f'JSON parse error: {e}. Response started with: {snippet!r}') from e
    return obj


def _make_diff(old: str, new: str) -> str:
    return '\n'.join(difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        lineterm='',
        fromfile='before',
        tofile='after',
    ))


def _format_site_index(pages: dict) -> str:
    """Render the site index compactly for the planner's cached system block.

    Accepts either the legacy {title: [categories]} shape or the richer
    {title: {'c': [categories], 'd': description}} shape. Pages are grouped one
    line per category, separated by ' · ', with "title — description" shown
    wherever a description is known. The dense layout keeps the (prompt-cached)
    block small while still conveying whole-site context.
    """
    if not pages:
        return 'SITE INDEX: (empty wiki — no pages yet)'

    def _cats_desc(value) -> tuple[list[str], str]:
        if isinstance(value, dict):
            return (value.get('c') or []), (value.get('d') or '').strip()
        return (value or []), ''

    by_cat: dict[str, list[str]] = {}
    uncategorized: list[str] = []
    for title, value in pages.items():
        cats, desc = _cats_desc(value)
        label = f'{title} — {desc}' if desc else title
        if cats:
            for cat in cats:
                by_cat.setdefault(cat, []).append(label)
        else:
            uncategorized.append(label)

    lines = [f'SITE INDEX: {len(pages)} pages '
             '(grouped by category; "title — description" where known)']
    for cat in sorted(by_cat):
        lines.append(f'{cat}: {" · ".join(sorted(by_cat[cat]))}')
    if uncategorized:
        lines.append(f'(uncategorized): {" · ".join(sorted(uncategorized))}')
    return '\n'.join(lines)


_FILE_REF_RE = re.compile(r'\[\[(?:File|Image):([^|\]]+)(\|[^\]]*)?(\]\])', re.IGNORECASE)
_IMAGE_PLACEHOLDER_RE = re.compile(r'\{\{COMMONS_IMAGE:([^|}]*?)(?:\|([^}]*))?\}\}', re.IGNORECASE)
MAX_PLACEHOLDER_IMAGES = 3


def _fix_file_references(content: str, wiki) -> str:
    matches = list(_FILE_REF_RE.finditer(content))
    if not matches:
        return content
    filenames = [m.group(1).strip() for m in matches]
    try:
        existing = wiki.check_commons_files_exist(filenames)
    except Exception:
        return content
    replacements = {}
    for m in matches:
        filename = m.group(1).strip()
        if filename in existing or filename in replacements:
            continue
        query = re.sub(r'\.[^.]+$', '', filename).replace('_', ' ').replace('-', ' ')
        try:
            results = wiki.search_commons_images(query, limit=3)
        except Exception:
            results = []
        replacements[filename] = results[0]['filename'] if results else None

    def apply_replacement(m):
        filename = m.group(1).strip()
        options = m.group(2) or ''
        closing = m.group(3)
        if filename in existing:
            return m.group(0)
        sub = replacements.get(filename)
        return f'[[File:{sub}{options}{closing}' if sub else ''

    return _FILE_REF_RE.sub(apply_replacement, content)


def _insert_image(content: str, filename: str, caption: str, placement: str) -> str:
    markup = f'[[File:{filename}|thumb|right|{caption}]]'
    lines = content.split('\n')
    if placement.startswith('section:'):
        section = placement[8:].strip()
        for i, line in enumerate(lines):
            if re.match(r'^==+\s*' + re.escape(section) + r'\s*==+\s*$', line):
                insert_pos = i + 1
                if insert_pos < len(lines) and not lines[insert_pos].strip():
                    insert_pos += 1
                lines.insert(insert_pos, markup)
                lines.insert(insert_pos, '')
                return '\n'.join(lines)
    if placement == 'end':
        cat_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('[[Category:')), len(lines))
        lines.insert(cat_idx, markup)
        lines.insert(cat_idx, '')
        return '\n'.join(lines)
    for i, line in enumerate(lines):
        if i > 0 and re.match(r'^==+', line.strip()):
            lines.insert(i, '')
            lines.insert(i, markup)
            return '\n'.join(lines)
    for i in range(1, len(lines)):
        if lines[i].strip() == '' and lines[i - 1].strip():
            lines.insert(i + 1, '')
            lines.insert(i + 1, markup)
            return '\n'.join(lines)
    return content + '\n\n' + markup


def _split_sections(content: str) -> list[dict]:
    """Split wikitext into the lead + each == Heading == block.

    Returns ordered dicts: {name, placement, text, has_image}. The lead uses
    name 'Lead' and placement 'after_lead'; headings use placement 'section:<name>'.
    has_image is True when the section body already contains a File/Image ref.
    """
    lines = content.split('\n')
    sections: list[dict] = []
    current = {'name': 'Lead', 'placement': 'after_lead', 'lines': []}
    heading_re = re.compile(r'^(==+)\s*(.*?)\s*==+\s*$')
    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            sections.append(current)
            name = m.group(2).strip()
            current = {'name': name, 'placement': f'section:{name}', 'lines': []}
        else:
            current['lines'].append(line)
    sections.append(current)

    result = []
    for sec in sections:
        text = '\n'.join(sec['lines']).strip()
        # Skip an empty lead (page that opens directly with a heading)
        if sec['name'] == 'Lead' and not text:
            continue
        result.append({
            'name': sec['name'],
            'placement': sec['placement'],
            'text': text,
            'has_image': bool(_FILE_REF_RE.search(text)),
        })
    return result


# ─── AGENT ───────────────────────────────────────────────────────────────────

class WikiAgent:
    def __init__(self, wiki: WikiClient, anthropic_client, system_prompt: str,
                 connection_id: str, site_index: dict | None = None,
                 context_pages: list | None = None, uploads_dir=None,
                 recent_pages: dict | None = None):
        self.wiki = wiki
        self.ai = anthropic_client
        self.connection_id = connection_id
        self.cancel_event = threading.Event()
        self._stream_callback: Optional[Callable] = None
        # Pages created/updated this session that aren't in the frozen site index
        # yet. They feed the uncached prompt tail (see generate_plan) so the model
        # knows they exist, and they count as "existing" for de-duplication and
        # link classification — without disturbing the cached site-index block.
        self._recent_pages: dict = recent_pages or {}
        existing = set(site_index.keys()) if site_index else set()
        existing |= set(self._recent_pages.keys())
        self._existing_titles: set[str] = existing
        self._context_docs: list[dict] = context_pages or []
        self._uploads_dir = uploads_dir

        # The planner prefix and the site index change rarely, so cache them for
        # an hour — repeat planning/execution sessions hit the cache instead of
        # re-paying for these (large, stable) blocks. Per-request context blocks
        # below keep the default 5-minute TTL.
        blocks = [{
            'type': 'text',
            'text': PLANNER_PREFIX + ('\n\n' + system_prompt if system_prompt else ''),
            'cache_control': {'type': 'ephemeral', 'ttl': '1h'},
        }]
        if site_index:
            blocks.append({
                'type': 'text',
                'text': _format_site_index(site_index),
                'cache_control': {'type': 'ephemeral', 'ttl': '1h'},
            })
        if context_pages:
            ctx_text = self._build_context_prefix(context_pages)
            if ctx_text:
                blocks.append({
                    'type': 'text',
                    'text': ctx_text,
                    'cache_control': {'type': 'ephemeral'},
                })
        self._system_blocks = blocks

    def _call_ai(self, user_message: str, max_tokens: int = 64000,
                 model: str = 'claude-sonnet-4-6') -> str:
        with self.ai.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=self._system_blocks,
            messages=[{'role': 'user', 'content': user_message}],
        ) as stream:
            return stream.get_final_text()

    def _build_context_prefix(self, context_pages: list[dict]) -> str:
        if not context_pages:
            return ''
        lines = []
        documents = [p for p in context_pages if p.get('is_document')]
        referenced = [p for p in context_pages if p.get('is_referenced') and not p.get('is_document')]
        general = [p for p in context_pages if not p.get('is_referenced') and not p.get('is_document')]
        if referenced:
            lines.append('REFERENCED WIKI PAGES (use their content as source material):')
            for p in referenced:
                lines.append(f"=== {p['title']} ===\n{p.get('content', '')}")
        if general:
            lines.append('ADDITIONAL CONTEXT PAGES:')
            for p in general[:5]:
                lines.append(f"=== {p['title']} ===\n{p.get('content', '')}")
        if documents:
            lines.append('UPLOADED DOCUMENTS / FETCHED URLS (use as reference material):')
            for p in documents[:10]:
                fname = p.get('filename', '')
                fname_note = f' [wiki filename: File:{fname}]' if fname else ''
                src_note = f' (source: {p["source_url"]})' if p.get('source_url') else ''
                lines.append(f"=== {p['title']}{fname_note}{src_note} ===\n{p.get('content', '')}")
        return '\n\n'.join(lines) + '\n\n---\n\n'

    def _emit(self, event: dict):
        if self._stream_callback:
            self._stream_callback(event)

    def _detect_referenced_pages(self, instruction: str) -> list[str]:
        # Explicitly quoted titles
        matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', instruction)
        titles = [title for pair in matches for title in pair if title]
        # Uploaded document titles are not wiki pages — exclude them from matching
        doc_titles = {p.get('title', '').lower() for p in self._context_docs if p.get('is_document')}
        # Also detect unquoted page titles from the site index that appear in the instruction
        if self._existing_titles:
            instr_lower = instruction.lower()
            for title in self._existing_titles:
                if title.lower() in doc_titles:
                    continue  # don't confuse document names with wiki page names
                if title not in titles and title.lower() in instr_lower:
                    titles.append(title)
        return titles

    # ── PHASE 1: GENERATE PLAN ────────────────────────────────────────────────

    def generate_plan(self, instruction: str) -> OperationPlan:
        plan = OperationPlan(connection_id=self.connection_id, operation_type='auto')

        # Auto-fetch URLs found in the instruction and add as context
        # Skip any URL already supplied as a context document (avoid duplicates)
        existing_urls = {p.get('source_url') for p in self._context_docs if p.get('source_url')}
        url_pattern = re.compile(r'https?://[^\s,>"\]]+')
        instruction_urls = [u for u in url_pattern.findall(instruction) if u not in existing_urls]
        if instruction_urls:
            fetched_docs = []
            for url in instruction_urls[:3]:
                try:
                    resp = requests.get(url, timeout=10, headers={'User-Agent': 'WikiGen/3.0'})
                    resp.raise_for_status()
                    content_type = resp.headers.get('Content-Type', '')
                    if 'html' in content_type:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(resp.content, 'lxml')
                        title_tag = soup.find('title')
                        title = title_tag.get_text(strip=True) if title_tag else url
                        for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
                            tag.decompose()
                        content_el = soup.find('main') or soup.find('article') or soup.find('body') or soup
                        text = content_el.get_text(separator='\n', strip=True)[:20000]
                    else:
                        text = resp.text.strip()[:20000]
                        title = url.rsplit('/', 1)[-1] or url
                    fetched_docs.append({'title': title, 'content': text, 'source_url': url, 'is_document': True})
                except Exception:
                    pass
            if fetched_docs:
                url_ctx = self._build_context_prefix(fetched_docs)
                if url_ctx:
                    self._system_blocks.append({
                        'type': 'text',
                        'text': url_ctx,
                        'cache_control': {'type': 'ephemeral'},
                    })

        # Build user message: auto-fetch referenced pages + links, then instruction
        parts = []
        referenced_titles = self._detect_referenced_pages(instruction)
        if referenced_titles:
            ref_sections = []
            for title in referenced_titles:
                page = self.wiki.get_page(title)
                if page.get('exists'):
                    section = f"=== {title} ===\n{page.get('content', '(empty)')}"
                    # Pre-compute missing vs. existing links so the AI doesn't have to cross-reference
                    try:
                        outgoing = self.wiki.get_links_from_page(title)
                        if outgoing:
                            missing = [l for l in outgoing if l not in self._existing_titles]
                            already_exist = [l for l in outgoing if l in self._existing_titles]
                            if missing:
                                section += '\n\nPages linked from this article that DO NOT EXIST yet (these are candidates to create):\n'
                                section += '\n'.join(f'  - {l}' for l in missing[:200])
                            if already_exist:
                                section += '\n\nPages linked from this article that ALREADY EXIST (do NOT create these):\n'
                                section += '\n'.join(f'  - {l}' for l in already_exist[:200])
                    except Exception:
                        pass
                    ref_sections.append(section)
            if ref_sections:
                parts.append('REFERENCED PAGES:\n\n' + '\n\n'.join(ref_sections))

        if self._recent_pages:
            recent_lines = ['RECENTLY CREATED OR UPDATED THIS SESSION '
                            '(these already exist — do NOT recreate them; they are '
                            'not yet listed in the SITE INDEX above):']
            for title in sorted(self._recent_pages):
                desc = (self._recent_pages[title] or {}).get('d', '')
                recent_lines.append(f'  - {title}' + (f' — {desc}' if desc else ''))
            parts.append('\n'.join(recent_lines))

        parts.append(f'INSTRUCTION: {instruction}')
        parts.append(PLAN_SCHEMA)
        user_message = '\n\n'.join(parts)

        try:
            raw = self._call_ai(user_message)
            if not raw or not raw.strip():
                raise ValueError('AI returned an empty response')
            data = _extract_json(raw)
        except Exception as e:
            plan.description = f'Planning failed: {e}'
            plan.status = 'failed'
            self._emit({'type': 'error', 'error': str(e)})
            return plan

        plan.description = data.get('description', instruction)
        plan.status = 'pending'

        for i, step_data in enumerate(data.get('steps', [])):
            step = OperationStep(
                id=step_data.get('id', f's{i}'),
                type=step_data.get('type', 'create'),
                title=step_data.get('title', ''),
                from_title=step_data.get('from_title') or None,
                description=step_data.get('description', ''),
                summary=step_data.get('summary', step_data.get('description', '')),
                depends_on=step_data.get('depends_on', []),
                source_url=step_data.get('source_url') or None,
            )
            plan.steps.append(step)
            self._emit({'type': 'step', 'step': step.to_dict()})

        # Filter out 'create' steps for pages that already exist in the site index
        original_count = len(plan.steps)
        skipped_creates = []
        filtered_steps = []
        for step in plan.steps:
            if step.type == 'create' and step.title in self._existing_titles:
                skipped_creates.append(step.title)
            else:
                filtered_steps.append(step)

        plan.steps = filtered_steps
        if skipped_creates:
            skip_msg = f"\n(Skipped {len(skipped_creates)} create step(s) for existing pages: {', '.join(skipped_creates)})"
            plan.description = (plan.description or '') + skip_msg

        self._emit({'type': 'done', 'plan_id': plan.id})
        return plan

    # ── PHASE 2: PER-STEP CONTENT GENERATION ─────────────────────────────────

    def _generate_page_content(self, title: str, instructions: str) -> dict:
        prompt = (
            f'Generate a complete wiki page for: **{title}**\n\n'
            f'Instructions: {instructions}\n\n'
            'Return raw MediaWiki wikitext only (no JSON, no explanation). Include:\n'
            '- A lead paragraph\n'
            '- == Section == headings\n'
            '- [[wikilinks]] to related topics\n'
            '- [[Category:...]] tags at the end\n\n'
            'For images: use {{COMMONS_IMAGE:search terms|caption text}} placeholders where '
            '"search terms" describes what image you want from Wikimedia Commons '
            '(e.g. {{COMMONS_IMAGE:ancient Roman amphitheater ruins|Roman amphitheater}}). '
            'The system will search Commons and pick the best matching file for each. '
            'Use up to 3 images. Do NOT use [[File:...]] with invented filenames.'
        )
        content = self._call_ai(prompt)
        content = self._resolve_image_placeholders(content)
        content = _fix_file_references(content, self.wiki)
        return {
            'content': content,
            'summary': f'Create: {title}',
        }

    def _edit_page_content(self, title: str, current_content: str, instructions: str) -> str:
        prompt = (
            f'Current content of [[{title}]]:\n\n{current_content}\n\n'
            f'Instruction: {instructions}\n\n'
            'Return the complete revised MediaWiki wikitext only (no JSON, no explanation).\n\n'
            'For any new images: use {{COMMONS_IMAGE:search terms|caption text}} placeholders '
            '(e.g. {{COMMONS_IMAGE:ancient Roman amphitheater ruins|Roman amphitheater}}). '
            'The system will search Wikimedia Commons and pick the best matching file for each. '
            'Use up to 3 new images. Do NOT use [[File:...]] with invented filenames.'
        )
        new_content = self._call_ai(prompt)
        new_content = self._resolve_image_placeholders(new_content)
        return _fix_file_references(new_content, self.wiki)

    def _execute_find_replace_step(self, step: OperationStep) -> dict:
        # Extract pairs from step description via AI
        prompt = (
            f'Extract find-and-replace pairs from this instruction: {step.description}\n\n'
            'Return JSON: {"replacements": [{"find": "...", "replace": "..."}]}'
        )
        try:
            data = _extract_json(self._call_ai(prompt))
        except Exception:
            return {'success': False, 'error': 'Could not parse find/replace pairs'}
        pairs = data.get('replacements', [])
        if not pairs and 'find' in data:
            pairs = [{'find': data['find'], 'replace': data['replace']}]

        page_changes: dict[str, dict] = {}
        for pair in pairs:
            find_term, replace_term = pair['find'], pair['replace']
            for result in self.wiki.search(find_term, limit=100):
                title = result['title']
                if title not in page_changes:
                    page = self.wiki.get_page(title)
                    if not page.get('content'):
                        continue
                    page_changes[title] = {
                        'original': page['content'],
                        'current': page['content'],
                        'summaries': [],
                    }
                current = page_changes[title]['current']
                if find_term not in current:
                    continue
                count = current.count(find_term)
                page_changes[title]['current'] = current.replace(find_term, replace_term)
                page_changes[title]['summaries'].append(f'"{find_term}" → "{replace_term}" ({count}×)')

        success_count = 0
        for title, change in page_changes.items():
            if change['current'] == change['original']:
                continue
            wiki_summary = 'Find/replace: ' + '; '.join(change['summaries'])
            result = self.wiki.write_page(title, change['current'], wiki_summary)
            if result.get('success'):
                success_count += 1

        step.summary = f'Replaced across {success_count} pages'
        return {'success': True, 'pages_modified': success_count}

    def _execute_ensure_disambig_step(self, step: OperationStep) -> dict:
        # Extract abbreviations from step description
        abbr_pattern = re.findall(r'\b([A-Z]{2,8}(?:s)?)\b', step.description or step.title)
        if not abbr_pattern:
            abbr_pattern = [step.title]

        all_pages = set(self.wiki.get_all_pages())
        written = 0
        for abbr in abbr_pattern:
            if abbr in all_pages:
                continue
            matching = [t for t in all_pages if abbr.lower() in t.lower().split()]
            if len(matching) == 1:
                content = DISAMBIG_TEMPLATE.format(target=matching[0])
                wiki_summary = f'Redirect: {abbr} → {matching[0]}'
            elif len(matching) > 1:
                items = '\n'.join(f'* [[{t}]]' for t in matching[:10])
                content = DISAMBIG_LISTING_TEMPLATE.format(abbr=abbr, items=items)
                wiki_summary = f'Disambiguation page for {abbr}'
            else:
                content = DISAMBIG_TEMPLATE.format(target=abbr + ' (disambiguation)')
                wiki_summary = f'Placeholder redirect for {abbr}'
            result = self.wiki.write_page(abbr, content, wiki_summary)
            if result.get('success'):
                written += 1
        return {'success': True, 'pages_written': written}

    def _prepare_add_image_content(self, step: OperationStep) -> None:
        """Populate step.content/diff/images without writing to wiki. Raises on failure.

        Section-aware: analyses the page's sections and adds 2–3 images, one per
        chosen section, skipping sections that already contain an image. Gracefully
        degrades to fewer images if some searches fail. All AI calls use Haiku
        (filenames/text only — no image pixel analysis) to keep cost down.
        """
        page = self.wiki.get_page(step.title)
        if not page.get('exists') or not page.get('content'):
            raise ValueError(f'Page "{step.title}" not found or empty')
        content = page['content']

        # Candidate sections = those that don't already have an image.
        sections = [s for s in _split_sections(content) if not s['has_image']]
        if not sections:
            raise ValueError(f'Every section of "{step.title}" already has an image')

        chosen_sections = self._select_image_sections(step, sections)
        if not chosen_sections:
            raise ValueError(f'No suitable section found for an image on "{step.title}"')

        # Search Commons per chosen section, then pick the best filename per section.
        placement_by_name = {s['name']: s['placement'] for s in sections}
        candidates = []  # [{name, placement, caption, results}]
        for entry in chosen_sections:
            name = entry.get('section', '').strip()
            query = (entry.get('query') or name or step.title).strip()
            placement = placement_by_name.get(name, 'after_lead')
            try:
                results = self.wiki.search_commons_images(query, limit=5)
            except Exception:
                results = []
            if results:
                candidates.append({
                    'name': name,
                    'placement': placement,
                    'caption': (entry.get('caption') or name or step.title).strip(),
                    'results': results,
                })

        if not candidates:
            raise ValueError(f'No images found on Commons for "{step.title}"')

        picks = self._pick_best_images(candidates)

        # Insert sequentially, deduping filenames across sections.
        new_content = content
        used: set[str] = set()
        images: list[dict] = []
        for cand, idx in zip(candidates, picks):
            results = cand['results']
            idx = max(0, min(idx, len(results) - 1))
            chosen = results[idx]
            # Skip a duplicate; fall back to the first unused result in this section.
            if chosen['filename'] in used:
                chosen = next((r for r in results if r['filename'] not in used), None)
                if not chosen:
                    continue
            used.add(chosen['filename'])
            new_content = _insert_image(new_content, chosen['filename'], cand['caption'], cand['placement'])
            images.append({
                'filename': chosen['filename'],
                'commons_url': chosen['commons_url'],
                'caption': cand['caption'],
                'section': cand['name'],
            })

        if not images:
            raise ValueError(f'No images found on Commons for "{step.title}"')

        step.content = new_content
        step.old_content = content
        step.diff = _make_diff(content, new_content)
        step.images = images
        step.image_file = images[0]['filename']
        step.commons_url = images[0]['commons_url']

    def _select_image_sections(self, step: OperationStep, sections: list[dict]) -> list[dict]:
        """Haiku call: choose 2–3 sections to illustrate, with a query + caption each."""
        section_blocks = '\n\n'.join(
            f'- Section: {s["name"]}\n  Content: {s["text"][:400]}' for s in sections
        )
        context_hint = f'\n\nUser request context: {step.description[:300]}' if step.description else ''
        prompt = (
            f'You are choosing where to add images to the wiki page "{step.title}".\n\n'
            f'These are the sections that do NOT yet have an image:\n\n{section_blocks}{context_hint}\n\n'
            'Choose 2 to 3 of these sections that would benefit from an illustrative image. '
            'Prioritize breadth and coverage. For each chosen section provide a concise '
            'Wikimedia Commons search query and a short caption.\n\n'
            'Return ONLY JSON: '
            '{"images": [{"section": "<exact section name>", '
            '"query": "commons search terms", "caption": "caption text"}]}'
        )
        try:
            data = _extract_json(self._call_ai(prompt, model='claude-haiku-4-5-20251001'))
            picks = data.get('images', []) if isinstance(data, dict) else data
            valid_names = {s['name'] for s in sections}
            picks = [p for p in picks if isinstance(p, dict) and p.get('section') in valid_names]
            return picks[:3]
        except Exception:
            # Fall back to picking 2-3 sections by position when Haiku fails.
            fallback = []
            for i, section in enumerate(sections[:3]):
                fallback.append({
                    'section': section['name'],
                    'query': section['name'],
                    'caption': section['name'],
                })
            return fallback

    def _resolve_image_placeholders(self, content: str) -> str:
        """Replace {{COMMONS_IMAGE:query|caption}} placeholders with real Commons file refs.

        Caps to the first MAX_PLACEHOLDER_IMAGES unique queries and uses the same Haiku
        best-of-candidates picker as the add_image step (rather than blindly taking the
        top Commons hit). Placeholders over the cap or with no results are removed.
        """
        matches = list(_IMAGE_PLACEHOLDER_RE.finditer(content))
        if not matches:
            return content

        # Ordered unique queries, capped.
        kept_queries: list[str] = []
        for m in matches:
            query = m.group(1).strip()
            if query and query not in kept_queries:
                kept_queries.append(query)
        kept_queries = kept_queries[:MAX_PLACEHOLDER_IMAGES]

        # Search Commons per kept query, then pick the best filename per query via Haiku.
        candidates = []  # [{name: query, results}]
        for query in kept_queries:
            try:
                results = self.wiki.search_commons_images(query, limit=5)
            except Exception:
                results = []
            if results:
                candidates.append({'name': query, 'results': results})

        picks = self._pick_best_images(candidates)
        chosen: dict[str, str] = {}
        for cand, idx in zip(candidates, picks):
            results = cand['results']
            idx = max(0, min(idx, len(results) - 1))
            chosen[cand['name']] = results[idx]['filename']

        def apply(m):
            query = m.group(1).strip()
            caption = (m.group(2) or query).strip()
            filename = chosen.get(query)
            if not filename:
                return ''
            return f'[[File:{filename}|thumb|right|{caption}]]'

        return _IMAGE_PLACEHOLDER_RE.sub(apply, content)

    def _pick_best_images(self, candidates: list[dict]) -> list[int]:
        """Haiku call: pick the best result index (0-based) per candidate group from filenames.

        Each candidate is {name, results}; returns one chosen index per candidate.
        """
        if not candidates:
            return []
        blocks = []
        for i, cand in enumerate(candidates):
            files = '\n'.join(f'    {j + 1}. {r["filename"]}' for j, r in enumerate(cand['results']))
            blocks.append(f'Option group {i + 1} ("{cand["name"]}") candidates:\n{files}')
        prompt = (
            'For each option group below, pick the most relevant image by its number, '
            'based on the filenames.\n\n' + '\n\n'.join(blocks) + '\n\n'
            'Return ONLY JSON mapping group number to chosen image number, e.g. '
            '{"picks": {"1": 2, "2": 1}}'
        )
        try:
            data = _extract_json(self._call_ai(prompt, model='claude-haiku-4-5-20251001'))
            raw = data.get('picks', {}) if isinstance(data, dict) else {}
            return [max(0, int(raw.get(str(i + 1), 1)) - 1) for i in range(len(candidates))]
        except Exception:
            return [0] * len(candidates)

    def _execute_add_image_step(self, step: OperationStep) -> dict:
        if not step.content:
            self._prepare_add_image_content(step)
        files = [img['filename'] for img in step.images] or ([step.image_file] if step.image_file else [])
        if len(files) > 1:
            summary = f'Add {len(files)} images: ' + ', '.join(files)
        else:
            summary = f'Add image: {files[0] if files else ""}'
        return self.wiki.write_page(step.title, step.content, summary)

    def _execute_upload_file_step(self, step: OperationStep) -> dict:
        filename = step.title.removeprefix('File:')
        description = step.description or step.summary
        import mimetypes

        # Preferred path: upload the actual attached document bytes.
        if step.upload_id and self._uploads_dir:
            path = Path(self._uploads_dir) / step.upload_id
            if path.exists():
                mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                return self.wiki.upload_file(filename, path.read_bytes(), mime, description)
            return {'success': False, 'error': f'Uploaded file for "{filename}" is no longer available'}

        if step.source_url:
            # Try wiki-side remote URL upload first (works only if $wgAllowCopyUploads is on)
            result = self.wiki.upload_file_from_url(filename, step.source_url, description)
            if not result.get('success'):
                # Fallback: fetch the file ourselves then upload the bytes.
                # Use browser-like headers — many CDNs (e.g. Google storage) return
                # 403 for non-browser User-Agents.
                try:
                    resp = requests.get(step.source_url, timeout=30, headers={
                        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                                       'Chrome/120.0.0.0 Safari/537.36'),
                        'Accept': '*/*',
                        'Accept-Language': 'en-US,en;q=0.9',
                    })
                    resp.raise_for_status()
                    mime = (resp.headers.get('Content-Type', '').split(';')[0].strip()
                            or mimetypes.guess_type(filename)[0] or 'application/octet-stream')
                    result = self.wiki.upload_file(filename, resp.content, mime, description)
                except Exception as e:
                    result = {'success': False, 'error': f'Could not fetch {step.source_url}: {e}'}
            return result

        return {'success': False, 'error': 'No attached file or source URL for this upload — re-attach the document and try again'}

    def generate_step_preview(self, step: OperationStep) -> None:
        """Populate step.content (and diff/summary) without writing to wiki. Raises on failure."""
        instruction = step.description or step.summary
        if step.type in ('create', 'write'):
            data = self._generate_page_content(step.title, instruction)
            step.content = data['content']
            step.summary = data['summary']
            step.links_to = self.wiki.extract_links_from_content(step.content)
        elif step.type == 'edit':
            page = self.wiki.get_page(step.title)
            if not page.get('exists'):
                raise ValueError(f'Page "{step.title}" does not exist')
            step.old_content = page['content']
            step.content = self._edit_page_content(step.title, step.old_content, instruction)
            step.diff = _make_diff(step.old_content, step.content)
            step.summary = step.summary or f'Edit: {step.title}'
        elif step.type == 'add_image':
            self._prepare_add_image_content(step)

    def _execute_step_with_content(self, step: OperationStep) -> dict:
        """Generate content if needed, then execute the wiki write. Used in Phase 2."""
        if self.cancel_event.is_set():
            step.status = 'rejected'
            step.error = 'Cancelled'
            return {'success': False, 'error': 'Cancelled'}

        step.status = 'executing'
        self._emit({'type': 'step_start', 'step_id': step.id, 'title': step.title})
        try:
            instruction = step.description or step.summary

            if step.type in ('create', 'write'):
                if not step.content:
                    data = self._generate_page_content(step.title, instruction)
                    step.content = data['content']
                    step.summary = data['summary']
                    step.links_to = self.wiki.extract_links_from_content(step.content)
                result = self.wiki.write_page(step.title, step.content, step.summary or f'Create: {step.title}')

            elif step.type == 'edit':
                if not step.content:
                    page = self.wiki.get_page(step.title)
                    if not page.get('exists'):
                        raise ValueError(f'Page "{step.title}" does not exist')
                    step.old_content = page['content']
                    step.content = self._edit_page_content(step.title, page['content'], instruction)
                    step.diff = _make_diff(step.old_content, step.content)
                step.summary = step.summary or f'Edit: {step.title}'
                result = self.wiki.write_page(step.title, step.content, step.summary)

            elif step.type in ('replace',):
                result = self.wiki.write_page(step.title, step.content, step.summary)

            elif step.type == 'find_replace':
                result = self._execute_find_replace_step(step)

            elif step.type == 'delete':
                result = self.wiki.delete_page(step.title, instruction or f'Delete: {step.title}')

            elif step.type == 'move':
                result = self.wiki.move_page(
                    step.from_title, step.title,
                    instruction or f'Move: {step.from_title} → {step.title}'
                )

            elif step.type == 'add_image':
                if step.content:
                    result = self.wiki.write_page(step.title, step.content, step.summary)
                else:
                    result = self._execute_add_image_step(step)

            elif step.type == 'ensure_disambig':
                result = self._execute_ensure_disambig_step(step)

            elif step.type == 'upload_file':
                result = self._execute_upload_file_step(step)

            else:
                result = {'success': False, 'error': f'Unknown step type: {step.type}'}

            step.status = 'done' if result.get('success') else 'failed'
            if not result.get('success'):
                step.error = result.get('error', 'Unknown error')

        except Exception as e:
            step.status = 'failed'
            step.error = str(e)
            result = {'success': False, 'error': str(e)}

        self._emit({'type': 'step_done', 'step': step.to_dict()})
        return result

    # ── PHASE 2: EXECUTE PLAN ─────────────────────────────────────────────────

    def execute_plan(self, plan: OperationPlan, only_approved: bool = True) -> list[dict]:
        approved = [s for s in plan.steps if not only_approved or s.status == 'approved']
        if not approved:
            plan.status = 'done'
            self._emit({'type': 'done', 'plan_id': plan.id})
            return []

        step_map = {s.id: s for s in approved}
        # remaining_deps[step_id] = set of dep ids still to complete
        remaining_deps = {s.id: set(s.depends_on) & step_map.keys() for s in approved}
        completed: dict[str, bool] = {}  # step_id -> success
        results = []
        lock = threading.Lock()
        ready_q: queue.Queue = queue.Queue()

        for s in approved:
            if not remaining_deps[s.id]:
                ready_q.put(s)

        active_futures: dict[str, concurrent.futures.Future] = {}

        def _on_done(future, step):
            with lock:
                try:
                    result = future.result()
                except Exception as e:
                    result = {'success': False, 'error': str(e)}
                    step.status = 'failed'
                    step.error = str(e)

                success = result.get('success', False)
                completed[step.id] = success
                results.append({'step_id': step.id, **result})
                active_futures.pop(step.id, None)

                # Unblock dependent steps
                for other in approved:
                    if step.id in remaining_deps.get(other.id, set()):
                        remaining_deps[other.id].discard(step.id)
                        if not remaining_deps[other.id] and other.id not in completed and other.id not in active_futures:
                            # Cascade failure if a dependency failed
                            failed_deps = set(step_map[other.id].depends_on) & {s for s, ok in completed.items() if not ok}
                            if failed_deps:
                                other.status = 'failed'
                                other.error = 'A dependency step failed'
                                completed[other.id] = False
                                results.append({'step_id': other.id, 'success': False, 'error': 'Dependency failed'})
                                self._emit({'type': 'step_done', 'step': other.to_dict()})
                            else:
                                ready_q.put(other)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Drain ready queue and submit, then wait for callbacks to feed more
            while True:
                if self.cancel_event.is_set():
                    break

                # Submit all currently ready steps
                submitted_any = False
                while True:
                    try:
                        step = ready_q.get_nowait()
                        with lock:
                            if step.id not in completed and step.id not in active_futures:
                                future = executor.submit(self._execute_step_with_content, step)
                                active_futures[step.id] = future
                                future.add_done_callback(lambda f, s=step: _on_done(f, s))
                                submitted_any = True
                    except queue.Empty:
                        break

                with lock:
                    all_accounted = len(completed) + len(active_futures) >= len(approved)
                    still_running = bool(active_futures)

                if all_accounted and not still_running:
                    break

                time.sleep(0.05)

        # Mark any unstarted approved steps as cancelled
        with lock:
            for step in approved:
                if step.id not in completed and step.id not in active_futures:
                    step.status = 'rejected'
                    step.error = 'Cancelled'
                    results.append({'step_id': step.id, 'success': False, 'error': 'Cancelled'})

        done_count = sum(1 for r in results if r.get('success'))
        total_count = len(results)
        plan.status = 'done' if done_count == total_count else ('partial' if done_count > 0 else 'failed')
        self._emit({'type': 'done', 'plan_id': plan.id})
        return results

    # ── BACKWARD-COMPAT EXECUTE (single step, content already set) ────────────

    def execute_step(self, step: OperationStep) -> dict:
        """Execute a single step whose content is already populated."""
        step.status = 'executing'
        try:
            if step.type in ('write', 'edit', 'replace', 'add_image', 'create'):
                result = self.wiki.write_page(step.title, step.content, step.summary)
            elif step.type == 'move':
                result = self.wiki.move_page(step.from_title, step.title, step.summary)
            elif step.type == 'delete':
                result = self.wiki.delete_page(step.title, step.summary)
            elif step.type == 'upload_file':
                result = self._execute_upload_file_step(step)
            else:
                result = {'success': False, 'error': f'Unknown step type: {step.type}'}
            step.status = 'done' if result.get('success') else 'failed'
            if not result.get('success'):
                step.error = result.get('error', 'Unknown error')
            return result
        except Exception as e:
            step.status = 'failed'
            step.error = str(e)
            return {'success': False, 'error': str(e)}
