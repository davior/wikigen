import concurrent.futures
import json
import queue
import re
import threading
import time
import uuid
import difflib
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

from wiki_client import WikiClient


# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

PLANNER_PREFIX = """You are a wiki management agent for a MediaWiki wiki.

You receive:
1. A full index of all current wiki pages grouped by category (in the system context)
2. Optionally, contents of specific wiki pages selected for reference (in the system context)
3. A natural language instruction (in the user message)

Analyse the instruction relative to the existing wiki structure and return a structured plan.

Step types you may produce:
- create: Create a new page from scratch (check the index — do not create pages that already exist)
- edit: Modify an existing page (must exist in the index)
- delete: Delete a page (must exist in the index)
- move: Rename a page; title = destination, from_title = source
- find_replace: Bulk text replacement across the wiki; title = "*"; describe find/replace pairs in description
- ensure_disambig: Create a redirect or disambiguation page for an abbreviation
- add_image: Source an image from Wikimedia Commons and embed it into an existing page

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
      "type": "create|edit|delete|move|find_replace|ensure_disambig|add_image",
      "title": "Page Title",
      "from_title": "Source Title",
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
- Use the site index to avoid creating pages that already exist
- Use depends_on when order matters (e.g. create pages before deleting source)"""

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


def _format_site_index(pages_with_cats: dict[str, list[str]]) -> str:
    if not pages_with_cats:
        return 'SITE INDEX: (empty wiki — no pages yet)'
    lines = [f'SITE INDEX: {len(pages_with_cats)} pages\n']
    by_cat: dict[str, list[str]] = {}
    uncategorized = []
    for title, cats in pages_with_cats.items():
        if cats:
            for cat in cats:
                by_cat.setdefault(cat, []).append(title)
        else:
            uncategorized.append(title)
    for cat, titles in sorted(by_cat.items()):
        lines.append(f'[{cat}]')
        lines.extend(f'  {t}' for t in sorted(titles))
        lines.append('')
    if uncategorized:
        lines.append('[Uncategorized]')
        lines.extend(f'  {t}' for t in sorted(uncategorized))
    return '\n'.join(lines)


_FILE_REF_RE = re.compile(r'\[\[(?:File|Image):([^|\]]+)(\|[^\]]*)?(\]\])', re.IGNORECASE)


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


# ─── AGENT ───────────────────────────────────────────────────────────────────

class WikiAgent:
    def __init__(self, wiki: WikiClient, anthropic_client, system_prompt: str,
                 connection_id: str, site_index: dict | None = None,
                 context_pages: list | None = None):
        self.wiki = wiki
        self.ai = anthropic_client
        self.connection_id = connection_id
        self.cancel_event = threading.Event()
        self._stream_callback: Optional[Callable] = None

        blocks = [{
            'type': 'text',
            'text': PLANNER_PREFIX + ('\n\n' + system_prompt if system_prompt else ''),
            'cache_control': {'type': 'ephemeral'},
        }]
        if site_index:
            blocks.append({
                'type': 'text',
                'text': _format_site_index(site_index),
                'cache_control': {'type': 'ephemeral'},
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

    def _call_ai(self, user_message: str, max_tokens: int = 8096) -> str:
        with self.ai.messages.stream(
            model='claude-sonnet-4-6',
            max_tokens=max_tokens,
            system=self._system_blocks,
            messages=[{'role': 'user', 'content': user_message}],
        ) as stream:
            return stream.get_final_text()

    def _build_context_prefix(self, context_pages: list[dict]) -> str:
        if not context_pages:
            return ''
        lines = []
        referenced = [p for p in context_pages if p.get('is_referenced')]
        general = [p for p in context_pages if not p.get('is_referenced')]
        if referenced:
            lines.append('REFERENCED WIKI PAGES (use their content as source material):')
            for p in referenced:
                lines.append(f"=== {p['title']} ===\n{p.get('content', '')}")
        if general:
            lines.append('ADDITIONAL CONTEXT PAGES:')
            for p in general[:5]:
                lines.append(f"=== {p['title']} ===\n{p.get('content', '')}")
        return '\n\n'.join(lines) + '\n\n---\n\n'

    def _emit(self, event: dict):
        if self._stream_callback:
            self._stream_callback(event)

    def _detect_referenced_pages(self, instruction: str) -> list[str]:
        matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', instruction)
        return [title for pair in matches for title in pair if title]

    # ── PHASE 1: GENERATE PLAN ────────────────────────────────────────────────

    def generate_plan(self, instruction: str) -> OperationPlan:
        plan = OperationPlan(connection_id=self.connection_id, operation_type='auto')

        # Build user message: auto-fetch referenced pages + links, then instruction
        parts = []
        referenced_titles = self._detect_referenced_pages(instruction)
        if referenced_titles:
            ref_sections = []
            for title in referenced_titles:
                page = self.wiki.get_page(title)
                if page.get('exists'):
                    section = f"=== {title} ===\n{page.get('content', '(empty)')}"
                    # Include outgoing links for "missing links" style queries
                    try:
                        outgoing = self.wiki.get_links_from_page(title)
                        if outgoing:
                            section += f"\n\nLinks from this page: {', '.join(outgoing[:100])}"
                    except Exception:
                        pass
                    ref_sections.append(section)
            if ref_sections:
                parts.append('REFERENCED PAGES:\n\n' + '\n\n'.join(ref_sections))

        parts.append(f'INSTRUCTION: {instruction}')
        parts.append(PLAN_SCHEMA)
        user_message = '\n\n'.join(parts)

        try:
            raw = self._call_ai(user_message, max_tokens=16000)
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
            )
            plan.steps.append(step)
            self._emit({'type': 'step', 'step': step.to_dict()})

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
            '- [[Category:...]] tags at the end'
        )
        content = self._call_ai(prompt, max_tokens=8096)
        content = _fix_file_references(content, self.wiki)
        return {
            'content': content,
            'summary': f'Create: {title}',
        }

    def _edit_page_content(self, title: str, current_content: str, instructions: str) -> str:
        prompt = (
            f'Current content of [[{title}]]:\n\n{current_content}\n\n'
            f'Instruction: {instructions}\n\n'
            'Return the complete revised MediaWiki wikitext only (no JSON, no explanation).'
        )
        new_content = self._call_ai(prompt, max_tokens=8096)
        return _fix_file_references(new_content, self.wiki)

    def _execute_find_replace_step(self, step: OperationStep) -> dict:
        # Extract pairs from step description via AI
        prompt = (
            f'Extract find-and-replace pairs from this instruction: {step.description}\n\n'
            'Return JSON: {"replacements": [{"find": "...", "replace": "..."}]}'
        )
        try:
            data = _extract_json(self._call_ai(prompt, max_tokens=512))
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

    def _execute_add_image_step(self, step: OperationStep) -> dict:
        page = self.wiki.get_page(step.title)
        if not page.get('exists') or not page.get('content'):
            return {'success': False, 'error': f'Page "{step.title}" not found or empty'}

        # Use page title as the primary Commons search term; description guides caption/placement
        results = []
        for query in [step.title, step.title.split()[0] if ' ' in step.title else None]:
            if not query:
                continue
            try:
                results = self.wiki.search_commons_images(query, limit=5)
            except Exception as e:
                return {'success': False, 'error': f'Commons search failed: {e}'}
            if results:
                break

        if not results:
            return {'success': False, 'error': f'No images found on Commons for "{step.title}"'}

        filenames_list = '\n'.join(f'{i + 1}. {r["filename"]}' for i, r in enumerate(results))
        context_hint = f'\n\nContext: {step.description[:300]}' if step.description else ''
        pick_prompt = (
            f'For the wiki page "{step.title}", pick the most relevant image:\n'
            f'{filenames_list}{context_hint}\n\n'
            f'Return JSON: {{"index": 1, "caption": "caption text", "placement": "after_lead"}}'
        )
        try:
            pick_data = _extract_json(self._call_ai(pick_prompt, max_tokens=256))
            idx = max(0, min(int(pick_data.get('index', 1)) - 1, len(results) - 1))
            caption = pick_data.get('caption', step.title)
            placement = pick_data.get('placement', 'after_lead')
        except Exception:
            idx, caption, placement = 0, step.title, 'after_lead'
        chosen = results[idx]
        new_content = _insert_image(page['content'], chosen['filename'], caption, placement)
        step.content = new_content
        step.old_content = page['content']
        step.diff = _make_diff(page['content'], new_content)
        step.image_file = chosen['filename']
        step.commons_url = chosen['commons_url']
        return self.wiki.write_page(step.title, new_content, f'Add image: {chosen["filename"]}')

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
