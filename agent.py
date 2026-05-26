import json
import re
import uuid
import difflib
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

from wiki_client import WikiClient


# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

PLANNER_PREFIX = """You are a wiki management agent for a MediaWiki wiki. Your job is to take natural language instructions and produce structured operation plans.

When asked to generate wiki pages, write complete, well-structured MediaWiki markup including:
- A lead paragraph summarising the topic
- Sections with == Heading == markers
- Internal [[wikilinks]] to related topics
- [[Category:Relevant Category]] tags at the end
- Inline citations and factual depth appropriate for a reference wiki

Operation types you handle:
- generate_pages: Create new wiki pages from scratch
- generate_recursive: Create a seed page then generate sub-pages for its links
- find_replace: Bulk text substitution across wiki pages
- ensure_disambig: Create missing disambiguation/redirect pages for abbreviations
- rename_pages: Move pages to correct titles (preserves history)
- edit_pages: Modify existing page content
- audit_pages: Read-only analysis, returns a report

When returning JSON plans, you MUST return ONLY valid JSON with no preamble, markdown fences, or explanation. The JSON must match the exact schema provided.
"""

GENERATE_SCHEMA = """Return a JSON object with this exact structure:
{
  "description": "one-sentence summary of what this plan does",
  "steps": [
    {
      "title": "Page Title",
      "content": "full MediaWiki markup here",
      "summary": "one-line edit summary",
      "links_to": ["Linked Page 1", "Linked Page 2"]
    }
  ]
}"""

EDIT_SCHEMA = """Return a JSON object with this exact structure:
{
  "description": "one-sentence summary of what was changed",
  "content": "complete revised MediaWiki markup",
  "summary": "one-line edit summary"
}"""

RENAME_SCHEMA = """Return a JSON object with this exact structure:
{
  "description": "one-sentence summary of the renames",
  "renames": [
    {"from_title": "Old Title", "to_title": "New Title", "reason": "why"}
  ]
}"""

AUDIT_SCHEMA = """Return a JSON object with this exact structure:
{
  "description": "your full audit report here as a detailed multi-paragraph string"
}"""

FIND_REPLACE_SCHEMA = """Return a JSON object with this exact structure:
{
  "find": "exact string to find",
  "replace": "replacement string"
}"""

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
    status: str = 'pending'
    error: Optional[str] = None
    links_to: list = field(default_factory=list)
    diff: Optional[str] = None

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
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _recover_generate_json(text: str) -> dict:
    """Extract whatever complete steps exist from a truncated generate response."""
    desc_match = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    description = desc_match.group(1) if desc_match else 'Plan (response truncated — partial results shown)'

    steps = []
    steps_match = re.search(r'"steps"\s*:\s*\[', text)
    if not steps_match:
        return {'description': description, 'steps': steps}

    decoder = json.JSONDecoder()
    pos = steps_match.end()
    while pos < len(text):
        while pos < len(text) and text[pos] in ' \t\n\r,':
            pos += 1
        if pos >= len(text) or text[pos] == ']':
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
            steps.append(obj)
        except json.JSONDecodeError:
            break

    return {'description': description, 'steps': steps}


def _make_diff(old: str, new: str) -> str:
    return '\n'.join(difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        lineterm='',
        fromfile='before',
        tofile='after',
    ))


def _detect_operation_type(instruction: str) -> str:
    lower = instruction.lower()
    if any(k in lower for k in ['replace', 'find and replace', 'substitute']):
        return 'find_replace'
    if any(k in lower for k in ['disambig', 'redirect', 'abbreviation', 'acronym']):
        return 'ensure_disambig'
    if any(k in lower for k in ['rename', ' move ', 'typo', 'misspell']):
        return 'rename_pages'
    if any(k in lower for k in ['recursive', 'sub-page', 'subpage', 'follow links', 'depth']):
        return 'generate_recursive'
    if any(k in lower for k in ['edit ', 'update ', 'expand ', 'rewrite ', 'fix ', 'add to ']):
        return 'edit_pages'
    if any(k in lower for k in ['audit', 'analyse', 'analyze', 'check all', 'review all', 'list all', 'which pages']):
        return 'audit_pages'
    if any(k in lower for k in ['copy', 'duplicate', 'clone']):
        return 'generate_pages'
    return 'generate_pages'


# ─── AGENT ───────────────────────────────────────────────────────────────────

class WikiAgent:
    def __init__(self, wiki: WikiClient, anthropic_client, system_prompt: str, connection_id: str):
        self.wiki = wiki
        self.ai = anthropic_client
        self.connection_id = connection_id
        self._system_blocks = [{
            'type': 'text',
            'text': PLANNER_PREFIX + ('\n\n' + system_prompt if system_prompt else ''),
            'cache_control': {'type': 'ephemeral'},
        }]
        self._stream_callback: Optional[Callable] = None

    def _call_ai(self, user_message: str, max_tokens: int = 8096) -> str:
        response = self.ai.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=max_tokens,
            system=self._system_blocks,
            messages=[{'role': 'user', 'content': user_message}],
        )
        return response.content[0].text

    def _build_context_prefix(self, context_pages: list[dict]) -> str:
        if not context_pages:
            return ''
        lines = []
        referenced = [p for p in context_pages if p.get('is_referenced')]
        general = [p for p in context_pages if not p.get('is_referenced')]

        if referenced:
            lines.append('REFERENCED WIKI PAGES (explicitly named in the instruction — use their content as source material):')
            for p in referenced:
                lines.append(f"=== {p['title']} ===\n{p.get('content', '')}")

        if general:
            lines.append('EXISTING WIKI PAGES FOR CONTEXT:')
            for p in general[:5]:
                lines.append(f"=== {p['title']} ===\n{p.get('content', '')}")

        return '\n\n'.join(lines) + '\n\n---\n\n'

    def _emit(self, event: dict):
        if self._stream_callback:
            self._stream_callback(event)

    def _detect_referenced_pages(self, instruction: str) -> list[str]:
        """Return page titles explicitly named in quotes within the instruction."""
        matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', instruction)
        return [title for pair in matches for title in pair if title]

    # ── PLAN ──────────────────────────────────────────────────────────────────

    def plan(self, instruction: str, operation_type: str | None = None,
             context_pages: list[dict] | None = None) -> OperationPlan:
        context_pages = list(context_pages or [])

        # Auto-fetch pages explicitly referenced by quoted title in the instruction
        referenced_titles = self._detect_referenced_pages(instruction)
        if referenced_titles:
            existing_titles = {p['title'] for p in context_pages}
            for title in referenced_titles:
                if title not in existing_titles:
                    page = self.wiki.get_page(title)
                    if page.get('exists'):
                        context_pages.insert(0, {
                            'title': title,
                            'content': page.get('content', ''),
                            'is_referenced': True,
                        })

        if not operation_type or operation_type == 'auto':
            operation_type = _detect_operation_type(instruction)

        planners = {
            'generate_pages': self._plan_generate_pages,
            'generate_recursive': self._plan_generate_recursive,
            'find_replace': self._plan_find_replace,
            'ensure_disambig': self._plan_ensure_disambig,
            'rename_pages': self._plan_rename_pages,
            'edit_pages': self._plan_edit_pages,
            'audit_pages': self._plan_audit_pages,
        }
        planner = planners.get(operation_type, self._plan_generate_pages)

        plan = OperationPlan(connection_id=self.connection_id, operation_type=operation_type)
        steps, description = planner(instruction, context_pages)
        plan.steps = steps
        plan.description = description
        self._emit({'type': 'done', 'plan_id': plan.id})
        return plan

    def _plan_generate_pages(self, instruction: str, context_pages: list[dict]) -> tuple[list, str]:
        prefix = self._build_context_prefix(context_pages)
        prompt = (
            f"{prefix}INSTRUCTION: {instruction}\n\n"
            f"{GENERATE_SCHEMA}"
        )
        raw = self._call_ai(prompt, max_tokens=64000)
        try:
            data = _extract_json(raw)
        except json.JSONDecodeError:
            data = _recover_generate_json(raw)
        steps = []
        for s in data.get('steps', []):
            content = s.get('content', '')
            links = s.get('links_to') or self.wiki.extract_links_from_content(content)
            step = OperationStep(
                type='write',
                title=s['title'],
                content=content,
                summary=s.get('summary', f'Create page: {s["title"]}'),
                links_to=links,
            )
            steps.append(step)
            self._emit({'type': 'step', 'step': step.to_dict()})
        return steps, data.get('description', f'Generate pages for: {instruction}')

    def _generate_single_page(self, title: str, context: dict[str, str],
                               parent_context: list[dict]) -> dict:
        parts = []
        parent_prefix = self._build_context_prefix(parent_context)
        if parent_prefix:
            parts.append(parent_prefix)

        if context:
            ctx_lines = ['PREVIOUSLY GENERATED PAGES (use as context and link to them):']
            for t, c in list(context.items())[-5:]:
                ctx_lines.append(f"=== {t} ===\n{c[:1000]}")
            parts.append('\n\n'.join(ctx_lines))

        parts.append(
            f"Generate a complete wiki page for the topic: **{title}**\n\n"
            f'Return JSON: {{"title": "{title}", "content": "<full wikitext>", '
            f'"summary": "one-line edit summary", "links_to": ["link1", "link2"]}}'
        )
        prompt = '\n\n'.join(parts)

        raw = self._call_ai(prompt, max_tokens=4096)
        return _extract_json(raw)

    def _plan_generate_recursive(self, instruction: str, context_pages: list[dict],
                                  max_depth: int = 2, max_pages: int = 20) -> tuple[list, str]:
        generated: dict[str, str] = {}
        queue: list[tuple[str, int]] = [(instruction, 0)]
        visited: set[str] = set()
        steps: list[OperationStep] = []

        while queue and len(steps) < max_pages:
            topic, depth = queue.pop(0)
            title = topic.strip()
            if title in visited:
                continue
            if self.wiki.page_exists(title):
                visited.add(title)
                continue
            visited.add(title)

            try:
                page = self._generate_single_page(title, generated, context_pages)
            except Exception as e:
                self._emit({'type': 'error', 'message': f'Failed to generate {title}: {e}'})
                continue

            content = page.get('content', '')
            generated[title] = content
            links = page.get('links_to') or self.wiki.extract_links_from_content(content)

            step = OperationStep(
                type='write',
                title=title,
                content=content,
                summary=page.get('summary', f'Create page: {title}'),
                links_to=links,
            )
            steps.append(step)
            self._emit({'type': 'step', 'step': step.to_dict()})

            if depth < max_depth:
                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))

        desc = f'Recursive generation from "{instruction}": {len(steps)} pages'
        return steps, desc

    def _plan_find_replace(self, instruction: str, context_pages: list[dict]) -> tuple[list, str]:
        prompt = (
            f"Extract the find and replace strings from this instruction: {instruction}\n\n"
            f"{FIND_REPLACE_SCHEMA}"
        )
        data = _extract_json(self._call_ai(prompt, max_tokens=512))
        find_term = data['find']
        replace_term = data['replace']

        results = self.wiki.search(find_term, limit=100)
        steps = []
        for result in results:
            page = self.wiki.get_page(result['title'])
            if not page.get('content') or find_term not in page['content']:
                continue
            new_content = page['content'].replace(find_term, replace_term)
            diff = _make_diff(page['content'], new_content)
            count = page['content'].count(find_term)
            step = OperationStep(
                type='replace',
                title=result['title'],
                content=new_content,
                old_content=page['content'],
                summary=f'Find/replace: "{find_term}" → "{replace_term}" ({count} occurrence{"s" if count != 1 else ""})',
                diff=diff,
            )
            steps.append(step)
            self._emit({'type': 'step', 'step': step.to_dict()})

        desc = f'Replace "{find_term}" → "{replace_term}" across {len(steps)} pages'
        return steps, desc

    def _plan_ensure_disambig(self, instruction: str, context_pages: list[dict]) -> tuple[list, str]:
        abbr_pattern = re.findall(r'\b([A-Z]{2,8}(?:s)?)\b', instruction)
        if not abbr_pattern:
            prompt = (
                f"Extract all abbreviations from this instruction: {instruction}\n\n"
                "Return JSON: {\"abbreviations\": [\"ABBR1\", \"ABBR2\"]}"
            )
            data = _extract_json(self._call_ai(prompt, max_tokens=512))
            abbr_pattern = data.get('abbreviations', [])

        all_pages = set(self.wiki.get_all_pages())
        steps = []
        for abbr in abbr_pattern:
            if abbr in all_pages:
                continue
            matching = [t for t in all_pages if abbr.lower() in t.lower().split()]
            if len(matching) == 1:
                content = DISAMBIG_TEMPLATE.format(target=matching[0])
                summary = f'Create redirect: {abbr} → {matching[0]}'
            elif len(matching) > 1:
                items = '\n'.join(f'* [[{t}]]' for t in matching[:10])
                content = DISAMBIG_LISTING_TEMPLATE.format(abbr=abbr, items=items)
                summary = f'Create disambiguation page for {abbr}'
            else:
                content = DISAMBIG_TEMPLATE.format(target=abbr + ' (disambiguation)')
                summary = f'Create placeholder redirect for {abbr}'

            step = OperationStep(
                type='write',
                title=abbr,
                content=content,
                summary=summary,
            )
            steps.append(step)
            self._emit({'type': 'step', 'step': step.to_dict()})

        desc = f'Ensure disambiguation pages for: {", ".join(abbr_pattern)}'
        return steps, desc

    def _plan_rename_pages(self, instruction: str, context_pages: list[dict]) -> tuple[list, str]:
        all_pages = self.wiki.get_all_pages()
        pages_list = '\n'.join(f'- {t}' for t in all_pages[:200])
        prefix = self._build_context_prefix(context_pages)
        prompt = (
            f"{prefix}"
            f"Existing wiki pages:\n{pages_list}\n\n"
            f"INSTRUCTION: {instruction}\n\n"
            f"{RENAME_SCHEMA}"
        )
        data = _extract_json(self._call_ai(prompt))
        steps = []
        for r in data.get('renames', []):
            step = OperationStep(
                type='move',
                title=r['to_title'],
                from_title=r['from_title'],
                summary=r.get('reason', f'Rename: {r["from_title"]} → {r["to_title"]}'),
            )
            steps.append(step)
            self._emit({'type': 'step', 'step': step.to_dict()})
        return steps, data.get('description', f'Rename pages per: {instruction}')

    def _plan_edit_pages(self, instruction: str, context_pages: list[dict]) -> tuple[list, str]:
        # Determine target pages from instruction
        prompt = (
            f"From this edit instruction, extract the wiki page title(s) to edit: {instruction}\n\n"
            "Return JSON: {{\"titles\": [\"Page Title 1\", \"Page Title 2\"]}}"
        )
        data = _extract_json(self._call_ai(prompt, max_tokens=512))
        titles = data.get('titles', [])

        prefix = self._build_context_prefix(context_pages)
        steps = []
        descriptions = []

        for title in titles:
            page = self.wiki.get_page(title)
            if not page.get('exists'):
                continue

            edit_prompt = (
                f"{prefix}"
                f"Current content of [[{title}]]:\n\n{page['content']}\n\n"
                f"INSTRUCTION: {instruction}\n\n"
                f"{EDIT_SCHEMA}"
            )
            edit_data = _extract_json(self._call_ai(edit_prompt))
            new_content = edit_data.get('content', page['content'])
            diff = _make_diff(page['content'], new_content)

            step = OperationStep(
                type='edit',
                title=title,
                content=new_content,
                old_content=page['content'],
                summary=edit_data.get('summary', f'Edit: {title}'),
                diff=diff,
            )
            steps.append(step)
            descriptions.append(edit_data.get('description', f'Edited {title}'))
            self._emit({'type': 'step', 'step': step.to_dict()})

        desc = '; '.join(descriptions) if descriptions else f'Edit pages per: {instruction}'
        return steps, desc

    def _plan_audit_pages(self, instruction: str, context_pages: list[dict]) -> tuple[list, str]:
        all_titles = self.wiki.get_all_pages()
        pages_list = '\n'.join(f'- {t}' for t in all_titles)
        prefix = self._build_context_prefix(context_pages)
        prompt = (
            f"{prefix}"
            f"All wiki pages:\n{pages_list}\n\n"
            f"INSTRUCTION: {instruction}\n\n"
            f"{AUDIT_SCHEMA}"
        )
        data = _extract_json(self._call_ai(prompt, max_tokens=4096))
        return [], data.get('description', 'Audit complete.')

    # ── EXECUTE ───────────────────────────────────────────────────────────────

    def execute_step(self, step: OperationStep) -> dict:
        step.status = 'executing'
        try:
            if step.type in ('write', 'edit', 'replace'):
                result = self.wiki.write_page(step.title, step.content, step.summary)
            elif step.type == 'move':
                result = self.wiki.move_page(step.from_title, step.title, step.summary)
            elif step.type == 'delete':
                result = self.wiki.delete_page(step.title, step.summary)
            else:
                result = {'success': False, 'error': f'Unknown step type: {step.type}'}

            if result.get('success'):
                step.status = 'done'
            else:
                step.status = 'failed'
                step.error = result.get('error', 'Unknown error')
            return result
        except Exception as e:
            step.status = 'failed'
            step.error = str(e)
            return {'success': False, 'error': str(e)}

    def execute_plan(self, plan: OperationPlan, only_approved: bool = True) -> list[dict]:
        results = []
        for step in plan.steps:
            if only_approved and step.status != 'approved':
                continue
            result = self.execute_step(step)
            results.append({'step_id': step.id, **result})

        done = sum(1 for r in results if r.get('success'))
        if done == len(results):
            plan.status = 'done'
        elif done > 0:
            plan.status = 'partial'
        else:
            plan.status = 'failed'
        return results
