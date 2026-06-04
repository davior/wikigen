# WikiGen v3

A Flask web application for managing MediaWiki wikis via AI. Enter a natural language instruction, review the generated operation plan, approve steps, and execute — all from a single browser interface.

![WikiGen v3](https://img.shields.io/badge/version-3.0-00d4ff?style=flat-square) ![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square) ![Flask](https://img.shields.io/badge/flask-3.1-green?style=flat-square) ![License](https://img.shields.io/badge/license-GPLv3-orange?style=flat-square)

---

## Features

- **7 operation types** — Generate, Recursive Generate, Edit, Find & Replace, Disambiguate, Rename, Audit
- **Plan → Review → Execute** — AI builds a step-by-step plan; you approve before anything touches the wiki
- **Live streaming** — Recursive generation streams steps to the UI in real time via SSE
- **Multi-wiki support** — Manage multiple MediaWiki instances via the connections manager
- **Context injection** — Load existing wiki pages as context for more coherent AI generation
- **Diff viewer** — Coloured before/after diff for Edit and Find & Replace operations
- **Knowledge graph** — D3.js force-directed graph showing page links
- **Wiki search** — Search the live wiki and load pages directly into the editor
- **Prompt caching** — Anthropic API prompt caching reduces cost and latency on repeated calls

---

## Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)
- A MediaWiki instance with a bot account (see [Bot Setup](#bot-setup) below)

---

## Installation

Create and activate a virtual environment before installing dependencies:

```bash
git clone https://github.com/davior/wikigen.git
cd wikigen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you come back later and `requirements.txt` has changed, reactivate the virtual environment and reinstall the dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer not to use a virtual environment, you can still install the requirements globally, but a venv is recommended.

---

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `FLASK_SECRET` | Recommended | Random string for Flask session signing |
| `DATA_DIR` | No | Directory for `connections.json` and `history.json` (default: `.`) |
| `PORT` | No | Port to listen on (default: `5055`) |
| `WIKI_URL` | No | Fallback wiki API URL if no connections are configured |
| `WIKI_USERNAME` | No | Fallback wiki username |
| `WIKI_PASSWORD` | No | Fallback wiki password |
| `WIKI_NAME` | No | Display name for the fallback connection |

Connections are managed through the UI and saved to `connections.json` — you don't need to set the wiki env vars if you add connections via the browser.

---

## Running

```bash
python app.py
# → http://localhost:5055
```

For production (Synology NAS or similar):

```bash
pip install gunicorn
gunicorn -k gthread -w 1 --threads 4 -b 0.0.0.0:5055 app:app
```

---

## Bot Setup

WikiGen uses the [MediaWiki Action API](https://www.mediawiki.org/wiki/API:Main_page) with bot credentials.

1. Log in to your wiki as an administrator
2. Go to **Special:BotPasswords**
3. Create a new bot with these permissions:
   - **Edit existing pages**
   - **Create, edit, and move pages**
   - **Move pages** (required for Rename operations)
   - **Delete pages** (optional, for Delete operations)
4. Use the format `Username@BotPasswordName` as the username in WikiGen

The wiki API URL should point to `api.php`, e.g.:
```
https://yourwiki.example.com/w/api.php
```

---

## Operation Types

| Type | What it does | Example instruction |
|---|---|---|
| **Auto** | Detects type from your instruction | — |
| **Generate** | Creates new pages from scratch | *"Create pages covering Transhumanism, the Singularity, and key figures"* |
| **Recursive** | Generates a seed page, then sub-pages for every link | *"Create a page on DEWs and follow all links two levels deep"* |
| **Edit** | Modifies existing page content | *"Expand the introduction on the Nanotechnology page"* |
| **Find & Replace** | Bulk text substitution across all pages | *"Replace 'Nanotech' with 'Nanotechnology' everywhere"* |
| **Disambig** | Creates redirect/disambiguation pages for abbreviations | *"Ensure disambiguation pages exist for DEW, V2K, RNM, TI, NWO"* |
| **Rename** | Moves pages, preserving edit history | *"Rename 'Lucerferianism' to 'Luciferianism' (fix the typo)"* |
| **Audit** | Read-only analysis, returns a report | *"Which pages are stubs and what topics are missing?"* |

---

## UI Overview

```
┌─────────────────┬──────────────────────────────┬──────────────────┐
│  LEFT PANEL     │  CENTRE PANEL                │  RIGHT PANEL     │
│                 │  [ PLAN ] [ GRAPH ] [ SEARCH]│                  │
│  Op type chips  │                              │  PREVIEW         │
│  Instruction    │  Step list with              │  EDIT MARKUP     │
│  textarea       │  approve/reject controls     │  DIFF            │
│                 │                              │  LINKS           │
│  Context badge  │  D3 knowledge graph          │                  │
│                 │                              │  + CONTEXT btn   │
│  PLAN button    │  Wiki search + load          │  ↗ WIKI btn      │
└─────────────────┴──────────────────────────────┴──────────────────┘
```

### Workflow

1. Select an **operation type** chip (or leave on AUTO)
2. Type your **instruction** in the textarea
3. Click **PLAN OPERATION** — the AI builds a step list
4. Review steps in the **PLAN tab**, click steps to preview content or diffs
5. Click **APPROVE ALL** (or approve individual steps)
6. Click **EXECUTE APPROVED** to write to the wiki

### Context Feature

Load existing wiki pages as context to improve generation quality:

1. Use the **SEARCH tab** to find an existing page
2. Click **+ CONTEXT** on the result, or open it in the editor and click the **+ CONTEXT** button
3. Loaded context pages are injected into every AI call until you clear them
4. The context badge in the left panel shows how many pages are loaded

---

## File Structure

```
wikigen/
├── app.py              # Flask backend — all routes and connections manager
├── wiki_client.py      # MediaWiki API client (auth, CRUD, search, pagination)
├── agent.py            # AI planner + executor, OperationStep/Plan dataclasses
├── requirements.txt
├── .gitignore
├── connections.json    # Created at runtime — saved wiki connections
├── history.json        # Created at runtime — completed operation log
├── plans/              # Created at runtime — persisted plan JSON files
└── templates/
    └── index.html      # Single-page frontend (vanilla JS + D3.js)
```

---

## API Reference

### Agent routes

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/agent/plan` | Build an operation plan (returns steps or `{status: "running"}` for recursive) |
| `GET` | `/api/agent/plan/stream/<plan_id>` | SSE stream for recursive generation progress |
| `GET` | `/api/agent/plan/<plan_id>` | Fetch a plan by ID |
| `POST` | `/api/agent/execute_step` | Execute a single step |
| `POST` | `/api/agent/execute_plan` | Execute all approved steps in a plan |

### Wiki read routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/wiki/page?title=X` | Fetch a page's content and metadata |
| `GET` | `/api/wiki/search?term=X` | Search wiki content |
| `GET` | `/api/wiki/all_pages` | List all page titles |

### Connections

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/connections` | List all connections |
| `POST` | `/api/connections` | Add a connection |
| `PUT` | `/api/connections/<id>` | Update a connection |
| `DELETE` | `/api/connections/<id>` | Delete a connection |
| `POST` | `/api/connections/<id>/activate` | Set active connection |
| `POST` | `/api/connections/<id>/test` | Test authentication |

---

## Development Notes

- **Recursive generation** is the only async operation — it runs in a background thread and streams steps via Server-Sent Events. All other operations block until complete.
- **Rate limiting** — WikiGen enforces a 1-second minimum between wiki write operations to stay within MediaWiki's default bot rate limit.
- **CSRF tokens** — Automatically refreshed on `badtoken` errors; no manual intervention needed.
- **Prompt caching** — The Anthropic system prompt uses `cache_control: ephemeral`. The stable blocks (planner rules + site index) are cached for **1 hour**; per-request context keeps the default 5-minute TTL — giving ~90% cost reduction on repeated calls.
- **Frozen site index** — The list of all pages (with categories and short descriptions) is built once (auto-populated on first use), stored per connection in a local sidecar file (`site_index_<id>.json`) and mirrored to a JSON page in the wiki (`User:<bot>/wikigen-index.json`, override per connection with `index_page`). It is then reused **byte-for-byte** on every operation so the planner's prompt-cache block stays warm across a whole content-generation session — instead of re-scanning `allpages` (and busting the cache) each time. It only rebuilds when you press **REINDEX**. Pages created mid-session are tracked separately and surfaced to the planner in the uncached prompt tail (so it won't recreate them) without touching the cached block. A cheap `recentchanges` check powers a passive "wiki changed since last refresh" hint in the connections UI. See `site_index.py`.
- **Plan persistence** — Plans are saved to `plans/<id>.json` on completion and survive server restarts.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
