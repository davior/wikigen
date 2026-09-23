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
- **AI illustrations** — Pages get 2–3 images generated with [fal.ai](https://fal.ai) in a consistent house style for each wiki, placed where they fit the content

---

## Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)
- Optional: a [fal.ai API key](https://fal.ai/dashboard/keys) for image generation
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
| `FAL_KEY` | No | fal.ai key used by connections that don't have their own (see [Image Generation](#image-generation)) |

Connections are managed through the UI and saved to `connections.json` — you don't need to set the wiki env vars if you add connections via the browser.

---

## Running

```bash
python app.py
# → http://localhost:5055
```

For production, run under gunicorn with a **single worker**. Plans and SSE queues live in process memory, so add threads, not workers:

```bash
pip install gunicorn
gunicorn -k gthread -w 1 --threads 8 --timeout 300 -b 0.0.0.0:5055 app:app
```

---

## Deploying on Synology (auto-updating)

Every push to `main` builds a Docker image and publishes it to `ghcr.io/davior/wikigen` (`.github/workflows/docker-publish.yml`). On the NAS, a [Watchtower](https://containrrr.dev/watchtower/) container picks up the new image and restarts WikiGen on it. Use the Watchtower you already run, or add one (below); run only one per Docker host, since a second unscoped instance shuts the first down. After the one-time setup, merging a PR is all it takes to deploy. The NAS needs no git access.

Requires DSM 7.2+ with **Container Manager** on an x86 NAS.

### One-time setup

1. **Folder.** In File Station, create `docker/wikigen/` (i.e. `/volume1/docker/wikigen/`) containing:
   - `.env`: copy of `.env.example` with your keys filled in. `DATA_DIR` is forced to `/data` by the compose file.
   - `data/`: empty folder. It holds `connections.json`, `history.json`, `plans/` and `uploads/`. Copy existing ones in here to keep them.
2. **User ID.** SSH into the NAS and run `id`. Put your `uid:gid` in the `user:` line of `deploy/docker-compose.yml` (default `1026:100`) so the container can write to `data/`.
3. **Image access.** After the first workflow run, open the package on GitHub (profile → Packages → wikigen). Then either:
   - set its visibility to **Public** (the image contains no secrets; those stay in `.env`), or
   - keep it private: create a classic PAT with only `read:packages`, then on the NAS run
     `docker login ghcr.io -u <github-user>` (paste the PAT) and copy `~/.docker/config.json` to `/volume1/docker/wikigen/config.json`. Mount it into your Watchtower container as `/config.json` so Watchtower can pull too. Container Manager → Registry → Settings → Add also lets DSM pull it.
4. **Project.** Container Manager → Project → Create → path `/volume1/docker/wikigen` → *Create docker-compose.yml* → paste `deploy/docker-compose.yml` → Done.
5. Browse to `http://<nas-ip>:5055`.
6. **Watchtower.** If you already run one, nothing to do: WikiGen carries the `com.centurylinklabs.watchtower.enable=true` label, which also covers a Watchtower running with `--label-enable`. If you don't, add this service to the project (host networking avoids the same DNS problem):
   ```yaml
     watchtower:
       image: containrrr/watchtower
       container_name: watchtower
       command: --label-enable --cleanup --schedule "0 0 17 * * *"
       network_mode: host
       volumes:
         - /var/run/docker.sock:/var/run/docker.sock
       restart: unless-stopped
   ```

**HTTPS (optional):** Control Panel → Login Portal → Advanced → Reverse Proxy. Map `https://wikigen.<your-domain>` to `http://localhost:5055`. Under *Custom Header*, add the WebSocket preset, and set a long proxy timeout so streaming plans aren't cut off.

### Day to day

- **Update:** merge to `main`, and your Watchtower installs the new image on its next check. How often that happens is set on your Watchtower, not here. An update restarts WikiGen, so a nightly schedule (e.g. `--schedule "0 0 17 * * *"`, 17:00 UTC) avoids restarts mid-session.
- **Update now:** over SSH on the NAS, run a one-off check for just this container:
  ```bash
  sudo docker run --rm --network host -v /var/run/docker.sock:/var/run/docker.sock \
    containrrr/watchtower --run-once --cleanup wikigen
  ```
  It pulls `latest` and recreates `wikigen` only if the image changed, keeping the container's settings. (For a private package add `-v /volume1/docker/wikigen/config.json:/config.json:ro`.)
- **Roll back:** change the image to a specific build, e.g. `ghcr.io/davior/wikigen:sha-1a2b3c4` (tags are listed on the package page), and rebuild the project. Switch back to `:latest` to resume auto-updates.

Updates restart the container and kill any generation or execution in progress, so don't force an update mid-run.

### Troubleshooting

Open `http://<nas-ip>:5055/api/check_connection` (or Container Manager → wikigen → Log) to see why the wiki connection fails:

| Error | Fix |
|---|---|
| `Cannot reach …: Failed to resolve` | The container can't do DNS. Make sure the project uses `network_mode: host` (as in `deploy/docker-compose.yml`) and rebuild it. |
| `Cannot reach …: Connection refused` | Wrong host or port in the wiki URL. Use the wiki's real hostname or LAN IP. |
| `Timed out reaching …` | If the wiki is on your LAN and addressed by its public domain, your router may lack NAT loopback; use its LAN address instead. Also check the DSM firewall. |
| `Login failed: …` | Network is fine; check the bot username/password in the connections manager. |

Test name resolution from the NAS over SSH:

```bash
sudo docker exec wikigen python -c "import socket; print(socket.gethostbyname('yourwiki.example.com'))"
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

## Image Generation

WikiGen illustrates pages with images generated by [fal.ai](https://fal.ai). It doesn't search the web for images.

**Setup:** Connections → **EDIT** → *Image generation (fal.ai)*:

| Setting | What it does |
|---|---|
| **fal.ai API Key** | Your key from fal.ai → Dashboard → Keys. Stored in `connections.json` and never sent back to the browser. Without one, pages are written without images. |
| **Model** | Any fal.ai text-to-image endpoint ID. Default `fal-ai/flux/dev`. |
| **Model Parameters** | JSON sent with every prompt. FLUX models take `image_size` (e.g. `{"image_size": "landscape_4_3"}`); most others take `aspect_ratio` (e.g. `{"aspect_ratio": "4:3"}`). Picking a preset model fills this in. |
| **Images / new page** | How many images new pages get (0–4, default 3; 0 turns them off). |
| **House Style Guide** | Added to every image prompt so the wiki's images share one look. **DRAFT FROM WIKI** writes one from the wiki's system prompt and site index; if you leave it blank, it's drafted and saved on first use. Edit it freely. |

**How it works:**

- **New pages.** While writing a page, the AI marks where images belong and describes what each should show, based on the text. Each description plus the house style becomes the image prompt. The images appear as thumbnails on the step.
- **Existing pages.** Ask for it in an instruction (*"Add images to the Tesla Coil page"*), which creates an **IMAGE** step, or open any page in the editor and click **🖼 IMAGES**. The AI picks 1–4 sections that have no image yet, and you can add a hint about what to show.
- **Upload timing.** Generated images are kept in `DATA_DIR/generated/` and only uploaded to the wiki when their page is published or its step executes, so rejected steps leave nothing on the wiki. Each file page records the caption, the prompt and the model, and is placed in `[[Category:AI-generated images]]`.
- **Failures.** If an image fails (bad key, no credit, blocked by the safety checker), the page is still written without it and the step shows why.

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
├── image_gen.py        # fal.ai image generation + store of generated images
├── requirements.txt
├── .gitignore
├── connections.json    # Created at runtime — saved wiki connections
├── history.json        # Created at runtime — completed operation log
├── plans/              # Created at runtime — persisted plan JSON files
├── generated/          # Created at runtime — generated images awaiting/after upload
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
| `GET` | `/api/connections` | List all connections (secrets replaced by `has_*` flags) |
| `POST` | `/api/connections` | Add a connection |
| `PUT` | `/api/connections/<id>` | Update a connection |
| `DELETE` | `/api/connections/<id>` | Delete a connection |
| `POST` | `/api/connections/<id>/activate` | Set active connection |
| `POST` | `/api/connections/<id>/test` | Test authentication |
| `POST` | `/api/connections/<id>/image_style/draft` | Draft a house image style from the wiki |

### Images

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/wiki/generate_images` | Add generated images to page content (returns new content + diff) |
| `GET` | `/api/image/<filename>` | Serve a generated image, or redirect to the wiki's copy of a file |

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
