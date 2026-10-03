"""The AI model a connection writes with: Claude (Anthropic) or DeepSeek.

Both go through the anthropic SDK, since DeepSeek serves the same Messages API at
api.deepseek.com/anthropic. API keys come from .env; each connection picks its
provider and, for DeepSeek, the model and how hard it thinks before writing.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass

import anthropic

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    key_env: str
    default_model: str
    max_output: int   # max_tokens for calls that don't set their own
    concurrency: int  # generations to run at once


PROVIDERS = {
    'anthropic': Provider('anthropic', 'Claude', 'ANTHROPIC_API_KEY', 'claude-sonnet-4-6', 64000, 3),
    # Reasoning tokens count against max_tokens, so leave room to think and still write a long page.
    'deepseek': Provider('deepseek', 'DeepSeek', 'DEEPSEEK_API_KEY', 'deepseek-v4-pro', 131072, 6),
}
DEEPSEEK_MODELS = ['deepseek-v4-pro', 'deepseek-flash']
EFFORTS = ['low', 'high', 'max']
DEFAULT_EFFORT = 'high'
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/anthropic'


class Cancelled(Exception):
    """The run was cancelled while the model was still writing."""


class OutputBudgetSpent(ValueError):
    """The model used its whole max_tokens without writing an answer."""


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    effort: str

    @property
    def spec(self) -> Provider:
        return PROVIDERS[self.provider]

    @property
    def label(self) -> str:
        if self.provider == 'deepseek':
            return f'DeepSeek · {self.model} · {self.effort} effort'
        return f'Claude · {self.model}'


def settings_for(conn: dict) -> Settings:
    """The AI settings a connection uses; connections saved before there was a choice use Claude."""
    provider = str(conn.get('ai_provider') or 'anthropic').strip().lower()
    if provider not in PROVIDERS:
        log.warning('Unknown ai_provider %r on connection %s; using Claude', provider, conn.get('id'))
        provider = 'anthropic'
    if provider != 'deepseek':
        return Settings(provider, PROVIDERS[provider].default_model, DEFAULT_EFFORT)
    model = str(conn.get('deepseek_model') or '').strip()
    if not model or model.lower().startswith('claude'):
        model = PROVIDERS['deepseek'].default_model
    effort = conn.get('deepseek_effort') if conn.get('deepseek_effort') in EFFORTS else DEFAULT_EFFORT
    return Settings('deepseek', model, effort)


def settings_problem(body: dict) -> str | None:
    """Why the AI settings in a connection being saved are unusable, or None. Blank means the default."""
    provider = body.get('ai_provider')
    if provider and (not isinstance(provider, str) or provider not in PROVIDERS):
        return f'Unknown AI provider "{provider}"'
    effort = body.get('deepseek_effort')
    if effort and (not isinstance(effort, str) or effort not in EFFORTS):
        return f'Reasoning effort must be one of: {", ".join(EFFORTS)}'
    model = str(body.get('deepseek_model') or '').strip()
    if model.lower().startswith('claude'):
        # DeepSeek quietly answers requests for Claude models with deepseek-flash.
        return f'"{model}" is a Claude model. Use a DeepSeek model, e.g. {DEEPSEEK_MODELS[0]}.'
    return None


def _api_key(provider: str) -> str:
    return (os.environ.get(PROVIDERS[provider].key_env) or '').strip()


def keys_present() -> dict[str, bool]:
    """Which providers have an API key in .env (never the keys themselves)."""
    return {name: bool(_api_key(name)) for name in PROVIDERS}


def key_problem(settings: Settings) -> str | None:
    """Why the connection's provider can't be called, checked before work starts; None if it can."""
    if _api_key(settings.provider):
        return None
    spec = settings.spec
    return (f'{spec.key_env} is not set in .env, and this connection uses {spec.label}. Add it to .env '
            '(on Synology: /volume1/docker/wikigen/.env), then rebuild the project in Container Manager; '
            'a Watchtower update alone keeps the old settings.')


_clients: dict[str, anthropic.Anthropic] = {}
_clients_lock = threading.Lock()


def client_for(provider: str) -> anthropic.Anthropic:
    key = _api_key(provider)
    if not key:
        # Never build a client without its key: the SDK would fall back to ANTHROPIC_API_KEY
        # and send it to whichever server the client points at.
        raise RuntimeError(key_problem(Settings(provider, PROVIDERS[provider].default_model, DEFAULT_EFFORT)))
    with _clients_lock:
        client = _clients.get(provider)
        if client is None or client.api_key != key:
            if provider == 'deepseek':
                client = anthropic.Anthropic(
                    api_key=key,
                    base_url=(os.environ.get('DEEPSEEK_BASE_URL') or '').strip() or DEEPSEEK_BASE_URL,
                    # Don't let an ANTHROPIC_AUTH_TOKEN in the environment ride along to DeepSeek.
                    default_headers={'Authorization': anthropic.Omit()},
                )
            else:
                client = anthropic.Anthropic(api_key=key)
            _clients[provider] = client
        return client


def _timeout_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get('AI_TIMEOUT_MINUTES', 30)) * 60)
    except ValueError:
        return 1800.0


class LLM:
    """Calls a connection's model; one instance per agent."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def provider(self) -> str:
        return self.settings.provider

    @property
    def concurrency(self) -> int:
        return self.settings.spec.concurrency

    def complete(self, system_blocks: list[dict], user_message: str, *, max_tokens: int | None = None,
                 thinking: bool = True, effort: str | None = None, cancel_event: threading.Event | None = None) -> str:
        """The model's answer to `user_message`.

        `thinking` and `effort` only apply to DeepSeek (Claude calls are sent as before);
        small calls turn thinking off so reasoning can't use up their max_tokens.
        """
        s = self.settings
        request = {
            'model': s.model,
            'max_tokens': max_tokens or s.spec.max_output,
            'system': system_blocks,
            'messages': [{'role': 'user', 'content': user_message}],
        }
        if s.provider == 'deepseek':
            # DeepSeek caches prompt prefixes by itself and ignores cache_control.
            request['system'] = [{'type': 'text', 'text': b['text']} for b in system_blocks]
            # Thinking is on unless disabled, and an effort sent with it disabled is a 400.
            request['extra_body'] = (
                {'thinking': {'type': 'enabled'}, 'output_config': {'effort': effort or s.effort}}
                if thinking else {'thinking': {'type': 'disabled'}})

        for attempt in (1, 2):
            msg = self._stream(request, cancel_event)
            text = ''.join(getattr(b, 'text', None) or '' for b in msg.content if b.type == 'text')
            if text.strip():
                return text
            if msg.stop_reason == 'max_tokens':
                hint = ' Try a lower reasoning effort.' if s.provider == 'deepseek' and thinking else ''
                raise OutputBudgetSpent(f'{s.spec.label} used its whole output budget ({request["max_tokens"]} '
                                        f'tokens) before writing an answer.{hint}')
            if attempt == 1:
                log.warning('%s returned an empty answer (stop_reason=%s); asking again', s.model, msg.stop_reason)
        raise ValueError(f'{s.spec.label} returned an empty answer twice')

    def _stream(self, request: dict, cancel_event: threading.Event | None):
        """Stream one request to the end, stopping early on cancel or after AI_TIMEOUT_MINUTES."""
        client = client_for(self.provider)
        timeout = _timeout_seconds()
        start = time.monotonic()
        stopped: list[str] = []  # why the watcher closed the stream
        done = threading.Event()

        with client.messages.stream(**request) as stream:
            # A model can sit in a queue sending only keep-alives, which reset the read
            # timeout and yield no events, so watch from another thread and close the stream.
            def watch():
                while not done.wait(0.5):
                    if cancel_event is not None and cancel_event.is_set():
                        stopped.append('cancelled')
                    elif time.monotonic() - start > timeout:
                        stopped.append('timeout')
                    else:
                        continue
                    stream.close()
                    return

            threading.Thread(target=watch, name='ai-stream-watch', daemon=True).start()
            try:
                for _event in stream:
                    if stopped:
                        break
                msg = None if stopped else stream.get_final_message()
            except Exception:
                if not stopped:
                    raise
            finally:
                done.set()

        if stopped and stopped[0] == 'cancelled':
            raise Cancelled('Cancelled')
        if stopped:
            raise TimeoutError(f'{self.settings.spec.label} took longer than {timeout / 60:g} minutes '
                               '(AI_TIMEOUT_MINUTES)')

        usage = msg.usage
        log.info('%s (asked for %s): %s in, %s out, %s cache-read tokens, %.0fs, stop=%s',
                 msg.model, request['model'], getattr(usage, 'input_tokens', '?'),
                 getattr(usage, 'output_tokens', '?'), getattr(usage, 'cache_read_input_tokens', None) or 0,
                 time.monotonic() - start, msg.stop_reason)
        return msg
