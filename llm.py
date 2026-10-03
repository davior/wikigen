"""The AI model a connection writes with: Claude (Anthropic) or DeepSeek.

Both go through the anthropic SDK, since DeepSeek serves the same Messages API at
api.deepseek.com/anthropic. Each connection picks its provider and, for DeepSeek,
the model and how hard it thinks before writing. API keys are saved on the
connection, with ANTHROPIC_API_KEY / DEEPSEEK_API_KEY in .env as the fallback for
connections without their own, like the fal.ai key.
"""

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import anthropic

import keys

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    key_field: str    # the connection field holding its API key
    key_env: str      # the .env fallback for connections without one
    key_url: str      # where keys are made
    billing_url: str
    default_model: str
    max_output: int   # max_tokens for calls that don't set their own
    concurrency: int  # generations to run at once


PROVIDERS = {
    'anthropic': Provider('anthropic', 'Claude', 'anthropic_key', 'ANTHROPIC_API_KEY',
                          'console.anthropic.com → API Keys', 'console.anthropic.com → Billing',
                          'claude-sonnet-4-6', 64000, 3),
    # Reasoning tokens count against max_tokens, so leave room to think and still write a long page.
    'deepseek': Provider('deepseek', 'DeepSeek', 'deepseek_key', 'DEEPSEEK_API_KEY',
                         'platform.deepseek.com → API keys', 'platform.deepseek.com → Top up',
                         'deepseek-v4-pro', 131072, 6),
}
DEEPSEEK_MODELS = ['deepseek-v4-pro', 'deepseek-flash']
EFFORTS = ['low', 'high', 'max']
DEFAULT_EFFORT = 'high'
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/anthropic'


class Cancelled(Exception):
    """The run was cancelled while the model was still writing."""


class OutputBudgetSpent(ValueError):
    """The model used its whole max_tokens without writing an answer."""


class ProviderKeyError(RuntimeError):
    """The provider refused the API key or the account behind it (401, 402 or 403)."""


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
    if model.lower().startswith('sk-'):
        return 'That looks like an API key, not a model name. Put it in the API key field.'
    return None


# The draft route lists here the AI keys it took from the form before they were saved.
TYPED_KEYS = '_typed_keys'


def key_for(conn: dict, provider: str) -> tuple[str, str]:
    """The API key a connection uses for `provider` and where it comes from; ('', '') if it has none."""
    spec = PROVIDERS[provider]
    key = str(conn.get(spec.key_field) or '').strip()
    if key:
        typed = spec.key_field in (conn.get(TYPED_KEYS) or ())
        return key, 'the key typed in the form' if typed else 'the connection settings'
    key = (os.environ.get(spec.key_env) or '').strip()
    if key:
        return key, f'{spec.key_env} in .env'
    return '', ''


def env_keys() -> dict[str, bool]:
    """Which providers have a fallback API key in .env (never the keys themselves)."""
    return {name: bool((os.environ.get(spec.key_env) or '').strip()) for name, spec in PROVIDERS.items()}


def _deepseek_base_url() -> str:
    return (os.environ.get('DEEPSEEK_BASE_URL') or '').strip() or DEEPSEEK_BASE_URL


def official_endpoint(provider: str) -> bool:
    """Whether calls go to the provider itself, so its key format applies, rather than a gateway."""
    if provider == 'deepseek':
        return _deepseek_base_url().rstrip('/') == DEEPSEEK_BASE_URL
    return not (os.environ.get('ANTHROPIC_BASE_URL') or '').strip()


def key_problem(conn: dict) -> str | None:
    """Why the connection's AI provider can't be called, checked before work starts; None if it can."""
    settings = settings_for(conn)
    spec = settings.spec
    key, source = key_for(conn, settings.provider)
    if not key:
        return (f'This connection has no {spec.label} API key. Add one in Connections → Edit → AI model '
                f'(or set {spec.key_env} in .env).')
    if problem := keys.ai_key_problem(settings.provider, key, conn.get('password'),
                                      strict=official_endpoint(settings.provider)):
        return (f'The {spec.label} key in {source} ({keys.mask_key(key)}) is unusable: {problem}. '
                'Re-enter it in Connections → Edit.')
    return None


_MAX_CLIENTS = 8  # one per key in use; replaced keys drop out
_clients: OrderedDict[tuple[str, str], anthropic.Anthropic] = OrderedDict()
_clients_lock = threading.Lock()


def client_for(provider: str, key: str) -> anthropic.Anthropic:
    if not key:
        # Never build a client without its key: the SDK would fall back to ANTHROPIC_API_KEY
        # and send it to whichever server the client points at.
        raise RuntimeError(f'No {PROVIDERS[provider].label} API key')
    cache_key = (provider, hashlib.sha256(key.encode()).hexdigest())
    with _clients_lock:
        client = _clients.get(cache_key)
        if client is not None:
            _clients.move_to_end(cache_key)
            return client
        if provider == 'deepseek':
            client = anthropic.Anthropic(
                api_key=key,
                base_url=_deepseek_base_url(),
                # Don't let an ANTHROPIC_AUTH_TOKEN in the environment ride along to DeepSeek.
                default_headers={'Authorization': anthropic.Omit()},
            )
        else:
            client = anthropic.Anthropic(api_key=key)
        _clients[cache_key] = client
        while len(_clients) > _MAX_CLIENTS:
            # Not closed here: a stream may still be using it; it closes once unreferenced.
            _clients.popitem(last=False)
        return client


def for_connection(conn: dict) -> 'LLM':
    """The model a connection writes with, and the key it calls it with."""
    settings = settings_for(conn)
    key, source = key_for(conn, settings.provider)
    return LLM(settings, key, source)


def _timeout_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get('AI_TIMEOUT_MINUTES', 30)) * 60)
    except ValueError:
        return 1800.0


class LLM:
    """Calls a connection's model; one instance per agent."""

    def __init__(self, settings: Settings, api_key: str, key_source: str):
        self.settings = settings
        self._api_key = api_key
        self.key_source = key_source

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
        client = client_for(self.provider, self._api_key)
        timeout = _timeout_seconds()
        start = time.monotonic()
        stopped: list[str] = []  # why the watcher closed the stream
        done = threading.Event()

        try:
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
        except anthropic.APIStatusError as e:
            # Raised as the stream opens. Not a ValueError, so callers don't ask again.
            if e.status_code in (401, 402, 403):
                raise ProviderKeyError(self._key_error(e)) from None
            raise

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

    def _key_error(self, e: anthropic.APIStatusError) -> str:
        """What a 401/402/403 means for this key, naming it masked and where it came from."""
        spec = self.settings.spec
        key = f'{keys.mask_key(self._api_key)} from {self.key_source}'
        if e.status_code == 401:
            reason = f'{spec.label} rejected the API key {key}; compare it with {spec.key_url}'
        elif e.status_code == 402:
            reason = f'{spec.label} refused the account behind key {key} for insufficient balance; top up at {spec.billing_url}'
        else:
            reason = f'{spec.label} refused the account behind key {key}; check it at {spec.key_url}'
        detail = str(getattr(e, 'message', '') or e)[:200]
        return f'{reason} ({e.status_code}: {detail})'
