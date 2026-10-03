"""Checks on a connection's secrets: its fal.ai and AI provider keys and wiki bot password."""

import re

# fal.ai keys are "<key id>:<key secret>"; the id is a UUID.
_FAL_KEY_RE = re.compile(r'^[^\s:]+:[^\s:]+$')
_FAL_KEY_STRICT_RE = re.compile(r'^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}:[0-9a-fA-F]{16,}$')


def mask_key(key: str | None) -> str:
    """Enough of a key to tell which one it is, e.g. against the provider's dashboard, but not to use it."""
    key = key or ''
    if not key:
        return '(none)'
    if len(key) < 16:
        return f'…{key[-2:]} ({len(key)} chars)'
    return f'{key[:8]}…{key[-4:]} ({len(key)} chars)'


def fal_key_problem(key: str, wiki_password: str | None = None) -> str | None:
    """Why `key` can't be a fal.ai key, or None if it looks like one."""
    if wiki_password and key == wiki_password.strip():
        return 'it is the same as the wiki password'
    if not _FAL_KEY_RE.match(key):
        return 'it isn\'t in fal.ai\'s key format ("<key id>:<key secret>")'
    return None


def ai_key_problem(provider: str, key: str, wiki_password: str | None = None, strict: bool = True) -> str | None:
    """Why `key` can't be the API key for `provider` ('anthropic' or 'deepseek'), or None.

    `strict` adds the provider's key format, which only holds when calls go to the
    provider itself rather than a gateway (ANTHROPIC_BASE_URL / DEEPSEEK_BASE_URL).
    """
    if wiki_password and key == wiki_password.strip():
        return 'it is the same as the wiki password'
    if re.search(r'\s', key):
        return 'it contains spaces'
    if ':' in key:
        return 'it looks like a fal.ai key ("<key id>:<key secret>")'
    if not strict:
        return None
    if provider == 'anthropic':
        if key.startswith('sk-ant-oat'):
            return 'it is a Claude subscription (OAuth) token, not an API key'
        if key.startswith('sk-ant-admin'):
            return 'it is an Admin API key, which can\'t write text; create a normal API key'
        if not key.startswith('sk-ant-'):
            return 'Anthropic API keys start with "sk-ant-"'
    elif provider == 'deepseek':
        if key.startswith('sk-ant-'):
            return 'it is an Anthropic key, not a DeepSeek key'
        if not key.startswith('sk-'):
            return 'DeepSeek API keys start with "sk-"'
    return None


def wiki_password_problem(password: str) -> str | None:
    """Why `password` can't be the wiki bot password, or None."""
    if _FAL_KEY_STRICT_RE.match(password.strip()):
        return 'it is a fal.ai key, not a wiki bot password'
    if password.strip().startswith('sk-ant-'):
        return 'it is an Anthropic API key, not a wiki bot password'
    return None
