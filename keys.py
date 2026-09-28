"""Checks on a connection's secrets: its fal.ai key and wiki bot password."""

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


def wiki_password_problem(password: str) -> str | None:
    """Why `password` can't be the wiki bot password, or None."""
    if _FAL_KEY_STRICT_RE.match(password.strip()):
        return 'it is a fal.ai key, not a wiki bot password'
    return None
