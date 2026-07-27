"""Load the licensed `cg` engine from data/ and expose it as modules.

The engine is a single-battle-at-a-time ctypes wrapper, so everything here is
process-global: `load()` puts the SDK dir on sys.path and imports `cg`.
"""
from __future__ import annotations

import sys
from functools import lru_cache

from ptcg import config

_loaded = None


def load():
    """Import and return the `cg` package (idempotent)."""
    global _loaded
    if _loaded is not None:
        return _loaded
    sdk_dir = config.find_sdk_dir()
    if sdk_dir is None:
        raise RuntimeError("cg engine not found under data/ (need **/cg/api.py)")
    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    import cg  # noqa: F401
    import cg.api  # noqa: F401
    import cg.game  # noqa: F401
    _loaded = cg
    return cg


def api():
    return load().api


def game():
    return load().game


@lru_cache(maxsize=1)
def card_db() -> dict:
    """cardId -> CardData for every card the engine knows."""
    return {c.cardId: c for c in api().all_card_data()}


@lru_cache(maxsize=1)
def attack_db() -> dict:
    """attackId -> Attack."""
    return {a.attackId: a for a in api().all_attack()}
