"""Runtime helpers for Genesis-based demos."""

from __future__ import annotations

import genesis as gs

_INITIALIZED = False


def init_genesis(backend=None, seed=0) -> None:
    """Initialize Genesis exactly once for this Python process."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    gs.init(backend=backend or gs.gpu, seed=seed)
    _INITIALIZED = True
