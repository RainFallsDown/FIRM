"""Genesis-first FIRM-Sim workspace."""


def init_genesis(*args, **kwargs):
    """Lazily import Genesis runtime helpers when actually needed."""
    from firm_sim.runtime import init_genesis as _init_genesis

    return _init_genesis(*args, **kwargs)


__all__ = ["init_genesis"]
