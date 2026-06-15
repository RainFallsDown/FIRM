"""Policy interfaces and placeholders."""

from firm_sim.policies.base import Policy
from firm_sim.policies.no_op import NoOpPolicy


POLICY_REGISTRY = {
    "no_op": NoOpPolicy,
}


def make_policy(name: str, seed: int = 0) -> Policy:
    try:
        return POLICY_REGISTRY[name](seed=seed)
    except KeyError as exc:
        available = ", ".join(sorted(POLICY_REGISTRY.keys()))
        raise KeyError(f"Unknown policy '{name}'. Available policies: {available}") from exc


__all__ = ["Policy", "NoOpPolicy", "make_policy"]
