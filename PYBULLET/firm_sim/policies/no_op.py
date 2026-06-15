"""A no-op policy for pipeline validation."""

from firm_sim.policies.base import Policy


class NoOpPolicy(Policy):
    def __init__(self, seed: int = 0):
        super().__init__(name="no_op", seed=seed)
