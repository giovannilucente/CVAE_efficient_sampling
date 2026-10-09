"""KL weight schedule of the CVAE training (KL annealing, Bowman et al. 2016).

beta rises from beta_start to beta_end over n_steps calls of step() (one per epoch in the
training scripts), linearly or along a sigmoid, and stays at beta_end afterwards.
"""
import math


class BetaAnnealer:
    def __init__(self, beta_start: float = 0.0, beta_end: float = 1.0, n_steps: int = 10,
                 schedule: str = "sigmoid", steepness: float = 10.0):
        if schedule not in ("sigmoid", "linear", "constant"):
            raise ValueError(f"unknown schedule {schedule}")
        self.beta_start, self.beta_end, self.n_steps = beta_start, beta_end, max(n_steps, 1)
        self.schedule, self.steepness = schedule, steepness
        self.t = 0

    def value(self, t: int) -> float:
        if self.schedule == "constant":
            return self.beta_end
        r = min(t / self.n_steps, 1.0)
        if self.schedule == "sigmoid":  # rescaled so that r = 0 -> 0 and r = 1 -> 1
            s = lambda x: 1.0 / (1.0 + math.exp(-self.steepness * (x - 0.5)))
            r = (s(r) - s(0.0)) / (s(1.0) - s(0.0))
        return self.beta_start + (self.beta_end - self.beta_start) * r

    def step(self) -> float:
        """beta of the next epoch."""
        self.t += 1
        return self.value(self.t)

    def state_dict(self) -> dict:
        return {"t": self.t}

    def load_state_dict(self, state: dict) -> None:
        self.t = state["t"]
