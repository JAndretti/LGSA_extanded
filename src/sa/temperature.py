"""Temperature schedulers for SA. Ported from LGSA_OLD/src/sa/scheduler.py."""
import math
from typing import Callable

import torch


class _CyclicLR:
    def __init__(self, T_max, T_min, step_max, step_size_up=20, mode="triangular2"):
        self.T_max, self.T_min, self.step_max = T_max, T_min, step_max
        self.step_size_up, self.mode = step_size_up, mode

    def step(self, step: int) -> torch.Tensor:
        step_t = torch.tensor(step, dtype=torch.float32)
        cycle = torch.floor(1 + step_t / (2 * self.step_size_up))
        x = torch.abs(step_t / self.step_size_up - 2 * cycle + 1)
        base = self.T_min + (self.T_max - self.T_min) * torch.clamp(1 - x, min=0.0)
        if self.mode == "triangular":
            return base.detach()
        if self.mode == "triangular2":
            return (base / (2 ** (cycle - 1))).detach()
        raise ValueError(f"Unknown cyclic mode: {self.mode}")


class _Cosine:
    def __init__(self, T_max, T_min, step_max, T_0=20, T_mult=1):
        self.T_max, self.T_min, self.step_max = T_max, T_min, step_max
        self.T_0, self.T_mult = T_0, T_mult

    def step(self, step: int) -> torch.Tensor:
        T_cur, T_i = step, self.T_0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        value = self.T_min + (self.T_max - self.T_min) * (
            1 + math.cos(math.pi * T_cur / T_i)
        ) / 2
        return torch.tensor(value, dtype=torch.float32)


class _Lambda:
    def __init__(self, T_max, T_min, step_max):
        self.T_max, self.T_min, self.step_max = T_max, T_min, step_max
        self.factor = (T_min / T_max) ** (1 / max(1, step_max))

    def step(self, step: int) -> torch.Tensor:
        return torch.tensor(self.T_max * (self.factor ** step), dtype=torch.float32)


class _Step:
    def __init__(self, T_max, T_min, step_max):
        self.T_max, self.T_min, self.step_max = T_max, T_min, step_max
        self.temp = T_max

    def step(self, step: int) -> torch.Tensor:
        if step == int(self.step_max / 2) or step == int(2 * self.step_max / 3) \
           or step == int(3 * self.step_max / 4):
            self.temp = self.temp / 2
        elif step == int(6 * self.step_max / 7):
            self.temp = self.T_min
        return torch.tensor(self.temp, dtype=torch.float32)


def make_schedule(
    name: str, T_max: float, T_min: float, step_max: int, **kwargs
) -> Callable[[int], torch.Tensor]:
    if name == "cyclic":
        sched = _CyclicLR(T_max=T_max, T_min=T_min, step_max=step_max, **kwargs)
    elif name == "cosine":
        sched = _Cosine(T_max=T_max, T_min=T_min, step_max=step_max, **kwargs)
    elif name == "lambda":
        sched = _Lambda(T_max=T_max, T_min=T_min, step_max=step_max)
    elif name == "step":
        sched = _Step(T_max=T_max, T_min=T_min, step_max=step_max)
    else:
        raise ValueError(f"Unknown temperature schedule: {name}")
    return sched.step
