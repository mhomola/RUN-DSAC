from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from tasks.signal_base import Signal


@dataclass
class Step(Signal):
    """
    Step function
    """
    def _signal(self, _: float) -> float:
        return 1.0

@dataclass
class Ramp(Signal):
    """
    Ramp function
    """
    offset: float = 0
    def _signal(self, t: float) -> float:
        return max((t-self.offset), 0)

@dataclass
class Parabolic(Signal):
    """
    Parabolic function
    """
    offset: float = 0.0

    def _signal(self, t: float) -> float:
        return (t - self.offset) * (t - self.offset) / 2.0


@dataclass
class Exponential(Signal):
    alpha:  float = 0.0
    offset: float = 0.0

    def _signal(self, t: float) -> float:
        return np.exp(self.alpha * (t - self.offset))


@dataclass
class Sinusoid(Signal):
    ampl:   float = 1.0
    freq:   float = 1.0
    phi:    float = 0.0

    def _signal(self, t: float) -> float:
        return self.ampl * np.sin(2.0 * np.pi * self.freq * t + self.phi)