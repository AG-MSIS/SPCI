from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt

from .base import System

if TYPE_CHECKING:
    from collections.abc import Iterable


class State1D:
    def __init__(self, position: float = 0.0, velocity: float = 0.0):
        """Creates a state object.

        :param position: The position of the particle [L].
        :param velocity: The velocity of the particle [L/T].
        """
        self.position = position
        self.velocity = velocity

    def __add__(self, other: npt.NDArray[np.float64]) -> State1D:
        return State1D(self.position + other[0], self.velocity + other[1])

    def __sub__(self, other: State1D) -> npt.NDArray[np.float64]:
        return np.array(
            [self.position - other.position, self.velocity - other.velocity], dtype=np.float64
        )

    def __repr__(self):
        return f"position: {self.position}, velocity: {self.velocity}"

    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.float64]) -> Self:
        return cls(array[0], array[1])

    def numpy(self) -> npt.NDArray[np.float64]:
        return np.array([self.position, self.velocity], dtype=np.float64)

    @classmethod
    def mean(cls, states: Iterable[State1D], *, weights: npt.NDArray[np.float64]) -> Self:
        mean = np.sum(np.stack([_.numpy() for _ in states], axis=-1) * weights, axis=-1)
        return cls.from_numpy(mean)


class Oscillator1D(System):
    def __init__(
        self,
        exponent: float = 1.0,
        dynamic_noise_stddev: float = 1.0,
        sensor_noise_stddev: float = 1.0,
    ):
        """Constructor.

        :param exponent: The exponent that determines how nonlinear the system is [1].
        :param dynamic_noise_stddev: The velocity noise standard deviation per second
            [L/T / sqrt(T)].
        :param sensor_noise_stddev: The sensor noise standard deviation [L].
        """
        super().__init__(2, 2, 1)
        self._exponent = exponent
        self._dynamic_noise_stddev = dynamic_noise_stddev
        self._sensor_noise_stddev = sensor_noise_stddev

    def dynamics(self, state: State1D, dt: float) -> State1D:
        # Velocity Verlet integration
        old_acceleration = -np.sign(state.position) * np.pow(np.abs(state.position), self._exponent)
        new_position = state.position + state.velocity * dt + 0.5 * old_acceleration * dt**2
        new_acceleration = -np.sign(new_position) * np.pow(np.abs(new_position), self._exponent)
        new_velocity = state.velocity + 0.5 * (old_acceleration + new_acceleration) * dt
        return State1D(new_position, new_velocity)

    def dynamics_covariance(self, dt):
        # Continuous time white acceleration noise
        return self._dynamic_noise_stddev**2 * np.array(
            [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], dtype=np.float64
        )

    def sensor(self, state) -> npt.NDArray[np.float64]:
        return np.array([state.position], dtype=np.float64)

    def sensor_covariance(self) -> npt.NDArray[np.float64]:
        return np.array([[self._sensor_noise_stddev**2]], dtype=np.float64)
