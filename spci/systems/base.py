from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

GENERATOR = np.random.Generator(np.random.PCG64())


class System(ABC):
    def __init__(self, state_dim, tangent_dim, measurement_dim):
        self.state_dim = state_dim
        self.tangent_dim = tangent_dim
        self.measurement_dim = measurement_dim

    @abstractmethod
    def dynamics(self, state, dt: float):
        raise NotImplementedError

    @abstractmethod
    def dynamics_covariance(self, dt: float) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def sensor(self, state):
        raise NotImplementedError

    @abstractmethod
    def sensor_covariance(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def step(self, state, dt: float, noisy: bool = False):
        state = self.dynamics(state, dt)
        if noisy:
            state = state + tangent_sample(self.dynamics_covariance(dt))
        return state


class SystemRunner:
    def __init__(self, system: System, state):
        self._system = system
        self._state = state

    def step(self, dt: float, noisy: bool = False):
        self._state = self._system.step(self._state, dt, noisy=noisy)

    def measure(self, noisy: bool = False):
        z = self._system.sensor(self._state)
        if noisy:
            z += tangent_sample(self._system.sensor_covariance())
        return z

    @property
    def state(self):
        return self._state


def tangent_sample(covariance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return GENERATOR.multivariate_normal(np.zeros_like(covariance[0, :]), covariance, method="eigh")
