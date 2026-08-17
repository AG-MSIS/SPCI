from enum import Enum
from typing import cast, override

import numpy as np
import numpy.typing as npt
import scipy.linalg

from ..systems import System
from .base import Filter
from .ukf_utils import (
    SigmaPointMethod,
    covariance_of_sigma_points,
    generate_sigma_points,
    generate_sigma_weights,
    mean_of_sigma_points,
)


class UpdateMethod(Enum):
    # Update each sigma point with its own residual, adapt covariance afterwards.
    PER_SIGMA_POINT = 0
    # Calculate standard UKF update and redraw sigma points afterwards.
    UKF_REDRAW = 1


class SPCIMethod(Enum):
    # Use Scipy's CARE method.
    CARE_SCIPY = 0
    # Use PPUKF-like SPCI but with optimal displacement.
    CONSTRAINED_OPTIMAL = 1
    # Use generalized optimal transport.
    UNCONSTRAINED_OPTIMAL = 2


def generalized_optimal_transport_matrix(
    points: npt.NDArray[np.float64],
    target_covariance: npt.NDArray[np.float64],
    metric: npt.NDArray[np.float64],
    *,
    weights: npt.NDArray[np.float64],
    factor4: float | None = None,
) -> npt.NDArray[np.float64]:
    """Generalized optimal transport formula on matrices (each point is a column). Keeps the mean of
    each row (i.e. mean over points for each state dimension) at 0, while setting the covariance to
    a given value, while minimizing the (weighted) sum of squared movement distance over points,
    measured by some metric tensor. Optionally, a gradient towards smaller 4th moments along
    cardinal axes is mixed in.

    :param points: Matrix of zero-centered points.
        (tangent_dim, num_of_points)
    :param target_covariance: The target for result @ diag(weights) @ result.T .
        (tangent_dim, tangent_dim)
    :param metric: The metric that measures distance between sigma points.
        (tangent_dim, tangent_dim)
    :param weights: Positive convex weight array (one for each point).
        (num_of_points,)
    :param factor4: Optional scaling factor for gradient of fourth moment.
    :return: New matrix of zero-centered points.
        (tangent_dim, num_of_points)
    """
    target_covariance_sqrt = scipy.linalg.sqrtm(target_covariance)
    matrix = metric @ points
    if factor4 is not None:
        target_covariance_diag_inv_square = 1 / np.square(np.diag(target_covariance))
        points_cubed = np.square(points) * points
        gradient4 = (
            4 / points.shape[0] * target_covariance_diag_inv_square[:, np.newaxis] * points_cubed
        )
        matrix -= 0.5 * factor4 * gradient4
    augmented_matrix = np.block(
        [
            [target_covariance_sqrt @ matrix * np.sqrt(weights)],
            [np.sqrt(weights)],
        ]
    )
    left_singular_vectors, _, right_singular_vectors_t = np.linalg.svd(
        augmented_matrix, compute_uv=True, full_matrices=False
    )
    return (
        target_covariance_sqrt @ left_singular_vectors[:-1] @ right_singular_vectors_t
    ) * np.sqrt(1 / weights)


# This method is called UO-SPCI in the paper.
def generalized_optimal_transport(
    points: list,
    target_covariance: npt.NDArray[np.float64],
    metric: npt.NDArray[np.float64],
    *,
    weights: npt.NDArray[np.float64],
    factor4: float | None = None,
    mean=None,
) -> list:
    """Generalized optimal transport for a list of states. This method converts to/from tangent
    space at the mean and lets generalized_optimal_transport_matrix handle the rest."""
    if mean is None:
        mean = mean_of_sigma_points(points, weights=weights)
    centered_point_matrix = np.stack(
        [_ - mean for _ in points], axis=-1
    )  # (tangent_dim, num_of_points)
    new_centered_point_matrix = generalized_optimal_transport_matrix(
        centered_point_matrix, target_covariance, metric, weights=weights, factor4=factor4
    )
    return [mean + new_centered_point_matrix[:, i] for i in range(len(points))]


def _care(
    p: npt.NDArray[np.float64],
    neg_q: npt.NDArray[np.float64],
    method: SPCIMethod,
    metric: npt.NDArray[np.float64] | None = None,
):
    p_2 = 0.5 * p
    match method:
        case SPCIMethod.CARE_SCIPY:
            return scipy.linalg.solve_continuous_are(
                -p_2.T, np.eye(p.shape[0]), neg_q, np.eye(p.shape[0])
            )
        case SPCIMethod.CONSTRAINED_OPTIMAL:
            assert metric is not None
            new_cov_sqrt = scipy.linalg.sqrtm(p_2 @ p_2.T + neg_q)
            u, _, v_t = np.linalg.svd(
                new_cov_sqrt @ metric @ p_2, compute_uv=True, full_matrices=False
            )
            return -p_2 + new_cov_sqrt @ u @ v_t
    raise ValueError("Unknown SPCI method.")


def spci(
    points: list,
    delta_covariance: npt.NDArray[np.float64],
    *,
    weights: npt.NDArray[np.float64],
    method: SPCIMethod,
    factor4: float | None = None,
) -> list:
    kwargs = {}
    if method == SPCIMethod.CONSTRAINED_OPTIMAL or method == SPCIMethod.UNCONSTRAINED_OPTIMAL:
        mean = mean_of_sigma_points(points, weights=weights)
        old_covariance = covariance_of_sigma_points(points, mean, weights=weights)

        metric = cast(
            "npt.NDArray[np.float64]", np.linalg.pinv(old_covariance + 0.5 * delta_covariance)
        )

        if method == SPCIMethod.UNCONSTRAINED_OPTIMAL:
            return generalized_optimal_transport(
                points,
                old_covariance + delta_covariance,
                metric,
                weights=weights,
                factor4=factor4,
                mean=mean,
            )

        kwargs = {"metric": metric}

    n = delta_covariance.shape[0]
    # This method is not as general as GOT: It really only works for sigma point sets where all
    # points after the first have the same weight.
    assert len(points) == 2 * n + 1
    assert np.all(weights[1] == weights[1:])
    assert factor4 is None
    x_tilde = np.stack(
        [points[1 + i] - points[1 + i + n] for i in range(n)], axis=-1
    )  # (tangent_dim, tangent_dim)
    u = _care(
        x_tilde,
        0.5 / weights[1] * delta_covariance,
        method,
        **kwargs,
    )  # (tangent_dim, tangent_dim)
    return [
        points[0],
        *[points[1 + i] + u[:, i] for i in range(n)],
        *[points[1 + i + n] + (-u[:, i]) for i in range(n)],
    ]


class PPUKF(Filter):
    def __init__(
        self,
        system: System,
        initial_mean,
        initial_covariance: npt.NDArray[np.float64],
        *,
        sigma_point_method: SigmaPointMethod,
        kappa: float,
        update_method: UpdateMethod,
        spci_method: SPCIMethod,
        factor4: float | None = None,
    ):
        super().__init__(system)
        self._sigma_points = generate_sigma_points(
            initial_mean, initial_covariance, sigma_point_method, kappa=kappa
        )
        self._weights = generate_sigma_weights(initial_covariance.shape[0], kappa=kappa)
        self._update_method = update_method
        self._spci_method = spci_method

        # Only needed if spci_method == UNCONSTRAINED_OPTIMAL
        self._factor4 = factor4

        # Only needed if update_method == UKF_REDRAW
        self._sigma_point_method = sigma_point_method
        self._kappa = kappa

    @override
    def predict(self, dt: float):
        sigma_points = [self._system.step(sp, dt) for sp in self._sigma_points]

        self._sigma_points = spci(
            sigma_points,
            self._system.dynamics_covariance(dt),
            weights=self._weights,
            method=self._spci_method,
            factor4=self._factor4,
        )

    def update(self, measurement, noise_covariance: npt.NDArray[np.float64]):
        sigma_measurements = [self._system.sensor(_) for _ in self._sigma_points]
        mean_of_sigma_measurements = mean_of_sigma_points(sigma_measurements, weights=self._weights)
        covariance_of_sigma_measurements = (
            covariance_of_sigma_points(
                sigma_measurements, mean_of_sigma_measurements, weights=self._weights
            )
            + noise_covariance
        )
        covariance_of_states_and_measurements = covariance_of_sigma_points(
            self._sigma_points,
            self.mean,
            sigma_measurements,
            mean_of_sigma_measurements,
            weights=self._weights,
        )
        kalman_gain = covariance_of_states_and_measurements @ np.linalg.inv(
            covariance_of_sigma_measurements
        )

        if self._update_method == UpdateMethod.PER_SIGMA_POINT:
            sigma_points = [
                _[0] + kalman_gain @ (measurement - _[1])
                for _ in zip(self._sigma_points, sigma_measurements, strict=True)
            ]

            delta_covariance = kalman_gain @ noise_covariance @ kalman_gain.T
            self._sigma_points = spci(
                sigma_points,
                delta_covariance,
                weights=self._weights,
                method=self._spci_method,
                factor4=self._factor4,
            )
        elif self._update_method == UpdateMethod.UKF_REDRAW:
            update = kalman_gain @ (measurement - mean_of_sigma_measurements)
            covariance = self.covariance - kalman_gain @ covariance_of_states_and_measurements.T
            self._sigma_points = generate_sigma_points(
                self.mean, covariance, self._sigma_point_method, offset=update, kappa=self._kappa
            )
        else:
            raise ValueError("Unknown update method.")

    @property
    @override
    def mean(self):
        return mean_of_sigma_points(self._sigma_points, weights=self._weights)

    @property
    @override
    def covariance(self) -> npt.NDArray[np.float64]:
        return covariance_of_sigma_points(self._sigma_points, self.mean, weights=self._weights)

    @property
    @override
    def sigma_points(self):
        return self._sigma_points
