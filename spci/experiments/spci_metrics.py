from typing import cast

import numpy as np
import numpy.typing as npt

from ..filters.ppukf import SPCIMethod, spci
from ..filters.ukf_utils import (
    covariance_of_sigma_points,
    generate_sigma_weights,
    mean_of_sigma_points,
)


def got_metric(
    points: list,
    other_points: list,
    metric: npt.NDArray[np.float64],
    *,
    weights: npt.NDArray[np.float64],
):
    def sqr(diff):
        return np.einsum("ni,ij,nj->n", diff, metric, diff)

    return np.sum(weights * sqr(np.stack([_[0] - _[1] for _ in zip(points, other_points)])))


def alpha_3(points: list, *, weights: npt.NDArray[np.float64]):
    mean = mean_of_sigma_points(points, weights=weights)  # (n,)
    variance = np.diag(covariance_of_sigma_points(points, mean, weights=weights))  # (n,)
    stddev = np.sqrt(variance)
    diff = np.stack(points, axis=-1) - mean[:, np.newaxis]  # (n, m)
    return np.sum(weights * (diff * np.square(diff)), axis=-1) / stddev**3


def alpha_4(points: list, *, weights: npt.NDArray[np.float64]):
    mean = mean_of_sigma_points(points, weights=weights)  # (n,)
    variance = np.diag(covariance_of_sigma_points(points, mean, weights=weights))  # (n,)
    diff = np.stack(points, axis=-1) - mean[:, np.newaxis]  # (n, m)
    return np.sum(weights * np.square(np.square(diff)), axis=-1) / np.square(variance)


def calculate_spci_metrics(epsilon, random_covariance):
    num_of_episodes = 10000
    dims = 2
    scale = np.array([[1, 0], [0, 1000]], dtype=np.float64)
    scale_inv = np.linalg.inv(scale)
    kappa = 1.0
    weights = generate_sigma_weights(dims, kappa=kappa)

    got_metric_data = np.zeros((num_of_episodes, 3), dtype=np.float64)
    scale_rmse_data = np.zeros((num_of_episodes, 3), dtype=np.float64)
    skewness_data = np.zeros((num_of_episodes, 3), dtype=np.float64)
    kurtosis_data = np.zeros((num_of_episodes, 3), dtype=np.float64)

    generator = np.random.default_rng()

    for i in range(num_of_episodes):
        points = [generator.normal(size=(dims,)) for _ in range(2 * dims + 1)]
        mean = mean_of_sigma_points(points, weights=weights)
        old_covariance = covariance_of_sigma_points(points, mean, weights=weights)

        if random_covariance:
            angle = generator.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            delta_covariance = (
                rotation @ np.diag(generator.uniform(0, epsilon, size=(dims,))) @ rotation.T
            )
        else:
            delta_covariance = epsilon * np.eye(dims)

        scaled_points = [scale @ _ for _ in points]

        scaled_delta_covariance = scale @ delta_covariance @ scale.T

        metric = cast(
            "npt.NDArray[np.float64]", np.linalg.pinv(old_covariance + 0.5 * delta_covariance)
        )

        spci_points = spci(points, delta_covariance, weights=weights, method=SPCIMethod.CARE_SCIPY)
        ospci_points = spci(
            points, delta_covariance, weights=weights, method=SPCIMethod.CONSTRAINED_OPTIMAL
        )
        uospci_points = spci(
            points, delta_covariance, weights=weights, method=SPCIMethod.UNCONSTRAINED_OPTIMAL
        )

        scaled_spci_points = spci(
            scaled_points, scaled_delta_covariance, weights=weights, method=SPCIMethod.CARE_SCIPY
        )
        scaled_ospci_points = spci(
            scaled_points,
            scaled_delta_covariance,
            weights=weights,
            method=SPCIMethod.CONSTRAINED_OPTIMAL,
        )
        scaled_uospci_points = spci(
            scaled_points,
            scaled_delta_covariance,
            weights=weights,
            method=SPCIMethod.UNCONSTRAINED_OPTIMAL,
        )

        got_metric_data[i, 0] = got_metric(points, spci_points, metric, weights=weights)
        got_metric_data[i, 1] = got_metric(points, ospci_points, metric, weights=weights)
        got_metric_data[i, 2] = got_metric(points, uospci_points, metric, weights=weights)

        # mean squared error (weighted) of points - scale_inv @ scaled_spci_points
        scale_rmse_data[i, 0] = np.sqrt(
            np.sum(
                np.square(
                    np.stack(
                        [_[0] - scale_inv @ _[1] for _ in zip(spci_points, scaled_spci_points)],
                        axis=-1,
                    )
                )
                * weights
            )
            / 2
        )
        scale_rmse_data[i, 1] = np.sqrt(
            np.sum(
                np.square(
                    np.stack(
                        [_[0] - scale_inv @ _[1] for _ in zip(ospci_points, scaled_ospci_points)],
                        axis=-1,
                    )
                )
                * weights
            )
            / 2
        )
        scale_rmse_data[i, 2] = np.sqrt(
            np.sum(
                np.square(
                    np.stack(
                        [_[0] - scale_inv @ _[1] for _ in zip(uospci_points, scaled_uospci_points)],
                        axis=-1,
                    )
                )
                * weights
            )
            / 2
        )

        skewness = alpha_3(points, weights=weights)
        skewness_data[i, 0] = np.mean(np.abs(alpha_3(spci_points, weights=weights) - skewness))
        skewness_data[i, 1] = np.mean(np.abs(alpha_3(ospci_points, weights=weights) - skewness))
        skewness_data[i, 2] = np.mean(np.abs(alpha_3(uospci_points, weights=weights) - skewness))

        kurtosis = alpha_4(points, weights=weights)
        kurtosis_data[i, 0] = np.mean(np.abs(alpha_4(spci_points, weights=weights) - kurtosis))
        kurtosis_data[i, 1] = np.mean(np.abs(alpha_4(ospci_points, weights=weights) - kurtosis))
        kurtosis_data[i, 2] = np.mean(np.abs(alpha_4(uospci_points, weights=weights) - kurtosis))

    print(f"epsilon={epsilon}, random={random_covariance}")
    print(
        "GOT Metric (SPCI(care), O-SPCI, UO-SPCI): ",
        np.mean(got_metric_data[:, 0]),
        np.mean(got_metric_data[:, 1]),
        np.mean(got_metric_data[:, 2]),
    )
    print(
        "Scale RMSE (SPCI(care), O-SPCI, UO-SPCI): ",
        np.mean(scale_rmse_data[:, 0]),
        np.mean(scale_rmse_data[:, 1]),
        np.mean(scale_rmse_data[:, 2]),
    )
    print(
        "Skewness Deviation (SPCI(care), O-SPCI, UO-SPCI): ",
        np.mean(skewness_data[:, 0]),
        np.mean(skewness_data[:, 1]),
        np.mean(skewness_data[:, 2]),
    )
    print(
        "Kurtosis Deviation (SPCI(care), O-SPCI, UO-SPCI): ",
        np.mean(kurtosis_data[:, 0]),
        np.mean(kurtosis_data[:, 1]),
        np.mean(kurtosis_data[:, 2]),
    )


if __name__ == "__main__":
    calculate_spci_metrics(1.0, False)
    calculate_spci_metrics(0.1, False)
    calculate_spci_metrics(1.0, True)
    calculate_spci_metrics(0.1, True)
