from enum import Enum

import numpy as np
import numpy.typing as npt


class SigmaPointMethod(Enum):
    CHOLESKY = 0
    SQRTM = 1


def _sqrtm(matrix: npt.NDArray[np.float64], tolerance: float = 1e-6) -> npt.NDArray[np.float64]:
    """Calculates a "safe" matrix square root of a symmetric positive semi-definite matrix. Negative
    eigenvalues are clipped as long as they aren't too negative.

    :param matrix: The matrix.
        (n, n)
    :param tolerance: Fraction of the largest eigenvalue by which the smallest one may be negative.
    :return: Its square root.
        (n, n)
    """
    eva, eve = np.linalg.eigh(matrix)
    # Smallest eigenvalue is more negative than a fraction of the largest:
    if eva[0] < -tolerance * np.abs(eva[-1]):
        raise ValueError(f"Matrix too far from positive semidefinite {eva[0]} {eva[-1]}")
    return (eve * np.sqrt(np.maximum(eva, 0))) @ eve.T


def generate_sigma_points(
    mean,
    covariance: npt.NDArray[np.float64],
    method: SigmaPointMethod,
    *,
    kappa: float,
    offset: npt.NDArray[np.float64] | None = None,
) -> list:
    """Generates sigma points around a mean state and a covariance matrix.

    :param mean: An element of the state manifold that will be the mean of the sigma points.
    :param covariance: A matrix that will be the covariance of the sigma points.
        (tangent_dim, tangent_dim)
    :param method: The method to use to draw sigma points.
    :param kappa: Additional spread of sigma points.
    :param offset: Additional offset (in tangent space) for each sigma point. This is needed for
        moving the reference on non-Euclidean boxplus manifolds.
        (tangent_dim,)
    :return: List of 2 * tangent_dim + 1 sigma points.
    """
    assert len(covariance.shape) == 2
    assert covariance.shape[0] == covariance.shape[1]
    assert kappa >= 0
    n = covariance.shape[0]
    match method:
        case SigmaPointMethod.CHOLESKY:
            directions = np.linalg.cholesky((n + kappa) * covariance)
        case SigmaPointMethod.SQRTM:
            directions = _sqrtm((n + kappa) * covariance)
    delta = np.concatenate([np.zeros((n, 1), dtype=np.float64), directions, -directions], axis=-1)
    if offset is not None:
        delta = delta + offset[:, np.newaxis]
    return [mean + delta[:, i] for i in range(delta.shape[-1])]


def generate_sigma_weights(n: int, *, kappa: float) -> npt.NDArray[np.float64]:
    """Generates weight vector to go with sigma points from generate_sigma_points.

    :param n: The dimension of the tangent space.
    :param kappa: Additional spread of sigma points.
    :return: Array of weights, corresponding to the result of generate_sigma_points.
        (2 * n + 1,)
    """
    assert kappa >= 0
    return np.array([kappa / (n + kappa)] + [1 / (2 * (n + kappa))] * 2 * n, dtype=np.float64)


def mean_of_sigma_points(elems: list, *, weights: npt.NDArray[np.float64]):
    """Calculates the mean of a list of sigma points, potentially on a [+]-manifold.

    :param elems: The list of elements.
    :param weights: Array of weights per element.
        (num_of_points,)
    :return: The mean.
    """
    assert len(elems) > 0
    assert len(weights.shape) == 1
    assert weights.shape[0] == len(elems)
    if isinstance(elems[0], np.ndarray):
        assert len(elems[0].shape) == 1
        return np.sum(np.stack(elems, axis=-1) * weights, axis=-1)
    elif hasattr(elems[0].__class__, "mean"):
        return elems[0].__class__.mean(elems, weights=weights)
    else:
        # Fixed point iteration. The 0th element is a good start since it is often the "mean" sigma
        # point.
        mean = elems[0]
        for _ in range(20):
            delta = np.sum(np.stack([_ - mean for _ in elems], axis=-1) * weights, axis=-1)
            if np.linalg.norm(delta) < 1e-7:
                return mean
            mean = mean + delta
        raise RuntimeError(f"[+] mean iteration did not converge {mean} {elems}.")


def covariance_of_sigma_points(
    elems1: list,
    mean1,
    elems2: list | None = None,
    mean2=None,
    *,
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates the covariance matrix over a list of sigma points, or the cross-covariance matrix
    with another list (of the same length, but not necessarily same dimension).

    :param elems1: The list of elements.
    :param mean1: The mean of the elements.
    :param elems2: Optional second list of elements.
    :param mean2: Optional mean of the second list of elements.
    :param weights: Array of weights per element.
        (num_of_points,)
    :return: The (cross-)covariance matrix.
        (tangent_dim1, tangent_dim2)
    """
    assert len(elems1) > 0
    if mean2 is None or elems2 is None:
        mean2 = mean1
        elems2 = elems1
    assert len(weights.shape) == 1
    assert weights.shape[0] == len(elems1)
    return np.sum(
        np.stack(
            [np.outer(_[0] - mean1, _[1] - mean2) for _ in zip(elems1, elems2, strict=True)],
            axis=-1,
        )
        * weights,
        axis=-1,
    )
