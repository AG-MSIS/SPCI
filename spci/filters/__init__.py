from .base import Filter
from .ppukf import PPUKF, SPCIMethod, UpdateMethod
from .ukf import UKF
from .ukf_utils import SigmaPointMethod

CONSTRUCTORS = {
    "ukf": (
        lambda system, mean, covariance: UKF(
            system, mean, covariance, sigma_point_method=SigmaPointMethod.SQRTM, kappa=1.0
        ),
        "UKF",
    ),
    "ppukf": (
        lambda system, mean, covariance: PPUKF(
            system,
            mean,
            covariance,
            sigma_point_method=SigmaPointMethod.SQRTM,
            kappa=1.0,
            update_method=UpdateMethod.PER_SIGMA_POINT,
            spci_method=SPCIMethod.CARE_SCIPY,
        ),
        "PPUKF",
    ),
    "o_ppukf": (
        lambda system, mean, covariance: PPUKF(
            system,
            mean,
            covariance,
            sigma_point_method=SigmaPointMethod.SQRTM,
            kappa=1.0,
            update_method=UpdateMethod.PER_SIGMA_POINT,
            spci_method=SPCIMethod.CONSTRAINED_OPTIMAL,
        ),
        "O-PPUKF",
    ),
    "or_ppukf": (
        lambda system, mean, covariance: PPUKF(
            system,
            mean,
            covariance,
            sigma_point_method=SigmaPointMethod.SQRTM,
            kappa=1.0,
            update_method=UpdateMethod.UKF_REDRAW,
            spci_method=SPCIMethod.CONSTRAINED_OPTIMAL,
        ),
        "OR-PPUKF",
    ),
    "uo_ppukf": (
        lambda system, mean, covariance: PPUKF(
            system,
            mean,
            covariance,
            sigma_point_method=SigmaPointMethod.SQRTM,
            kappa=1.0,
            update_method=UpdateMethod.PER_SIGMA_POINT,
            spci_method=SPCIMethod.UNCONSTRAINED_OPTIMAL,
        ),
        "UO-PPUKF",
    ),
    "uor_ppukf": (
        lambda system, mean, covariance: PPUKF(
            system,
            mean,
            covariance,
            sigma_point_method=SigmaPointMethod.SQRTM,
            kappa=1.0,
            update_method=UpdateMethod.UKF_REDRAW,
            spci_method=SPCIMethod.UNCONSTRAINED_OPTIMAL,
        ),
        "UOR-PPUKF",
    ),
}

__all__ = ["CONSTRUCTORS", "Filter"]
