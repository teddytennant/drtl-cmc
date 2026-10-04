"""Doubly robust target inference with a completely missing covariate."""

from drtl_cmc._jaxsetup import ensure_x64

ensure_x64()

from drtl_cmc.estimators import FitResult, fit_dr, fit_glm, fit_imp, fit_iw
from drtl_cmc.links import IDENTITY, LOGISTIC
from drtl_cmc.nuisance import (
    DensityRatioFit,
    GaussianConditional,
    density_ratio_foc,
    density_ratio_weights,
    fit_density_ratio,
    fit_gaussian_conditional,
)
from drtl_cmc.simulate import simulate_cmc

__all__ = [
    "IDENTITY",
    "LOGISTIC",
    "DensityRatioFit",
    "FitResult",
    "GaussianConditional",
    "density_ratio_foc",
    "density_ratio_weights",
    "fit_density_ratio",
    "fit_dr",
    "fit_gaussian_conditional",
    "fit_glm",
    "fit_imp",
    "fit_iw",
    "simulate_cmc",
]
