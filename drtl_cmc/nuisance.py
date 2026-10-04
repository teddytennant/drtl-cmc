"""Nuisance fits from Section 3.

Density ratio working model (eq. 11):
    omega(y, z; eta) = exp(y * eta_y + z' eta_z)

Empirical objective (eq. 13), source 1..n and target n+1..n+N:
    (1/n) sum_source exp(y_i eta_y + z_i' eta_z)
        - (1/N) sum_target (y_i eta_y + z_i' eta_z)

Conditional covariate model used in the simulation (eq. 15):
    q(x | y, z) = Normal(B' (y, z), Sigma)
fitted by Gaussian MLE on the source sample only.
"""

from typing import NamedTuple, Optional

import jax.numpy as jnp

from drtl_cmc._jaxsetup import ensure_x64

ensure_x64()

Array = jnp.ndarray


class DensityRatioFit(NamedTuple):
    """eta stacks (eta_y, eta_z). Z includes its intercept column."""

    eta: Array
    foc_norm: Array
    n_iter: int


class GaussianConditional(NamedTuple):
    """B is (1 + q, p) so that E_q[X | y, z] = (y, z) @ B. Sigma is (p, p)."""

    B: Array
    Sigma: Array


def features(y: Array, z: Array) -> Array:
    """Columns (y, z_1, ..., z_q). z is (n, q) and already contains an intercept."""
    y = jnp.asarray(y).reshape(-1)
    z = jnp.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    return jnp.concatenate([y.reshape(-1, 1), z], axis=1)


def density_ratio_weights(eta: Array, y: Array, z: Array, clip: float = 40.0) -> Array:
    """omega(y, z; eta). Linear predictor is clipped only as an overflow guard.

    At a converged fit the clip does not bind for the designs in Section 5.
    """
    lin = features(y, z) @ jnp.asarray(eta)
    if clip is not None:
        lin = jnp.clip(lin, -clip, clip)
    return jnp.exp(lin)


def density_ratio_foc(
    eta: Array,
    y_source: Array,
    z_source: Array,
    y_target: Array,
    z_target: Array,
) -> Array:
    """First-order condition of eq. 13, without clipping.

    (1/n) sum_source omega_i (y_i, z_i) - (1/N) sum_target (y_i, z_i) = 0.
    """
    fs = features(y_source, z_source)
    ft = features(y_target, z_target)
    omega = jnp.exp(fs @ jnp.asarray(eta))
    return (omega[:, None] * fs).mean(axis=0) - ft.mean(axis=0)


def density_ratio_objective(
    eta: Array,
    y_source: Array,
    z_source: Array,
    y_target: Array,
    z_target: Array,
) -> Array:
    """Empirical objective (eq. 13)."""
    fs = features(y_source, z_source)
    ft = features(y_target, z_target)
    eta = jnp.asarray(eta)
    return jnp.mean(jnp.exp(fs @ eta)) - jnp.mean(ft @ eta)


def _grad_hess(eta: Array, fs: Array, target_mean: Array):
    lin = fs @ eta
    omega = jnp.exp(lin)
    grad = (omega[:, None] * fs).mean(axis=0) - target_mean
    hess = jnp.einsum("n,ni,nj->ij", omega, fs, fs) / fs.shape[0]
    return grad, hess, omega


def fit_density_ratio(
    y_source: Array,
    z_source: Array,
    y_target: Array,
    z_target: Array,
    *,
    eta_init: Optional[Array] = None,
    tol: float = 1e-12,
    max_iter: int = 40,
    max_lin: float = 40.0,
) -> DensityRatioFit:
    """Minimize eq. 13 by damped Newton on the convex exponential-tilt objective.

    Steps whose source linear predictor would overflow are rejected. The
    returned eta is the unclipped critical point: its FOC norm is `foc_norm`.
    """
    fs = features(y_source, z_source)
    ft = features(y_target, z_target)
    target_mean = ft.mean(axis=0)
    if eta_init is None:
        eta = jnp.zeros(fs.shape[1], dtype=fs.dtype)
    else:
        eta = jnp.asarray(eta_init, dtype=fs.dtype)

    eye = jnp.eye(fs.shape[1], dtype=fs.dtype)
    foc_norm = jnp.asarray(jnp.inf)
    it = 0
    for it in range(max_iter):
        grad, hess, _ = _grad_hess(eta, fs, target_mean)
        foc_norm = jnp.linalg.norm(grad)
        if float(foc_norm) < tol:
            break
        # Tiny ridge keeps the Newton step defined if a feature is constant.
        step = jnp.linalg.solve(hess + 1e-12 * eye, grad)
        scale = 1.0
        accepted = False
        base = float(foc_norm)
        for _ in range(20):
            eta_try = eta - scale * step
            lin = fs @ eta_try
            if float(jnp.max(lin)) > max_lin or not bool(jnp.all(jnp.isfinite(lin))):
                scale *= 0.5
                continue
            grad_try, _, _ = _grad_hess(eta_try, fs, target_mean)
            nrm_try = float(jnp.linalg.norm(grad_try))
            if nrm_try < base * (1.0 - 1e-4 * scale) or nrm_try < tol:
                eta = eta_try
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    # Recompute the unclipped FOC at the returned point.
    foc = density_ratio_foc(eta, y_source, z_source, y_target, z_target)
    return DensityRatioFit(eta=eta, foc_norm=jnp.linalg.norm(foc), n_iter=it + 1)


def _as_design(y: Array, x: Array, z: Array):
    y = jnp.asarray(y).reshape(-1)
    z = jnp.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] != y.shape[0] or z.shape[0] != y.shape[0]:
        raise ValueError("y, x, and z must share the sample axis")
    return y, x, z


def fit_gaussian_conditional(y: Array, x: Array, z: Array) -> GaussianConditional:
    """Gaussian MLE of q(x | y, z) = Normal(B' (y, z), Sigma) on the source sample.

    B solves the OLS normal equations. Sigma is the MLE residual covariance
    (divisor n, not n - dim).
    """
    y, x, z = _as_design(y, x, z)
    W = features(y, z)
    B, _, _, _ = jnp.linalg.lstsq(W, x, rcond=None)
    resid = x - W @ B
    n = x.shape[0]
    Sigma = (resid.T @ resid) / n
    Sigma = 0.5 * (Sigma + Sigma.T)
    return GaussianConditional(B=B, Sigma=Sigma)
