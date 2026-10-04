"""Importance-weighted, imputation, and doubly robust estimating equations.

Psi_IW(beta)  = (1/n) sum_source omega_i phi(y_i, x_i, z_i; beta)
Psi_IMP(beta) = (1/N) sum_target phi_bar_q(y_i, z_i; beta)
Psi_DR(beta)  = Psi_IMP(beta)
              + (1/n) sum_source omega_i (phi_i - phi_bar_q(y_i, z_i; beta))

Each equation is solved by damped Newton. Logistic conditional functionals use
a Monte Carlo sample drawn once and held fixed across Newton steps.
"""

from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from drtl_cmc._jaxsetup import ensure_x64
from drtl_cmc.links import IDENTITY, Link
from drtl_cmc.nuisance import GaussianConditional
from drtl_cmc.scores import (
    complete_score_and_jac,
    identity_imputed_score_and_jac,
    mc_imputed_score_and_jac,
)

ensure_x64()

Array = jnp.ndarray


class FitResult(NamedTuple):
    beta: Array
    score_norm: Array
    n_iter: int


def _as_x(x: Array, n: int) -> Array:
    x = jnp.asarray(x)
    if x.ndim == 1:
        if x.shape[0] != n:
            raise ValueError("x length does not match y")
        return x.reshape(n, 1)
    return x


def _as_z(z: Array) -> Array:
    z = jnp.asarray(z)
    if z.ndim == 1:
        return z.reshape(-1, 1)
    return z


def _as_omega(omega: Array, n: int) -> Array:
    omega = jnp.asarray(omega)
    if omega.ndim == 0:
        return jnp.full((n,), omega)
    if omega.shape[0] != n:
        raise ValueError("omega length does not match the source sample")
    return omega


@partial(jax.jit, static_argnames=("link_id",))
def iw_score_jac(
    beta: Array,
    y: Array,
    x: Array,
    z: Array,
    omega: Array,
    link_id: int,
) -> Tuple[Array, Array]:
    phi, jac = complete_score_and_jac(y, x, z, beta, link_id)
    psi = (omega[:, None] * phi).mean(axis=0)
    J = (omega[:, None, None] * jac).mean(axis=0)
    return psi, J


@partial(jax.jit, static_argnames=("link_id",))
def imp_score_jac(
    beta: Array,
    y: Array,
    z: Array,
    B: Array,
    Sigma: Array,
    eps: Array,
    link_id: int,
) -> Tuple[Array, Array]:
    conditional = GaussianConditional(B, Sigma)
    if link_id == 0:
        phi, jac = identity_imputed_score_and_jac(y, z, beta, conditional)
    else:
        phi, jac = mc_imputed_score_and_jac(y, z, beta, conditional, eps, link_id)
    return phi.mean(axis=0), jac.mean(axis=0)


@partial(jax.jit, static_argnames=("link_id",))
def dr_score_jac(
    beta: Array,
    y_s: Array,
    x_s: Array,
    z_s: Array,
    y_t: Array,
    z_t: Array,
    omega: Array,
    B: Array,
    Sigma: Array,
    eps_s: Array,
    eps_t: Array,
    link_id: int,
) -> Tuple[Array, Array]:
    conditional = GaussianConditional(B, Sigma)
    phi_s, jac_s = complete_score_and_jac(y_s, x_s, z_s, beta, link_id)
    if link_id == 0:
        bar_s, jbar_s = identity_imputed_score_and_jac(y_s, z_s, beta, conditional)
        bar_t, jbar_t = identity_imputed_score_and_jac(y_t, z_t, beta, conditional)
    else:
        bar_s, jbar_s = mc_imputed_score_and_jac(
            y_s, z_s, beta, conditional, eps_s, link_id
        )
        bar_t, jbar_t = mc_imputed_score_and_jac(
            y_t, z_t, beta, conditional, eps_t, link_id
        )
    psi = bar_t.mean(axis=0) + (omega[:, None] * (phi_s - bar_s)).mean(axis=0)
    J = jbar_t.mean(axis=0) + (omega[:, None, None] * (jac_s - jbar_s)).mean(axis=0)
    return psi, J


def newton(
    score_jac,
    beta: Array,
    *,
    tol: float = 1e-10,
    max_iter: int = 30,
    ridge: float = 1e-12,
) -> FitResult:
    """Damped Newton. `score_jac(beta)` returns (psi, d psi / d beta')."""
    beta = jnp.asarray(beta)
    eye = jnp.eye(beta.shape[0], dtype=beta.dtype)
    nrm = jnp.asarray(jnp.inf)
    it = 0
    for it in range(max_iter):
        psi, jac = score_jac(beta)
        nrm = jnp.linalg.norm(psi)
        if not bool(jnp.isfinite(nrm)):
            break
        if float(nrm) < tol:
            return FitResult(beta=beta, score_norm=nrm, n_iter=it)
        step = jnp.linalg.solve(jac - ridge * eye, psi)
        if not bool(jnp.all(jnp.isfinite(step))):
            step, _, _, _ = jnp.linalg.lstsq(jac, psi, rcond=None)
        scale = 1.0
        accepted = False
        base = float(nrm)
        for _ in range(16):
            beta_try = beta - scale * step
            psi_try, _ = score_jac(beta_try)
            nrm_try = float(jnp.linalg.norm(psi_try))
            if nrm_try <= base * (1.0 - 1e-4 * scale) or nrm_try < tol:
                beta = beta_try
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    psi, _ = score_jac(beta)
    return FitResult(beta=beta, score_norm=jnp.linalg.norm(psi), n_iter=it + 1)


def _draw_eps(key: Optional[Array], K: int, n: int, p: int, dtype) -> Array:
    if key is None:
        raise ValueError("logistic moments require a PRNG key")
    return jax.random.normal(key, (K, n, p), dtype=dtype)


def _dummy_eps(n: int, p: int, dtype) -> Array:
    return jnp.zeros((1, n, p), dtype=dtype)


def fit_glm(
    y: Array,
    x: Array,
    z: Array,
    link: Link = IDENTITY,
    *,
    beta_init: Optional[Array] = None,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> FitResult:
    """Complete-data GLM MLE, the unweighted score E[phi] = 0."""
    y = jnp.asarray(y).reshape(-1)
    x = _as_x(x, y.shape[0])
    z = _as_z(z)
    omega = jnp.ones(y.shape[0], dtype=y.dtype)
    if beta_init is None:
        beta_init = jnp.zeros(x.shape[1] + z.shape[1], dtype=y.dtype)
    return newton(
        lambda beta: iw_score_jac(beta, y, x, z, omega, link.link_id),
        beta_init,
        tol=tol,
        max_iter=max_iter,
    )


def fit_iw(
    y: Array,
    x: Array,
    z: Array,
    omega: Array,
    link: Link = IDENTITY,
    *,
    beta_init: Optional[Array] = None,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> FitResult:
    """Solve the importance-weighted source equation."""
    y = jnp.asarray(y).reshape(-1)
    x = _as_x(x, y.shape[0])
    z = _as_z(z)
    omega = _as_omega(omega, y.shape[0]).astype(y.dtype)
    if beta_init is None:
        beta_init = jnp.zeros(x.shape[1] + z.shape[1], dtype=y.dtype)
    return newton(
        lambda beta: iw_score_jac(beta, y, x, z, omega, link.link_id),
        beta_init,
        tol=tol,
        max_iter=max_iter,
    )


def fit_imp(
    y: Array,
    z: Array,
    conditional: GaussianConditional,
    link: Link = IDENTITY,
    *,
    key: Optional[Array] = None,
    K: int = 128,
    beta_init: Optional[Array] = None,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> FitResult:
    """Solve the target imputation equation."""
    y = jnp.asarray(y).reshape(-1)
    z = _as_z(z)
    p = conditional.B.shape[1]
    if link.closed_form_gaussian:
        eps = _dummy_eps(y.shape[0], p, y.dtype)
    else:
        eps = _draw_eps(key, K, y.shape[0], p, y.dtype)
    if beta_init is None:
        beta_init = jnp.zeros(p + z.shape[1], dtype=y.dtype)
    B = conditional.B
    Sigma = conditional.Sigma
    link_id = link.link_id
    return newton(
        lambda beta: imp_score_jac(beta, y, z, B, Sigma, eps, link_id),
        beta_init,
        tol=tol,
        max_iter=max_iter,
    )


def fit_dr(
    y_source: Array,
    x_source: Array,
    z_source: Array,
    y_target: Array,
    z_target: Array,
    omega: Array,
    conditional: GaussianConditional,
    link: Link = IDENTITY,
    *,
    key: Optional[Array] = None,
    K: int = 128,
    beta_init: Optional[Array] = None,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> FitResult:
    """Solve the doubly robust augmented equation (eq. 18)."""
    y_source = jnp.asarray(y_source).reshape(-1)
    y_target = jnp.asarray(y_target).reshape(-1)
    x_source = _as_x(x_source, y_source.shape[0])
    z_source = _as_z(z_source)
    z_target = _as_z(z_target)
    omega = _as_omega(omega, y_source.shape[0]).astype(y_source.dtype)
    p = conditional.B.shape[1]
    if link.closed_form_gaussian:
        eps_s = _dummy_eps(y_source.shape[0], p, y_source.dtype)
        eps_t = _dummy_eps(y_target.shape[0], p, y_target.dtype)
    else:
        if key is None:
            raise ValueError("logistic DR requires a PRNG key")
        k_s, k_t = jax.random.split(key)
        eps_s = _draw_eps(k_s, K, y_source.shape[0], p, y_source.dtype)
        eps_t = _draw_eps(k_t, K, y_target.shape[0], p, y_target.dtype)
    if beta_init is None:
        beta_init = jnp.zeros(p + z_source.shape[1], dtype=y_source.dtype)
    B = conditional.B
    Sigma = conditional.Sigma
    link_id = link.link_id
    return newton(
        lambda beta: dr_score_jac(
            beta,
            y_source,
            x_source,
            z_source,
            y_target,
            z_target,
            omega,
            B,
            Sigma,
            eps_s,
            eps_t,
            link_id,
        ),
        beta_init,
        tol=tol,
        max_iter=max_iter,
    )


def source_glm_start(
    y_source: Array,
    x_source: Array,
    z_source: Array,
    link: Link,
) -> Array:
    """Complete-data GLM on the source sample, a reasonable Newton start."""
    return fit_glm(y_source, x_source, z_source, link).beta
