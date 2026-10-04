"""Complete-data score, imputed score, and conditional functionals.

phi(y, x, z; beta) = (x, z) * (y - g(x' beta_x + z' beta_z))

Working functionals under q (eq. 9), with Monte Carlo (eq. 17) when g is not
linear. For the identity link and Gaussian q the moments are closed form:

    m1 = B' (y, z)
    m2 = m1' beta_x + z' beta_z
    m3 = (Sigma + m1 m1') beta_x + m1 * (z' beta_z)

Imputed score:
    phi_bar_q(y, z; beta) = (m1 * y - m3, z * (y - m2))
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from drtl_cmc._jaxsetup import ensure_x64
from drtl_cmc.links import IDENTITY, LOGISTIC, Link
from drtl_cmc.nuisance import GaussianConditional, features

ensure_x64()

Array = jnp.ndarray


def covariate_matrix(x: Array, z: Array) -> Array:
    x = jnp.asarray(x)
    z = jnp.asarray(z)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    return jnp.concatenate([x, z], axis=1)


def split_beta(beta: Array, p: int) -> Tuple[Array, Array]:
    beta = jnp.asarray(beta)
    return beta[:p], beta[p:]


def linear_predictor(x: Array, z: Array, beta: Array) -> Array:
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    p = x.shape[-1]
    beta_x, beta_z = split_beta(beta, p)
    return x @ beta_x + jnp.asarray(z) @ beta_z


def complete_score(y: Array, x: Array, z: Array, beta: Array, link: Link) -> Array:
    """Rows of phi(y, x, z; beta). Shape (n, p + q)."""
    y = jnp.asarray(y).reshape(-1)
    u = covariate_matrix(x, z)
    resid = y - link.g(u @ jnp.asarray(beta))
    return u * resid[:, None]


def gaussian_m1(y: Array, z: Array, conditional: GaussianConditional) -> Array:
    """E_q[X | y, z] = (y, z) @ B. Shape (n, p)."""
    return features(y, z) @ conditional.B


def identity_moments(
    y: Array,
    z: Array,
    beta: Array,
    conditional: GaussianConditional,
) -> Tuple[Array, Array, Array]:
    """Closed-form m1, m2, m3 for the identity link and Gaussian q."""
    m1 = gaussian_m1(y, z, conditional)
    p = m1.shape[-1]
    beta_x, beta_z = split_beta(beta, p)
    z = jnp.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    z_lin = z @ beta_z
    m2 = m1 @ beta_x + z_lin
    # (Sigma + m1 m1') beta_x + m1 * (z' beta_z)
    m3 = (conditional.Sigma @ beta_x) + m1 * (m1 @ beta_x + z_lin)[:, None]
    return m1, m2, m3


def _cholesky(Sigma: Array) -> Array:
    p = Sigma.shape[0]
    jitter = jnp.eye(p, dtype=Sigma.dtype) * 1e-12
    return jnp.linalg.cholesky(0.5 * (Sigma + Sigma.T) + jitter)


def mc_moments(
    y: Array,
    z: Array,
    beta: Array,
    conditional: GaussianConditional,
    link: Link,
    key: Array,
    K: int,
    eps: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    """Monte Carlo m1, m2, m3 (eq. 17).

    m1 is the exact Gaussian mean (closed form is available). m2 and m3 are
    averages of K draws X^{(k)} ~ q(. | y, z). Pass `eps` of shape (K, n, p)
    to reuse draws; otherwise `key` generates them.
    """
    y = jnp.asarray(y).reshape(-1)
    z = jnp.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    m1 = gaussian_m1(y, z, conditional)
    n, p = m1.shape
    if eps is None:
        if key is None:
            raise ValueError("mc_moments requires key or eps")
        eps = jax.random.normal(key, (K, n, p), dtype=m1.dtype)
    else:
        eps = jnp.asarray(eps)
        K = eps.shape[0]
    L = _cholesky(conditional.Sigma)
    X = m1[None, :, :] + eps @ L.T
    beta_x, beta_z = split_beta(beta, p)
    eta = jnp.einsum("knp,p->kn", X, beta_x) + (z @ beta_z)[None, :]
    g_val = link.g(eta)
    m2 = g_val.mean(axis=0)
    m3 = jnp.einsum("kn,knp->np", g_val, X) / K
    return m1, m2, m3


def conditional_moments(
    y: Array,
    z: Array,
    beta: Array,
    conditional: GaussianConditional,
    link: Link,
    *,
    key: Optional[Array] = None,
    K: int = 128,
    eps: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    """m1, m2, m3. Identity + Gaussian q uses the closed form; logistic uses MC."""
    if link.closed_form_gaussian and eps is None:
        return identity_moments(y, z, beta, conditional)
    if link.name == "identity" and eps is None and key is None:
        return identity_moments(y, z, beta, conditional)
    return mc_moments(y, z, beta, conditional, link, key, K, eps=eps)


def imputed_score_from_moments(
    y: Array,
    z: Array,
    m1: Array,
    m2: Array,
    m3: Array,
) -> Array:
    """phi_bar = (m1 * y - m3, z * (y - m2))."""
    y = jnp.asarray(y).reshape(-1)
    z = jnp.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    x_part = m1 * y[:, None] - m3
    z_part = z * (y - m2)[:, None]
    return jnp.concatenate([x_part, z_part], axis=1)


def imputed_score(
    y: Array,
    z: Array,
    beta: Array,
    conditional: GaussianConditional,
    link: Link,
    *,
    key: Optional[Array] = None,
    K: int = 128,
    eps: Optional[Array] = None,
) -> Array:
    m1, m2, m3 = conditional_moments(
        y, z, beta, conditional, link, key=key, K=K, eps=eps
    )
    return imputed_score_from_moments(y, z, m1, m2, m3)


def _link_apply(eta: Array, link_id: int) -> Tuple[Array, Array]:
    if link_id == 0:
        return eta, jnp.ones_like(eta)
    if link_id == 1:
        p = jax.nn.sigmoid(eta)
        return p, p * (1.0 - p)
    raise ValueError(f"unknown link_id {link_id}")


def complete_score_and_jac(
    y: Array,
    x: Array,
    z: Array,
    beta: Array,
    link_id: int,
) -> Tuple[Array, Array]:
    """phi and d phi / d beta' for each row. Shapes (n, d) and (n, d, d)."""
    y = jnp.asarray(y).reshape(-1)
    u = covariate_matrix(x, z)
    eta = u @ jnp.asarray(beta)
    g_val, g_prime = _link_apply(eta, link_id)
    phi = u * (y - g_val)[:, None]
    jac = -g_prime[:, None, None] * (u[:, :, None] * u[:, None, :])
    return phi, jac


def identity_imputed_score_and_jac(
    y: Array,
    z: Array,
    beta: Array,
    conditional: GaussianConditional,
) -> Tuple[Array, Array]:
    """Imputed score and Jacobian under identity link + Gaussian q.

    phi_bar is affine in beta, so the Jacobian does not depend on beta:
    J = -v v' - blockdiag(Sigma, 0), v = (m1, z).
    """
    y = jnp.asarray(y).reshape(-1)
    z = jnp.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    m1 = gaussian_m1(y, z, conditional)
    p = m1.shape[1]
    d = p + z.shape[1]
    v = jnp.concatenate([m1, z], axis=1)
    m1, m2, m3 = identity_moments(y, z, beta, conditional)
    phi = imputed_score_from_moments(y, z, m1, m2, m3)
    vv = v[:, :, None] * v[:, None, :]
    extra = jnp.zeros((d, d), dtype=v.dtype).at[:p, :p].set(conditional.Sigma)
    jac = -vv - extra
    return phi, jac


def mc_imputed_score_and_jac(
    y: Array,
    z: Array,
    beta: Array,
    conditional: GaussianConditional,
    eps: Array,
    link_id: int,
) -> Tuple[Array, Array]:
    """Imputed score and Jacobian for a fixed Monte Carlo sample `eps` (K, n, p)."""
    y = jnp.asarray(y).reshape(-1)
    z = jnp.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    m1 = gaussian_m1(y, z, conditional)
    n, p = m1.shape
    q = z.shape[1]
    K = eps.shape[0]
    L = _cholesky(conditional.Sigma)
    X = m1[None, :, :] + eps @ L.T
    beta_x, beta_z = split_beta(beta, p)
    eta = jnp.einsum("knp,p->kn", X, beta_x) + (z @ beta_z)[None, :]
    g_val, g_prime = _link_apply(eta, link_id)
    m2 = g_val.mean(axis=0)
    m3 = jnp.einsum("kn,knp->np", g_val, X) / K
    phi = imputed_score_from_moments(y, z, m1, m2, m3)

    dm2_bx = jnp.einsum("kn,knp->np", g_prime, X) / K
    dm2_bz = jnp.einsum("kn,nq->nq", g_prime, z) / K
    dm3_bx = jnp.einsum("kn,knp,knr->npr", g_prime, X, X) / K
    dm3_bz = jnp.einsum("kn,knp,nq->npq", g_prime, X, z) / K

    jac_xx = -dm3_bx
    jac_xz = -dm3_bz
    jac_zx = -z[:, :, None] * dm2_bx[:, None, :]
    jac_zz = -z[:, :, None] * dm2_bz[:, None, :]
    top = jnp.concatenate([jac_xx, jac_xz], axis=-1)
    bot = jnp.concatenate([jac_zx, jac_zz], axis=-1)
    jac = jnp.concatenate([top, bot], axis=1)
    return phi, jac


def link_from_id(link_id: int) -> Link:
    if link_id == 0:
        return IDENTITY
    if link_id == 1:
        return LOGISTIC
    raise ValueError(f"unknown link_id {link_id}")
