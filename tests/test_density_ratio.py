"""Density-ratio fit on data from the exponential-tilt model (eq. 11, eq. 13)."""

import jax
import jax.numpy as jnp

from drtl_cmc.nuisance import (
    density_ratio_foc,
    density_ratio_objective,
    density_ratio_weights,
    fit_density_ratio,
)


def test_exponential_tilt_foc_and_weights():
    # Source: (Y, Z2, Z3) independent N(0, 1), Z = (1, Z2, Z3).
    # Target is the exponential tilt exp(a Y + b Z2 + c Z3) / E1[exp(...)].
    # Because Z contains an intercept, the working model matches
    # w = p0/p1 = exp(eta_y Y + eta_z' Z) exactly, with
    # eta_y = a, eta_z2 = b, eta_z3 = c,
    # eta_intercept = -0.5 (a^2 + b^2 + c^2).
    a, b, c = 0.45, -0.35, 0.25
    eta_true = jnp.array([a, -0.5 * (a**2 + b**2 + c**2), b, c])

    n = 8000
    key = jax.random.PRNGKey(11)
    ks, kt = jax.random.split(key)
    y_s = jax.random.normal(jax.random.fold_in(ks, 1), (n,))
    z2_s = jax.random.normal(jax.random.fold_in(ks, 2), (n,))
    z3_s = jax.random.normal(jax.random.fold_in(ks, 3), (n,))
    z_s = jnp.stack([jnp.ones(n), z2_s, z3_s], axis=1)

    y_t = a + jax.random.normal(jax.random.fold_in(kt, 1), (n,))
    z2_t = b + jax.random.normal(jax.random.fold_in(kt, 2), (n,))
    z3_t = c + jax.random.normal(jax.random.fold_in(kt, 3), (n,))
    z_t = jnp.stack([jnp.ones(n), z2_t, z3_t], axis=1)

    fit = fit_density_ratio(y_s, z_s, y_t, z_t, tol=1e-12)
    foc = density_ratio_foc(fit.eta, y_s, z_s, y_t, z_t)
    assert float(jnp.linalg.norm(foc)) < 1e-8
    assert float(fit.foc_norm) < 1e-8

    # Gradient of eq. 13 matches a numerical derivative at eta_hat and nearby.
    def obj(eta):
        return density_ratio_objective(eta, y_s, z_s, y_t, z_t)

    eta_probe = eta_true * 0.5
    base = obj(eta_probe)
    eps = 1e-6
    num = []
    for j in range(eta_probe.size):
        num.append((obj(eta_probe.at[j].add(eps)) - base) / eps)
    assert jnp.allclose(
        jnp.stack(num),
        density_ratio_foc(eta_probe, y_s, z_s, y_t, z_t),
        atol=1e-5,
    )

    assert jnp.allclose(fit.eta, eta_true, atol=0.06)

    # omega close to the true density ratio on a fresh source draw.
    y_h = jax.random.normal(jax.random.PRNGKey(21), (2000,))
    z2_h = jax.random.normal(jax.random.PRNGKey(22), (2000,))
    z3_h = jax.random.normal(jax.random.PRNGKey(23), (2000,))
    z_h = jnp.stack([jnp.ones(2000), z2_h, z3_h], axis=1)
    omega = density_ratio_weights(fit.eta, y_h, z_h, clip=None)
    w_true = density_ratio_weights(eta_true, y_h, z_h, clip=None)
    rel = jnp.abs(omega - w_true) / w_true
    assert float(jnp.mean(rel)) < 0.05
    assert float(jnp.quantile(rel, 0.95)) < 0.12
