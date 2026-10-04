"""Theorem 1 with known nuisances, identity link.

Sub-population shift: X | (Y, Z) is the same linear Gaussian in both
populations, while the law of (Y, Z) differs. (Y, Z) is a three-component
Gaussian mixture with different weights, so w = p0/p1 is bounded and known.
beta0 is the target-population OLS coefficient.
"""

import jax
import jax.numpy as jnp

from drtl_cmc.estimators import fit_dr, fit_glm, fit_imp, fit_iw
from drtl_cmc.links import IDENTITY
from drtl_cmc.nuisance import GaussianConditional
from drtl_cmc.scores import complete_score, imputed_score

MEANS = jnp.array(
    [
        [2.2, 1.0],
        [0.0, 0.0],
        [-2.2, -1.0],
    ]
)
SDS = jnp.array([0.5, 0.5, 0.5])
W_TARGET = jnp.array([0.45, 0.10, 0.45])
W_SOURCE = jnp.array([0.12, 0.76, 0.12])

# X = 0.2 + 0.8 Y - 0.5 Z2 + 0.45 eps, features (Y, 1, Z2).
B_TRUE = jnp.array([[0.8], [0.2], [-0.5]])
SIGMA_TRUE = jnp.array([[0.45**2]])
# Drops the Y dependence. IMP with this q does not recover beta_x.
B_WRONG = jnp.array([[0.0], [0.2], [0.0]])
SIGMA_WRONG = jnp.array([[1.0]])


def _sample_pop(n, key, weights):
    k1, k2, k3 = jax.random.split(key, 3)
    comp = jax.random.choice(k1, 3, (n,), p=weights)
    noise = jax.random.normal(k2, (n, 2)) * SDS[comp][:, None]
    v = MEANS[comp] + noise
    y, z2 = v[:, 0], v[:, 1]
    z = jnp.stack([jnp.ones(n), z2], axis=1)
    x = 0.2 + 0.8 * y - 0.5 * z2 + 0.45 * jax.random.normal(k3, (n,))
    return y, x.reshape(-1, 1), z


def _log_mix(v, weights):
    logs = []
    for k in range(3):
        d = v - MEANS[k]
        lp = (
            jnp.log(weights[k])
            - jnp.sum(d**2, axis=-1) / (2.0 * SDS[k] ** 2)
            - jnp.log(2.0 * jnp.pi * SDS[k] ** 2)
        )
        logs.append(lp)
    return jnp.logaddexp(jnp.logaddexp(logs[0], logs[1]), logs[2])


def _true_w(y, z):
    v = jnp.stack([y, z[:, 1]], axis=1)
    return jnp.exp(_log_mix(v, W_TARGET) - _log_mix(v, W_SOURCE))


def _beta0():
    y, x, z = _sample_pop(400_000, jax.random.PRNGKey(100), W_TARGET)
    u = jnp.concatenate([x, z], axis=1)
    beta, _, _, _ = jnp.linalg.lstsq(u, y, rcond=None)
    return beta


def _dr_score(y_s, x_s, z_s, y_t, z_t, beta, omega, conditional):
    phi = complete_score(y_s, x_s, z_s, beta, IDENTITY)
    bar_s = imputed_score(y_s, z_s, beta, conditional, IDENTITY)
    bar_t = imputed_score(y_t, z_t, beta, conditional, IDENTITY)
    return bar_t.mean(0) + (omega[:, None] * (phi - bar_s)).mean(0)


def test_theorem1_known_nuisances_identity():
    beta0 = _beta0()
    n = 80_000
    y_s, x_s, z_s = _sample_pop(n, jax.random.PRNGKey(201), W_SOURCE)
    y_t, x_t, z_t = _sample_pop(n, jax.random.PRNGKey(202), W_TARGET)
    w = _true_w(y_s, z_s)
    # E1[w] = 1 when w = p0/p1. A large departure would mean the ratio is wrong.
    assert abs(float(w.mean()) - 1.0) < 0.03

    q_right = GaussianConditional(B=B_TRUE, Sigma=SIGMA_TRUE)
    q_wrong = GaussianConditional(B=B_WRONG, Sigma=SIGMA_WRONG)
    omega_wrong = jnp.ones(n)

    score_cw = _dr_score(y_s, x_s, z_s, y_t, z_t, beta0, w, q_wrong)
    score_ww = _dr_score(y_s, x_s, z_s, y_t, z_t, beta0, omega_wrong, q_right)
    assert float(jnp.linalg.norm(score_cw)) < 0.05
    assert float(jnp.linalg.norm(score_ww)) < 0.02

    # Start at zero so recovery is the solver, not a warm start at beta0.
    dr_cw = fit_dr(
        y_s, x_s, z_s, y_t, z_t, w, q_wrong, IDENTITY, beta_init=jnp.zeros(3)
    )
    dr_ww = fit_dr(
        y_s, x_s, z_s, y_t, z_t, omega_wrong, q_right, IDENTITY, beta_init=jnp.zeros(3)
    )
    assert float(dr_cw.score_norm) < 1e-8
    assert float(dr_ww.score_norm) < 1e-8
    assert jnp.allclose(dr_cw.beta, beta0, atol=0.04)
    assert jnp.allclose(dr_ww.beta, beta0, atol=0.03)

    # Both nuisances wrong: the DR equation need not recover beta0.
    # This design does not; the check is that we do not treat that as success.
    dr_neither = fit_dr(
        y_s,
        x_s,
        z_s,
        y_t,
        z_t,
        omega_wrong,
        q_wrong,
        IDENTITY,
        beta_init=jnp.zeros(3),
    )
    assert float(jnp.max(jnp.abs(dr_neither.beta - beta0))) > 0.4


def test_single_nuisance_failure_identity():
    beta0 = _beta0()
    n = 80_000
    y_s, x_s, z_s = _sample_pop(n, jax.random.PRNGKey(301), W_SOURCE)
    y_t, _, z_t = _sample_pop(n, jax.random.PRNGKey(302), W_TARGET)
    q_wrong = GaussianConditional(B=B_WRONG, Sigma=SIGMA_WRONG)

    iw = fit_iw(y_s, x_s, z_s, jnp.ones(n), IDENTITY, beta_init=jnp.zeros(3))
    imp = fit_imp(y_t, z_t, q_wrong, IDENTITY, beta_init=jnp.zeros(3))
    # Source GLM with omega ≡ 1 is not the target coefficient.
    iw_err = float(jnp.max(jnp.abs(iw.beta - beta0)))
    imp_err = float(jnp.max(jnp.abs(imp.beta - beta0)))
    assert iw_err > 0.06
    assert imp_err > 0.4
    # And the failure is not "the solver returned the start".
    assert float(iw.score_norm) < 1e-8
    assert float(imp.score_norm) < 1e-8
    assert float(jnp.linalg.norm(iw.beta)) > 0.2
    assert float(jnp.linalg.norm(imp.beta)) > 0.2


def test_identity_glm_matches_ols():
    y, x, z = _sample_pop(5000, jax.random.PRNGKey(9), W_TARGET)
    beta_ols, _, _, _ = jnp.linalg.lstsq(jnp.concatenate([x, z], 1), y, rcond=None)
    fit = fit_glm(y, x, z, IDENTITY, beta_init=jnp.zeros(3))
    assert float(fit.score_norm) < 1e-8
    assert jnp.allclose(fit.beta, beta_ols, atol=1e-6)
