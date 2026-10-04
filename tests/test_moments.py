"""Identity-link conditional functionals match a Monte Carlo average under q."""

import jax
import jax.numpy as jnp
import pytest

from drtl_cmc.links import IDENTITY, LOGISTIC
from drtl_cmc.nuisance import GaussianConditional
from drtl_cmc.scores import identity_moments, mc_moments


def test_identity_moments_match_monte_carlo():
    # p=1 and p=2, fixed (y, z, beta), not a shape check.
    B = jnp.array(
        [
            [0.40, -0.20],
            [1.00, 0.30],
            [-0.60, 0.10],
            [0.20, 0.50],
        ]
    )
    Sigma = jnp.array([[0.25, 0.05], [0.05, 0.16]])
    conditional = GaussianConditional(B=B, Sigma=Sigma)
    y = jnp.array([0.0, 1.0, 0.5, -0.3])
    z = jnp.array(
        [
            [1.0, 0.2, -0.3],
            [1.0, -1.0, 0.4],
            [1.0, 0.5, 0.5],
            [1.0, -0.2, 1.1],
        ]
    )
    beta = jnp.array([1.2, -0.7, -0.5, 0.3, 0.8])
    m1, m2, m3 = identity_moments(y, z, beta, conditional)

    # m1 is the Gaussian mean, not an average.
    feat = jnp.concatenate([y[:, None], z], axis=1)
    assert jnp.allclose(m1, feat @ B, atol=1e-12)

    key = jax.random.PRNGKey(7)
    _, m2_mc, m3_mc = mc_moments(
        y, z, beta, conditional, IDENTITY, key, K=200_000
    )
    assert jnp.allclose(m2_mc, m2, atol=2e-2)
    assert jnp.allclose(m3_mc, m3, atol=2e-2)

    # Scalar-X formula used by the simulation: m3 = (sigma^2 + m1^2) beta_x + m1 (z' beta_z).
    B1 = B[:, :1]
    Sigma1 = Sigma[:1, :1]
    beta1 = jnp.concatenate([beta[:1], beta[2:]])
    cond1 = GaussianConditional(B=B1, Sigma=Sigma1)
    m1_1, m2_1, m3_1 = identity_moments(y, z, beta1, cond1)
    z_lin = z @ beta1[1:]
    m3_manual = (Sigma1[0, 0] + m1_1[:, 0] ** 2) * beta1[0] + m1_1[:, 0] * z_lin
    assert jnp.allclose(m3_1[:, 0], m3_manual, atol=1e-12)
    assert jnp.allclose(m2_1, m1_1[:, 0] * beta1[0] + z_lin, atol=1e-12)


def test_logistic_mc_depends_on_beta():
    """A smoke check that logistic m2 is not the identity closed form."""
    conditional = GaussianConditional(
        B=jnp.array([[0.5], [0.2], [-0.3]]),
        Sigma=jnp.array([[0.25]]),
    )
    y = jnp.array([0.0, 1.0])
    z = jnp.array([[1.0, 0.4], [1.0, -0.2]])
    beta = jnp.array([1.5, -0.4, 0.3])
    m1_id, m2_id, _ = identity_moments(y, z, beta, conditional)
    _, m2_mc, _ = mc_moments(
        y, z, beta, conditional, LOGISTIC, jax.random.PRNGKey(1), K=20_000
    )
    assert jnp.allclose(m1_id, jnp.concatenate([y[:, None], z], 1) @ conditional.B)
    assert float(jnp.max(jnp.abs(m2_mc - m2_id))) > 0.05
    assert jnp.all((m2_mc > 0.0) & (m2_mc < 1.0))


@pytest.mark.parametrize("p", [1, 2])
def test_identity_jacobian_matches_finite_difference(p):
    from drtl_cmc.scores import identity_imputed_score_and_jac

    q = 3
    key = jax.random.PRNGKey(3)
    B = jax.random.normal(key, (1 + q, p))
    A = jax.random.normal(jax.random.PRNGKey(4), (p, p))
    Sigma = A @ A.T / p + 0.2 * jnp.eye(p)
    conditional = GaussianConditional(B=B, Sigma=Sigma)
    y = jnp.array([0.2, 1.0, -0.4, 0.7])
    z = jnp.array(
        [
            [1.0, 0.3, -0.2],
            [1.0, -0.5, 0.4],
            [1.0, 0.1, 0.8],
            [1.0, -1.0, -0.3],
        ]
    )
    beta = jnp.linspace(-0.4, 0.6, p + q)

    phi0, jac = identity_imputed_score_and_jac(y, z, beta, conditional)
    eps = 1e-6

    def mean_score(b):
        return identity_imputed_score_and_jac(y, z, b, conditional)[0].mean(0)

    cols = []
    for j in range(beta.size):
        cols.append((mean_score(beta.at[j].add(eps)) - mean_score(beta)) / eps)
    jac_fd = jnp.stack(cols, axis=1)
    assert jnp.allclose(jac.mean(0), jac_fd, atol=1e-5)
    assert jnp.all(jnp.isfinite(phi0))
